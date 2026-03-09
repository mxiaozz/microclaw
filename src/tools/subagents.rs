use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use tokio::sync::Semaphore;
use tracing::{info, warn};

use super::{
    auth_context_from_input, authorize_chat_access, schema_object, Tool, ToolAuthContext,
    ToolRegistry, ToolResult,
};
use crate::config::Config;
use microclaw_channels::channel::deliver_and_store_bot_message;
use microclaw_channels::channel_adapter::ChannelRegistry;
use microclaw_core::llm_types::{
    ContentBlock, Message, MessageContent, ResponseContentBlock, ToolDefinition,
};
use microclaw_storage::db::{call_blocking, Database};

const MAX_SUB_AGENT_ITERATIONS: usize = 16;

#[derive(Debug, Clone, Copy)]
struct SubagentRuntimeMeta {
    depth: i64,
}

fn subagent_runtime_meta_from_input(input: &serde_json::Value) -> Option<SubagentRuntimeMeta> {
    let meta = input.get("__subagent_runtime")?;
    let depth = meta.get("depth").and_then(|v| v.as_i64()).unwrap_or(0);
    Some(SubagentRuntimeMeta { depth })
}

struct SubagentRuntime {
    semaphore: Semaphore,
    cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl SubagentRuntime {
    fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Semaphore::new(max_concurrent.max(1)),
            cancel_flags: Mutex::new(HashMap::new()),
        }
    }

    fn register_run(&self, run_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut guard) = self.cancel_flags.lock() {
            guard.insert(run_id.to_string(), flag.clone());
        }
        flag
    }

    fn cancel_run(&self, run_id: &str) {
        if let Ok(guard) = self.cancel_flags.lock() {
            if let Some(flag) = guard.get(run_id) {
                flag.store(true, Ordering::Relaxed);
            }
        }
    }

    fn remove_run(&self, run_id: &str) {
        if let Ok(mut guard) = self.cancel_flags.lock() {
            guard.remove(run_id);
        }
    }
}

static RUNTIME: LazyLock<Mutex<Option<Arc<SubagentRuntime>>>> = LazyLock::new(|| Mutex::new(None));

fn subagent_runtime(config: &Config) -> Arc<SubagentRuntime> {
    let mut guard = match RUNTIME.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(existing) = guard.as_ref() {
        return existing.clone();
    }
    let runtime = Arc::new(SubagentRuntime::new(config.subagents.max_concurrent));
    *guard = Some(runtime.clone());
    runtime
}

async fn log_subagent_event(
    db: Arc<Database>,
    run_id: &str,
    event_type: &str,
    detail: Option<String>,
) {
    let run_id = run_id.to_string();
    let event_type = event_type.to_string();
    let _ = call_blocking(db, move |db| {
        db.append_subagent_event(&run_id, &event_type, detail.as_deref())
    })
    .await;
}

async fn is_cancelled(
    db: Arc<Database>,
    run_id: &str,
    local_flag: &Arc<AtomicBool>,
) -> Result<bool, String> {
    if local_flag.load(Ordering::Relaxed) {
        return Ok(true);
    }
    let run_id_owned = run_id.to_string();
    let db_cancel = call_blocking(db, move |db| db.is_subagent_cancel_requested(&run_id_owned))
        .await
        .map_err(|e| format!("Failed checking cancel state: {e}"))?;
    Ok(db_cancel)
}

async fn run_sub_agent_task(
    config: Config,
    db: Arc<Database>,
    channel_registry: Arc<ChannelRegistry>,
    auth_context: ToolAuthContext,
    run_id: String,
    depth: i64,
    task: String,
    context: String,
    local_cancel: Arc<AtomicBool>,
) -> Result<(String, i64, i64), String> {
    let llm = crate::llm::create_provider(&config);
    let allow_session_tools = depth < config.subagents.max_spawn_depth as i64;
    let tools = ToolRegistry::new_sub_agent(
        &config,
        db.clone(),
        Some(channel_registry),
        allow_session_tools,
    );
    let tool_defs = tools.definitions().to_vec();

    let system_prompt = "You are a sub-agent assistant. Complete the given task thoroughly and return a clear, concise result. You have access to tools for file operations, search, and web access. Focus on the task and provide actionable output.".to_string();

    let user_content = if context.is_empty() {
        task.to_string()
    } else {
        format!("Context: {context}\n\nTask: {task}")
    };

    let mut messages = vec![Message {
        role: "user".into(),
        content: MessageContent::Text(user_content),
    }];
    let mut input_tokens_sum = 0_i64;
    let mut output_tokens_sum = 0_i64;

    for _ in 0..MAX_SUB_AGENT_ITERATIONS {
        if is_cancelled(db.clone(), &run_id, &local_cancel).await? {
            return Err("cancelled".into());
        }

        let response = llm
            .send_message(&system_prompt, messages.clone(), Some(tool_defs.clone()))
            .await
            .map_err(|e| format!("Sub-agent API error: {e}"))?;

        if let Some(usage) = &response.usage {
            let input_tokens = i64::from(usage.input_tokens);
            let output_tokens = i64::from(usage.output_tokens);
            input_tokens_sum += input_tokens;
            output_tokens_sum += output_tokens;

            let channel = auth_context.caller_channel.clone();
            let provider = config.llm_provider.clone();
            let model = config.model.clone();
            let chat_id = auth_context.caller_chat_id;
            let _ = call_blocking(db.clone(), move |db| {
                db.log_llm_usage(
                    chat_id,
                    &channel,
                    &provider,
                    &model,
                    input_tokens,
                    output_tokens,
                    "subagent_run",
                )
                .map(|_| ())
            })
            .await;
        }

        let stop_reason = response.stop_reason.as_deref().unwrap_or("end_turn");
        if stop_reason == "end_turn" || stop_reason == "max_tokens" {
            let text = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ResponseContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            let final_text = if text.is_empty() {
                "(sub-agent produced no output)".to_string()
            } else {
                text
            };
            return Ok((final_text, input_tokens_sum, output_tokens_sum));
        }

        if stop_reason == "tool_use" {
            let assistant_content: Vec<ContentBlock> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ResponseContentBlock::Text { text } => {
                        Some(ContentBlock::Text { text: text.clone() })
                    }
                    ResponseContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        thought_signature,
                    } => Some(ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                        thought_signature: thought_signature.clone(),
                    }),
                    ResponseContentBlock::Other => None,
                })
                .collect();

            messages.push(Message {
                role: "assistant".into(),
                content: MessageContent::Blocks(assistant_content),
            });

            let mut tool_results = Vec::new();
            for block in &response.content {
                if let ResponseContentBlock::ToolUse {
                    id, name, input, ..
                } = block
                {
                    log_subagent_event(
                        db.clone(),
                        &run_id,
                        "tool_use",
                        Some(format!("tool={name}")),
                    )
                    .await;
                    let mut tool_input = input.clone();
                    if let Some(obj) = tool_input.as_object_mut() {
                        obj.insert(
                            "__subagent_runtime".to_string(),
                            json!({
                                "run_id": run_id.clone(),
                                "depth": depth,
                            }),
                        );
                    }
                    let result = tools
                        .execute_with_auth(name, tool_input, &auth_context)
                        .await;
                    tool_results.push(ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: result.content,
                        is_error: if result.is_error { Some(true) } else { None },
                    });
                }
            }

            messages.push(Message {
                role: "user".into(),
                content: MessageContent::Blocks(tool_results),
            });
            continue;
        }

        let text = response
            .content
            .iter()
            .filter_map(|block| match block {
                ResponseContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let final_text = if text.is_empty() {
            "(sub-agent produced no output)".to_string()
        } else {
            text
        };
        return Ok((final_text, input_tokens_sum, output_tokens_sum));
    }

    Err("Sub-agent reached maximum iterations without completing the task.".into())
}

async fn build_announce_payload(
    db: Arc<Database>,
    chat_id: i64,
    run_id: &str,
) -> Result<String, String> {
    let run_id_owned = run_id.to_string();
    let run = match call_blocking(db.clone(), move |db| {
        db.get_subagent_run(&run_id_owned, chat_id)
    })
    .await
    {
        Ok(Some(run)) => run,
        Ok(None) => return Err("run_not_found".into()),
        Err(e) => return Err(format!("failed_loading_run: {e}")),
    };

    let status_emoji = match run.status.as_str() {
        "completed" => "✅",
        "cancelled" => "🛑",
        "timed_out" => "⏱️",
        _ => "❌",
    };

    let mut text = format!(
        "{status_emoji} Subagent `{}` finished\nstatus: {}\ninput_tokens: {}\noutput_tokens: {}",
        run.run_id, run.status, run.input_tokens, run.output_tokens
    );
    if let Some(err) = &run.error_text {
        text.push_str(&format!("\nerror: {err}"));
    }
    if let Some(result) = &run.result_text {
        let clipped: String = result.chars().take(2400).collect();
        text.push_str("\nresult:\n");
        text.push_str(&clipped);
    }
    Ok(text)
}

async fn flush_pending_announces(
    config: &Config,
    channel_registry: Arc<ChannelRegistry>,
    db: Arc<Database>,
    max_batch: usize,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let rows = match call_blocking(db.clone(), move |db| {
        db.list_due_subagent_announces(&now, max_batch)
    })
    .await
    {
        Ok(v) => v,
        Err(e) => {
            warn!("failed to list due subagent announces: {e}");
            return;
        }
    };

    for row in rows {
        let bot_username = config.bot_username_for_channel(&row.caller_channel);
        let delivery = deliver_and_store_bot_message(
            channel_registry.as_ref(),
            db.clone(),
            &bot_username,
            row.chat_id,
            &row.payload_text,
        )
        .await;
        match delivery {
            Ok(_) => {
                let id = row.id;
                let _ =
                    call_blocking(db.clone(), move |db| db.mark_subagent_announce_sent(id)).await;
            }
            Err(err) => {
                let next_attempts = row.attempts + 1;
                let terminal = next_attempts >= 5;
                let delay_secs = (1_i64 << next_attempts.min(6)) as i64;
                let next_at = if terminal {
                    None
                } else {
                    Some((chrono::Utc::now() + chrono::Duration::seconds(delay_secs)).to_rfc3339())
                };
                let id = row.id;
                let err_text = err;
                let _ = call_blocking(db.clone(), move |db| {
                    db.mark_subagent_announce_retry(
                        id,
                        next_attempts,
                        next_at.as_deref(),
                        &err_text,
                        terminal,
                    )
                })
                .await;
            }
        }
    }
}

pub struct SessionsSpawnTool {
    config: Config,
    db: Arc<Database>,
    channel_registry: Arc<ChannelRegistry>,
}

impl SessionsSpawnTool {
    pub fn new(config: &Config, db: Arc<Database>, channel_registry: Arc<ChannelRegistry>) -> Self {
        Self {
            config: config.clone(),
            db,
            channel_registry,
        }
    }
}

#[async_trait]
impl Tool for SessionsSpawnTool {
    fn name(&self) -> &str {
        "sessions_spawn"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "sessions_spawn".into(),
            description: "Spawn an asynchronous sub-agent run for long tasks. Returns immediately with a run id. Use subagents_list/subagents_info/subagents_kill to manage runs.".into(),
            input_schema: schema_object(
                json!({
                    "task": {
                        "type": "string",
                        "description": "Task for the spawned sub-agent"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional extra context passed to the sub-agent"
                    },
                    "chat_id": {
                        "type": "integer",
                        "description": "Target chat id. Defaults to current chat."
                    }
                }),
                &["task"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth =
            match auth_context_from_input(&input) {
                Some(v) => v,
                None => return ToolResult::error(
                    "sessions_spawn requires caller auth context; run from an active chat session"
                        .into(),
                ),
            };

        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }

        let task = match input.get("task").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: task".into()),
        };
        let context = input
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let parent_meta = subagent_runtime_meta_from_input(&input);
        let parent_depth = parent_meta.map(|m| m.depth).unwrap_or(0);
        let child_depth = if parent_depth > 0 {
            parent_depth + 1
        } else {
            1
        };
        if child_depth as usize > self.config.subagents.max_spawn_depth {
            return ToolResult::error(format!(
                "subagent spawn depth exceeded: requested depth {}, max {}",
                child_depth, self.config.subagents.max_spawn_depth
            ));
        }
        let parent_run_id = input
            .get("__subagent_runtime")
            .and_then(|v| v.get("run_id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let db_for_count = self.db.clone();
        let active_count = match call_blocking(db_for_count, move |db| {
            db.count_active_subagent_runs_for_chat(chat_id)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(format!("Failed checking active subagent runs: {e}"));
            }
        };
        if active_count as usize >= self.config.subagents.max_active_per_chat {
            return ToolResult::error(format!(
                "Too many active subagent runs for this chat (limit: {})",
                self.config.subagents.max_active_per_chat
            ));
        }
        if let Some(parent_id) = parent_run_id.as_ref() {
            let parent_id_for_count = parent_id.clone();
            let active_children = match call_blocking(self.db.clone(), move |db| {
                db.count_active_subagent_children(&parent_id_for_count)
            })
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    return ToolResult::error(format!(
                        "Failed checking active subagent child runs: {e}"
                    ));
                }
            };
            if active_children as usize >= self.config.subagents.max_children_per_run {
                return ToolResult::error(format!(
                    "Too many active child runs for this parent (limit: {})",
                    self.config.subagents.max_children_per_run
                ));
            }
        }

        let run_id = format!("subrun-{}", uuid::Uuid::new_v4());
        let provider = self.config.llm_provider.clone();
        let model = self.config.model.clone();

        let run_id_for_insert = run_id.clone();
        let task_for_insert = task.clone();
        let context_for_insert = context.clone();
        let caller_channel_for_insert = auth.caller_channel.clone();
        let parent_for_insert = parent_run_id.clone();
        if let Err(e) = call_blocking(self.db.clone(), move |db| {
            db.create_subagent_run(
                &run_id_for_insert,
                parent_for_insert.as_deref(),
                child_depth,
                chat_id,
                &caller_channel_for_insert,
                &task_for_insert,
                &context_for_insert,
                &provider,
                &model,
            )
        })
        .await
        {
            return ToolResult::error(format!("Failed creating subagent run: {e}"));
        }
        log_subagent_event(
            self.db.clone(),
            &run_id,
            "accepted",
            Some(format!("depth={child_depth}")),
        )
        .await;

        let runtime = subagent_runtime(&self.config);
        let local_cancel = runtime.register_run(&run_id);
        let db = self.db.clone();
        let cfg = self.config.clone();
        let run_id_async = run_id.clone();
        let task_async = task.clone();
        let context_async = context.clone();
        let auth_async = ToolAuthContext {
            caller_channel: auth.caller_channel.clone(),
            caller_chat_id: chat_id,
            control_chat_ids: auth.control_chat_ids.clone(),
            env_files: auth.env_files.clone(),
        };
        let channel_registry = self.channel_registry.clone();
        let subagent_channel_registry = self.channel_registry.clone();
        tokio::spawn(async move {
            let run_id_for_finish = run_id_async.clone();
            let _ = call_blocking(db.clone(), {
                let run_id = run_id_async.clone();
                move |db| db.mark_subagent_queued(&run_id)
            })
            .await;
            log_subagent_event(db.clone(), &run_id_async, "queued", None).await;

            let _permit = match runtime.semaphore.acquire().await {
                Ok(p) => p,
                Err(_) => {
                    let _ = call_blocking(db.clone(), move |db| {
                        db.mark_subagent_finished(
                            &run_id_for_finish,
                            "failed",
                            Some("subagent runtime is shutting down"),
                            None,
                            0,
                            0,
                        )
                    })
                    .await;
                    runtime.remove_run(&run_id_async);
                    return;
                }
            };

            let _ = call_blocking(db.clone(), {
                let run_id = run_id_async.clone();
                move |db| db.mark_subagent_running(&run_id)
            })
            .await;
            log_subagent_event(db.clone(), &run_id_async, "running", None).await;

            let timeout_secs = cfg.subagents.run_timeout_secs;
            let run_future = run_sub_agent_task(
                cfg.clone(),
                db.clone(),
                subagent_channel_registry,
                auth_async,
                run_id_async.clone(),
                child_depth,
                task_async,
                context_async,
                local_cancel,
            );

            let final_outcome = if timeout_secs > 0 {
                match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), run_future)
                    .await
                {
                    Ok(result) => result,
                    Err(_) => Err("timed_out".to_string()),
                }
            } else {
                run_future.await
            };

            match final_outcome {
                Ok((result, input_tokens, output_tokens)) => {
                    let rid = run_id_for_finish.clone();
                    let _ = call_blocking(db.clone(), move |db| {
                        db.mark_subagent_finished(
                            &rid,
                            "completed",
                            None,
                            Some(&result),
                            input_tokens,
                            output_tokens,
                        )
                    })
                    .await;
                    log_subagent_event(db.clone(), &run_id_for_finish, "completed", None).await;
                }
                Err(err) if err == "cancelled" => {
                    let rid = run_id_for_finish.clone();
                    let _ = call_blocking(db.clone(), move |db| {
                        db.mark_subagent_finished(
                            &rid,
                            "cancelled",
                            Some("Cancelled by user"),
                            None,
                            0,
                            0,
                        )
                    })
                    .await;
                    log_subagent_event(db.clone(), &run_id_for_finish, "cancelled", None).await;
                }
                Err(err) if err == "timed_out" => {
                    let rid = run_id_for_finish.clone();
                    let _ = call_blocking(db.clone(), move |db| {
                        db.mark_subagent_finished(
                            &rid,
                            "timed_out",
                            Some("Sub-agent run exceeded configured timeout"),
                            None,
                            0,
                            0,
                        )
                    })
                    .await;
                    log_subagent_event(db.clone(), &run_id_for_finish, "timed_out", None).await;
                }
                Err(err) => {
                    let rid = run_id_for_finish.clone();
                    let err_for_db = err.clone();
                    let _ = call_blocking(db.clone(), move |db| {
                        db.mark_subagent_finished(&rid, "failed", Some(&err_for_db), None, 0, 0)
                    })
                    .await;
                    log_subagent_event(db.clone(), &run_id_for_finish, "failed", Some(err)).await;
                }
            }

            runtime.remove_run(&run_id_async);

            if cfg.subagents.announce_to_chat {
                match build_announce_payload(db.clone(), chat_id, &run_id_async).await {
                    Ok(payload) => {
                        let rid = run_id_async.clone();
                        let caller_channel = auth.caller_channel.clone();
                        let _ = call_blocking(db.clone(), move |db| {
                            db.enqueue_subagent_announce(&rid, chat_id, &caller_channel, &payload)
                        })
                        .await;
                        flush_pending_announces(&cfg, channel_registry, db, 10).await;
                    }
                    Err(e) => {
                        warn!("failed to build announce payload for run {run_id_async}: {e}");
                    }
                }
            }
        });

        info!("subagent accepted run_id={run_id} chat_id={chat_id}");
        ToolResult::success(
            json!({
                "status": "accepted",
                "run_id": run_id,
                "chat_id": chat_id,
                "depth": child_depth,
                "parent_run_id": parent_run_id,
            })
            .to_string(),
        )
    }
}

pub struct SubagentsListTool {
    db: Arc<Database>,
}

impl SubagentsListTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsListTool {
    fn name(&self) -> &str {
        "subagents_list"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_list".into(),
            description: "List recent subagent runs for the current chat.".into(),
            input_schema: schema_object(
                json!({
                    "chat_id": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                }),
                &[],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth =
            match auth_context_from_input(&input) {
                Some(v) => v,
                None => return ToolResult::error(
                    "subagents_list requires caller auth context; run from an active chat session"
                        .into(),
                ),
            };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .clamp(1, 100) as usize;

        let rows = match call_blocking(self.db.clone(), move |db| {
            db.list_subagent_runs(chat_id, limit)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Failed listing subagent runs: {e}")),
        };
        let payload: Vec<serde_json::Value> = rows
            .into_iter()
            .map(|r| {
                json!({
                    "run_id": r.run_id,
                    "parent_run_id": r.parent_run_id,
                    "depth": r.depth,
                    "status": r.status,
                    "created_at": r.created_at,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "cancel_requested": r.cancel_requested,
                    "task": r.task,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                })
            })
            .collect();

        ToolResult::success(json!({"chat_id": chat_id, "runs": payload}).to_string())
    }
}

pub struct SubagentsInfoTool {
    db: Arc<Database>,
}

impl SubagentsInfoTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsInfoTool {
    fn name(&self) -> &str {
        "subagents_info"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_info".into(),
            description: "Get detailed information for one subagent run.".into(),
            input_schema: schema_object(
                json!({
                    "run_id": {"type": "string"},
                    "chat_id": {"type": "integer"}
                }),
                &["run_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth =
            match auth_context_from_input(&input) {
                Some(v) => v,
                None => return ToolResult::error(
                    "subagents_info requires caller auth context; run from an active chat session"
                        .into(),
                ),
            };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let run_id = match input.get("run_id").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: run_id".into()),
        };

        let run = match call_blocking(self.db.clone(), move |db| {
            db.get_subagent_run(&run_id, chat_id)
        })
        .await
        {
            Ok(Some(v)) => v,
            Ok(None) => return ToolResult::error("Subagent run not found".into()),
            Err(e) => return ToolResult::error(format!("Failed reading subagent run: {e}")),
        };

        ToolResult::success(
            json!({
                "run_id": run.run_id,
                "parent_run_id": run.parent_run_id,
                "depth": run.depth,
                "chat_id": run.chat_id,
                "caller_channel": run.caller_channel,
                "task": run.task,
                "context": run.context,
                "status": run.status,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "cancel_requested": run.cancel_requested,
                "error_text": run.error_text,
                "result_text": run.result_text,
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "total_tokens": run.total_tokens,
                "provider": run.provider,
                "model": run.model,
            })
            .to_string(),
        )
    }
}

pub struct SubagentsKillTool {
    config: Config,
    db: Arc<Database>,
}

impl SubagentsKillTool {
    pub fn new(config: &Config, db: Arc<Database>) -> Self {
        Self {
            config: config.clone(),
            db,
        }
    }
}

#[async_trait]
impl Tool for SubagentsKillTool {
    fn name(&self) -> &str {
        "subagents_kill"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_kill".into(),
            description: "Request cancellation for one running subagent run, or all active runs in current chat with run_id=all.".into(),
            input_schema: schema_object(
                json!({
                    "run_id": {"type": "string", "description": "Run id or 'all'"},
                    "chat_id": {"type": "integer"}
                }),
                &["run_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth =
            match auth_context_from_input(&input) {
                Some(v) => v,
                None => return ToolResult::error(
                    "subagents_kill requires caller auth context; run from an active chat session"
                        .into(),
                ),
            };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let run_id = match input.get("run_id").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: run_id".into()),
        };

        let runtime = subagent_runtime(&self.config);

        if run_id.eq_ignore_ascii_case("all") {
            let rows = match call_blocking(self.db.clone(), move |db| {
                db.list_subagent_runs(chat_id, 200)
            })
            .await
            {
                Ok(v) => v,
                Err(e) => return ToolResult::error(format!("Failed listing subagent runs: {e}")),
            };
            let mut cancelled = 0usize;
            for row in rows {
                if matches!(row.status.as_str(), "accepted" | "queued" | "running") {
                    let rid = row.run_id.clone();
                    let requested = call_blocking(self.db.clone(), move |db| {
                        db.request_subagent_cancel(&rid, chat_id)
                    })
                    .await
                    .unwrap_or(false);
                    if requested {
                        runtime.cancel_run(&row.run_id);
                        log_subagent_event(
                            self.db.clone(),
                            &row.run_id,
                            "cancel_requested",
                            Some("kill_all".to_string()),
                        )
                        .await;
                        cancelled += 1;
                    }
                }
            }
            return ToolResult::success(
                json!({"status": "ok", "cancelled": cancelled, "chat_id": chat_id}).to_string(),
            );
        }

        let run_id_for_db = run_id.clone();
        let requested = match call_blocking(self.db.clone(), move |db| {
            db.request_subagent_cancel(&run_id_for_db, chat_id)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(format!("Failed requesting cancellation: {e}"));
            }
        };

        if !requested {
            return ToolResult::error("Subagent run not found or already finished".into());
        }
        runtime.cancel_run(&run_id);
        log_subagent_event(
            self.db.clone(),
            &run_id,
            "cancel_requested",
            Some("kill_one".to_string()),
        )
        .await;
        ToolResult::success(json!({"status": "ok", "run_id": run_id}).to_string())
    }
}

pub struct SubagentsRetryAnnouncesTool {
    config: Config,
    db: Arc<Database>,
    channel_registry: Arc<ChannelRegistry>,
}

pub struct SubagentsFocusTool {
    db: Arc<Database>,
}

impl SubagentsFocusTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsFocusTool {
    fn name(&self) -> &str {
        "subagents_focus"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_focus".into(),
            description: "Bind the current chat to a subagent run for follow-up actions.".into(),
            input_schema: schema_object(
                json!({
                    "run_id": {"type":"string"},
                    "chat_id": {"type":"integer"}
                }),
                &["run_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => {
                return ToolResult::error("subagents_focus requires caller auth context".into())
            }
        };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let run_id = match input.get("run_id").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: run_id".into()),
        };
        let run_id_for_check = run_id.clone();
        let exists = match call_blocking(self.db.clone(), move |db| {
            db.get_subagent_run(&run_id_for_check, chat_id)
        })
        .await
        {
            Ok(v) => v.is_some(),
            Err(e) => return ToolResult::error(format!("Failed reading subagent run: {e}")),
        };
        if !exists {
            return ToolResult::error("Subagent run not found".into());
        }
        let run_id_for_set = run_id.clone();
        if let Err(e) = call_blocking(self.db.clone(), move |db| {
            db.set_subagent_focus(chat_id, &run_id_for_set)
        })
        .await
        {
            return ToolResult::error(format!("Failed setting subagent focus: {e}"));
        }
        ToolResult::success(json!({"status":"ok","chat_id":chat_id,"run_id":run_id}).to_string())
    }
}

pub struct SubagentsUnfocusTool {
    db: Arc<Database>,
}

impl SubagentsUnfocusTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsUnfocusTool {
    fn name(&self) -> &str {
        "subagents_unfocus"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_unfocus".into(),
            description: "Clear focused subagent binding for the current chat.".into(),
            input_schema: schema_object(
                json!({
                    "chat_id": {"type":"integer"}
                }),
                &[],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => {
                return ToolResult::error("subagents_unfocus requires caller auth context".into())
            }
        };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        if let Err(e) =
            call_blocking(self.db.clone(), move |db| db.clear_subagent_focus(chat_id)).await
        {
            return ToolResult::error(format!("Failed clearing subagent focus: {e}"));
        }
        ToolResult::success(json!({"status":"ok","chat_id":chat_id}).to_string())
    }
}

pub struct SubagentsFocusedTool {
    db: Arc<Database>,
}

impl SubagentsFocusedTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsFocusedTool {
    fn name(&self) -> &str {
        "subagents_focused"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_focused".into(),
            description: "Show focused subagent binding for current chat.".into(),
            input_schema: schema_object(
                json!({
                    "chat_id": {"type":"integer"}
                }),
                &[],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => {
                return ToolResult::error("subagents_focused requires caller auth context".into())
            }
        };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let focused =
            match call_blocking(self.db.clone(), move |db| db.get_subagent_focus(chat_id)).await {
                Ok(v) => v,
                Err(e) => return ToolResult::error(format!("Failed loading subagent focus: {e}")),
            };
        ToolResult::success(json!({"chat_id":chat_id,"run_id":focused}).to_string())
    }
}

pub struct SubagentsSendTool {
    config: Config,
    db: Arc<Database>,
    channel_registry: Arc<ChannelRegistry>,
}

impl SubagentsSendTool {
    pub fn new(config: &Config, db: Arc<Database>, channel_registry: Arc<ChannelRegistry>) -> Self {
        Self {
            config: config.clone(),
            db,
            channel_registry,
        }
    }
}

#[async_trait]
impl Tool for SubagentsSendTool {
    fn name(&self) -> &str {
        "subagents_send"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_send".into(),
            description:
                "Send follow-up work to focused subagent by spawning a child continuation run."
                    .into(),
            input_schema: schema_object(
                json!({
                    "message": {"type":"string"},
                    "chat_id": {"type":"integer"}
                }),
                &["message"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => return ToolResult::error("subagents_send requires caller auth context".into()),
        };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let message = match input.get("message").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: message".into()),
        };
        let focused_run =
            match call_blocking(self.db.clone(), move |db| db.get_subagent_focus(chat_id)).await {
                Ok(Some(v)) => v,
                Ok(None) => return ToolResult::error("No focused subagent for this chat".into()),
                Err(e) => return ToolResult::error(format!("Failed loading subagent focus: {e}")),
            };
        let focused_run_for_load = focused_run.clone();
        let parent = match call_blocking(self.db.clone(), move |db| {
            db.get_subagent_run(&focused_run_for_load, chat_id)
        })
        .await
        {
            Ok(Some(v)) => v,
            Ok(None) => return ToolResult::error("Focused subagent run not found".into()),
            Err(e) => return ToolResult::error(format!("Failed loading focused subagent: {e}")),
        };

        let spawn_tool =
            SessionsSpawnTool::new(&self.config, self.db.clone(), self.channel_registry.clone());
        let spawn_input = json!({
            "task": format!("Continuation request: {message}"),
            "context": format!("This is a follow-up sent to focused run {}. Continue the work based on prior run context and produce actionable output.", focused_run),
            "__microclaw_auth": {
                "caller_channel": auth.caller_channel,
                "caller_chat_id": chat_id,
                "control_chat_ids": auth.control_chat_ids,
                "env_files": auth.env_files,
            },
            "__subagent_runtime": {
                "run_id": parent.run_id,
                "depth": parent.depth,
            }
        });
        spawn_tool.execute(spawn_input).await
    }
}

pub struct SubagentsLogTool {
    db: Arc<Database>,
}

impl SubagentsLogTool {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SubagentsLogTool {
    fn name(&self) -> &str {
        "subagents_log"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_log".into(),
            description: "Get timeline events for one subagent run.".into(),
            input_schema: schema_object(
                json!({
                    "run_id": {"type":"string"},
                    "chat_id": {"type":"integer"},
                    "limit": {"type":"integer", "minimum":1, "maximum":200}
                }),
                &["run_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => return ToolResult::error("subagents_log requires caller auth context".into()),
        };
        let chat_id = input
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(auth.caller_chat_id);
        if let Err(e) = authorize_chat_access(&input, chat_id) {
            return ToolResult::error(e);
        }
        let run_id = match input.get("run_id").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return ToolResult::error("Missing required parameter: run_id".into()),
        };
        let run_id_for_check = run_id.clone();
        let run_exists = match call_blocking(self.db.clone(), move |db| {
            db.get_subagent_run(&run_id_for_check, chat_id)
        })
        .await
        {
            Ok(v) => v.is_some(),
            Err(e) => return ToolResult::error(format!("Failed reading subagent run: {e}")),
        };
        if !run_exists {
            return ToolResult::error("Subagent run not found".into());
        }
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50)
            .clamp(1, 200) as usize;
        let run_id_for_events = run_id.clone();
        let events = match call_blocking(self.db.clone(), move |db| {
            db.list_subagent_events(&run_id_for_events, limit)
        })
        .await
        {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Failed listing subagent events: {e}")),
        };
        let payload: Vec<serde_json::Value> = events
            .into_iter()
            .map(|e| {
                json!({
                    "id": e.id,
                    "run_id": e.run_id,
                    "event_type": e.event_type,
                    "detail": e.detail,
                    "created_at": e.created_at
                })
            })
            .collect();
        ToolResult::success(json!({"run_id": run_id, "events": payload}).to_string())
    }
}

impl SubagentsRetryAnnouncesTool {
    pub fn new(config: &Config, db: Arc<Database>, channel_registry: Arc<ChannelRegistry>) -> Self {
        Self {
            config: config.clone(),
            db,
            channel_registry,
        }
    }
}

#[async_trait]
impl Tool for SubagentsRetryAnnouncesTool {
    fn name(&self) -> &str {
        "subagents_retry_announces"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagents_retry_announces".into(),
            description:
                "Manually flush pending subagent completion announcements (control chats only)."
                    .into(),
            input_schema: schema_object(
                json!({
                    "batch": {"type": "integer", "minimum": 1, "maximum": 200}
                }),
                &[],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let auth = match auth_context_from_input(&input) {
            Some(v) => v,
            None => {
                return ToolResult::error(
                    "subagents_retry_announces requires caller auth context".into(),
                )
            }
        };
        if !auth.is_control_chat() {
            return ToolResult::error(
                "Permission denied: subagents_retry_announces requires control chat".into(),
            );
        }
        let batch = input
            .get("batch")
            .and_then(|v| v.as_u64())
            .unwrap_or(50)
            .clamp(1, 200) as usize;
        flush_pending_announces(
            &self.config,
            self.channel_registry.clone(),
            self.db.clone(),
            batch,
        )
        .await;
        ToolResult::success(json!({"status":"ok","batch":batch}).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkingDirIsolation;

    fn test_config() -> Config {
        let mut cfg = Config::test_defaults();
        cfg.model = "claude-test".into();
        cfg.max_tokens = 2048;
        cfg.data_dir = "/tmp".into();
        cfg.working_dir = "/tmp".into();
        cfg.working_dir_isolation = WorkingDirIsolation::Shared;
        cfg.web_enabled = false;
        cfg
    }

    fn test_db() -> Arc<Database> {
        let dir = std::env::temp_dir().join(format!(
            "microclaw_subagents_tool_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(Database::new(dir.to_str().unwrap()).unwrap())
    }

    #[tokio::test]
    async fn test_sessions_spawn_requires_task() {
        let tool =
            SessionsSpawnTool::new(&test_config(), test_db(), Arc::new(ChannelRegistry::new()));
        let result = tool
            .execute(json!({"__microclaw_auth": {"caller_channel":"web", "caller_chat_id": 1}}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("task"));
    }

    #[tokio::test]
    async fn test_subagents_info_requires_run_id() {
        let tool = SubagentsInfoTool::new(test_db());
        let result = tool
            .execute(json!({"__microclaw_auth": {"caller_channel":"web", "caller_chat_id": 1}}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("run_id"));
    }
}
