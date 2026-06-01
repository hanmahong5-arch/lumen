//! Deterministic replay engine.
//!
//! Reads AgentTrace JSON files and reconstructs the step-by-step
//! sequence of LLM decisions and tool executions from a past agent run.
//!
//! Phase 1: Replay from trace JSON files (step-level detail).
//! Phase 2 (future): WAL-level replay from AgentDirective type=28/29 records
//! for full deterministic LLM decision reconstruction.

use std::path::{Path, PathBuf};

use crate::trace_types::{AgentTrace, TraceStatus, TraceStepType};

use crate::LumenError;

/// A single step in a replayed agent run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayStep {
    /// Step number (1-indexed).
    pub step: u32,
    /// What the LLM decided.
    pub decision: ReplayDecision,
    /// Timestamp of the original execution (ms since epoch).
    pub timestamp_ms: u64,
    /// Duration of this step in milliseconds.
    pub duration_ms: u64,
}

/// What happened at each step.
#[derive(Debug, Clone, serde::Serialize)]
pub enum ReplayDecision {
    /// LLM called one or more tools.
    ToolCalls {
        /// Tool calls made.
        calls: Vec<ReplayToolCall>,
    },
    /// LLM produced a final answer.
    FinalAnswer {
        /// The content of the final answer.
        content: String,
    },
}

/// A single tool call in a replay.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayToolCall {
    /// Tool name.
    pub tool_name: String,
    /// Call ID.
    pub call_id: String,
    /// Whether the tool succeeded.
    pub success: bool,
    /// Tool output preview.
    pub output_preview: String,
}

/// Crash-recovery provenance for a replayed run.
///
/// Present only when this run was resumed from a crashed predecessor (Kova
/// sets `parent_trace_id` and emits a recovery-marker Checkpoint as the first
/// step). Lets Lumen surface Kova's durability story: "resumed from X at N".
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecoveryInfo {
    /// Trace id of the crashed run this one resumed from.
    pub parent_trace_id: String,
    /// Iteration the run resumed at, from the recovery Checkpoint metadata
    /// (`resumed_at_iteration`). `None` if the marker was absent/unparseable.
    pub resumed_at_iteration: Option<u32>,
}

/// Full replay of an agent run.
#[derive(Debug, serde::Serialize)]
pub struct ReplayTrace {
    /// Trace ID.
    pub trace_id: String,
    /// Agent name.
    pub agent_name: String,
    /// All steps in order.
    pub steps: Vec<ReplayStep>,
    /// Total cost of the original run in USD.
    pub original_cost_usd: f64,
    /// Total LLM iterations.
    pub total_iterations: u32,
    /// Whether the run completed successfully.
    pub success: bool,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Crash-recovery provenance, if this run resumed from a crashed one.
    pub recovery: Option<RecoveryInfo>,
}

/// Replay engine that reads trace JSON files to reconstruct agent runs.
pub struct ReplayEngine {
    trace_dir: PathBuf,
}

impl ReplayEngine {
    /// Create a new replay engine pointing at a trace directory.
    #[must_use]
    pub fn new(trace_dir: impl AsRef<Path>) -> Self {
        Self {
            trace_dir: trace_dir.as_ref().to_path_buf(),
        }
    }

    /// Replay an agent run by trace ID.
    ///
    /// Reconstructs the step-by-step decision sequence from
    /// the trace JSON file. No LLM calls are made.
    ///
    /// # Errors
    ///
    /// Returns `LumenError::TraceNotFound` if no trace matches the ID.
    pub fn replay(&self, trace_id: &str) -> Result<ReplayTrace, LumenError> {
        let traces = crate::trace_reader::load_traces(&self.trace_dir)?;
        let trace = traces
            .iter()
            .find(|t| t.trace_id == trace_id)
            .ok_or_else(|| LumenError::TraceNotFound(trace_id.to_string()))?;

        let steps = build_replay_steps(trace);
        let success = matches!(trace.status, TraceStatus::Completed);
        let total_duration_ms = trace
            .completed_at_ms
            .unwrap_or(trace.started_at_ms)
            .saturating_sub(trace.started_at_ms);

        Ok(ReplayTrace {
            trace_id: trace.trace_id.clone(),
            agent_name: trace.agent_id.clone(),
            steps,
            original_cost_usd: trace.total_cost_usd,
            total_iterations: trace
                .steps
                .iter()
                .filter(|s| matches!(s.step_type, TraceStepType::LlmCall { .. }))
                .count() as u32,
            success,
            total_duration_ms,
            recovery: extract_recovery(trace),
        })
    }

    /// List all available trace IDs.
    ///
    /// # Errors
    ///
    /// Returns `LumenError::Io` if the trace directory cannot be read.
    pub fn list_traces(&self) -> Result<Vec<String>, LumenError> {
        let traces = crate::trace_reader::load_traces(&self.trace_dir)?;
        Ok(traces.into_iter().map(|t| t.trace_id).collect())
    }
}

/// Extract crash-recovery provenance from a trace.
///
/// Recovery is signalled by `parent_trace_id` being set. The `resumed_at_iteration`
/// is read from the first step's recovery-marker metadata
/// (`[("recovery","true"),("resumed_at_iteration","N")]`) when present.
fn extract_recovery(trace: &AgentTrace) -> Option<RecoveryInfo> {
    let parent_trace_id = trace.parent_trace_id.clone()?;
    let resumed_at_iteration = trace
        .steps
        .first()
        .and_then(|s| {
            s.metadata
                .iter()
                .find(|(k, _)| k == "resumed_at_iteration")
                .map(|(_, v)| v)
        })
        .and_then(|v| v.parse::<u32>().ok());
    Some(RecoveryInfo {
        parent_trace_id,
        resumed_at_iteration,
    })
}

/// Build replay steps by grouping LLM calls with their subsequent tool calls.
///
/// Pattern: [LlmCall(tool_use) → ToolCall → ToolCall] = one iteration step.
/// Final [LlmCall(stop)] = final answer step.
fn build_replay_steps(trace: &AgentTrace) -> Vec<ReplayStep> {
    let mut steps = Vec::new();
    let mut step_num: u32 = 0;
    let mut pending_tools: Vec<ReplayToolCall> = Vec::new();
    let mut pending_timestamp: u64 = 0;
    let mut pending_duration: u64 = 0;

    for trace_step in &trace.steps {
        match &trace_step.step_type {
            TraceStepType::LlmCall {
                finish_reason,
                model: _,
            } => {
                // Flush any accumulated tool calls as one iteration
                if !pending_tools.is_empty() {
                    step_num += 1;
                    steps.push(ReplayStep {
                        step: step_num,
                        decision: ReplayDecision::ToolCalls {
                            calls: std::mem::take(&mut pending_tools),
                        },
                        timestamp_ms: pending_timestamp,
                        duration_ms: pending_duration,
                    });
                    pending_duration = 0;
                }

                // Terminal LLM call → final answer
                if finish_reason == "stop" || finish_reason == "end_turn" {
                    step_num += 1;
                    steps.push(ReplayStep {
                        step: step_num,
                        decision: ReplayDecision::FinalAnswer {
                            content: format!("Agent completed ({finish_reason})"),
                        },
                        timestamp_ms: trace_step.started_at_ms,
                        duration_ms: trace_step.duration_ms,
                    });
                } else {
                    // tool_use → tools will follow
                    pending_timestamp = trace_step.started_at_ms;
                    pending_duration += trace_step.duration_ms;
                }
            }
            TraceStepType::ToolCall { tool_name, success } => {
                if pending_timestamp == 0 {
                    pending_timestamp = trace_step.started_at_ms;
                }
                pending_duration += trace_step.duration_ms;
                pending_tools.push(ReplayToolCall {
                    tool_name: tool_name.clone(),
                    call_id: format!("step_{}", trace_step.step_num),
                    success: *success,
                    output_preview: if *success {
                        format!("ok ({}ms)", trace_step.duration_ms)
                    } else {
                        "failed".to_string()
                    },
                });
            }
            TraceStepType::Error { message } => {
                // Flush pending tools before error
                if !pending_tools.is_empty() {
                    step_num += 1;
                    steps.push(ReplayStep {
                        step: step_num,
                        decision: ReplayDecision::ToolCalls {
                            calls: std::mem::take(&mut pending_tools),
                        },
                        timestamp_ms: pending_timestamp,
                        duration_ms: pending_duration,
                    });
                    pending_duration = 0;
                }
                step_num += 1;
                steps.push(ReplayStep {
                    step: step_num,
                    decision: ReplayDecision::FinalAnswer {
                        content: format!("Error: {message}"),
                    },
                    timestamp_ms: trace_step.started_at_ms,
                    duration_ms: trace_step.duration_ms,
                });
            }
            TraceStepType::Checkpoint { .. } => {
                // Checkpoints are internal, skip for replay display
            }
        }
    }

    // Flush remaining tool calls
    if !pending_tools.is_empty() {
        step_num += 1;
        steps.push(ReplayStep {
            step: step_num,
            decision: ReplayDecision::ToolCalls {
                calls: pending_tools,
            },
            timestamp_ms: pending_timestamp,
            duration_ms: pending_duration,
        });
    }

    steps
}
