//! Trace listing and management.
//!
//! Provides an index of all agent runs stored as trace JSON files,
//! with summary statistics for quick scanning.

use std::path::{Path, PathBuf};

use crate::trace_types::{TraceStatus, TraceStepType};

use crate::LumenError;

/// Summary of an agent trace.
#[derive(Debug, serde::Serialize)]
pub struct TraceSummary {
    /// Trace ID.
    pub trace_id: String,
    /// Agent name.
    pub agent_name: String,
    /// Whether the run succeeded.
    pub success: bool,
    /// Total LLM iterations.
    pub iterations: u32,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Start timestamp (ms since epoch).
    pub started_at_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Prompt tokens used.
    pub prompt_tokens: u32,
    /// Completion tokens used.
    pub completion_tokens: u32,
}

/// List and manage agent traces.
pub struct TraceStore {
    trace_dir: PathBuf,
}

impl TraceStore {
    /// Create a new trace store pointing at a directory of trace JSON files.
    #[must_use]
    pub fn new(trace_dir: impl AsRef<Path>) -> Self {
        Self {
            trace_dir: trace_dir.as_ref().to_path_buf(),
        }
    }

    /// List all traces, newest first.
    ///
    /// # Errors
    ///
    /// Returns `LumenError::Io` if the trace directory cannot be read.
    pub fn list(&self) -> Result<Vec<TraceSummary>, LumenError> {
        let traces = crate::trace_reader::load_traces(&self.trace_dir)?;

        Ok(traces
            .iter()
            .map(|t| {
                let success = matches!(t.status, TraceStatus::Completed);
                let duration_ms = t
                    .completed_at_ms
                    .unwrap_or(t.started_at_ms)
                    .saturating_sub(t.started_at_ms);
                // Count LLM call steps as iterations
                let iterations = t
                    .steps
                    .iter()
                    .filter(|s| matches!(s.step_type, TraceStepType::LlmCall { .. }))
                    .count();

                TraceSummary {
                    trace_id: t.trace_id.clone(),
                    agent_name: t.agent_id.clone(),
                    success,
                    iterations: iterations as u32,
                    cost_usd: t.total_cost_usd,
                    started_at_ms: t.started_at_ms,
                    duration_ms,
                    prompt_tokens: t.total_tokens.prompt_tokens,
                    completion_tokens: t.total_tokens.completion_tokens,
                }
            })
            .collect())
    }

    /// Get the trace directory path.
    #[must_use]
    pub fn trace_dir(&self) -> &Path {
        &self.trace_dir
    }
}
