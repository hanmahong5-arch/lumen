//! LLM cost tracking and aggregation.
//!
//! Scans trace JSON files to aggregate token usage and USD costs
//! across agents, models, and time windows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kova_types::trace::TraceStepType;

use crate::LumenError;

/// Cost record for a single agent run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunCost {
    /// Trace ID.
    pub trace_id: String,
    /// Agent name.
    pub agent_name: String,
    /// Primary model used.
    pub model: String,
    /// Prompt tokens consumed.
    pub prompt_tokens: u64,
    /// Completion tokens consumed.
    pub completion_tokens: u64,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Number of LLM iterations.
    pub iterations: u32,
    /// Start timestamp (ms since epoch).
    pub timestamp_ms: u64,
}

/// Aggregated cost report.
#[derive(Debug, serde::Serialize)]
pub struct CostReport {
    /// Total cost in USD.
    pub total_usd: f64,
    /// Total agent runs.
    pub total_runs: u64,
    /// Total prompt tokens.
    pub total_prompt_tokens: u64,
    /// Total completion tokens.
    pub total_completion_tokens: u64,
    /// Cost breakdown by agent name.
    pub by_agent: HashMap<String, f64>,
    /// Cost breakdown by model name.
    pub by_model: HashMap<String, f64>,
    /// Runs with anomalous cost (>2x average).
    pub anomalies: Vec<CostAnomaly>,
}

/// A run flagged for anomalous cost.
#[derive(Debug, serde::Serialize)]
pub struct CostAnomaly {
    /// Trace ID.
    pub trace_id: String,
    /// Agent name.
    pub agent_name: String,
    /// Cost of this run.
    pub cost_usd: f64,
    /// Multiple of average cost.
    pub multiplier: f64,
    /// Suspected reason.
    pub reason: String,
}

/// Get or estimate cost for a trace.
///
/// If `total_cost_usd > 0`, uses it directly.
/// Otherwise estimates from per-step token usage and model pricing.
fn trace_cost(trace: &kova_types::trace::AgentTrace) -> f64 {
    if trace.total_cost_usd > 0.0 {
        return trace.total_cost_usd;
    }

    // Estimate from individual LLM call steps
    let mut total = 0.0_f64;
    for step in &trace.steps {
        if let TraceStepType::LlmCall { ref model, .. } = step.step_type
            && let Some(tokens) = step.tokens
        {
            total += crate::pricing::estimate_cost(
                model,
                tokens.prompt_tokens,
                tokens.completion_tokens,
            );
        }
    }

    // Fallback: estimate from aggregate tokens
    if total == 0.0 && trace.total_tokens.total() > 0 {
        total = crate::pricing::estimate_cost(
            "unknown",
            trace.total_tokens.prompt_tokens,
            trace.total_tokens.completion_tokens,
        );
    }

    total
}

/// Cost tracker that reads trace JSON files to aggregate LLM spending.
pub struct CostTracker {
    trace_dir: PathBuf,
}

impl CostTracker {
    /// Create a new cost tracker.
    #[must_use]
    pub fn new(trace_dir: impl AsRef<Path>) -> Self {
        Self {
            trace_dir: trace_dir.as_ref().to_path_buf(),
        }
    }

    /// Generate a cost report for runs since `since_ms` (epoch milliseconds).
    ///
    /// # Errors
    ///
    /// Returns `LumenError::Io` if the trace directory cannot be read.
    pub fn report(&self, since_ms: u64) -> Result<CostReport, LumenError> {
        let traces = crate::trace_reader::load_traces(&self.trace_dir)?;

        // Filter by time window
        let filtered: Vec<_> = traces
            .iter()
            .filter(|t| t.started_at_ms >= since_ms)
            .collect();

        let total_runs = filtered.len() as u64;
        let total_usd: f64 = filtered.iter().map(|t| trace_cost(t)).sum();
        let total_prompt: u64 = filtered
            .iter()
            .map(|t| u64::from(t.total_tokens.prompt_tokens))
            .sum();
        let total_completion: u64 = filtered
            .iter()
            .map(|t| u64::from(t.total_tokens.completion_tokens))
            .sum();

        // Precompute per-trace estimated costs
        let trace_costs: Vec<f64> = filtered.iter().map(|t| trace_cost(t)).collect();

        // Group by agent
        let mut by_agent: HashMap<String, f64> = HashMap::new();
        for (t, &cost) in filtered.iter().zip(trace_costs.iter()) {
            *by_agent.entry(t.agent_id.clone()).or_insert(0.0) += cost;
        }

        // Group by model (extract from LlmCall steps)
        let mut by_model: HashMap<String, f64> = HashMap::new();
        for (t, &cost) in filtered.iter().zip(trace_costs.iter()) {
            let llm_steps: Vec<_> = t
                .steps
                .iter()
                .filter_map(|s| {
                    if let TraceStepType::LlmCall { ref model, .. } = s.step_type {
                        Some((model.clone(), s.tokens))
                    } else {
                        None
                    }
                })
                .collect();

            let llm_count = llm_steps.len();
            if llm_count > 0 {
                let per_call = cost / llm_count as f64;
                for (model, _) in &llm_steps {
                    let m: String = model.clone();
                    *by_model.entry(m).or_insert(0.0) += per_call;
                }
            }
        }

        // Detect anomalies: runs > 2x average cost
        let avg_cost = if total_runs > 0 {
            total_usd / total_runs as f64
        } else {
            0.0
        };

        let anomalies = filtered
            .iter()
            .zip(trace_costs.iter())
            .filter(|(_, cost)| **cost > avg_cost * 2.0 && **cost > 0.01)
            .map(|(t, cost)| {
                let c = *cost;
                let multiplier = if avg_cost > 0.0 { c / avg_cost } else { 0.0 };
                let step_count = t.steps.len();
                CostAnomaly {
                    trace_id: t.trace_id.clone(),
                    agent_name: t.agent_id.clone(),
                    cost_usd: c,
                    multiplier,
                    reason: format!("{step_count} steps, {:.0}x avg", multiplier),
                }
            })
            .collect();

        Ok(CostReport {
            total_usd,
            total_runs,
            total_prompt_tokens: total_prompt,
            total_completion_tokens: total_completion,
            by_agent,
            by_model,
            anomalies,
        })
    }
}
