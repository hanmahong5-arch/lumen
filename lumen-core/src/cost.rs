//! LLM cost tracking and aggregation.
//!
//! Scans trace JSON files to aggregate token usage and USD costs
//! across agents, models, and time windows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::LumenError;
use crate::trace_types::{AgentTrace, TraceStepType};

/// Multiplier over the mean run cost above which a run is flagged a **cost
/// outlier**.
///
/// This is a cheap, offline, *per-run cross-sectional* check ("this run cost
/// more than `N`× the mean of the runs on disk"). It is deliberately distinct
/// from Netdata's *per-metric, per-timestep* ML anomaly detection that the Lumen
/// dashboard surfaces on its Metrics tab: Netdata has no `trace_id` dimension,
/// so a fleet-level metric anomaly cannot be mapped back to one run. This flag
/// stays as the no-Netdata fallback signal.
///
/// The dashboard JS mirrors this literal in three places; the `dashboard.rs`
/// drift-guard test keeps them in lockstep.
pub const COST_ANOMALY_MULTIPLIER: f64 = 2.0;

/// Floor (USD) below which a run is never flagged a cost outlier, so trivially
/// cheap runs don't trip the multiplier on rounding noise.
pub const COST_ANOMALY_MIN_USD: f64 = 0.01;

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
    /// Runs flagged as cost outliers (cost > `COST_ANOMALY_MULTIPLIER`× the mean
    /// run cost). An offline, per-run heuristic — not Netdata ML anomaly.
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
fn trace_cost(trace: &AgentTrace) -> f64 {
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
        // Per-trace real cost, computed ONCE. `total_usd` is derived from this
        // vector so `trace_cost()` is not evaluated twice per trace.
        let trace_costs: Vec<f64> = filtered.iter().map(|t| trace_cost(t)).collect();
        let total_usd: f64 = trace_costs.iter().sum();
        let total_prompt: u64 = filtered
            .iter()
            .map(|t| u64::from(t.total_tokens.prompt_tokens))
            .sum();
        let total_completion: u64 = filtered
            .iter()
            .map(|t| u64::from(t.total_tokens.completion_tokens))
            .sum();

        // Group by agent
        let mut by_agent: HashMap<String, f64> = HashMap::new();
        for (t, &cost) in filtered.iter().zip(trace_costs.iter()) {
            *by_agent.entry(t.agent_id.clone()).or_insert(0.0) += cost;
        }

        // Group by model. Attribute each trace's REAL cost (`trace_cost` — the
        // engine-reported `total_cost_usd` when present) across the models it
        // used, split proportionally by each model's estimated share. This keeps
        // `sum(by_model) == total_usd` exactly. Re-estimating per step instead
        // (the previous behavior) diverged from the real billed cost and
        // exploded for models absent from the pricing table — they fall back to
        // a placeholder $1/$3 per-1k price, so e.g. a 5k-token `deepseek-chat`
        // run reported ~$6 against a real cost of ~$0.002.
        let mut by_model: HashMap<String, f64> = HashMap::new();
        for (t, &real_cost) in filtered.iter().zip(trace_costs.iter()) {
            // Per-model weight = estimated step cost, falling back to raw token
            // count when pricing yields zero, so the split is always defined.
            let mut weights: HashMap<String, f64> = HashMap::new();
            for s in &t.steps {
                if let TraceStepType::LlmCall { ref model, .. } = s.step_type {
                    let w = s.tokens.map_or(0.0, |tok| {
                        let est = crate::pricing::estimate_cost(
                            model,
                            tok.prompt_tokens,
                            tok.completion_tokens,
                        );
                        if est > 0.0 {
                            est
                        } else {
                            f64::from(tok.prompt_tokens) + f64::from(tok.completion_tokens)
                        }
                    });
                    *weights.entry(model.clone()).or_insert(0.0) += w;
                }
            }
            if weights.is_empty() {
                // Trace has a real cost but no LLM-call step to attribute it to
                // (e.g. a tool-only run, or the aggregate-token fallback path in
                // `trace_cost`). Bucket it under "unknown" so the invariant
                // `sum(by_model) == total_usd` still holds, instead of silently
                // dropping this trace's cost from the model breakdown.
                if real_cost != 0.0 {
                    *by_model.entry("unknown".to_string()).or_insert(0.0) += real_cost;
                }
                continue;
            }
            let weight_total: f64 = weights.values().sum();
            if weight_total > 0.0 {
                for (model, w) in weights {
                    *by_model.entry(model).or_insert(0.0) += real_cost * (w / weight_total);
                }
            } else {
                // All steps had zero tokens: split the real cost evenly so the
                // model still appears and the invariant `sum(by_model)==total` holds.
                let share = real_cost / weights.len() as f64;
                for model in weights.into_keys() {
                    *by_model.entry(model).or_insert(0.0) += share;
                }
            }
        }

        // Flag cost outliers: runs > COST_ANOMALY_MULTIPLIER× the mean run cost
        // (with a small absolute floor so trivially cheap runs don't trip it).
        // This is the offline per-run signal, NOT Netdata ML anomaly.
        let avg_cost = if total_runs > 0 {
            total_usd / total_runs as f64
        } else {
            0.0
        };

        let anomalies = filtered
            .iter()
            .zip(trace_costs.iter())
            .filter(|(_, cost)| {
                **cost > avg_cost * COST_ANOMALY_MULTIPLIER && **cost > COST_ANOMALY_MIN_USD
            })
            .map(|(t, cost)| {
                let c = *cost;
                let multiplier = if avg_cost > 0.0 { c / avg_cost } else { 0.0 };
                let step_count = t.steps.len();
                CostAnomaly {
                    trace_id: t.trace_id.clone(),
                    agent_name: t.agent_id.clone(),
                    cost_usd: c,
                    multiplier,
                    reason: format!("cost outlier: {step_count} steps, {multiplier:.0}× run mean"),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Write one trace JSON into a fresh temp dir and return its path.
    fn write_trace(name: &str, json: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen_cost_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{name}.json")), json).unwrap();
        dir
    }

    // Regression: `by_model` must attribute the trace's REAL cost, so
    // `sum(by_model) == total_usd`. Previously it re-estimated per step and a
    // `deepseek-chat` run (absent from the pricing table → $1/$3 fallback)
    // reported ~$6 against a real cost of $0.0017.
    #[test]
    fn by_model_sums_to_real_total() {
        let json = r#"{
            "trace_id":"t1","agent_id":"a1","task_id":1,
            "steps":[{"step_num":0,"step_type":{"LlmCall":{"model":"deepseek-chat","finish_reason":"stop"}},
                "started_at_ms":100,"duration_ms":10,"tokens":{"prompt_tokens":4000,"completion_tokens":400},"metadata":[]}],
            "status":"Completed","total_tokens":{"prompt_tokens":4000,"completion_tokens":400},
            "total_cost_usd":0.0017,"started_at_ms":100,"completed_at_ms":200,"parent_trace_id":null,"schema_version":1
        }"#;
        let dir = write_trace("real", json);
        let report = CostTracker::new(&dir).report(0).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let by_model_sum: f64 = report.by_model.values().sum();
        assert!(
            (report.total_usd - 0.0017).abs() < 1e-9,
            "uses real engine cost, not estimate"
        );
        assert!(
            (by_model_sum - report.total_usd).abs() < 1e-9,
            "by_model must sum to total_usd"
        );
        assert!(
            (report.by_model.get("deepseek-chat").copied().unwrap_or(0.0) - 0.0017).abs() < 1e-9
        );
    }

    // A trace mixing two models must split the real cost proportionally and
    // still sum to the total (no double counting, no smearing to zero).
    #[test]
    fn by_model_splits_mixed_models() {
        let json = r#"{
            "trace_id":"t2","agent_id":"a2","task_id":2,
            "steps":[
                {"step_num":0,"step_type":{"LlmCall":{"model":"gpt-4o","finish_reason":"stop"}},
                 "started_at_ms":100,"duration_ms":10,"tokens":{"prompt_tokens":1000,"completion_tokens":100},"metadata":[]},
                {"step_num":1,"step_type":{"LlmCall":{"model":"gpt-4o-mini","finish_reason":"stop"}},
                 "started_at_ms":120,"duration_ms":10,"tokens":{"prompt_tokens":1000,"completion_tokens":100},"metadata":[]}
            ],
            "status":"Completed","total_tokens":{"prompt_tokens":2000,"completion_tokens":200},
            "total_cost_usd":1.0,"started_at_ms":100,"completed_at_ms":200,"parent_trace_id":null,"schema_version":1
        }"#;
        let dir = write_trace("mixed", json);
        let report = CostTracker::new(&dir).report(0).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let by_model_sum: f64 = report.by_model.values().sum();
        assert!(
            (by_model_sum - 1.0).abs() < 1e-9,
            "split of real cost must sum to total"
        );
        // gpt-4o is ~10x pricier than gpt-4o-mini, so it must carry the larger share.
        let big = report.by_model.get("gpt-4o").copied().unwrap_or(0.0);
        let small = report.by_model.get("gpt-4o-mini").copied().unwrap_or(0.0);
        assert!(
            big > small,
            "pricier model carries the larger share of the real cost"
        );
    }

    // A trace that has a real cost but NO LlmCall step (tool-only run) must
    // still appear in by_model — under "unknown" — so sum(by_model) == total_usd
    // never silently under-counts. Regression for the weights.is_empty() drop.
    #[test]
    fn by_model_keeps_invariant_when_no_llm_steps() {
        let json = r#"{
            "trace_id":"t3","agent_id":"a3","task_id":3,
            "steps":[
                {"step_num":0,"step_type":{"ToolCall":{"tool_name":"search","success":true}},
                 "started_at_ms":100,"duration_ms":10,"tokens":null,"metadata":[]},
                {"step_num":1,"step_type":{"Checkpoint":{"iteration":1}},
                 "started_at_ms":120,"duration_ms":1,"tokens":null,"metadata":[]}
            ],
            "status":"Completed","total_tokens":{"prompt_tokens":0,"completion_tokens":0},
            "total_cost_usd":0.0042,"started_at_ms":100,"completed_at_ms":200,"parent_trace_id":null,"schema_version":1
        }"#;
        let dir = write_trace("nollm", json);
        let report = CostTracker::new(&dir).report(0).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let by_model_sum: f64 = report.by_model.values().sum();
        assert!((report.total_usd - 0.0042).abs() < 1e-9);
        assert!(
            (by_model_sum - report.total_usd).abs() < 1e-9,
            "by_model must sum to total_usd even with no LLM steps"
        );
        assert!((report.by_model.get("unknown").copied().unwrap_or(0.0) - 0.0042).abs() < 1e-9);
    }
}
