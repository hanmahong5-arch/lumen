//! Vendored read-structs for the kova-rest **workflow-type flow** endpoint.
//!
//! Mirrors `GET /api/v1/workflows/types/{type}/flow` — the Markov state-transition
//! graph + Bayesian outcome priors kova computes **server-side** (the single
//! source both Lumen and Forge render, so the two surfaces can't drift). Lumen is
//! a post-hoc/read viewer: every field is `#[serde(default)]` so backend drift (a
//! renamed/dropped field) degrades to a zero value instead of failing the read.
//!
//! Keep in sync with `2b-svc-kova`:
//! - `kova-rest/src/routes/analytics.rs::{WorkflowTypeFlowResponse, FlowNode,
//!   FlowEdge, FlowOutcomes, OutcomePrior}`
//! - `doc/coord/contracts.md` ("Workflow-type flow" section)
//!
//! **Honesty rails** (these are properties of the *server's* numbers — never
//! re-derive them client-side): `probability` is a Laplace-smoothed posterior
//! mean `(successes+1)/(N+2)`, NOT a raw frequency (so `N=0 → 0.5`, and a small
//! sample never reports a fake `1.000`/`0`); `(ci_lo,ci_hi)` is a Wilson 95%
//! interval; `insufficient` is the server's `N<10` flag — render "insufficient
//! data (N=k)", never a point estimate.

use serde::{Deserialize, Serialize};

/// One node in the flow graph. `state` is `step:0…step:N`, a terminal sink
/// (`completed`/`failed`/`cancelled`), or a holding node
/// (`awaiting_input|awaiting_timer|awaiting_retry|awaiting_signal`, `paused`).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FlowNode {
    /// State label.
    #[serde(default)]
    pub state: String,
    /// `"step"` | `"terminal"` | `"holding"` — lets the client color without
    /// re-deriving the vocabulary.
    #[serde(default)]
    pub kind: String,
    /// Visit count: how many trajectories passed through this node.
    #[serde(default)]
    pub n: u64,
}

/// One directed transition. `p` is the empirical Markov transition probability
/// out of `from` (`count / Σ(edges leaving from)`, 4-dp).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FlowEdge {
    /// Source node label.
    #[serde(default)]
    pub from: String,
    /// Destination node label.
    #[serde(default)]
    pub to: String,
    /// Trajectories that traversed this transition.
    #[serde(default)]
    pub count: u64,
    /// Transition probability `count / Σ(out-edges of from)`, 4-dp.
    #[serde(default)]
    pub p: f64,
}

/// Unconditional terminal tallies (CAN chains counted once, by tail status).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FlowOutcomes {
    /// Trajectories that reached `completed`.
    #[serde(default)]
    pub completed: u64,
    /// Trajectories that reached `failed`.
    #[serde(default)]
    pub failed: u64,
    /// Trajectories that reached `cancelled`.
    #[serde(default)]
    pub cancelled: u64,
    /// `= completed + failed + cancelled` (the terminal denominator).
    #[serde(default)]
    pub n: u64,
}

/// One Bayesian outcome prior. See the module-level honesty rails: `probability`
/// is a Laplace-smoothed posterior mean, NOT a raw frequency.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OutcomePrior {
    /// `"completed"` | `"failed"` | `"cancelled"`.
    #[serde(default)]
    pub outcome: String,
    /// `None` = unconditional `P(outcome)`; `Some(k)` = `P(outcome | reached
    /// step k)`. The client conditions on the run's current step.
    #[serde(default)]
    pub conditioned_on_step: Option<u32>,
    /// Laplace-smoothed posterior mean `(successes+1)/(N+2)`, 4-dp.
    #[serde(default)]
    pub probability: f64,
    /// Wilson 95% interval lower bound.
    #[serde(default)]
    pub ci_lo: f64,
    /// Wilson 95% interval upper bound.
    #[serde(default)]
    pub ci_hi: f64,
    /// Denominator N (terminal trajectories in this conditioning slice).
    #[serde(default)]
    pub sample_size: u64,
    /// `N < 10` — render "insufficient data (N=k)" instead of a point estimate.
    #[serde(default)]
    pub insufficient: bool,
}

/// `GET /api/v1/workflows/types/{type}/flow` response.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FlowGraph {
    /// The workflow type this projection summarizes.
    #[serde(default)]
    pub workflow_type: String,
    /// Logical trajectories of this type (a CAN chain counts as **one**).
    #[serde(default)]
    pub total_runs: u64,
    /// Trajectories that reached a terminal status.
    #[serde(default)]
    pub terminal_runs: u64,
    /// `= terminal_runs` — denominator for the unconditional priors.
    #[serde(default)]
    pub sample_size: u64,
    /// `sample_size < 10` — render "insufficient data (N=k)".
    #[serde(default)]
    pub insufficient: bool,
    /// A non-terminal (running/awaiting/paused) trajectory is present.
    #[serde(default)]
    pub has_in_flight_runs: bool,
    /// More than one distinct spec version in the population → warn; transitions
    /// across spec versions aren't strictly comparable (`?spec_hash=` narrows).
    #[serde(default)]
    pub mixed_spec_versions: bool,
    /// Graph nodes (step-progression / terminal / holding).
    #[serde(default)]
    pub nodes: Vec<FlowNode>,
    /// Directed transitions with empirical probabilities.
    #[serde(default)]
    pub edges: Vec<FlowEdge>,
    /// Unconditional terminal tallies.
    #[serde(default)]
    pub outcomes: FlowOutcomes,
    /// Bayesian outcome priors (unconditional + per reached step).
    #[serde(default)]
    pub outcome_priors: Vec<OutcomePrior>,
}

impl FlowGraph {
    /// Pick the prior for `outcome` conditioned on `step` (or the unconditional
    /// one when `step` is `None`). Returns `None` when the server didn't emit a
    /// matching row — the client then shows "insufficient data" / nothing rather
    /// than fabricating a number. This is the cheap client-side join the
    /// contract describes (the run detail is already loaded, so the run's current
    /// step is known): the server returns priors as a property of the *type*; the
    /// client overlays the *run* by conditioning here.
    #[must_use]
    pub fn prior_for(&self, outcome: &str, step: Option<u32>) -> Option<&OutcomePrior> {
        self.outcome_priors
            .iter()
            .find(|p| p.outcome == outcome && p.conditioned_on_step == step)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const FULL: &str = r#"{
        "workflow_type":"expense","total_runs":4,"terminal_runs":4,"sample_size":4,
        "insufficient":true,"has_in_flight_runs":false,"mixed_spec_versions":false,
        "nodes":[
            {"state":"step:0","kind":"step","n":4},
            {"state":"completed","kind":"terminal","n":3},
            {"state":"failed","kind":"terminal","n":1}
        ],
        "edges":[
            {"from":"step:0","to":"step:1","count":4,"p":1.0},
            {"from":"step:1","to":"completed","count":3,"p":0.75},
            {"from":"step:1","to":"failed","count":1,"p":0.25}
        ],
        "outcomes":{"completed":3,"failed":1,"cancelled":0,"n":4},
        "outcome_priors":[
            {"outcome":"completed","conditioned_on_step":null,"probability":0.6667,
             "ci_lo":0.3009,"ci_hi":0.9637,"sample_size":4,"insufficient":true},
            {"outcome":"failed","conditioned_on_step":1,"probability":0.3333,
             "ci_lo":0.0676,"ci_hi":0.7673,"sample_size":4,"insufficient":true}
        ]
    }"#;

    #[test]
    fn full_response_parses() {
        let fg: FlowGraph = serde_json::from_str(FULL).unwrap();
        assert_eq!(fg.workflow_type, "expense");
        assert_eq!(fg.total_runs, 4);
        assert!(fg.insufficient);
        assert_eq!(fg.nodes.len(), 3);
        assert_eq!(fg.edges.len(), 3);
        assert_eq!(fg.outcomes.completed, 3);
        // A smoothed prior is never a bare 1.0 / 0.0.
        let c = fg.prior_for("completed", None).unwrap();
        assert!(c.probability < 1.0 && c.probability > 0.0);
        // The Wilson interval must bracket the point estimate (`||` here was a
        // tautology — for any finite floats at least one side holds, so it
        // never caught an interval that fails to contain `probability`).
        assert!(c.ci_lo <= c.probability && c.probability <= c.ci_hi);
    }

    #[test]
    fn older_producer_degrades_to_defaults() {
        // A minimal/older payload must still parse via serde(default) — Lumen
        // stays tolerant of producer drift.
        let fg: FlowGraph = serde_json::from_str(r#"{"workflow_type":"x"}"#).unwrap();
        assert_eq!(fg.workflow_type, "x");
        assert_eq!(fg.total_runs, 0);
        assert!(!fg.insufficient); // absent → false (the client treats 0-node graphs as empty)
        assert!(fg.nodes.is_empty());
        assert!(fg.outcome_priors.is_empty());
    }

    #[test]
    fn prior_lookup_conditions_on_step() {
        let fg: FlowGraph = serde_json::from_str(FULL).unwrap();
        // Conditional lookup finds the step-1 failed prior.
        let f1 = fg.prior_for("failed", Some(1)).unwrap();
        assert_eq!(f1.sample_size, 4);
        assert!((f1.probability - 0.3333).abs() < 1e-4);
        // A step with no emitted prior → None (client shows nothing, not a fake).
        assert!(fg.prior_for("failed", Some(99)).is_none());
        assert!(fg.prior_for("cancelled", None).is_none());
    }
}
