//! Minimal vendored read-structs for the kova-rest lifecycle endpoints.
//!
//! These mirror just the fields lumen reads from three kova-rest surfaces
//! (beyond the per-trace `AgentTrace` in [`crate::trace_types`]):
//!
//! - `GET /agents/{id}/history?format=causal` → a bare `[CausalEvent]` array
//!   (the `AgentHistoryPayload`/`HistoryResponse` envelopes are
//!   `#[serde(untagged)]`, so the unpaged causal response is just the array).
//! - `GET /workflows/runs/{id}` → [`WorkflowRunDetail`] (subset).
//! - `GET /swarm/{id}/graph` (+ `/trace` usage merged in by the fetcher) →
//!   [`SwarmGraph`].
//!
//! Every field is `#[serde(default)]` so that backend drift (a renamed or
//! dropped field) degrades to a zero value instead of failing the whole read —
//! lumen is a post-hoc viewer and must stay tolerant of older/newer producers.
//!
//! Keep in sync with `2b-svc-kova`:
//! - `kova/src/visibility/causal_projection.rs::CausalEvent`
//! - `kova-rest/src/routes/workflow.rs::{WorkflowRunDetail, WorkflowRunStep}`
//! - `kova-rest/src/routes/swarm_trace.rs::{SwarmGraph, GraphNode, GraphEdge,
//!   SwarmTrace, AgentTraceEntry}`

use serde::{Deserialize, Serialize};

/// One element of the `format=causal` stream.
///
/// `event_id` is the zero-based index into the agent-filtered WAL slice;
/// `caused_by` (when `Some`) references an earlier `event_id` in the same
/// slice. `record_type` is the kova `TaskEventType` variant **name** (e.g.
/// `"AgentDirective"`) — the enum derives serde with no repr remap.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CausalEvent {
    /// Monotonic index in the agent-filtered record stream.
    #[serde(default)]
    pub event_id: u64,
    /// Kova record-type variant name.
    #[serde(default)]
    pub record_type: String,
    /// Wall-clock nanoseconds mirrored from the WAL record.
    #[serde(default)]
    pub timestamp_ns: u64,
    /// `event_id` of the upstream cause, or `None` for a root / undetected edge.
    #[serde(default)]
    pub caused_by: Option<u64>,
    /// Opaque correlation key (diagnostic; lumen does not parse it).
    #[serde(default)]
    pub correlation_key: Option<String>,
    /// Short UI-safe summary.
    #[serde(default)]
    pub summary: String,
}

/// One per-step output row from `GET /workflows/runs/{id}`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WorkflowStep {
    /// 0-based step index (skips reflect a G9 branch path).
    #[serde(default)]
    pub step_index: u32,
    /// Lossy-UTF-8 preview of the step output (capped server-side).
    #[serde(default)]
    pub output_preview: String,
    /// Unix-epoch ms when this step started.
    #[serde(default)]
    pub started_at_ms: u64,
    /// Wall-clock duration in ms (includes wait time for await/sleep steps).
    #[serde(default)]
    pub duration_ms: u64,
}

/// Subset of `GET /workflows/runs/{id}` lumen needs to render a workflow-only
/// run (one with no `AgentTrace`, e.g. the kill-9 declarative demo).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WorkflowRunDetail {
    /// Workflow execution id (matches the URL `:id`).
    #[serde(default)]
    pub workflow_id: u64,
    /// Workflow type string (registry key at start time).
    #[serde(default)]
    pub workflow_type: String,
    /// Run status (`completed` / `failed` / `awaiting_input` / …).
    #[serde(default)]
    pub status: String,
    /// Total step count from the `WorkflowStart` record.
    #[serde(default)]
    pub total_steps: u32,
    /// Number of steps with a checkpoint.
    #[serde(default)]
    pub completed_step_count: u32,
    /// Per-step outputs in execution order.
    #[serde(default)]
    pub steps: Vec<WorkflowStep>,
    /// Suspended step (only when `status == awaiting_input`).
    #[serde(default)]
    pub awaiting_step: Option<u32>,
    /// Step where failure was triggered (only when `status == failed`).
    #[serde(default)]
    pub failed_step: Option<u32>,
    /// Error message of the failing step.
    #[serde(default)]
    pub error_message: Option<String>,
    /// Unix-epoch ms when the workflow started.
    #[serde(default)]
    pub started_at_ms: u64,
    /// Unix-epoch ms when the workflow reached a terminal state.
    #[serde(default)]
    pub terminal_at_ms: Option<u64>,
    /// Step indices that have had compensation run.
    #[serde(default)]
    pub compensated_steps: Vec<u32>,
    /// Step the workflow is sleeping at (only when `status == awaiting_timer`).
    #[serde(default)]
    pub sleeping_step: Option<u32>,
    /// Attempt number the next retry will use (only when `awaiting_retry`).
    #[serde(default)]
    pub retry_attempt: Option<u32>,
    /// Name of the signal this run is parked on (only when
    /// `status == awaiting_signal`). Surfaced so the visualizer can show *which*
    /// signal a run waits for, not just that it waits.
    #[serde(default)]
    pub awaiting_signal_name: Option<String>,
    /// Step index parked awaiting a signal (only when `status == awaiting_signal`).
    #[serde(default)]
    pub awaiting_signal_step: Option<u32>,
    /// `true` when this run was terminated by Continue-As-New (it handed off to a
    /// successor) rather than a plain completion. Both render as `status ==
    /// completed`, so this flag is the only signal distinguishing a continuation
    /// from a finish. Defaults to `false` for older producers that don't emit it.
    #[serde(default)]
    pub continued_as_new: bool,
    /// **CAN chain** — the ordered run-id continuation chain `[head, …, tail]`
    /// (e.g. `[1,2,3]`) this run belongs to, computed server-side by kova-rest.
    /// **Empty** for a non-CAN run; a single-element `[id]` when sibling runs
    /// were compacted away. Rendered as a chain (like the recovery chain) when
    /// `len > 1`, NOT stitched from `parent_trace_id`. Defaults to empty for
    /// older producers that don't emit it.
    #[serde(default)]
    pub continuation_chain: Vec<u64>,
}

/// A node in the swarm delegation graph (`GET /swarm/{id}/graph`).
///
/// `prompt_tokens` / `completion_tokens` are not part of the `/graph` response;
/// the fetcher merges them in from `/swarm/{id}/trace` so the lifecycle view
/// can show per-agent token spend. They default to `0` when no trace was
/// merged.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SwarmNode {
    /// Agent id (unique within the graph).
    #[serde(default)]
    pub id: String,
    /// `"supervisor"` / `"worker"`.
    #[serde(default)]
    pub agent_type: String,
    /// Agent status.
    #[serde(default)]
    pub status: String,
    /// Real prompt tokens (merged from `/swarm/{id}/trace`).
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Real completion tokens (merged from `/swarm/{id}/trace`).
    #[serde(default)]
    pub completion_tokens: u32,
}

/// An edge in the swarm delegation graph.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SwarmEdge {
    /// Source agent id.
    #[serde(default)]
    pub from: String,
    /// Target agent id.
    #[serde(default)]
    pub to: String,
    /// `"delegation"` / `"handoff"`.
    #[serde(default)]
    pub edge_type: String,
}

/// Swarm delegation graph for a swarm execution.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SwarmGraph {
    /// Agent nodes.
    #[serde(default)]
    pub nodes: Vec<SwarmNode>,
    /// Delegation/handoff edges.
    #[serde(default)]
    pub edges: Vec<SwarmEdge>,
}

/// One agent row from `GET /swarm/{id}/trace` — used by the fetcher to merge
/// per-agent token counts into [`SwarmNode`]s.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SwarmAgentEntry {
    /// Agent id.
    #[serde(default)]
    pub agent_id: String,
    /// Real prompt tokens.
    #[serde(default)]
    pub tokens_prompt: u32,
    /// Real completion tokens.
    #[serde(default)]
    pub tokens_completion: u32,
}

/// Subset of `GET /swarm/{id}/trace` the fetcher reads for token merge.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SwarmTrace {
    /// Per-agent trace entries (carry real token counts post-A3).
    #[serde(default)]
    pub agents: Vec<SwarmAgentEntry>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn causal_event_array_parses() {
        let json = r#"[
            {"event_id":0,"record_type":"AgentDirective","timestamp_ns":1000,
             "caused_by":null,"correlation_key":"call-1","summary":"tool: reconcile"},
            {"event_id":1,"record_type":"AgentDirectiveResult","timestamp_ns":2000,
             "caused_by":0,"correlation_key":"call-1","summary":"ok"}
        ]"#;
        let events: Vec<CausalEvent> = serde_json::from_str(json).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].caused_by, Some(0));
        assert_eq!(events[0].record_type, "AgentDirective");
    }

    #[test]
    fn workflow_detail_tolerates_missing_optionals() {
        // A minimal payload (older producer) must still parse via serde(default).
        let json = r#"{"workflow_id":42,"status":"completed","total_steps":3,
            "steps":[{"step_index":0,"started_at_ms":10,"duration_ms":5}]}"#;
        let detail: WorkflowRunDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.workflow_id, 42);
        assert_eq!(detail.steps.len(), 1);
        assert_eq!(detail.terminal_at_ms, None);
        assert!(detail.compensated_steps.is_empty());
    }

    #[test]
    fn workflow_detail_parses_awaiting_signal() {
        let json = r#"{"workflow_id":5,"status":"awaiting_signal","total_steps":2,
            "steps":[{"step_index":1,"started_at_ms":10,"duration_ms":0}],
            "awaiting_signal_step":1,"awaiting_signal_name":"approve_payment"}"#;
        let detail: WorkflowRunDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.awaiting_signal_step, Some(1));
        assert_eq!(
            detail.awaiting_signal_name.as_deref(),
            Some("approve_payment")
        );
        // Absent in an older producer → None via serde(default).
        let older = r#"{"workflow_id":5,"status":"completed","total_steps":1,"steps":[]}"#;
        let d2: WorkflowRunDetail = serde_json::from_str(older).unwrap();
        assert_eq!(d2.awaiting_signal_step, None);
        assert_eq!(d2.awaiting_signal_name, None);
    }

    #[test]
    fn workflow_detail_parses_continued_as_new() {
        // A Continue-As-New run reports status "completed" PLUS the flag and the
        // server-computed continuation chain.
        let json = r#"{"workflow_id":7,"status":"completed","total_steps":2,"steps":[],
            "continued_as_new":true,"continuation_chain":[7,8,9]}"#;
        let detail: WorkflowRunDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.status, "completed");
        assert!(detail.continued_as_new);
        assert_eq!(detail.continuation_chain, vec![7, 8, 9]);
        // Absent in an older producer → false / empty via serde(default).
        let older = r#"{"workflow_id":7,"status":"completed","total_steps":2,"steps":[]}"#;
        let d2: WorkflowRunDetail = serde_json::from_str(older).unwrap();
        assert!(!d2.continued_as_new);
        assert!(d2.continuation_chain.is_empty());
    }

    #[test]
    fn swarm_graph_parses_and_usage_defaults_zero() {
        let json = r#"{"nodes":[{"id":"lead","agent_type":"supervisor","status":"completed"}],
            "edges":[{"from":"lead","to":"coder","edge_type":"delegation"}]}"#;
        let g: SwarmGraph = serde_json::from_str(json).unwrap();
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].prompt_tokens, 0); // not in /graph → default 0
        assert_eq!(g.edges[0].to, "coder");
    }
}
