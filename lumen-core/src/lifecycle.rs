//! The `Lifecycle` — an agent run's whole story as a set of **complementary
//! views**, not one forced graph.
//!
//! A single run produces several kinds of evidence that don't share one node
//! set: a step timeline (from [`AgentTrace`](crate::trace_types::AgentTrace)),
//! a causal DAG (from the `format=causal` stream), declarative workflow steps
//! (from `WorkflowRunDetail`), and swarm delegations (from `SwarmGraph`). The
//! `Lifecycle` holds each as its own view so the renderer can show them
//! side-by-side. It is built by [`crate::lifecycle_builder::build`] and is
//! `Serialize`/`Deserialize` so it can be embedded verbatim into the
//! self-contained export HTML and re-read by the live dashboard route.

use serde::{Deserialize, Serialize};

/// Lifecycle schema version, stamped into [`Provenance::schema_version`].
pub const LIFECYCLE_SCHEMA_VERSION: u32 = 1;

/// Lane a [`TimelineNode`] belongs to (the swimlane it renders in).
pub mod lane {
    /// LLM completion call.
    pub const LLM: &str = "Llm";
    /// Tool execution.
    pub const TOOL: &str = "Tool";
    /// Wait / checkpoint (durable sleep, await, plain checkpoint).
    pub const WAIT: &str = "Wait";
    /// Declarative workflow step.
    pub const WORKFLOW: &str = "Workflow";
    /// Swarm delegation activity.
    pub const SWARM: &str = "Swarm";
    /// Crash→resume marker (the synthetic "RESUMED HERE" divider).
    pub const RECOVERY: &str = "Recovery";
}

/// One bar in the swimlane timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineNode {
    /// Human-readable label (e.g. `llm_call(deepseek-chat)` / `RESUMED HERE`).
    pub phase: String,
    /// Lane id (see [`lane`]).
    pub lane: String,
    /// Start time, Unix-epoch ms.
    pub t_start: u64,
    /// Duration in ms.
    pub dur_ms: u64,
    /// Tokens for this step (`0` for non-LLM steps).
    pub tokens: u32,
    /// Per-step USD (only LLM steps that hit a priced model).
    pub cost_usd: Option<f64>,
    /// `"ok"` / `"error"` / `"resumed"`.
    pub status: String,
    /// Trace id this node belongs to (which run in the recovery chain).
    pub run_id: String,
}

/// A node in the causal DAG (one per `CausalEvent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// `event_id` from the causal stream (DAG node id).
    pub id: u64,
    /// Kova record-type variant name.
    pub record_type: String,
    /// Wall-clock nanoseconds.
    pub t_ns: u64,
    /// Short UI-safe summary.
    pub summary: String,
}

/// A directed causal edge (`from` caused `to`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Upstream cause `event_id`.
    pub from: u64,
    /// Downstream effect `event_id`.
    pub to: u64,
}

/// The causal DAG view (empty for a pure-text run with no detected edges).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalGraph {
    /// DAG nodes.
    pub nodes: Vec<CausalNode>,
    /// DAG edges.
    pub edges: Vec<CausalEdge>,
}

/// A declarative-workflow step view (sleep / branch / await / retry / etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfStep {
    /// 0-based step index (gaps reflect a branch path).
    pub index: u32,
    /// Derived kind: `step` / `await` / `sleep` / `retry` / `compensation` /
    /// `failed`.
    pub kind: String,
    /// Short label (output preview, capped).
    pub label: String,
    /// Start time, Unix-epoch ms.
    pub t_start: u64,
    /// Duration in ms.
    pub dur_ms: u64,
    /// `"ok"` / `"error"` / `"await"` / `"sleep"` / `"retry"`.
    pub status: String,
}

/// A swarm graph node in the lifecycle view (real per-agent token spend).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmViewNode {
    /// Agent id.
    pub id: String,
    /// `"supervisor"` / `"worker"`.
    pub agent_type: String,
    /// Agent status.
    pub status: String,
    /// Real prompt tokens.
    pub prompt_tokens: u32,
    /// Real completion tokens.
    pub completion_tokens: u32,
}

/// A swarm delegation/handoff edge in the lifecycle view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmViewEdge {
    /// Source agent id.
    pub from: String,
    /// Target agent id.
    pub to: String,
    /// `"delegation"` / `"handoff"`.
    pub edge_type: String,
}

/// The swarm view (present only when the run delegated to a swarm).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmView {
    /// Agent nodes with real token spend.
    pub nodes: Vec<SwarmViewNode>,
    /// Delegation edges.
    pub edges: Vec<SwarmViewEdge>,
}

/// Where each view came from + the tamper-evidence note, for the provenance
/// badge on the exported artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// [`LIFECYCLE_SCHEMA_VERSION`] at build time.
    pub schema_version: u32,
    /// Which sources actually contributed (`trace` / `causal` / `workflow` /
    /// `swarm`).
    pub sources: Vec<String>,
    /// Human note on WAL authentication (HMAC) for the badge.
    pub wal_hmac_note: String,
}

/// The whole run, as complementary views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lifecycle {
    /// Run id (newest trace id, or the workflow id for a workflow-only run).
    pub run_id: String,
    /// Agent id (empty for a workflow-only run).
    pub agent_id: String,
    /// Overall status string.
    pub status: String,
    /// Earliest start across the recovery chain, Unix-epoch ms.
    pub started_ms: u64,
    /// Latest completion, Unix-epoch ms (`None` while running).
    pub completed_ms: Option<u64>,
    /// Total tokens across the whole story.
    pub total_tokens: u32,
    /// Total USD across the whole story.
    pub total_cost_usd: f64,
    /// Swimlane timeline (from `AgentTrace` steps, or workflow steps).
    pub timeline: Vec<TimelineNode>,
    /// Causal DAG (from the `format=causal` stream).
    pub causal: CausalGraph,
    /// Declarative workflow steps (from `WorkflowRunDetail`).
    pub workflow_steps: Vec<WfStep>,
    /// Swarm delegations, when present.
    pub swarm: Option<SwarmView>,
    /// Recovery chain of trace ids, oldest→newest (len 1 = no crash).
    pub recovery_chain: Vec<String>,
    /// `true` when the run handed off via Continue-As-New (workflow flag,
    /// derived from [`WorkflowRunDetail::continued_as_new`]). Rendered as a
    /// distinct continuation pill — NOT a recovery-chain hop (that chain is
    /// agent-trace-derived; a workflow-only CAN never enters it). Defaults to
    /// `false`, so older exports keep loading.
    #[serde(default)]
    pub continued_as_new: bool,
    /// Provenance + tamper-evidence note.
    pub provenance: Provenance,
}
