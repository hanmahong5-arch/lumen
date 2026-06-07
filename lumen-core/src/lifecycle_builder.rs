//! Build a [`Lifecycle`] from the four kova read-surfaces.
//!
//! Pure (no I/O): the caller fetches/loads the inputs (live REST or on-disk
//! sidecars) and assembles the recovery chain (oldest→newest, cycle-guarded);
//! this module just projects them into the complementary views. Each view is
//! built independently and degrades gracefully when its source is absent — a
//! plain tool-less run yields an empty causal DAG, a workflow-only run yields a
//! timeline built from workflow steps, etc.

use crate::causal_types::{CausalEvent, SwarmGraph, WorkflowRunDetail};
use crate::lifecycle::{
    CausalEdge, CausalGraph, CausalNode, LIFECYCLE_SCHEMA_VERSION, Lifecycle, Provenance,
    SwarmView, SwarmViewEdge, SwarmViewNode, TimelineNode, WfStep, lane,
};
use crate::trace_types::{AgentTrace, TraceStepType};

/// Max characters kept for a step/output label.
const LABEL_CAP: usize = 80;

/// Honest provenance note: lumen reads kova-rest JSON; WAL integrity is a
/// server-side property of the deployment, not something lumen re-verifies.
const HMAC_NOTE: &str = "Views rendered from kova-rest trace/causal/workflow/swarm reads. \
     WAL integrity is SM3-HMAC tamper-evident server-side when the deployment sets \
     KOVA_WAL_HMAC_KEY; lumen surfaces, but does not itself re-verify, that property.";

/// Build the [`Lifecycle`] for a run.
///
/// `traces` is the recovery chain oldest→newest (empty for a workflow-only
/// run). `causal` / `workflow` / `swarm` are optional companion reads.
#[must_use]
pub fn build(
    traces: &[&AgentTrace],
    causal: Option<&[CausalEvent]>,
    workflow: Option<&WorkflowRunDetail>,
    swarm: Option<&SwarmGraph>,
) -> Lifecycle {
    let causal_graph = build_causal(causal.unwrap_or(&[]));
    let workflow_steps = workflow.map(build_workflow_steps).unwrap_or_default();
    let swarm_view = swarm.and_then(build_swarm_view);

    let (
        run_id,
        agent_id,
        status,
        started_ms,
        completed_ms,
        total_tokens,
        total_cost_usd,
        recovery_chain,
        timeline,
    ) = if traces.is_empty() {
        build_from_workflow(workflow, &workflow_steps)
    } else {
        build_from_traces(traces)
    };

    let provenance = Provenance {
        schema_version: LIFECYCLE_SCHEMA_VERSION,
        sources: collect_sources(traces, causal, workflow, swarm),
        wal_hmac_note: HMAC_NOTE.to_string(),
    };

    Lifecycle {
        run_id,
        agent_id,
        status,
        started_ms,
        completed_ms,
        total_tokens,
        total_cost_usd,
        timeline,
        causal: causal_graph,
        workflow_steps,
        swarm: swarm_view,
        recovery_chain,
        // T2: a workflow flag — true only when the run handed off via
        // Continue-As-New. Rendered as a continuation pill, not a recovery hop.
        continued_as_new: workflow.is_some_and(|w| w.continued_as_new),
        provenance,
    }
}

/// Header fields + a step-by-step swimlane built from the trace recovery chain.
#[allow(clippy::type_complexity)]
fn build_from_traces(
    traces: &[&AgentTrace],
) -> (
    String,
    String,
    String,
    u64,
    Option<u64>,
    u32,
    f64,
    Vec<String>,
    Vec<TimelineNode>,
) {
    // newest run drives the run-level identity/status; oldest drives start.
    let newest = traces[traces.len() - 1];
    let oldest = traces[0];

    let mut total_tokens: u32 = 0;
    let mut total_cost_usd: f64 = 0.0;
    let mut recovery_chain: Vec<String> = Vec::with_capacity(traces.len());
    let mut timeline: Vec<TimelineNode> = Vec::new();

    for (i, trace) in traces.iter().enumerate() {
        recovery_chain.push(trace.trace_id.clone());
        total_tokens = total_tokens.saturating_add(trace.total_tokens.total());
        total_cost_usd += trace.total_cost_usd;

        // A synthetic divider between runs makes the crash→resume visible.
        if i > 0 {
            timeline.push(TimelineNode {
                phase: "RESUMED HERE".to_string(),
                lane: lane::RECOVERY.to_string(),
                t_start: trace.started_at_ms,
                dur_ms: 0,
                tokens: 0,
                cost_usd: None,
                status: "resumed".to_string(),
                run_id: trace.trace_id.clone(),
            });
        }

        for step in &trace.steps {
            timeline.push(timeline_node_from_step(step, &trace.trace_id));
        }
    }

    (
        newest.trace_id.clone(),
        newest.agent_id.clone(),
        newest.status.to_string(),
        oldest.started_at_ms,
        newest.completed_at_ms,
        total_tokens,
        total_cost_usd,
        recovery_chain,
        timeline,
    )
}

/// Map one trace step to a swimlane node.
fn timeline_node_from_step(step: &crate::trace_types::TraceStep, trace_id: &str) -> TimelineNode {
    let is_recovery_marker = step
        .metadata
        .iter()
        .any(|(k, v)| k == "recovery" && v == "true");
    let (lane_id, status) = match &step.step_type {
        TraceStepType::LlmCall { .. } => (lane::LLM, "ok".to_string()),
        TraceStepType::ToolCall { success, .. } => (
            lane::TOOL,
            if *success { "ok" } else { "error" }.to_string(),
        ),
        TraceStepType::Checkpoint { .. } => (
            lane::WAIT,
            if is_recovery_marker { "resumed" } else { "ok" }.to_string(),
        ),
        TraceStepType::Error { .. } => (lane::TOOL, "error".to_string()),
    };
    TimelineNode {
        phase: step.step_type.to_string(),
        lane: lane_id.to_string(),
        t_start: step.started_at_ms,
        dur_ms: step.duration_ms,
        tokens: step.tokens.map_or(0, |t| t.total()),
        cost_usd: step.cost_usd,
        status,
        run_id: trace_id.to_string(),
    }
}

/// Header fields + a timeline built from workflow steps (no `AgentTrace`).
#[allow(clippy::type_complexity)]
fn build_from_workflow(
    workflow: Option<&WorkflowRunDetail>,
    steps: &[WfStep],
) -> (
    String,
    String,
    String,
    u64,
    Option<u64>,
    u32,
    f64,
    Vec<String>,
    Vec<TimelineNode>,
) {
    let Some(wf) = workflow else {
        // Nothing at all — an honest empty lifecycle.
        return (
            "unknown".to_string(),
            String::new(),
            "unknown".to_string(),
            0,
            None,
            0,
            0.0,
            vec!["unknown".to_string()],
            Vec::new(),
        );
    };
    let run_id = wf.workflow_id.to_string();
    let timeline: Vec<TimelineNode> = steps
        .iter()
        .map(|s| TimelineNode {
            // Prefer the step's output preview (label) so a declarative run
            // reads as its real story (e.g. "category=billing_dispute") rather
            // than a generic "step". Fall back to the derived kind when the step
            // produced no preview; `status` already carries await/sleep/error.
            phase: if s.label.is_empty() {
                format!("step {} · {}", s.index, s.kind)
            } else {
                format!("step {} · {}", s.index, s.label)
            },
            lane: lane::WORKFLOW.to_string(),
            t_start: s.t_start,
            dur_ms: s.dur_ms,
            tokens: 0,
            cost_usd: None,
            status: s.status.clone(),
            run_id: run_id.clone(),
        })
        .collect();
    (
        run_id.clone(),
        String::new(),
        wf.status.clone(),
        wf.started_at_ms,
        wf.terminal_at_ms,
        0,
        0.0,
        vec![run_id],
        timeline,
    )
}

/// Project the `format=causal` stream into a DAG. Empty in → empty out.
fn build_causal(events: &[CausalEvent]) -> CausalGraph {
    let nodes = events
        .iter()
        .map(|e| CausalNode {
            id: e.event_id,
            record_type: e.record_type.clone(),
            t_ns: e.timestamp_ns,
            summary: truncate(&e.summary, LABEL_CAP),
        })
        .collect();
    let edges = events
        .iter()
        .filter_map(|e| {
            e.caused_by.map(|from| CausalEdge {
                from,
                to: e.event_id,
            })
        })
        .collect();
    CausalGraph { nodes, edges }
}

/// Derive per-step kinds from the run-level workflow markers.
fn build_workflow_steps(wf: &WorkflowRunDetail) -> Vec<WfStep> {
    wf.steps
        .iter()
        .map(|s| {
            let (kind, status) = if wf.awaiting_step == Some(s.step_index) {
                ("await", "await")
            } else if wf.awaiting_signal_step == Some(s.step_index) {
                ("await_signal", "await_signal")
            } else if wf.sleeping_step == Some(s.step_index) {
                ("sleep", "sleep")
            } else if wf.failed_step == Some(s.step_index) {
                ("failed", "error")
            } else if wf.compensated_steps.contains(&s.step_index) {
                ("compensation", "ok")
            } else {
                ("step", "ok")
            };
            // A parked-on-signal step has no output yet; surface *which* signal it
            // waits for instead of an empty preview.
            let label = if kind == "await_signal" {
                match wf.awaiting_signal_name.as_deref() {
                    Some(n) if !n.is_empty() => truncate(&format!("await signal: {n}"), LABEL_CAP),
                    _ => "awaiting signal".to_string(),
                }
            } else {
                truncate(&s.output_preview, LABEL_CAP)
            };
            WfStep {
                index: s.step_index,
                kind: kind.to_string(),
                label,
                t_start: s.started_at_ms,
                dur_ms: s.duration_ms,
                status: status.to_string(),
            }
        })
        .collect()
}

/// Map the vendored `SwarmGraph` into the lifecycle swarm view. `None` when the
/// graph has no nodes (nothing to show).
fn build_swarm_view(graph: &SwarmGraph) -> Option<SwarmView> {
    if graph.nodes.is_empty() {
        return None;
    }
    Some(SwarmView {
        nodes: graph
            .nodes
            .iter()
            .map(|n| SwarmViewNode {
                id: n.id.clone(),
                agent_type: n.agent_type.clone(),
                status: n.status.clone(),
                prompt_tokens: n.prompt_tokens,
                completion_tokens: n.completion_tokens,
            })
            .collect(),
        edges: graph
            .edges
            .iter()
            .map(|e| SwarmViewEdge {
                from: e.from.clone(),
                to: e.to.clone(),
                edge_type: e.edge_type.clone(),
            })
            .collect(),
    })
}

/// Which sources actually contributed data (for the provenance badge).
fn collect_sources(
    traces: &[&AgentTrace],
    causal: Option<&[CausalEvent]>,
    workflow: Option<&WorkflowRunDetail>,
    swarm: Option<&SwarmGraph>,
) -> Vec<String> {
    let mut out = Vec::new();
    if !traces.is_empty() {
        out.push("trace".to_string());
    }
    if causal.is_some_and(|c| !c.is_empty()) {
        out.push("causal".to_string());
    }
    if workflow.is_some() {
        out.push("workflow".to_string());
    }
    if swarm.is_some_and(|s| !s.nodes.is_empty()) {
        out.push("swarm".to_string());
    }
    out
}

/// Truncate `s` to at most `max` chars on a char boundary (UI-safe).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::causal_types::{SwarmEdge, SwarmNode, WorkflowStep};
    use crate::trace_types::{AgentTrace, TokenUsage, TraceStatus, TraceStep, TraceStepType};

    fn llm_step(num: u32, t: u64, model: &str, p: u32, c: u32, cost: Option<f64>) -> TraceStep {
        TraceStep {
            step_num: num,
            step_type: TraceStepType::LlmCall {
                model: model.into(),
                finish_reason: "stop".into(),
            },
            started_at_ms: t,
            duration_ms: 100,
            tokens: Some(TokenUsage {
                prompt_tokens: p,
                completion_tokens: c,
            }),
            cost_usd: cost,
            metadata: vec![],
        }
    }

    fn tool_step(num: u32, t: u64, name: &str, ok: bool) -> TraceStep {
        TraceStep {
            step_num: num,
            step_type: TraceStepType::ToolCall {
                tool_name: name.into(),
                success: ok,
            },
            started_at_ms: t,
            duration_ms: 20,
            tokens: None,
            cost_usd: None,
            metadata: vec![],
        }
    }

    fn recovery_checkpoint(num: u32, t: u64, iter: u32) -> TraceStep {
        TraceStep {
            step_num: num,
            step_type: TraceStepType::Checkpoint { iteration: iter },
            started_at_ms: t,
            duration_ms: 0,
            tokens: None,
            cost_usd: None,
            metadata: vec![
                ("recovery".into(), "true".into()),
                ("resumed_at_iteration".into(), iter.to_string()),
            ],
        }
    }

    fn trace(id: &str, parent: Option<&str>, start: u64, steps: Vec<TraceStep>) -> AgentTrace {
        let mut tokens = TokenUsage::default();
        let mut cost = 0.0;
        for s in &steps {
            if let Some(t) = s.tokens {
                tokens.accumulate(t);
            }
            cost += s.cost_usd.unwrap_or(0.0);
        }
        AgentTrace {
            trace_id: id.into(),
            agent_id: "agent-1".into(),
            task_id: 0,
            steps,
            status: TraceStatus::Completed,
            total_tokens: tokens,
            total_cost_usd: cost,
            started_at_ms: start,
            completed_at_ms: Some(start + 1000),
            parent_trace_id: parent.map(str::to_string),
            schema_version: 1,
        }
    }

    #[test]
    fn timeline_from_single_trace_maps_lanes_and_cost() {
        let t = trace(
            "t-1",
            None,
            1000,
            vec![
                llm_step(0, 1000, "deepseek-chat", 200, 50, Some(0.0012)),
                tool_step(1, 1100, "reconcile", true),
            ],
        );
        let lc = build(&[&t], None, None, None);
        assert_eq!(lc.run_id, "t-1");
        assert_eq!(lc.timeline.len(), 2);
        assert_eq!(lc.timeline[0].lane, lane::LLM);
        assert_eq!(lc.timeline[0].tokens, 250);
        assert_eq!(lc.timeline[0].cost_usd, Some(0.0012));
        assert_eq!(lc.timeline[1].lane, lane::TOOL);
        assert_eq!(lc.timeline[1].status, "ok");
        assert_eq!(lc.total_tokens, 250);
        assert!((lc.total_cost_usd - 0.0012).abs() < 1e-9);
        // No causal edges → empty DAG (pure-text-ish run), not an error.
        assert!(lc.causal.nodes.is_empty());
        assert_eq!(lc.recovery_chain, vec!["t-1".to_string()]);
        assert_eq!(lc.provenance.sources, vec!["trace".to_string()]);
    }

    #[test]
    fn recovery_chain_inserts_resumed_divider() {
        let run_a = trace("t-a", None, 1000, vec![llm_step(0, 1000, "m", 10, 5, None)]);
        let run_b = trace(
            "t-b",
            Some("t-a"),
            5000,
            vec![
                recovery_checkpoint(0, 5000, 1),
                llm_step(1, 5100, "m", 20, 10, None),
            ],
        );
        let lc = build(&[&run_a, &run_b], None, None, None);
        assert_eq!(
            lc.recovery_chain,
            vec!["t-a".to_string(), "t-b".to_string()]
        );
        // One divider between the two runs.
        let dividers: Vec<&TimelineNode> = lc
            .timeline
            .iter()
            .filter(|n| n.lane == lane::RECOVERY && n.phase == "RESUMED HERE")
            .collect();
        assert_eq!(dividers.len(), 1);
        assert_eq!(dividers[0].t_start, 5000);
        // start = oldest run start; total spans both runs.
        assert_eq!(lc.started_ms, 1000);
        assert_eq!(lc.total_tokens, 45);
    }

    #[test]
    fn causal_edges_built_from_caused_by() {
        let events = vec![
            CausalEvent {
                event_id: 0,
                record_type: "AgentDirective".into(),
                timestamp_ns: 1000,
                caused_by: None,
                correlation_key: Some("c1".into()),
                summary: "tool: reconcile".into(),
            },
            CausalEvent {
                event_id: 1,
                record_type: "AgentDirectiveResult".into(),
                timestamp_ns: 2000,
                caused_by: Some(0),
                correlation_key: Some("c1".into()),
                summary: "ok".into(),
            },
        ];
        let t = trace("t-1", None, 1000, vec![llm_step(0, 1000, "m", 1, 1, None)]);
        let lc = build(&[&t], Some(&events), None, None);
        assert_eq!(lc.causal.nodes.len(), 2);
        assert_eq!(lc.causal.edges.len(), 1);
        assert_eq!(lc.causal.edges[0].from, 0);
        assert_eq!(lc.causal.edges[0].to, 1);
        assert!(lc.provenance.sources.contains(&"causal".to_string()));
    }

    #[test]
    fn workflow_only_run_builds_timeline_from_steps() {
        let wf = WorkflowRunDetail {
            workflow_id: 77,
            workflow_type: "reconcile-flow".into(),
            status: "completed".into(),
            total_steps: 2,
            completed_step_count: 2,
            steps: vec![
                WorkflowStep {
                    step_index: 0,
                    output_preview: "fetched".into(),
                    started_at_ms: 10,
                    duration_ms: 5,
                },
                WorkflowStep {
                    step_index: 1,
                    output_preview: "reconciled".into(),
                    started_at_ms: 15,
                    duration_ms: 8,
                },
            ],
            started_at_ms: 10,
            terminal_at_ms: Some(23),
            ..Default::default()
        };
        let lc = build(&[], None, Some(&wf), None);
        assert_eq!(lc.run_id, "77");
        assert_eq!(lc.timeline.len(), 2);
        assert_eq!(lc.timeline[0].lane, lane::WORKFLOW);
        assert_eq!(lc.workflow_steps.len(), 2);
        assert_eq!(lc.workflow_steps[0].kind, "step");
        assert_eq!(lc.started_ms, 10);
        assert_eq!(lc.completed_ms, Some(23));
        assert!(lc.provenance.sources.contains(&"workflow".to_string()));
        // A plain completion does NOT set the continuation flag.
        assert!(!lc.continued_as_new);
    }

    #[test]
    fn continue_as_new_sets_lifecycle_flag_not_recovery_chain() {
        // A Continue-As-New run reports status "completed" PLUS the flag; it is a
        // workflow flag, NOT a recovery-chain hop (that chain is trace-derived).
        let wf = WorkflowRunDetail {
            workflow_id: 88,
            workflow_type: "billing-cycle".into(),
            status: "completed".into(),
            total_steps: 1,
            completed_step_count: 1,
            steps: vec![WorkflowStep {
                step_index: 0,
                output_preview: "handed off".into(),
                started_at_ms: 10,
                duration_ms: 3,
            }],
            started_at_ms: 10,
            terminal_at_ms: Some(13),
            continued_as_new: true,
            ..Default::default()
        };
        let lc = build(&[], None, Some(&wf), None);
        assert!(lc.continued_as_new, "the CAN flag flows from the detail");
        // CAN is a workflow flag, not a crash → the recovery chain stays trivial.
        assert_eq!(lc.recovery_chain, vec!["88".to_string()]);
    }

    #[test]
    fn workflow_step_kinds_reflect_markers() {
        let wf = WorkflowRunDetail {
            workflow_id: 9,
            status: "failed".into(),
            total_steps: 3,
            steps: vec![
                WorkflowStep {
                    step_index: 0,
                    ..Default::default()
                },
                WorkflowStep {
                    step_index: 1,
                    ..Default::default()
                },
                WorkflowStep {
                    step_index: 2,
                    ..Default::default()
                },
            ],
            failed_step: Some(2),
            compensated_steps: vec![1],
            ..Default::default()
        };
        let lc = build(&[], None, Some(&wf), None);
        assert_eq!(lc.workflow_steps[1].kind, "compensation");
        assert_eq!(lc.workflow_steps[2].kind, "failed");
        assert_eq!(lc.workflow_steps[2].status, "error");
    }

    #[test]
    fn awaiting_signal_step_is_kind_await_signal_with_name() {
        let wf = WorkflowRunDetail {
            workflow_id: 5,
            status: "awaiting_signal".into(),
            total_steps: 2,
            steps: vec![
                WorkflowStep {
                    step_index: 0,
                    output_preview: "started".into(),
                    ..Default::default()
                },
                WorkflowStep {
                    step_index: 1,
                    ..Default::default()
                },
            ],
            awaiting_signal_step: Some(1),
            awaiting_signal_name: Some("approve_payment".into()),
            ..Default::default()
        };
        let lc = build(&[], None, Some(&wf), None);
        assert_eq!(lc.workflow_steps[1].kind, "await_signal");
        assert_eq!(lc.workflow_steps[1].status, "await_signal");
        // The signal name is surfaced in the label (which signal, not just that
        // it waits).
        assert!(
            lc.workflow_steps[1].label.contains("approve_payment"),
            "label surfaces the signal name: {}",
            lc.workflow_steps[1].label
        );
        // A missing name degrades to a plain marker, never an empty label.
        let wf2 = WorkflowRunDetail {
            workflow_id: 6,
            status: "awaiting_signal".into(),
            total_steps: 1,
            steps: vec![WorkflowStep {
                step_index: 0,
                ..Default::default()
            }],
            awaiting_signal_step: Some(0),
            ..Default::default()
        };
        let lc2 = build(&[], None, Some(&wf2), None);
        assert_eq!(lc2.workflow_steps[0].label, "awaiting signal");
    }

    #[test]
    fn swarm_view_built_from_graph() {
        let g = SwarmGraph {
            nodes: vec![
                SwarmNode {
                    id: "lead".into(),
                    agent_type: "supervisor".into(),
                    status: "completed".into(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
                SwarmNode {
                    id: "coder".into(),
                    agent_type: "worker".into(),
                    status: "completed".into(),
                    prompt_tokens: 200,
                    completion_tokens: 80,
                },
            ],
            edges: vec![SwarmEdge {
                from: "lead".into(),
                to: "coder".into(),
                edge_type: "delegation".into(),
            }],
        };
        let t = trace("t-1", None, 1000, vec![llm_step(0, 1000, "m", 1, 1, None)]);
        let lc = build(&[&t], None, None, Some(&g));
        let sv = lc.swarm.expect("swarm view present");
        assert_eq!(sv.nodes.len(), 2);
        assert_eq!(sv.edges.len(), 1);
        assert_eq!(sv.nodes[1].prompt_tokens, 200);
        assert!(lc.provenance.sources.contains(&"swarm".to_string()));
    }

    #[test]
    fn combined_fixture_all_sources() {
        // 4-step trace with a parent + 2 causal events — the plan's combined case.
        let parent = trace(
            "t-parent",
            None,
            1000,
            vec![llm_step(0, 1000, "m", 5, 5, None)],
        );
        let child = trace(
            "t-child",
            Some("t-parent"),
            5000,
            vec![
                recovery_checkpoint(0, 5000, 1),
                llm_step(1, 5100, "deepseek-chat", 100, 40, Some(0.001)),
                tool_step(2, 5200, "reconcile", true),
                llm_step(3, 5300, "deepseek-chat", 50, 20, Some(0.0005)),
            ],
        );
        let events = vec![
            CausalEvent {
                event_id: 0,
                record_type: "AgentDirective".into(),
                timestamp_ns: 5_200_000_000,
                caused_by: None,
                correlation_key: None,
                summary: "reconcile".into(),
            },
            CausalEvent {
                event_id: 1,
                record_type: "AgentDirectiveResult".into(),
                timestamp_ns: 5_250_000_000,
                caused_by: Some(0),
                correlation_key: None,
                summary: "ok".into(),
            },
        ];
        let lc = build(&[&parent, &child], Some(&events), None, None);
        assert_eq!(lc.recovery_chain.len(), 2);
        // parent(1) + divider(1) + child(4) = 6 timeline nodes.
        assert_eq!(lc.timeline.len(), 6);
        assert_eq!(lc.causal.edges.len(), 1);
        assert_eq!(lc.total_tokens, 5 + 5 + 100 + 40 + 50 + 20);
        assert!((lc.total_cost_usd - 0.0015).abs() < 1e-9);
        assert_eq!(
            lc.provenance.sources,
            vec!["trace".to_string(), "causal".to_string()]
        );
    }

    #[test]
    fn empty_everything_is_honest_not_panic() {
        let lc = build(&[], None, None, None);
        assert_eq!(lc.run_id, "unknown");
        assert!(lc.timeline.is_empty());
        assert!(lc.causal.nodes.is_empty());
        assert!(lc.provenance.sources.is_empty());
    }
}
