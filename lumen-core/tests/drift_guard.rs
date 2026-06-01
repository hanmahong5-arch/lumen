//! Rust↔Python drift guard.
//!
//! Parses the SAME shared fixture (`tests/fixtures/trace_examples.json`) that the
//! Python test suite parses (`lumen-sdk/tests/test_drift_guard.py`) and asserts
//! identical key fields. If the vendored Rust `AgentTrace` and the Python
//! `trace_reader` dataclass diverge in how they read the on-disk Kova trace
//! shape, one of the two suites breaks here.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use lumen_core::trace_types::{AgentTrace, TraceStatus, TraceStepType};

/// Load the shared fixture as a map of name -> AgentTrace.
///
/// The fixture is a JSON object whose values are canonical AgentTrace examples
/// (plus a leading `_comment` string that we skip).
fn load_examples() -> std::collections::HashMap<String, AgentTrace> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("trace_examples.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&data).unwrap();

    let mut out = std::collections::HashMap::new();
    for (name, value) in raw {
        if name.starts_with('_') {
            continue; // skip _comment
        }
        let trace: AgentTrace = serde_json::from_value(value)
            .unwrap_or_else(|e| panic!("parse fixture example {name}: {e}"));
        out.insert(name, trace);
    }
    out
}

#[test]
fn normal_multi_step_parses() {
    let ex = load_examples();
    let t = &ex["normal_multi_step"];
    assert_eq!(t.trace_id, "trace_normal_001");
    assert_eq!(t.agent_id, "research-agent");
    assert_eq!(t.status, TraceStatus::Completed);
    assert_eq!(t.steps.len(), 3);
    assert_eq!(t.total_tokens.total(), 1550);
    // A non-zero total_cost_usd must survive the round-trip verbatim.
    assert!((t.total_cost_usd - 0.0123).abs() < f64::EPSILON);
    // Fresh run: no parent.
    assert_eq!(t.parent_trace_id, None);
    // First step metadata preserved.
    assert_eq!(
        t.steps[0].metadata,
        vec![("phase".to_string(), "plan".to_string())]
    );
}

#[test]
fn recovered_run_has_parent_and_recovery_marker() {
    let ex = load_examples();
    let t = &ex["recovered_with_parent"];
    assert_eq!(t.trace_id, "trace_recovered_002");
    // parent_trace_id links back to the crashed run.
    assert_eq!(t.parent_trace_id.as_deref(), Some("trace_normal_001"));

    // First step is a recovery Checkpoint carrying the recovery markers.
    let first = &t.steps[0];
    assert!(
        matches!(first.step_type, TraceStepType::Checkpoint { iteration: 4 }),
        "expected recovery Checkpoint at iteration 4, got {:?}",
        first.step_type
    );
    assert_eq!(
        first.metadata,
        vec![
            ("recovery".to_string(), "true".to_string()),
            ("resumed_at_iteration".to_string(), "4".to_string()),
        ]
    );
}

#[test]
fn legacy_trace_without_parent_field_still_loads() {
    let ex = load_examples();
    let t = &ex["legacy_no_parent_field"];
    assert_eq!(t.trace_id, "trace_legacy_003");
    // The JSON has no `parent_trace_id` key at all; serde(default) → None.
    assert_eq!(t.parent_trace_id, None);
    assert_eq!(t.status, TraceStatus::Completed);
}
