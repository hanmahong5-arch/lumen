//! Rust↔Python behavioral parity for cost + replay.
//!
//! Materializes each example from `fixtures/trace_examples.json` into a temp
//! trace dir, runs `CostTracker` / `ReplayEngine`, and asserts against the
//! shared `fixtures/behavior_expectations.json` that the Python suite also
//! loads. Catches algorithm drift (not just parse drift).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};

use lumen_core::cost::CostTracker;
use lumen_core::replay::ReplayEngine;
use lumen_core::trace_types::AgentTrace;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Expectation {
    trace_id: String,
    cost_usd: f64,
    replay_success: bool,
    replay_iterations: u32,
    replay_original_cost_usd: f64,
    replay_parent: Option<String>,
    replay_resumed_at: Option<u32>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_examples() -> Vec<(String, AgentTrace, serde_json::Value)> {
    let path = fixtures_dir().join("trace_examples.json");
    let data = fs::read_to_string(&path).unwrap();
    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&data).unwrap();
    let mut out = Vec::new();
    for (name, value) in raw {
        if name.starts_with('_') {
            continue;
        }
        let trace: AgentTrace = serde_json::from_value(value.clone()).unwrap();
        out.push((name, trace, value));
    }
    out
}

fn load_expectations() -> std::collections::HashMap<String, Expectation> {
    let path = fixtures_dir().join("behavior_expectations.json");
    let data = fs::read_to_string(&path).unwrap();
    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&data).unwrap();
    let mut out = std::collections::HashMap::new();
    for (name, value) in raw {
        if name.starts_with('_') {
            continue;
        }
        out.insert(name, serde_json::from_value(value).unwrap());
    }
    out
}

fn materialize(dir: &Path, body: &serde_json::Value, trace_id: &str) {
    let path = dir.join(format!("{trace_id}.json"));
    fs::write(path, serde_json::to_string_pretty(body).unwrap()).unwrap();
}

#[test]
fn cost_and_replay_match_shared_expectations() {
    let expectations = load_expectations();
    let examples = load_examples();
    assert_eq!(
        examples.len(),
        expectations.len(),
        "every fixture example needs an expectation entry"
    );

    for (name, trace, body) in examples {
        let exp = expectations
            .get(&name)
            .unwrap_or_else(|| panic!("missing expectation for {name}"));
        assert_eq!(trace.trace_id, exp.trace_id);

        let dir = std::env::temp_dir().join(format!("lumen_parity_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        materialize(&dir, &body, &trace.trace_id);

        // CostTracker is the public surface (trace_cost is crate-private).
        let report = CostTracker::new(&dir).report(0).unwrap();
        assert_eq!(report.total_runs, 1, "{name}");
        assert!(
            (report.total_usd - exp.cost_usd).abs() < 1e-9,
            "{name}: report total {} != {}",
            report.total_usd,
            exp.cost_usd
        );

        let replay = ReplayEngine::new(&dir).replay(&trace.trace_id).unwrap();
        assert_eq!(replay.success, exp.replay_success, "{name}");
        assert_eq!(replay.total_iterations, exp.replay_iterations, "{name}");
        assert!(
            (replay.original_cost_usd - exp.replay_original_cost_usd).abs() < 1e-9,
            "{name}: original_cost {} != {}",
            replay.original_cost_usd,
            exp.replay_original_cost_usd
        );
        let parent = replay.recovery.as_ref().map(|r| r.parent_trace_id.clone());
        let resumed = replay
            .recovery
            .as_ref()
            .and_then(|r| r.resumed_at_iteration);
        assert_eq!(parent, exp.replay_parent, "{name}");
        assert_eq!(resumed, exp.replay_resumed_at, "{name}");

        let _ = fs::remove_dir_all(&dir);
    }
}
