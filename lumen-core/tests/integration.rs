//! Integration tests for lumen-core using mock trace files.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use lumen_core::cost::CostTracker;
use lumen_core::replay::ReplayEngine;
use lumen_core::trace::TraceStore;

use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a temp directory with test trace files.
fn setup_trace_dir() -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("lumen_test_{}_{id}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a successful trace
    let trace1 = serde_json::json!({
        "trace_id": "trace_abc123",
        "agent_id": "research-agent",
        "task_id": 42,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "LlmCall": { "model": "claude-sonnet-4-6", "finish_reason": "tool_use" } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 1200,
                "tokens": { "prompt_tokens": 500, "completion_tokens": 100 },
                "metadata": []
            },
            {
                "step_num": 1,
                "step_type": { "ToolCall": { "tool_name": "search_web", "success": true } },
                "started_at_ms": 1710000001200_u64,
                "duration_ms": 800,
                "tokens": null,
                "metadata": []
            },
            {
                "step_num": 2,
                "step_type": { "LlmCall": { "model": "claude-sonnet-4-6", "finish_reason": "tool_use" } },
                "started_at_ms": 1710000002000_u64,
                "duration_ms": 900,
                "tokens": { "prompt_tokens": 800, "completion_tokens": 150 },
                "metadata": []
            },
            {
                "step_num": 3,
                "step_type": { "ToolCall": { "tool_name": "read_url", "success": true } },
                "started_at_ms": 1710000002900_u64,
                "duration_ms": 500,
                "tokens": null,
                "metadata": []
            },
            {
                "step_num": 4,
                "step_type": { "LlmCall": { "model": "claude-sonnet-4-6", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000003400_u64,
                "duration_ms": 1500,
                "tokens": { "prompt_tokens": 1200, "completion_tokens": 300 },
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 2500, "completion_tokens": 550 },
        "total_cost_usd": 0.12,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000004900_u64
    });

    // Write a failed trace (older)
    let trace2 = serde_json::json!({
        "trace_id": "trace_def456",
        "agent_id": "summary-agent",
        "task_id": 99,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "LlmCall": { "model": "gpt-4o", "finish_reason": "tool_use" } },
                "started_at_ms": 1709990000000_u64,
                "duration_ms": 2000,
                "tokens": { "prompt_tokens": 1000, "completion_tokens": 200 },
                "metadata": []
            },
            {
                "step_num": 1,
                "step_type": { "ToolCall": { "tool_name": "run_code", "success": false } },
                "started_at_ms": 1709990002000_u64,
                "duration_ms": 5000,
                "tokens": null,
                "metadata": []
            },
            {
                "step_num": 2,
                "step_type": { "Error": { "message": "tool timeout after 5000ms" } },
                "started_at_ms": 1709990007000_u64,
                "duration_ms": 0,
                "tokens": null,
                "metadata": []
            }
        ],
        "status": { "Failed": "tool timeout" },
        "total_tokens": { "prompt_tokens": 1000, "completion_tokens": 200 },
        "total_cost_usd": 0.08,
        "started_at_ms": 1709990000000_u64,
        "completed_at_ms": 1709990007000_u64
    });

    std::fs::write(
        dir.join("trace_abc123.json"),
        serde_json::to_string_pretty(&trace1).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("trace_def456.json"),
        serde_json::to_string_pretty(&trace2).unwrap(),
    )
    .unwrap();

    // Write a non-trace JSON file (should be skipped)
    std::fs::write(dir.join("config.json"), r#"{"not": "a trace"}"#).unwrap();

    // Write a non-JSON file (should be skipped)
    std::fs::write(dir.join("notes.txt"), "some notes").unwrap();

    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_trace_listing() {
    let dir = setup_trace_dir();

    let store = TraceStore::new(&dir);
    let traces = store.list().unwrap();

    assert_eq!(traces.len(), 2, "should find exactly 2 traces");

    // Newest first
    assert_eq!(traces[0].trace_id, "trace_abc123");
    assert_eq!(traces[0].agent_name, "research-agent");
    assert!(traces[0].success);
    assert_eq!(traces[0].iterations, 3); // 3 LLM calls
    assert!((traces[0].cost_usd - 0.12).abs() < f64::EPSILON);
    assert_eq!(traces[0].duration_ms, 4900);
    assert_eq!(traces[0].prompt_tokens, 2500);
    assert_eq!(traces[0].completion_tokens, 550);

    assert_eq!(traces[1].trace_id, "trace_def456");
    assert_eq!(traces[1].agent_name, "summary-agent");
    assert!(!traces[1].success);

    cleanup(&dir);
}

#[test]
fn test_replay_success() {
    let dir = setup_trace_dir();

    let engine = ReplayEngine::new(&dir);
    let replay = engine.replay("trace_abc123").unwrap();

    assert_eq!(replay.trace_id, "trace_abc123");
    assert_eq!(replay.agent_name, "research-agent");
    assert!(replay.success);
    assert_eq!(replay.total_iterations, 3);
    assert!((replay.original_cost_usd - 0.12).abs() < f64::EPSILON);

    // Step 1: LLM(tool_use) → search_web
    assert_eq!(replay.steps.len(), 3);
    assert_eq!(replay.steps[0].step, 1);
    match &replay.steps[0].decision {
        lumen_core::replay::ReplayDecision::ToolCalls { calls } => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_name, "search_web");
            assert!(calls[0].success);
        }
        _ => panic!("expected ToolCalls"),
    }

    // Step 2: LLM(tool_use) → read_url
    match &replay.steps[1].decision {
        lumen_core::replay::ReplayDecision::ToolCalls { calls } => {
            assert_eq!(calls[0].tool_name, "read_url");
        }
        _ => panic!("expected ToolCalls"),
    }

    // Step 3: LLM(end_turn) → Final answer
    match &replay.steps[2].decision {
        lumen_core::replay::ReplayDecision::FinalAnswer { content } => {
            assert!(content.contains("end_turn"));
        }
        _ => panic!("expected FinalAnswer"),
    }

    cleanup(&dir);
}

#[test]
fn test_replay_failed_run() {
    let dir = setup_trace_dir();

    let engine = ReplayEngine::new(&dir);
    let replay = engine.replay("trace_def456").unwrap();

    assert!(!replay.success);
    assert_eq!(replay.agent_name, "summary-agent");

    // Should have tool call step + error step
    assert!(replay.steps.len() >= 2);

    // Last step should be the error
    let last = replay.steps.last().unwrap();
    match &last.decision {
        lumen_core::replay::ReplayDecision::FinalAnswer { content } => {
            assert!(content.contains("timeout"));
        }
        _ => panic!("expected error as FinalAnswer"),
    }

    cleanup(&dir);
}

#[test]
fn test_replay_not_found() {
    let dir = setup_trace_dir();

    let engine = ReplayEngine::new(&dir);
    let result = engine.replay("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("nonexistent"));

    cleanup(&dir);
}

#[test]
fn test_cost_report() {
    let dir = setup_trace_dir();

    let tracker = CostTracker::new(&dir);
    // Report for all time (since_ms = 0)
    let report = tracker.report(0).unwrap();

    assert_eq!(report.total_runs, 2);
    assert!((report.total_usd - 0.20).abs() < 0.01);
    assert_eq!(report.total_prompt_tokens, 3500); // 2500 + 1000
    assert_eq!(report.total_completion_tokens, 750); // 550 + 200

    // By agent
    assert_eq!(report.by_agent.len(), 2);
    assert!((report.by_agent["research-agent"] - 0.12).abs() < f64::EPSILON);
    assert!((report.by_agent["summary-agent"] - 0.08).abs() < f64::EPSILON);

    // By model
    assert!(report.by_model.contains_key("claude-sonnet-4-6"));
    assert!(report.by_model.contains_key("gpt-4o"));

    cleanup(&dir);
}

#[test]
fn test_cost_report_time_filter() {
    let dir = setup_trace_dir();

    let tracker = CostTracker::new(&dir);
    // Only recent trace (started >= 1710000000000)
    let report = tracker.report(1710000000000).unwrap();

    assert_eq!(report.total_runs, 1);
    assert!((report.total_usd - 0.12).abs() < f64::EPSILON);

    cleanup(&dir);
}

/// Helper: write a single trace JSON into a fresh temp dir and return the dir.
fn dir_with_one_trace(name: &str, trace: &serde_json::Value) -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("lumen_cost_{}_{name}_{id}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let trace_id = trace["trace_id"].as_str().unwrap();
    std::fs::write(
        dir.join(format!("{trace_id}.json")),
        serde_json::to_string_pretty(trace).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn test_replay_surfaces_recovery_marker() {
    // A crash-recovered run: parent_trace_id set + recovery-marker Checkpoint.
    let trace = serde_json::json!({
        "trace_id": "trace_resumed",
        "agent_id": "durable-agent",
        "task_id": 5,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "Checkpoint": { "iteration": 4 } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 0,
                "tokens": null,
                "metadata": [["recovery", "true"], ["resumed_at_iteration", "4"]]
            },
            {
                "step_num": 1,
                "step_type": { "LlmCall": { "model": "gpt-4o", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000000050_u64,
                "duration_ms": 1100,
                "tokens": { "prompt_tokens": 900, "completion_tokens": 120 },
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 900, "completion_tokens": 120 },
        "total_cost_usd": 0.05,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000001150_u64,
        "parent_trace_id": "trace_original"
    });
    let dir = dir_with_one_trace("recovery", &trace);

    let replay = ReplayEngine::new(&dir).replay("trace_resumed").unwrap();
    let recovery = replay
        .recovery
        .expect("recovered run should carry recovery info");
    assert_eq!(recovery.parent_trace_id, "trace_original");
    assert_eq!(recovery.resumed_at_iteration, Some(4));

    // A fresh run (no parent_trace_id) carries no recovery info.
    let fresh_dir = setup_trace_dir();
    let fresh = ReplayEngine::new(&fresh_dir)
        .replay("trace_abc123")
        .unwrap();
    assert!(fresh.recovery.is_none());

    cleanup(&fresh_dir);
    cleanup(&dir);
}

#[test]
fn test_cost_uses_kova_total_verbatim_when_nonzero() {
    // total_cost_usd is non-zero AND deliberately disagrees with what the
    // pricing table would estimate from the tokens. Kova's number must win.
    let trace = serde_json::json!({
        "trace_id": "trace_realcost",
        "agent_id": "billed-agent",
        "task_id": 1,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "LlmCall": { "model": "claude-opus-4-6", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 1000,
                // 10k/5k tokens on opus would estimate to (10000*15 + 5000*75)/1000 = $525,
                // nowhere near the authoritative 0.4242 below.
                "tokens": { "prompt_tokens": 10000, "completion_tokens": 5000 },
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 10000, "completion_tokens": 5000 },
        "total_cost_usd": 0.4242,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000001000_u64
    });
    let dir = dir_with_one_trace("verbatim", &trace);

    let report = CostTracker::new(&dir).report(0).unwrap();
    assert_eq!(report.total_runs, 1);
    // Used verbatim — NOT the ~$525 token estimate.
    assert!(
        (report.total_usd - 0.4242).abs() < 1e-9,
        "expected verbatim 0.4242, got {}",
        report.total_usd
    );
    assert!((report.by_agent["billed-agent"] - 0.4242).abs() < 1e-9);

    cleanup(&dir);
}

#[test]
fn test_cost_falls_back_to_estimate_when_total_is_zero() {
    // Legacy trace: total_cost_usd == 0.0 → estimate from per-step tokens.
    let trace = serde_json::json!({
        "trace_id": "trace_legacy_cost",
        "agent_id": "legacy-agent",
        "task_id": 2,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "LlmCall": { "model": "gpt-4o", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 1000,
                "tokens": { "prompt_tokens": 1000, "completion_tokens": 500 },
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 1000, "completion_tokens": 500 },
        "total_cost_usd": 0.0,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000001000_u64
    });
    let dir = dir_with_one_trace("estimate", &trace);

    let report = CostTracker::new(&dir).report(0).unwrap();
    assert_eq!(report.total_runs, 1);
    // gpt-4o = $2.50/$10.00 per 1k → (1000*2.5 + 500*10)/1000 = 2.5 + 5.0 = $7.50
    let expected = (1000.0 * 2.50 + 500.0 * 10.0) / 1000.0;
    assert!(
        (report.total_usd - expected).abs() < 1e-9,
        "expected estimate {expected}, got {}",
        report.total_usd
    );
    assert!(
        report.total_usd > 0.0,
        "fallback must produce a non-zero estimate"
    );

    cleanup(&dir);
}

#[test]
fn test_cost_by_model_uses_per_step_tokens_not_equal_split() {
    // One trace with two LLM steps: one cheap (gpt-4o-mini), one expensive
    // (claude-opus-4-6). Equal split would give each 50% of the trace total;
    // per-step attribution must give each step its own token-derived cost.
    let trace = serde_json::json!({
        "trace_id": "trace_mixed_models",
        "agent_id": "mixed-agent",
        "task_id": 1,
        "steps": [
            {
                "step_num": 0,
                // gpt-4o-mini: $0.15/$0.60 per 1k. 100 prompt + 50 completion
                // = (100*0.15 + 50*0.60)/1000 = (15 + 30)/1000 = $0.045
                "step_type": { "LlmCall": { "model": "gpt-4o-mini", "finish_reason": "tool_use" } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 500,
                "tokens": { "prompt_tokens": 100, "completion_tokens": 50 },
                "metadata": []
            },
            {
                "step_num": 1,
                // claude-opus-4-6: $15/$75 per 1k. 200 prompt + 100 completion
                // = (200*15 + 100*75)/1000 = (3000 + 7500)/1000 = $10.50
                "step_type": { "LlmCall": { "model": "claude-opus-4-6", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000000500_u64,
                "duration_ms": 1500,
                "tokens": { "prompt_tokens": 200, "completion_tokens": 100 },
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 300, "completion_tokens": 150 },
        "total_cost_usd": 0.0,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000002000_u64
    });
    let dir = dir_with_one_trace("mixed_models", &trace);
    let report = CostTracker::new(&dir).report(0).unwrap();

    let mini_cost = (100.0 * 0.15 + 50.0 * 0.60) / 1000.0; // $0.045
    let opus_cost = (200.0 * 15.0 + 100.0 * 75.0) / 1000.0; // $10.50

    let actual_mini = report.by_model["gpt-4o-mini"];
    let actual_opus = report.by_model["claude-opus-4-6"];

    assert!(
        (actual_mini - mini_cost).abs() < 1e-9,
        "gpt-4o-mini: expected {mini_cost}, got {actual_mini}"
    );
    assert!(
        (actual_opus - opus_cost).abs() < 1e-9,
        "claude-opus-4-6: expected {opus_cost}, got {actual_opus}"
    );

    // The two must NOT be equal (the old equal-split bug would have made them
    // half of the trace total each).
    assert!(
        (actual_mini - actual_opus).abs() > 1.0,
        "per-model costs should differ by >$1, not be equal-split equal"
    );

    cleanup(&dir);
}

#[test]
fn test_cost_by_model_none_tokens_attributes_zero() {
    // A step with tokens: null must not smear any cost onto the model — it
    // contributes $0.0 to by_model for that model.
    let trace = serde_json::json!({
        "trace_id": "trace_no_tokens",
        "agent_id": "agent-x",
        "task_id": 2,
        "steps": [
            {
                "step_num": 0,
                "step_type": { "LlmCall": { "model": "gpt-4o", "finish_reason": "end_turn" } },
                "started_at_ms": 1710000000000_u64,
                "duration_ms": 100,
                "tokens": null,
                "metadata": []
            }
        ],
        "status": "Completed",
        "total_tokens": { "prompt_tokens": 0, "completion_tokens": 0 },
        "total_cost_usd": 0.0,
        "started_at_ms": 1710000000000_u64,
        "completed_at_ms": 1710000000100_u64
    });
    let dir = dir_with_one_trace("no_tokens", &trace);
    let report = CostTracker::new(&dir).report(0).unwrap();

    // Model key may exist but cost must be exactly 0.
    if let Some(&cost) = report.by_model.get("gpt-4o") {
        assert_eq!(cost, 0.0, "null-token step must attribute $0 to by_model");
    }

    cleanup(&dir);
}

#[test]
fn test_empty_directory() {
    let dir = std::env::temp_dir().join(format!("lumen_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let store = TraceStore::new(&dir);
    assert!(store.list().unwrap().is_empty());

    let engine = ReplayEngine::new(&dir);
    assert!(engine.list_traces().unwrap().is_empty());

    let tracker = CostTracker::new(&dir);
    let report = tracker.report(0).unwrap();
    assert_eq!(report.total_runs, 0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_nonexistent_directory() {
    let dir = PathBuf::from("/tmp/lumen_nonexistent_dir_12345");

    // Should return empty, not error
    let store = TraceStore::new(&dir);
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn test_list_traces() {
    let dir = setup_trace_dir();

    let engine = ReplayEngine::new(&dir);
    let ids = engine.list_traces().unwrap();

    assert_eq!(ids.len(), 2);
    // Newest first
    assert_eq!(ids[0], "trace_abc123");
    assert_eq!(ids[1], "trace_def456");

    cleanup(&dir);
}
