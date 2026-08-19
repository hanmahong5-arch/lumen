//! The offline sample run behind `lumen demo --sample`.
//!
//! `lumen demo`'s live path spins up a `kova-rest` and drives a real agent — a
//! great second look, but it needs a backend binary and an LLM key, so from a
//! clean `git clone` it cannot run at all. This module is the first look that
//! always works: a checked-in two-run recovery chain, materialized into a temp
//! trace directory and handed to the same export path the live run uses.
//!
//! It is *sample data*, never presented as a real run — the exported lifecycle
//! carries the same provenance the live path does, and `demo` prints the run
//! ids so the follow-on commands (`traces` / `cost` / `replay` / `export`) work
//! on it verbatim.
//!
//! The story it tells is the one the tool exists for:
//!   1. a run whose cost is 95% one step (per-step attribution vs equal-split),
//!   2. a tool call refused by policy — not the same thing as a failure,
//!   3. a crash, then a resume from the checkpoint that does *not* redo the
//!      expensive step (the recovery chain stitched into one story).
//!
//! Timestamps in the embedded JSON are rebased at write time so the run reads
//! as "a couple of minutes ago"; otherwise `lumen cost`, which windows on the
//! last 24h, would report an empty report right after the demo wrote the files.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// The run the demo exports — the *recovered* run, whose `parent_trace_id`
/// pulls the crashed first attempt into the same lifecycle.
pub const RUN_ID: &str = "lumen-sample-run-b";

/// The crashed first attempt, referenced by [`RUN_ID`]'s `parent_trace_id`.
pub const PARENT_RUN_ID: &str = "lumen-sample-run-a";

const TRACE_A: &str = include_str!("sample/lumen-sample-run-a.json");
const TRACE_B: &str = include_str!("sample/lumen-sample-run-b.json");
const CAUSAL: &str = include_str!("sample/lumen-sample.causal.json");

/// Latest `completed_at_ms` in the embedded fixtures. Rebasing is relative to
/// this, so editing the fixtures only requires keeping this in sync with the
/// last timestamp in them.
const FIXTURE_END_MS: i64 = 1_755_500_183_076;

/// How far in the past the rebased run should end. Two minutes reads as "just
/// now" while staying comfortably inside `lumen cost`'s 24h window.
const ENDS_AGO_MS: i64 = 120_000;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(FIXTURE_END_MS, |d| {
            i64::try_from(d.as_millis()).unwrap_or(FIXTURE_END_MS)
        })
}

/// Shift every wall-clock field in `v` by `shift_ms`.
///
/// Walks the whole document rather than the known field paths so that adding a
/// step or a causal event to the fixtures needs no change here. Millisecond
/// fields are matched by the `_ms` suffix; `timestamp_ns` is the one nanosecond
/// field in the causal sidecar.
fn rebase(v: &mut Value, shift_ms: i64) {
    match v {
        Value::Object(map) => {
            for (k, child) in map.iter_mut() {
                if let Some(n) = child.as_i64() {
                    if k.ends_with("_at_ms") {
                        *child = Value::from(n + shift_ms);
                        continue;
                    }
                    if k == "timestamp_ns" {
                        *child = Value::from(n + shift_ms * 1_000_000);
                        continue;
                    }
                }
                rebase(child, shift_ms);
            }
        }
        Value::Array(items) => {
            for item in items {
                rebase(item, shift_ms);
            }
        }
        _ => {}
    }
}

fn write_rebased(dir: &Path, name: &str, raw: &str, shift_ms: i64) -> Result<(), String> {
    let mut v: Value = serde_json::from_str(raw)
        .map_err(|e| format!("embedded sample {name} is not JSON: {e}"))?;
    rebase(&mut v, shift_ms);
    let body = serde_json::to_vec_pretty(&v)
        .map_err(|e| format!("re-serializing embedded sample {name}: {e}"))?;
    std::fs::write(dir.join(name), body).map_err(|e| format!("writing {name}: {e}"))
}

/// Write the sample trace chain into `dir` and return the run id to export.
///
/// Creates `dir` if needed. Returns `Err` with a human-readable reason on any
/// I/O or parse failure — never a partial success the caller could mistake for
/// a working run.
pub fn materialize(dir: &Path) -> Result<&'static str, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    let shift_ms = now_ms() - ENDS_AGO_MS - FIXTURE_END_MS;

    write_rebased(dir, "lumen-sample-run-a.json", TRACE_A, shift_ms)?;
    write_rebased(dir, "lumen-sample-run-b.json", TRACE_B, shift_ms)?;
    // The causal sidecar is read as `{run_id}.causal.json`; both runs get a copy
    // so exporting either end of the chain shows the same DAG.
    write_rebased(dir, "lumen-sample-run-a.causal.json", CAUSAL, shift_ms)?;
    write_rebased(dir, "lumen-sample-run-b.causal.json", CAUSAL, shift_ms)?;

    Ok(RUN_ID)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn embedded_fixtures_are_valid_json() {
        for (name, raw) in [("run-a", TRACE_A), ("run-b", TRACE_B), ("causal", CAUSAL)] {
            serde_json::from_str::<Value>(raw)
                .unwrap_or_else(|e| panic!("embedded sample {name} is not valid JSON: {e}"));
        }
    }

    #[test]
    fn fixture_end_ms_matches_the_fixtures() {
        // Guards the one hand-maintained constant: if someone extends the
        // sample run without updating FIXTURE_END_MS, rebasing would silently
        // place the run at the wrong time instead of failing.
        let b: Value = serde_json::from_str(TRACE_B).unwrap();
        assert_eq!(
            b["completed_at_ms"].as_i64(),
            Some(FIXTURE_END_MS),
            "FIXTURE_END_MS is out of sync with the embedded sample"
        );
    }

    #[test]
    fn chain_is_linked_and_tells_the_intended_story() {
        let a: Value = serde_json::from_str(TRACE_A).unwrap();
        let b: Value = serde_json::from_str(TRACE_B).unwrap();

        assert_eq!(a["trace_id"].as_str(), Some(PARENT_RUN_ID));
        assert_eq!(b["trace_id"].as_str(), Some(RUN_ID));
        assert_eq!(
            b["parent_trace_id"].as_str(),
            Some(PARENT_RUN_ID),
            "the recovered run must point at the crashed one, or the lifecycle \
             shows two unrelated runs"
        );

        // The cost story: one step dominates the crashed run's total.
        let steps = a["steps"].as_array().unwrap();
        let max_step = steps
            .iter()
            .filter_map(|s| s["cost_usd"].as_f64())
            .fold(0.0_f64, f64::max);
        let total = a["total_cost_usd"].as_f64().unwrap();
        assert!(
            max_step / total > 0.9,
            "sample should show one step dominating cost (got {max_step} of {total})"
        );

        // The policy story: a denied tool call, distinct from a plain failure.
        let denied = steps.iter().any(|s| {
            s["metadata"]
                .as_array()
                .is_some_and(|m| m.iter().any(|kv| kv[0] == "policy_denied"))
        });
        assert!(denied, "sample should contain a policy-denied tool call");
    }

    #[test]
    fn rebase_shifts_both_ms_and_ns_fields() {
        let mut v: Value = serde_json::json!({
            "started_at_ms": 1_000_i64,
            "completed_at_ms": 2_000_i64,
            "duration_ms": 500_i64,
            "nested": [{ "timestamp_ns": 1_000_000_000_i64 }],
        });
        rebase(&mut v, 10);

        assert_eq!(v["started_at_ms"], 1_010);
        assert_eq!(v["completed_at_ms"], 2_010);
        // A duration is not a wall-clock instant and must survive untouched.
        assert_eq!(v["duration_ms"], 500);
        assert_eq!(v["nested"][0]["timestamp_ns"], 1_010_000_000_i64);
    }

    #[test]
    fn materialize_writes_a_loadable_chain() {
        let dir = std::env::temp_dir().join(format!("lumen-sample-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let run_id = materialize(&dir).expect("materialize should succeed");
        assert_eq!(run_id, RUN_ID);

        for name in [
            "lumen-sample-run-a.json",
            "lumen-sample-run-b.json",
            "lumen-sample-run-a.causal.json",
            "lumen-sample-run-b.causal.json",
        ] {
            assert!(dir.join(name).is_file(), "{name} was not written");
        }

        // Rebased into the recent past, so `lumen cost`'s 24h window includes it.
        let b: Value =
            serde_json::from_slice(&std::fs::read(dir.join("lumen-sample-run-b.json")).unwrap())
                .unwrap();
        let end = b["completed_at_ms"].as_i64().unwrap();
        let age = now_ms() - end;
        assert!(
            (0..24 * 60 * 60 * 1000).contains(&age),
            "rebased run should sit inside the last 24h (age {age}ms)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
