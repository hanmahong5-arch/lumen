//! Load a [`Lifecycle`] for a run — from on-disk trace files (+ deep sidecars)
//! or by fetching a single run live from a kova-rest server.
//!
//! The dashboard's `/api/traces/{id}/lifecycle` route and the `lumen export`
//! command both go through here, so the offline artifact and the live view are
//! built from the exact same projection ([`lumen_core::lifecycle_builder`]).

use std::collections::HashSet;
use std::path::Path;

use lumen_core::causal_types::{CausalEvent, SwarmAgentEntry, SwarmGraph, WorkflowRunDetail};
use lumen_core::lifecycle::Lifecycle;
use lumen_core::lifecycle_builder;
use lumen_core::trace_types::AgentTrace;

use crate::pull::TraceFetcher;

/// Max recovery-chain depth we walk before stopping (defensive bound).
const MAX_CHAIN: usize = 64;

/// Whether `id` is a safe file-stem: non-empty, bounded, `[A-Za-z0-9_.-]`, no
/// `..`. Gates building a `{dir}/{id}.json` path from untrusted input (the
/// dashboard route takes the id from the URL) — a hard path-traversal guard.
#[must_use]
pub fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && !id.contains("..")
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Load the target trace `{trace_dir}/{id}.json` (None on any read/parse error).
fn read_trace(trace_dir: &Path, id: &str) -> Option<AgentTrace> {
    let bytes = std::fs::read(trace_dir.join(format!("{id}.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Read + parse a `{run_id}.{kind}.json` sidecar, or `None` on any failure
/// (missing / unreadable / unparseable — sidecars are optional, degrade quietly).
fn read_causal(trace_dir: &Path, run_id: &str) -> Option<Vec<CausalEvent>> {
    let bytes = std::fs::read(trace_dir.join(format!("{run_id}.causal.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_workflow(trace_dir: &Path, run_id: &str) -> Option<WorkflowRunDetail> {
    let bytes = std::fs::read(trace_dir.join(format!("{run_id}.workflow.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_swarm(trace_dir: &Path, run_id: &str) -> Option<SwarmGraph> {
    let bytes = std::fs::read(trace_dir.join(format!("{run_id}.swarm.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Walk `parent_trace_id` from `target` to build the recovery chain
/// oldest→newest, loading each parent from disk. Cycle-guarded (a malformed
/// `parent==self` or A→B→A loop stops instead of spinning) and depth-bounded.
fn chain_from_disk(trace_dir: &Path, target: AgentTrace) -> Vec<AgentTrace> {
    let mut newest_first = vec![target];
    let mut seen: HashSet<String> = HashSet::new();
    if let Some(t) = newest_first.first() {
        seen.insert(t.trace_id.clone());
    }
    while newest_first.len() < MAX_CHAIN {
        let Some(parent_id) = newest_first
            .last()
            .and_then(|t| t.parent_trace_id.clone())
            .filter(|p| !p.is_empty() && !seen.contains(p))
        else {
            break;
        };
        let Some(parent) = read_trace(trace_dir, &parent_id) else {
            break;
        };
        seen.insert(parent.trace_id.clone());
        newest_first.push(parent);
    }
    newest_first.reverse(); // oldest → newest
    newest_first
}

/// Build a [`Lifecycle`] for `run_id` from `trace_dir`, reading the trace +
/// recovery chain + the `{run_id}.{causal,workflow,swarm}.json` sidecars.
///
/// Returns `None` only when there is nothing to show — no trace AND no workflow
/// sidecar for `run_id`.
#[must_use]
pub fn build_from_dir(trace_dir: &Path, run_id: &str) -> Option<Lifecycle> {
    if !safe_id(run_id) {
        return None;
    }
    let chain = read_trace(trace_dir, run_id)
        .map(|t| chain_from_disk(trace_dir, t))
        .unwrap_or_default();

    let causal = read_causal(trace_dir, run_id);
    let workflow = read_workflow(trace_dir, run_id);
    let swarm = read_swarm(trace_dir, run_id);

    if chain.is_empty() && workflow.is_none() {
        return None;
    }

    let chain_refs: Vec<&AgentTrace> = chain.iter().collect();
    Some(lifecycle_builder::build(
        &chain_refs,
        causal.as_deref(),
        workflow.as_ref(),
        swarm.as_ref(),
    ))
}

/// Outcome of a live single-run fetch.
#[derive(Debug)]
pub enum LoadError {
    /// The run id is not a safe file stem.
    BadId,
    /// The run could not be found live (no trace and no workflow).
    NotFound(String),
    /// A directory write failed.
    Io(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadId => write!(f, "run id contains characters outside [A-Za-z0-9_.-]"),
            Self::NotFound(id) => write!(f, "no trace or workflow run found for `{id}`"),
            Self::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Fetch just one run (its trace + recovery chain + sidecars) from a live
/// kova-rest server into `trace_dir`, so [`build_from_dir`] can then assemble
/// it. `swarm_id`, when given, also fetches + merges the swarm graph/trace.
///
/// # Errors
/// [`LoadError`] when the id is unsafe, the run is absent, or a write fails.
pub fn fetch_run_into_dir(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    run_id: &str,
    swarm_id: Option<&str>,
) -> Result<(), LoadError> {
    if !safe_id(run_id) {
        return Err(LoadError::BadId);
    }
    std::fs::create_dir_all(trace_dir)
        .map_err(|e| LoadError::Io(format!("create {}: {e}", trace_dir.display())))?;

    let mut wrote_anything = false;

    // 1. Target trace (+ walk the recovery chain live).
    if let Ok(body) = fetcher.fetch_trace(run_id) {
        write_file(trace_dir, &format!("{run_id}.json"), &body)?;
        wrote_anything = true;

        let (agent_id, task_id, _parent) = trace_fields(&body);

        // Causal sidecar (by agent id), keyed by the target run id.
        if let Some(agent_id) = agent_id.as_deref()
            && let Ok(Some(bytes)) = fetcher.fetch_causal(agent_id)
        {
            write_file(trace_dir, &format!("{run_id}.causal.json"), &bytes)?;
        }
        // Workflow sidecar when the run is a workflow.
        if let Some(wid) = task_id.filter(|w| *w != 0)
            && let Ok(Some(bytes)) = fetcher.fetch_workflow(wid)
        {
            write_file(trace_dir, &format!("{run_id}.workflow.json"), &bytes)?;
        }
        // Walk parents (cycle-guarded) and fetch each.
        fetch_chain(fetcher, trace_dir, &body)?;
    } else if let Ok(wid) = run_id.parse::<u64>()
        && let Ok(Some(bytes)) = fetcher.fetch_workflow(wid)
    {
        // Workflow-only run: the id is a workflow id, no agent trace exists.
        write_file(trace_dir, &format!("{run_id}.workflow.json"), &bytes)?;
        wrote_anything = true;
    }

    // 2. Optional swarm sidecar (graph + trace token merge).
    if let Some(sid) = swarm_id
        && let Some(bytes) = fetch_merged_swarm(fetcher, sid)
    {
        write_file(trace_dir, &format!("{run_id}.swarm.json"), &bytes)?;
        wrote_anything = true;
    }

    if wrote_anything {
        Ok(())
    } else {
        Err(LoadError::NotFound(run_id.to_string()))
    }
}

/// Follow `parent_trace_id` from a just-fetched trace body, fetching + writing
/// each parent. Cycle-guarded + depth-bounded.
fn fetch_chain(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    target_body: &[u8],
) -> Result<(), LoadError> {
    let mut seen: HashSet<String> = HashSet::new();
    let (_, _, mut parent) = trace_fields(target_body);
    let mut depth = 0;
    while let Some(pid) = parent
        .take()
        .filter(|p| !p.is_empty() && safe_id(p) && !seen.contains(p))
    {
        if depth >= MAX_CHAIN {
            break;
        }
        depth += 1;
        seen.insert(pid.clone());
        let Ok(body) = fetcher.fetch_trace(&pid) else {
            break;
        };
        write_file(trace_dir, &format!("{pid}.json"), &body)?;
        let (_, _, next) = trace_fields(&body);
        parent = next;
    }
    Ok(())
}

/// Fetch `/swarm/{id}/graph` and merge per-agent token counts from
/// `/swarm/{id}/trace` into the node usage fields. Returns the merged graph
/// JSON bytes, or `None` if the graph isn't available.
fn fetch_merged_swarm(fetcher: &dyn TraceFetcher, swarm_id: &str) -> Option<Vec<u8>> {
    let graph_bytes = fetcher.fetch_swarm_graph(swarm_id).ok().flatten()?;
    let mut graph: SwarmGraph = serde_json::from_slice(&graph_bytes).ok()?;

    if let Ok(Some(trace_bytes)) = fetcher.fetch_swarm_trace(swarm_id)
        && let Ok(value) = serde_json::from_slice::<serde_json::Value>(&trace_bytes)
    {
        let agents: Vec<SwarmAgentEntry> = value
            .get("agents")
            .and_then(|a| serde_json::from_value(a.clone()).ok())
            .unwrap_or_default();
        for node in &mut graph.nodes {
            if let Some(a) = agents.iter().find(|a| a.agent_id == node.id) {
                node.prompt_tokens = a.tokens_prompt;
                node.completion_tokens = a.tokens_completion;
            }
        }
    }
    serde_json::to_vec(&graph).ok()
}

/// Extract `(agent_id, task_id, parent_trace_id)` from a raw trace body.
fn trace_fields(body: &[u8]) -> (Option<String>, Option<u64>, Option<String>) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return (None, None, None);
    };
    let agent_id = v
        .get("agent_id")
        .and_then(|x| x.as_str())
        .map(str::to_string);
    let task_id = v.get("task_id").and_then(serde_json::Value::as_u64);
    let parent = v
        .get("parent_trace_id")
        .and_then(|x| x.as_str())
        .map(str::to_string);
    (agent_id, task_id, parent)
}

/// Write `{dir}/{name}` atomically (tmp + rename).
fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), LoadError> {
    let final_path = dir.join(name);
    let tmp_path = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp_path, bytes)
        .map_err(|e| LoadError::Io(format!("write {}: {e}", tmp_path.display())))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| LoadError::Io(format!("rename to {}: {e}", final_path.display())))?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("lumen_lcload_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn trace_json(id: &str, parent: Option<&str>, step_cost: Option<f64>) -> String {
        let parent_field = parent.map_or("null".to_string(), |p| format!("\"{p}\""));
        let cost = step_cost.map_or("null".to_string(), |c| c.to_string());
        format!(
            r#"{{"trace_id":"{id}","agent_id":"reconcile-agent","task_id":0,
              "steps":[{{"step_num":0,"step_type":{{"LlmCall":{{"model":"deepseek-chat","finish_reason":"stop"}}}},
                "started_at_ms":1000,"duration_ms":100,"tokens":{{"prompt_tokens":200,"completion_tokens":50}},
                "cost_usd":{cost},"metadata":[]}}],
              "status":"Completed","total_tokens":{{"prompt_tokens":200,"completion_tokens":50}},
              "total_cost_usd":0.0012,"started_at_ms":1000,"completed_at_ms":2000,
              "parent_trace_id":{parent_field},"schema_version":1}}"#
        )
    }

    #[test]
    fn safe_id_rejects_traversal_and_slashes() {
        assert!(safe_id("trace_abc-123.v1"));
        assert!(!safe_id("../etc/passwd"));
        assert!(!safe_id("a/b"));
        assert!(!safe_id(""));
        assert!(!safe_id("a..b"));
    }

    #[test]
    fn build_from_dir_assembles_trace_causal_and_chain() {
        let dir = tmp_dir("assemble");
        write(
            &dir,
            "child.json",
            &trace_json("child", Some("parent"), Some(0.001)),
        );
        write(
            &dir,
            "parent.json",
            &trace_json("parent", None, Some(0.0005)),
        );
        write(
            &dir,
            "child.causal.json",
            r#"[{"event_id":0,"record_type":"AgentDirective","timestamp_ns":1,"caused_by":null,"summary":"reconcile"},
                {"event_id":1,"record_type":"AgentDirectiveResult","timestamp_ns":2,"caused_by":0,"summary":"ok"}]"#,
        );

        let lc = build_from_dir(&dir, "child").expect("lifecycle built");
        // Chain walked oldest→newest.
        assert_eq!(
            lc.recovery_chain,
            vec!["parent".to_string(), "child".to_string()]
        );
        // Causal edges present.
        assert_eq!(lc.causal.edges.len(), 1);
        // Timeline: parent step + RESUMED divider + child step = 3.
        assert_eq!(lc.timeline.len(), 3);
        assert!(lc.provenance.sources.contains(&"causal".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_from_dir_workflow_only() {
        let dir = tmp_dir("wf_only");
        write(
            &dir,
            "77.workflow.json",
            r#"{"workflow_id":77,"status":"completed","total_steps":1,
                "steps":[{"step_index":0,"output_preview":"done","started_at_ms":10,"duration_ms":5}],
                "started_at_ms":10,"terminal_at_ms":15}"#,
        );
        let lc = build_from_dir(&dir, "77").expect("workflow-only lifecycle");
        assert_eq!(lc.run_id, "77");
        assert_eq!(lc.workflow_steps.len(), 1);
        assert!(lc.timeline.len() == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_from_dir_missing_returns_none() {
        let dir = tmp_dir("missing");
        assert!(build_from_dir(&dir, "nope").is_none());
        assert!(build_from_dir(&dir, "../escape").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
