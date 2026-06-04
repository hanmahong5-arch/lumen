//! `lumen pull` — fetch traces from a running Kova into a local trace dir.
//!
//! Today the replay/cost/traces commands only read `{trace-dir}/{id}.json`
//! files. `pull` bridges that gap: it lists trace ids from a live Kova
//! (`GET /api/v1/traces`), fetches each full trace
//! (`GET /api/v1/traces/{id}`), and writes them as local JSON files so the
//! existing commands work unchanged against pulled data.
//!
//! Design: the network seam is the [`TraceFetcher`] trait so the
//! orchestration ([`pull_into`]) and parsing ([`extract_trace_ids`]) are pure
//! and unit-testable without a socket. Only [`HttpFetcher`] touches the wire.

use std::path::Path;
use std::time::Duration;

/// Error from a pull operation.
#[derive(Debug)]
pub enum PullError {
    /// Could not reach Kova or list the traces (whole pull aborts).
    List(String),
    /// Could not create or write the trace directory.
    Io(String),
}

impl std::fmt::Display for PullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::List(m) => write!(f, "could not list traces from Kova: {m}"),
            Self::Io(m) => write!(f, "trace directory error: {m}"),
        }
    }
}

impl std::error::Error for PullError {}

/// Outcome of a successful pull (the list step worked; individual fetches may
/// still have been skipped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PullSummary {
    /// Number of trace ids advertised by Kova.
    pub listed: usize,
    /// Number of traces successfully written locally.
    pub written: usize,
    /// Ids that failed to fetch/write and were skipped (not fatal).
    pub skipped: Vec<String>,
    /// `--deep`: number of `{id}.causal.json` sidecars written.
    pub causal_written: usize,
    /// `--deep`: number of `{id}.workflow.json` sidecars written.
    pub workflow_written: usize,
    /// `--deep`: per-sidecar failures (`"{id}.causal: <msg>"`); non-fatal.
    pub sidecar_skipped: Vec<String>,
}

/// Default limit for `lumen pull --limit` (number of traces to fetch).
pub const DEFAULT_PULL_LIMIT: usize = 200;
/// Maximum value accepted for `--limit`.
pub const MAX_PULL_LIMIT: usize = 10_000;

/// Abstraction over fetching traces from Kova — the single network seam.
///
/// Implemented by [`HttpFetcher`] in production and by stubs in tests so the
/// orchestration logic can be exercised without a real server.
pub trait TraceFetcher {
    /// List up to `limit` trace ids known to Kova. A failure here aborts the
    /// whole pull.
    ///
    /// # Errors
    /// Returns a message when the server is unreachable or the response is
    /// not a parseable trace list.
    fn list_trace_ids(&self, limit: usize) -> Result<Vec<String>, String>;

    /// Fetch one full trace as raw JSON bytes. A failure here is non-fatal:
    /// the caller skips this id and continues.
    ///
    /// # Errors
    /// Returns a message when the trace can't be fetched.
    fn fetch_trace(&self, trace_id: &str) -> Result<Vec<u8>, String>;

    /// Fetch the `format=causal` stream for `agent_id`
    /// (`GET /agents/{id}/history?format=causal`). `Ok(None)` means the server
    /// has nothing for this agent (e.g. 404) — not an error.
    ///
    /// Default is `Ok(None)` so non-HTTP fetchers (test stubs) opt in only when
    /// they care about the deep sidecars.
    ///
    /// # Errors
    /// Returns a message on a transport/HTTP error other than 404.
    fn fetch_causal(&self, _agent_id: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }

    /// Fetch a workflow run detail (`GET /workflows/runs/{id}`). `Ok(None)` for
    /// a 404 (the run isn't a workflow / not found).
    ///
    /// # Errors
    /// Returns a message on a transport/HTTP error other than 404.
    fn fetch_workflow(&self, _workflow_id: u64) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }

    /// Fetch a swarm delegation graph (`GET /swarm/{id}/graph`). `Ok(None)` for
    /// a 404.
    ///
    /// # Errors
    /// Returns a message on a transport/HTTP error other than 404.
    fn fetch_swarm_graph(&self, _swarm_id: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }

    /// Fetch a swarm execution trace (`GET /swarm/{id}/trace`). `Ok(None)` for
    /// a 404. Used to merge per-agent token counts into the graph nodes.
    ///
    /// # Errors
    /// Returns a message on a transport/HTTP error other than 404.
    fn fetch_swarm_trace(&self, _swarm_id: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }
}

/// Parse the `GET /api/v1/traces` response body into a list of trace ids.
///
/// Kova returns a JSON array of summary objects, each carrying a `trace_id`
/// field. Elements missing a string `trace_id` are skipped (defensive against
/// schema drift) rather than failing the whole list.
///
/// # Errors
/// Returns an error string if the body is not a JSON array.
pub fn extract_trace_ids(body: &[u8]) -> Result<Vec<String>, String> {
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("invalid trace-list JSON: {e}"))?;
    let arr = value
        .as_array()
        .ok_or_else(|| "trace list response was not a JSON array".to_string())?;
    Ok(arr
        .iter()
        .filter_map(|item| item.get("trace_id").and_then(|v| v.as_str()))
        .map(str::to_string)
        .collect())
}

/// Validate that fetched bytes are a trace for `trace_id` before writing.
///
/// Guards against a Kova response that is JSON but not the trace we asked for
/// (e.g. an error envelope). We only require it to be an object carrying a
/// string `trace_id`; we deliberately do not re-validate the full schema here
/// (the readers tolerate extra/missing optional fields via serde defaults).
///
/// # Errors
/// Returns a message if the body is not a trace object.
fn validate_trace_body(trace_id: &str, body: &[u8]) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("invalid trace JSON: {e}"))?;
    match value.get("trace_id").and_then(|v| v.as_str()) {
        Some(id) if id == trace_id => Ok(()),
        Some(other) => Err(format!(
            "response trace_id `{other}` != requested `{trace_id}`"
        )),
        None => Err("response had no string trace_id".to_string()),
    }
}

/// Orchestrate a pull: list ids (up to `limit`), fetch each, write
/// `{trace_dir}/{id}.json`. Prints per-trace progress to stdout.
///
/// Behavior:
/// - List failure → [`PullError::List`] (whole pull aborts, non-zero exit).
/// - Empty list → no-op, returns a summary with `listed == 0`.
/// - Per-trace fetch/validate/write failure → skip + record in
///   `summary.skipped` (does NOT abort the rest of the pull).
///
/// This function is pure with respect to the network: all I/O to Kova goes
/// through `fetcher`, so it is unit-tested with a stub fetcher.
///
/// # Errors
/// Returns [`PullError`] when the trace directory cannot be created or the
/// initial list call fails.
pub fn pull_into(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    limit: usize,
) -> Result<PullSummary, PullError> {
    pull_into_inner(fetcher, trace_dir, limit, false)
}

/// Like [`pull_into`] but also fetches the deep lifecycle sidecars for each
/// trace (`{id}.causal.json` / `{id}.workflow.json`), so `lumen export` can
/// render the full lifecycle offline. Per-sidecar failures are non-fatal and
/// recorded in [`PullSummary::sidecar_skipped`].
///
/// Note: the swarm sidecar (`{id}.swarm.json`) is **not** written here — a
/// swarm id is not derivable from an agent trace, so it is fetched on demand by
/// `lumen export --swarm <id>` instead. (Documented deferral.)
///
/// # Errors
/// Same as [`pull_into`] (list/dir failures abort; per-trace failures skip).
pub fn pull_into_deep(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    limit: usize,
) -> Result<PullSummary, PullError> {
    pull_into_inner(fetcher, trace_dir, limit, true)
}

fn pull_into_inner(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    limit: usize,
    deep: bool,
) -> Result<PullSummary, PullError> {
    let ids = fetcher.list_trace_ids(limit).map_err(PullError::List)?;

    let mut summary = PullSummary {
        listed: ids.len(),
        ..PullSummary::default()
    };

    if ids.is_empty() {
        return Ok(summary);
    }

    std::fs::create_dir_all(trace_dir)
        .map_err(|e| PullError::Io(format!("create {}: {e}", trace_dir.display())))?;

    let total = ids.len();
    for (idx, id) in ids.into_iter().enumerate() {
        eprintln!("  [{}/{}] {}", idx + 1, total, id);
        match fetch_and_write_one(fetcher, trace_dir, &id) {
            Ok(body) => {
                summary.written += 1;
                if deep {
                    write_sidecars(fetcher, trace_dir, &id, &body, &mut summary);
                }
            }
            Err(msg) => {
                eprintln!("\x1b[33mwarning: skipping trace {id}: {msg}\x1b[0m");
                summary.skipped.push(id);
            }
        }
    }

    Ok(summary)
}

/// `--deep`: fetch + write the causal/workflow sidecars next to `{id}.json`.
///
/// Keys are derived from the just-written trace body: causal by `agent_id`,
/// workflow by `task_id` (only when non-zero — `0` is a standalone agent run,
/// not a workflow). Each sidecar is independent; a failure records into
/// `summary.sidecar_skipped` and never aborts the pull.
fn write_sidecars(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    trace_id: &str,
    trace_body: &[u8],
    summary: &mut PullSummary,
) {
    let value: serde_json::Value = match serde_json::from_slice(trace_body) {
        Ok(v) => v,
        Err(e) => {
            summary
                .sidecar_skipped
                .push(format!("{trace_id}.sidecars: trace body not JSON: {e}"));
            return;
        }
    };
    let agent_id = value.get("agent_id").and_then(|v| v.as_str());
    let task_id = value.get("task_id").and_then(serde_json::Value::as_u64);

    // Causal sidecar (by agent_id).
    if let Some(agent_id) = agent_id {
        match fetcher.fetch_causal(agent_id) {
            Ok(Some(bytes)) => match write_sidecar(trace_dir, trace_id, "causal", &bytes) {
                Ok(()) => summary.causal_written += 1,
                Err(e) => summary
                    .sidecar_skipped
                    .push(format!("{trace_id}.causal: {e}")),
            },
            Ok(None) => {}
            Err(e) => summary
                .sidecar_skipped
                .push(format!("{trace_id}.causal: {e}")),
        }
    }

    // Workflow sidecar (by task_id, only when this run IS a workflow).
    if let Some(wid) = task_id.filter(|w| *w != 0) {
        match fetcher.fetch_workflow(wid) {
            Ok(Some(bytes)) => match write_sidecar(trace_dir, trace_id, "workflow", &bytes) {
                Ok(()) => summary.workflow_written += 1,
                Err(e) => summary
                    .sidecar_skipped
                    .push(format!("{trace_id}.workflow: {e}")),
            },
            Ok(None) => {}
            Err(e) => summary
                .sidecar_skipped
                .push(format!("{trace_id}.workflow: {e}")),
        }
    }
}

/// Write `{trace_dir}/{trace_id}.{kind}.json` atomically (tmp + rename).
fn write_sidecar(trace_dir: &Path, trace_id: &str, kind: &str, bytes: &[u8]) -> Result<(), String> {
    let final_path = trace_dir.join(format!("{trace_id}.{kind}.json"));
    let tmp_path = trace_dir.join(format!(".{trace_id}.{kind}.json.tmp"));
    std::fs::write(&tmp_path, bytes).map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("rename to {}: {e}", final_path.display()))?;
    Ok(())
}

/// Fetch a single trace, validate it, and write it atomically-ish.
///
/// Writes to `{id}.json.tmp` then renames to `{id}.json` so a crash mid-write
/// never leaves a half-written trace that the readers would choke on (they
/// skip dotfiles and `.tmp`).
fn fetch_and_write_one(
    fetcher: &dyn TraceFetcher,
    trace_dir: &Path,
    trace_id: &str,
) -> Result<Vec<u8>, String> {
    let body = fetcher.fetch_trace(trace_id)?;
    validate_trace_body(trace_id, &body)?;

    let final_path = trace_dir.join(format!("{trace_id}.json"));
    let tmp_path = trace_dir.join(format!(".{trace_id}.json.tmp"));
    std::fs::write(&tmp_path, &body).map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("rename to {}: {e}", final_path.display()))?;
    Ok(body)
}

/// Production [`TraceFetcher`] backed by a synchronous `ureq` HTTP client.
pub struct HttpFetcher {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

/// `X-API-Key` header name (Kova auth).
const API_KEY_HEADER: &str = "X-API-Key";
/// Per-request timeout. Pulls are interactive; fail fast rather than hang.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

impl HttpFetcher {
    /// Build a fetcher for a Kova base URL (e.g. `http://100.122.83.20:3010`).
    ///
    /// `api_key` is sent as `X-API-Key` when present. A trailing slash on
    /// `base_url` is trimmed so path joining is predictable.
    #[must_use]
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(REQUEST_TIMEOUT).build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            agent,
        }
    }

    /// Issue a GET and read the full body, attaching auth when configured.
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let mut req = self.agent.get(url);
        if let Some(ref key) = self.api_key {
            req = req.set(API_KEY_HEADER, key);
        }
        let resp = req.call().map_err(|e| describe_ureq_error(&e))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading response body: {e}"))?;
        Ok(buf)
    }

    /// GET that treats `404`/`501` as "no data" (`Ok(None)`) rather than an
    /// error — for optional sidecars (a run that isn't a workflow 404s
    /// `/workflows/runs/{id}`; a control-plane-only server 501s on WAL reads).
    fn get_opt(&self, url: &str) -> Result<Option<Vec<u8>>, String> {
        let mut req = self.agent.get(url);
        if let Some(ref key) = self.api_key {
            req = req.set(API_KEY_HEADER, key);
        }
        match req.call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                resp.into_reader()
                    .read_to_end(&mut buf)
                    .map_err(|e| format!("reading response body: {e}"))?;
                Ok(Some(buf))
            }
            Err(ureq::Error::Status(404 | 501, _)) => Ok(None),
            Err(e) => Err(describe_ureq_error(&e)),
        }
    }
}

impl TraceFetcher for HttpFetcher {
    fn list_trace_ids(&self, limit: usize) -> Result<Vec<String>, String> {
        let url = format!("{}/api/v1/traces?limit={limit}", self.base_url);
        let body = self.get(&url)?;
        extract_trace_ids(&body)
    }

    fn fetch_trace(&self, trace_id: &str) -> Result<Vec<u8>, String> {
        // trace ids from Kova are opaque; encode any slashes defensively so a
        // malformed id can't escape the traces path.
        let safe_id = trace_id.replace('/', "%2F");
        let url = format!("{}/api/v1/traces/{safe_id}", self.base_url);
        self.get(&url)
    }

    fn fetch_causal(&self, agent_id: &str) -> Result<Option<Vec<u8>>, String> {
        let safe = agent_id.replace('/', "%2F");
        // Unpaged (no `limit`) → bare `[CausalEvent]` array.
        let url = format!(
            "{}/api/v1/agents/{safe}/history?format=causal",
            self.base_url
        );
        self.get_opt(&url)
    }

    fn fetch_workflow(&self, workflow_id: u64) -> Result<Option<Vec<u8>>, String> {
        let url = format!("{}/api/v1/workflows/runs/{workflow_id}", self.base_url);
        self.get_opt(&url)
    }

    fn fetch_swarm_graph(&self, swarm_id: &str) -> Result<Option<Vec<u8>>, String> {
        let safe = swarm_id.replace('/', "%2F");
        let url = format!("{}/api/v1/swarm/{safe}/graph", self.base_url);
        self.get_opt(&url)
    }

    fn fetch_swarm_trace(&self, swarm_id: &str) -> Result<Option<Vec<u8>>, String> {
        let safe = swarm_id.replace('/', "%2F");
        let url = format!("{}/api/v1/swarm/{safe}/trace", self.base_url);
        self.get_opt(&url)
    }
}

use std::io::Read;

/// Turn a ureq error into a single human-readable line (HTTP status vs
/// transport failure), so the CLI can show why a pull failed.
fn describe_ureq_error(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        ureq::Error::Transport(t) => format!("connection failed: {t}"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// In-memory stub fetcher — no network. Records calls so tests can assert
    /// which traces were requested and what limit was passed.
    struct StubFetcher {
        list: Result<Vec<String>, String>,
        traces: HashMap<String, Result<Vec<u8>, String>>,
        causal: HashMap<String, Result<Option<Vec<u8>>, String>>,
        workflows: HashMap<u64, Result<Option<Vec<u8>>, String>>,
        fetched: RefCell<Vec<String>>,
        /// Last `limit` value passed to `list_trace_ids`.
        last_limit: RefCell<usize>,
    }

    impl StubFetcher {
        fn new(ids: &[&str]) -> Self {
            Self {
                list: Ok(ids.iter().map(|s| (*s).to_string()).collect()),
                traces: HashMap::new(),
                causal: HashMap::new(),
                workflows: HashMap::new(),
                fetched: RefCell::new(Vec::new()),
                last_limit: RefCell::new(0),
            }
        }

        fn with_trace(mut self, id: &str, body: &str) -> Self {
            self.traces
                .insert(id.to_string(), Ok(body.as_bytes().to_vec()));
            self
        }

        fn with_fetch_error(mut self, id: &str, msg: &str) -> Self {
            self.traces.insert(id.to_string(), Err(msg.to_string()));
            self
        }

        fn with_causal(mut self, agent_id: &str, body: &str) -> Self {
            self.causal
                .insert(agent_id.to_string(), Ok(Some(body.as_bytes().to_vec())));
            self
        }

        fn with_workflow(mut self, workflow_id: u64, body: &str) -> Self {
            self.workflows
                .insert(workflow_id, Ok(Some(body.as_bytes().to_vec())));
            self
        }
    }

    impl TraceFetcher for StubFetcher {
        fn list_trace_ids(&self, limit: usize) -> Result<Vec<String>, String> {
            *self.last_limit.borrow_mut() = limit;
            self.list.clone()
        }

        fn fetch_trace(&self, trace_id: &str) -> Result<Vec<u8>, String> {
            self.fetched.borrow_mut().push(trace_id.to_string());
            self.traces
                .get(trace_id)
                .cloned()
                .unwrap_or_else(|| Err("no such trace in stub".to_string()))
        }

        fn fetch_causal(&self, agent_id: &str) -> Result<Option<Vec<u8>>, String> {
            self.causal.get(agent_id).cloned().unwrap_or(Ok(None))
        }

        fn fetch_workflow(&self, workflow_id: u64) -> Result<Option<Vec<u8>>, String> {
            self.workflows
                .get(&workflow_id)
                .cloned()
                .unwrap_or(Ok(None))
        }
    }

    fn trace_json(id: &str) -> String {
        format!(
            r#"{{"trace_id":"{id}","agent_id":"a","task_id":0,"steps":[],"status":"Completed","total_tokens":{{"prompt_tokens":0,"completion_tokens":0}},"total_cost_usd":0.0,"started_at_ms":1}}"#
        )
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("lumen_pull_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn extract_ids_from_summary_array() {
        let body = br#"[{"trace_id":"t1","agent_id":"a"},{"trace_id":"t2","agent_id":"b"}]"#;
        let ids = extract_trace_ids(body).unwrap();
        assert_eq!(ids, vec!["t1", "t2"]);
    }

    #[test]
    fn extract_ids_skips_elements_without_trace_id() {
        let body = br#"[{"trace_id":"t1"},{"agent_id":"no-id"},{"trace_id":"t3"}]"#;
        let ids = extract_trace_ids(body).unwrap();
        assert_eq!(ids, vec!["t1", "t3"]);
    }

    #[test]
    fn extract_ids_rejects_non_array() {
        let body = br#"{"error":"unauthorized"}"#;
        assert!(extract_trace_ids(body).is_err());
    }

    #[test]
    fn pull_writes_each_trace_to_disk() {
        let dir = tmp_dir("write");
        let fetcher = StubFetcher::new(&["t1", "t2"])
            .with_trace("t1", &trace_json("t1"))
            .with_trace("t2", &trace_json("t2"));

        let summary = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary.listed, 2);
        assert_eq!(summary.written, 2);
        assert!(summary.skipped.is_empty());

        // Files exist and are loadable as traces by the core reader path.
        for id in ["t1", "t2"] {
            let p = dir.join(format!("{id}.json"));
            assert!(p.is_file(), "{} should exist", p.display());
            let bytes = std::fs::read(&p).unwrap();
            assert!(
                extract_trace_ids(format!("[{}]", String::from_utf8(bytes).unwrap()).as_bytes())
                    .is_ok()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_empty_list_is_noop() {
        let dir = tmp_dir("empty");
        let fetcher = StubFetcher::new(&[]);
        let summary = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary, PullSummary::default());
        // No directory side effects required for an empty list.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_skips_failed_fetch_but_continues() {
        let dir = tmp_dir("skip");
        let fetcher = StubFetcher::new(&["good", "bad", "good2"])
            .with_trace("good", &trace_json("good"))
            .with_fetch_error("bad", "HTTP 500")
            .with_trace("good2", &trace_json("good2"));

        let summary = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary.listed, 3);
        assert_eq!(summary.written, 2);
        assert_eq!(summary.skipped, vec!["bad".to_string()]);
        // All three were attempted (the bad one did not abort the loop).
        assert_eq!(fetcher.fetched.borrow().len(), 3);
        assert!(dir.join("good.json").is_file());
        assert!(dir.join("good2.json").is_file());
        assert!(!dir.join("bad.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_skips_body_with_mismatched_trace_id() {
        let dir = tmp_dir("mismatch");
        // Server returns a trace whose id doesn't match what we asked for.
        let fetcher = StubFetcher::new(&["want"]).with_trace("want", &trace_json("other"));
        let summary = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary.written, 0);
        assert_eq!(summary.skipped, vec!["want".to_string()]);
        assert!(!dir.join("want.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_list_failure_aborts() {
        let dir = tmp_dir("listfail");
        let fetcher = StubFetcher {
            list: Err("connection refused".to_string()),
            traces: HashMap::new(),
            causal: HashMap::new(),
            workflows: HashMap::new(),
            fetched: RefCell::new(Vec::new()),
            last_limit: RefCell::new(0),
        };
        let err = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap_err();
        assert!(matches!(err, PullError::List(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deep_pull_writes_causal_and_workflow_sidecars() {
        let dir = tmp_dir("deep");
        // "wf" is a workflow run (task_id != 0) so its workflow sidecar is
        // fetched; "ag" is a plain agent run (task_id 0) so only causal.
        let wf_trace = r#"{"trace_id":"wf","agent_id":"flow-agent","task_id":77,"steps":[],"status":"Completed","total_tokens":{"prompt_tokens":0,"completion_tokens":0},"total_cost_usd":0.0,"started_at_ms":1}"#;
        let ag_trace = r#"{"trace_id":"ag","agent_id":"tool-agent","task_id":0,"steps":[],"status":"Completed","total_tokens":{"prompt_tokens":0,"completion_tokens":0},"total_cost_usd":0.0,"started_at_ms":1}"#;
        let fetcher = StubFetcher::new(&["wf", "ag"])
            .with_trace("wf", wf_trace)
            .with_trace("ag", ag_trace)
            .with_causal(
                "flow-agent",
                r#"[{"event_id":0,"record_type":"WorkflowStart"}]"#,
            )
            .with_causal(
                "tool-agent",
                r#"[{"event_id":0,"record_type":"AgentDirective","caused_by":null}]"#,
            )
            .with_workflow(77, r#"{"workflow_id":77,"status":"completed","steps":[]}"#);

        let summary = pull_into_deep(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary.written, 2);
        // Both agents had a causal sidecar; only the workflow run had a wf sidecar.
        assert_eq!(summary.causal_written, 2);
        assert_eq!(summary.workflow_written, 1);
        assert!(summary.sidecar_skipped.is_empty());
        assert!(dir.join("wf.causal.json").is_file());
        assert!(dir.join("wf.workflow.json").is_file());
        assert!(dir.join("ag.causal.json").is_file());
        // The plain agent run (task_id 0) gets NO workflow sidecar.
        assert!(!dir.join("ag.workflow.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shallow_pull_writes_no_sidecars() {
        let dir = tmp_dir("shallow");
        let fetcher = StubFetcher::new(&["ag"])
            .with_trace("ag", &trace_json("ag"))
            .with_causal("a", r#"[]"#);
        let summary = pull_into(&fetcher, &dir, DEFAULT_PULL_LIMIT).unwrap();
        assert_eq!(summary.written, 1);
        assert_eq!(summary.causal_written, 0);
        assert!(!dir.join("ag.causal.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pull_limit_is_forwarded_to_fetcher() {
        // The limit value passed to pull_into must reach list_trace_ids.
        let dir = tmp_dir("limit");
        let fetcher = StubFetcher::new(&[]);
        pull_into(&fetcher, &dir, 42).unwrap();
        assert_eq!(*fetcher.last_limit.borrow(), 42, "limit must be forwarded");

        // Default and max constants are sane.
        assert_eq!(DEFAULT_PULL_LIMIT, 200);
        assert_eq!(MAX_PULL_LIMIT, 10_000);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
