//! `lumen demo` — one command, one picture.
//!
//! The whole point: a person evaluating Kova should type **one** thing and see
//! **one** real lifecycle visualization — not learn a 6-step `pull → traces →
//! cost → replay → export` pipeline first. This module is the orchestration that
//! collapses that pipeline:
//!
//! 1. Get a Kova to talk to — either an already-running one (`--kova-url` /
//!    `LUMEN_KOVA_URL`), or an **ephemeral** `kova-rest` this command spawns on a
//!    free loopback port with a throwaway WAL + generated key, and tears down on
//!    exit.
//! 2. Run one canned scenario (a finance `reconcile` agent — it exercises a real
//!    tool call, so the exported lifecycle has a real causal edge + real cost).
//! 3. Hand the resulting run id back to the existing export path, which fetches
//!    the trace + sidecars and writes the self-contained `.html`.
//!
//! Lumen stays a thin client: it does **not** depend on `kova-rest` at build
//! time. It only *locates and execs* the binary at runtime (env / PATH / a few
//! dev-tree locations); absent that, it points the user at the one thing to set.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::kova::{HttpKovaClient, KovaControlClient};

/// Canned agent id for the demo scenario. Stable (not per-run) so a repeat
/// `lumen demo` against the same Kova reuses the agent rather than littering.
pub const DEMO_AGENT_ID: &str = "lumen-demo-reconcile";

/// System prompt — ASCII only (a multi-byte char in a JSON string field has bit
/// us before via shells; keeping it ASCII removes a whole class of surprise).
/// Phrased to degrade gracefully: if the `reconcile` tool is not registered on
/// the target Kova, the agent just answers in text (still a valid lifecycle,
/// only without the tool-call causal edge).
const DEMO_SYSTEM_PROMPT: &str = "You are a finance reconciliation assistant. \
If a reconcile tool is available you MUST call it to compare the bank statement \
against the ledger; never compute the matches yourself. After the tool returns, \
summarize the discrepancies in ONE short sentence.";

/// The canned task: a 3-vs-3 bank/ledger with exactly one mismatch (-50 vs -55),
/// so the reconcile tool returns one discrepancy — a concrete, legible result.
const DEMO_MESSAGE: &str = "Reconcile these. \
Bank statement: [{\"id\":\"b1\",\"amount\":-100.00,\"memo\":\"rent\"},\
{\"id\":\"b2\",\"amount\":-50.00,\"memo\":\"power\"},\
{\"id\":\"b3\",\"amount\":-25.00,\"memo\":\"water\"}]. \
Ledger: [{\"id\":\"l1\",\"amount\":-100.00},{\"id\":\"l2\",\"amount\":-55.00},\
{\"id\":\"l3\",\"amount\":-25.00}]. Which entries do not match?";

/// Agent iteration ceiling (tool call + result + final answer fits in 4).
const DEMO_MAX_ITERATIONS: u32 = 4;
/// Cost budget (USD). The scenario spends ~$0.001; this is a generous fence.
const DEMO_COST_BUDGET_USD: f64 = 0.05;
/// Daily LLM budget (CNY) for the EPHEMERAL instance only — squeezed so the
/// demo's own LLM spend (~0.002–0.006 CNY on deepseek-chat) crosses the soft
/// (80%) threshold and kova's governor emits a real `budget_*` advisory: the
/// observe→advise feedback loop, demonstrated with measured numbers, in one
/// run. Never applied to a `--kova-url` instance (we don't touch real config).
const DEMO_DAILY_BUDGET_CNY: &str = "0.003";

/// How long to wait for a freshly-spawned `kova-rest` to answer `/status`.
const READY_TIMEOUT: Duration = Duration::from_secs(25);
/// How long to wait for the canned run to reach a terminal trace.
const RUN_TIMEOUT: Duration = Duration::from_secs(90);
/// Poll cadence for both waits.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// An ephemeral `kova-rest` this command owns. Dropping it kills the child and
/// removes the throwaway WAL dir — so the demo leaves nothing running or on disk.
pub struct SpawnedKova {
    child: Child,
    wal_dir: PathBuf,
    /// Base URL (e.g. `http://127.0.0.1:34567`).
    pub url: String,
    /// The generated `X-API-Key` for this instance.
    pub api_key: String,
}

impl Drop for SpawnedKova {
    fn drop(&mut self) {
        // Best-effort teardown: kill the child, give the OS a moment to release
        // the WAL writer lock, then remove the temp dir. Errors are ignored —
        // this is cleanup, not a correctness path.
        let _ = self.child.kill();
        let _ = self.child.wait();
        std::thread::sleep(Duration::from_millis(300));
        let _ = std::fs::remove_dir_all(&self.wal_dir);
    }
}

/// LLM configuration pulled from the caller's environment, passed through to the
/// spawned Kova so its in-process worker can actually execute the agent run.
pub struct LlmEnv {
    pub api_key: String,
    pub provider: String,
    pub model: String,
    pub base_url: String,
}

impl LlmEnv {
    /// Read `KOVA_LLM_*` from the environment. `None` if no API key is set — the
    /// agent loop is LLM-driven, so without a key there is nothing real to show
    /// and we must say so rather than spawn a worker-less instance that hangs.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let api_key = non_empty(std::env::var("KOVA_LLM_API_KEY").ok())?;
        Some(Self {
            api_key,
            provider: non_empty(std::env::var("KOVA_LLM_PROVIDER").ok())
                .unwrap_or_else(|| "openai".to_string()),
            model: non_empty(std::env::var("KOVA_LLM_MODEL").ok())
                .unwrap_or_else(|| "deepseek-chat".to_string()),
            base_url: non_empty(std::env::var("KOVA_LLM_BASE_URL").ok())
                .unwrap_or_else(|| "https://newapi.lurus.cn/v1".to_string()),
        })
    }
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.filter(|v| !v.trim().is_empty())
}

/// Locate the `kova-rest` binary without a build-time dependency on it.
///
/// Search order (first hit wins): explicit `--kova-bin`, `KOVA_REST_BIN`, the
/// system `PATH`, then a few well-known dev-tree locations relative to this
/// executable and the current directory (the sibling `2b-svc-kova` checkout).
#[must_use]
pub fn locate_kova_rest(explicit: Option<&str>) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "kova-rest.exe"
    } else {
        "kova-rest"
    };

    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        return path.is_file().then_some(path);
    }
    if let Some(p) = non_empty(std::env::var("KOVA_REST_BIN").ok()) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    // PATH lookup.
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let cand = dir.join(exe_name);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    // Dev-tree fallbacks: build the candidate roots, then try debug + release.
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        // .../2c-cli-lumen/target/<profile>/lumen → ../../../2b-svc-kova
        if let Some(p) = exe.ancestors().nth(3) {
            roots.push(p.join("2b-svc-kova"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.clone()); // already inside 2b-svc-kova
        roots.push(cwd.join("2b-svc-kova"));
        if let Some(parent) = cwd.parent() {
            roots.push(parent.join("2b-svc-kova"));
        }
    }
    for root in roots {
        for profile in ["debug", "release"] {
            let cand = root.join("target").join(profile).join(exe_name);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Spawn an ephemeral `kova-rest` on a free loopback port and wait until it
/// answers `/status`. The returned [`SpawnedKova`] tears the instance down on
/// drop.
///
/// # Errors
/// Returns a human-readable message if a free port can't be found, the process
/// fails to spawn, or it doesn't become ready (with worker running) in time.
pub fn spawn_ephemeral(bin: &Path, llm: &LlmEnv) -> Result<SpawnedKova, String> {
    let port = free_loopback_port().ok_or("could not find a free loopback port")?;
    let api_key = generate_api_key();
    let wal_dir = std::env::temp_dir().join(format!("lumen-demo-kova-{port}"));
    std::fs::create_dir_all(&wal_dir)
        .map_err(|e| format!("creating temp WAL dir {}: {e}", wal_dir.display()))?;
    let url = format!("http://127.0.0.1:{port}");

    let child = Command::new(bin)
        .env("KOVA_REST_PORT", port.to_string())
        .env("KOVA_WAL_DIR", &wal_dir)
        .env("KOVA_TRACE_DB", wal_dir.join("traces.db"))
        .env("KOVA_JWT_SECRET", "lumen-demo-ephemeral")
        .env("KOVA_API_KEYS", format!("{api_key}:admin"))
        // The reconcile tool is what gives the lifecycle a real tool-call causal
        // edge; enable it on the instance we own.
        .env("KOVA_RECONCILE_ENABLED", "1")
        // Squeeze the daily budget so the run's real spend trips the governor's
        // budget early-warning — the demo then shows a live advisory.
        .env("KOVA_LLM_COST_DAILY_BUDGET_CNY", DEMO_DAILY_BUDGET_CNY)
        .env("KOVA_LLM_API_KEY", &llm.api_key)
        .env("KOVA_LLM_PROVIDER", &llm.provider)
        .env("KOVA_LLM_MODEL", &llm.model)
        .env("KOVA_LLM_BASE_URL", &llm.base_url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawning {}: {e}", bin.display()))?;

    let mut spawned = SpawnedKova {
        child,
        wal_dir,
        url,
        api_key,
    };

    match wait_until_ready(&spawned.url, &spawned.api_key) {
        Ok(()) => Ok(spawned),
        Err(e) => {
            // Surface an early exit (e.g. a missing feature / bad config) plainly.
            if let Ok(Some(status)) = spawned.child.try_wait() {
                return Err(format!(
                    "kova-rest exited before becoming ready ({status}); {e}"
                ));
            }
            Err(e)
        }
    }
}

/// Poll `GET /status` until the instance is ready AND its LLM worker is running
/// (no worker ⇒ the agent run would never complete, so fail fast with the why).
fn wait_until_ready(url: &str, api_key: &str) -> Result<(), String> {
    let client = HttpKovaClient::new(url, Some(api_key.to_string()));
    let deadline = Instant::now() + READY_TIMEOUT;
    let mut last = String::from("no response yet");
    while Instant::now() < deadline {
        match client.send("GET", "/status", None) {
            Ok((200, body)) => {
                let worker = body
                    .get("llm")
                    .and_then(|l| l.get("worker"))
                    .and_then(Value::as_str);
                if worker == Some("running") {
                    return Ok(());
                }
                last = format!(
                    "ready but LLM worker is '{}' (the kova-rest build may lack the `llm` feature)",
                    worker.unwrap_or("absent")
                );
            }
            Ok((code, _)) => last = format!("/status returned {code}"),
            Err(e) => last = e,
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    Err(format!(
        "kova-rest not ready within {READY_TIMEOUT:?}: {last}"
    ))
}

/// Create the demo agent (idempotent — a `409 already exists` is fine) and
/// dispatch the canned reconcile run, then poll until a terminal trace appears.
/// Returns the completed run's trace id.
///
/// # Errors
/// Returns a message on a non-success create/run response, or if no terminal
/// trace appears within [`RUN_TIMEOUT`].
pub fn run_canned_scenario(client: &HttpKovaClient) -> Result<String, String> {
    // Create (idempotent).
    let create_body = json!({
        "agent_id": DEMO_AGENT_ID,
        "system_prompt": DEMO_SYSTEM_PROMPT,
        "max_iterations": DEMO_MAX_ITERATIONS,
        "cost_budget_usd": DEMO_COST_BUDGET_USD,
    });
    match client.send("POST", "/api/v1/agents", Some(&create_body)) {
        Ok((s, _)) if s == 200 || s == 201 || s == 409 => {}
        Ok((s, body)) => {
            return Err(format!(
                "creating demo agent failed ({s}): {}",
                terse(&body)
            ));
        }
        Err(e) => return Err(e),
    }

    // Dispatch the run. A 409 here means an identical run is already in flight
    // (15s idempotency window) — fine, we poll for its trace either way.
    let run_path = format!("/api/v1/agents/{DEMO_AGENT_ID}/run");
    match client.send("POST", &run_path, Some(&json!({ "message": DEMO_MESSAGE }))) {
        Ok((s, _)) if s == 200 || s == 202 || s == 409 => {}
        Ok((s, body)) => {
            return Err(format!(
                "dispatching demo run failed ({s}): {}",
                terse(&body)
            ));
        }
        Err(e) => return Err(e),
    }

    // Poll the agent panorama for the newest terminal trace.
    let get_path = format!("/api/v1/agents/{DEMO_AGENT_ID}");
    let deadline = Instant::now() + RUN_TIMEOUT;
    while Instant::now() < deadline {
        if let Ok((200, body)) = client.send("GET", &get_path, None)
            && let Some(trace) = newest_terminal_trace(&body)
        {
            return Ok(trace);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    Err(format!(
        "demo run did not complete within {RUN_TIMEOUT:?} (LLM slow or unreachable)"
    ))
}

/// Fetch the governor's active advisory set (best-effort, read-only). Returns
/// the advisory array — empty when none fired, `None` when the endpoint is
/// absent (older kova) or unreachable. The demo prints whatever ACTUALLY
/// happened; it never fabricates an advisory.
#[must_use]
pub fn governor_advisories(client: &HttpKovaClient) -> Option<Vec<Value>> {
    match client.send("GET", "/api/v1/governor/advisories", None) {
        Ok((200, body)) => Some(
            body.get("advisories")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        ),
        _ => None,
    }
}

/// Render one advisory as the demo's two-line summary (icon + scenario +
/// prediction). Pure; unit-tested.
#[must_use]
pub fn advisory_line(advisory: &Value) -> String {
    let s = |k: &str| advisory.get(k).and_then(Value::as_str).unwrap_or("");
    let icon = match s("severity") {
        "critical" => "✗",
        "warn" => "⚠",
        _ => "ℹ",
    };
    format!(
        "{icon} {} [{}] — {}",
        s("scenario"),
        s("subject"),
        s("prediction")
    )
}

/// Extract the newest `recent_traces[*]` whose status is terminal
/// (`completed`/`failed`), returning its `trace_id`. `recent_traces` is only
/// populated once a run finishes, so its presence is the completion signal.
fn newest_terminal_trace(agent: &Value) -> Option<String> {
    let traces = agent.get("recent_traces")?.as_array()?;
    for t in traces {
        let status = t.get("status").and_then(Value::as_str).unwrap_or("");
        if (status == "completed" || status == "failed")
            && let Some(id) = t.get("trace_id").and_then(Value::as_str)
        {
            return Some(id.to_string());
        }
    }
    None
}

/// Pick a free loopback TCP port by binding `:0` and reading the assigned port.
/// There's a tiny window between drop and the child's bind, acceptable for a
/// local demo.
fn free_loopback_port() -> Option<u16> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
}

/// Generate an `sk-kova-<64-hex>` key for the throwaway instance. Not
/// cryptographic — it's a loopback secret that lives for seconds — but seeded
/// from time + pid so two concurrent demos don't collide.
fn generate_api_key() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = nanos ^ (u64::from(std::process::id()) << 17);
    let mut hex = String::with_capacity(64);
    // 4 splitmix64 draws → 4 * 16 hex = 64 chars = 32 bytes.
    for _ in 0..4 {
        state = splitmix64(&mut state);
        hex.push_str(&format!("{state:016x}"));
    }
    format!("sk-kova-{hex}")
}

/// One round of the splitmix64 PRNG (deterministic, no deps).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A short one-line form of an error body for messages.
fn terse(body: &Value) -> String {
    let s = body
        .get("error")
        .and_then(Value::as_str)
        .map_or_else(|| body.to_string(), str::to_string);
    if s.len() > 200 {
        format!("{}...", &s[..200])
    } else {
        s
    }
}

/// Read a bounded amount from a reader (used by tests / future helpers).
#[allow(dead_code)]
fn read_capped(mut r: impl Read, cap: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = r.by_ref().take(cap).read_to_end(&mut buf);
    buf
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn generated_key_has_expected_shape() {
        let k = generate_api_key();
        assert!(k.starts_with("sk-kova-"), "{k}");
        let hex = &k["sk-kova-".len()..];
        assert_eq!(hex.len(), 64, "64 hex chars: {k}");
        assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()), "all hex: {k}");
    }

    #[test]
    fn newest_terminal_trace_picks_completed() {
        let agent = json!({
            "recent_traces": [
                {"trace_id": "trace-2", "status": "completed"},
                {"trace_id": "trace-1", "status": "completed"}
            ]
        });
        assert_eq!(newest_terminal_trace(&agent), Some("trace-2".to_string()));
    }

    #[test]
    fn newest_terminal_trace_none_while_running() {
        // No recent_traces yet (run still in flight) ⇒ keep polling.
        let agent = json!({ "recent_traces": [], "current_trace": {"status": "running"} });
        assert_eq!(newest_terminal_trace(&agent), None);
        let missing = json!({ "agent_id": "x" });
        assert_eq!(newest_terminal_trace(&missing), None);
    }

    #[test]
    fn newest_terminal_trace_accepts_failed() {
        let agent = json!({ "recent_traces": [{"trace_id": "t9", "status": "failed"}] });
        assert_eq!(newest_terminal_trace(&agent), Some("t9".to_string()));
    }

    #[test]
    fn locate_returns_none_for_bad_explicit() {
        assert!(locate_kova_rest(Some("/no/such/kova-rest-binary-xyz")).is_none());
    }

    #[test]
    fn advisory_line_renders_severity_and_prediction() {
        let a = json!({
            "severity": "warn",
            "scenario": "budget_soft_warn",
            "subject": "llm-cost-budget",
            "prediction": "hard cap in ~2.1 h (linear extrapolation)"
        });
        let line = advisory_line(&a);
        assert!(line.starts_with('⚠'), "{line}");
        assert!(
            line.contains("budget_soft_warn [llm-cost-budget]"),
            "{line}"
        );
        assert!(line.contains("linear extrapolation"), "{line}");
    }

    #[test]
    fn demo_budget_squeeze_is_a_positive_cny_amount() {
        // The squeezed budget must parse and be > 0 — kova treats <= 0 as
        // "tracker not constructed", which would silently kill the demo beat.
        let v: f64 = DEMO_DAILY_BUDGET_CNY.parse().unwrap();
        assert!(v > 0.0);
    }

    #[test]
    fn llm_env_requires_key() {
        // We can't safely mutate process env in parallel tests; just assert the
        // empty-string filter via the helper it relies on.
        assert_eq!(non_empty(Some(String::new())), None);
        assert_eq!(non_empty(Some("  ".to_string())), None);
        assert_eq!(non_empty(Some("x".to_string())), Some("x".to_string()));
    }
}
