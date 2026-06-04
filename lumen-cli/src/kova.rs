//! Kova control client + command interpreter — the seam Lumen uses to drive a
//! running Kova from the dashboard's Terminal tab (and the `lumen kova` one-shot).
//!
//! Kova is API-first: every control verb is a discrete request/response REST
//! call (`POST /api/v1/agents/{id}/run`, `POST /workflows/runs/{id}/cancel`, …).
//! So the "terminal" is **not** a shell — it is a REST interpreter. A typed line
//! is parsed into one **whitelisted** [`KovaCommand`], which maps to exactly one
//! [`Plan`] `(method, path, body, mutating, destructive)`. The HTTP seam
//! ([`KovaControlClient`]) is only ever called with an interpreter-built path, so
//! the whitelist is the single chokepoint — no arbitrary path / method / host /
//! shell is reachable from the browser.
//!
//! Design mirrors [`crate::pull`] / [`crate::netdata`]: the network seam is a
//! trait so parsing ([`parse_command`]), request-mapping ([`KovaCommand::to_request`])
//! and rendering ([`render_result`]) are pure and unit-testable without a socket.
//! Only [`HttpKovaClient`] touches the wire (ureq, `X-API-Key`, `redirects(0)`,
//! bounded read — the same discipline as `pull.rs`).

use std::io::Read;
use std::time::Duration;

use serde_json::{Value, json};

/// `X-API-Key` header name (Kova auth). The key is held server-side by Lumen and
/// attached here; the browser never sees it.
const API_KEY_HEADER: &str = "X-API-Key";
/// Per-request timeout. Console commands are interactive; fail fast rather than
/// hang the dashboard's connection thread. Matches `pull.rs`'s 30 s budget
/// (some control verbs touch the WAL).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Response body cap (~2 MiB). Bounds memory against a hostile or huge reply; a
/// list/detail response is a few KiB, so this is generous headroom.
const MAX_RESPONSE_BYTES: u64 = 2 * 1024 * 1024;
/// Reachability-probe timeout — far shorter than [`REQUEST_TIMEOUT`] so a
/// stalled-but-TCP-up Kova can't freeze the dashboard's status check (and the
/// Terminal tab's "connecting…") for the whole command budget.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// The top-level verbs the console accepts. Drives `help` + client-side tab
/// completion in the dashboard; a Rust test asserts the dashboard JS lists the
/// same set (drift guard).
pub const KOVA_VERBS: &[&str] = &[
    "status",
    "config",
    "agents",
    "agent",
    "workflows",
    "workflow",
    "awaiting",
    "approvals",
    "schedules",
    "schedule",
    "tools",
    "queues",
    "traces",
    "trace",
    "llm",
    "help",
    "clear",
];

/// Reachability of the configured Kova endpoint (cheap `GET /health` probe).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KovaStatus {
    /// Kova answered — the API is live.
    Reachable,
    /// Kova is configured but did not answer (down / wrong URL / network).
    Unreachable,
}

/// Outcome of interpreting one console line. The dashboard maps these to a
/// `{kind, text}` JSON envelope; the one-shot CLI maps them to stdout/stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsoleOutcome {
    /// A rendered result (read output, mutation ack, or HTTP-error line).
    Output(String),
    /// A parse / unknown-verb error (shown red in the UI).
    Error(String),
    /// A destructive command that needs an explicit confirm before sending.
    /// The interpreter has **not** touched the client.
    Confirm(String),
}

/// A parse / validation failure for a console line.
#[derive(Debug)]
pub struct ParseError(String);

impl ParseError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// Lifecycle verbs that map to a parameterless `POST /agents/{id}/{verb}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentAction {
    Stop,
    Pause,
    Resume,
    Restart,
}

impl AgentAction {
    fn segment(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Restart => "restart",
        }
    }
}

/// A single whitelisted console command. The only commands reachable — there is
/// no free-form path / method.
#[derive(Debug, Clone, PartialEq)]
pub enum KovaCommand {
    /// Local: print the help text (no client call).
    Help,
    /// Local: clear the scrollback (no client call).
    Clear,
    /// `GET /analytics/overview`
    Status,
    /// `GET /config` — resolved config + deployment profile + validation (W0-A).
    Config,
    /// `GET /agents`
    Agents,
    /// `GET /agents/{id}`
    AgentGet(String),
    /// `GET /agents/{id}/traces`
    AgentTraces(String),
    /// `GET /agents/{id}/history`
    AgentHistory(String),
    /// `GET /agents/{id}/status`
    AgentStatus(String),
    /// `POST /agents/{id}/run`
    AgentRun { id: String, message: String },
    /// `POST /agents/{id}/{stop|pause|resume|restart}`
    AgentLifecycle { id: String, action: AgentAction },
    /// `POST /agents/{id}/{approve|deny}`
    AgentApproval {
        id: String,
        approve: bool,
        comment: Option<String>,
    },
    /// `POST /agents/{id}/reset` (destructive)
    AgentReset { id: String, reason: String },
    /// `POST /agents/{id}/terminate` (destructive)
    AgentTerminate { id: String, reason: String },
    /// `DELETE /agents/{id}` (destructive)
    AgentDelete(String),
    /// `GET /workflows/runs`
    Workflows,
    /// `GET /workflows/runs/{id}`
    WorkflowGet(String),
    /// `POST /workflows/runs/{id}/cancel`
    WorkflowCancel { id: String, reason: Option<String> },
    /// `POST /workflows/{id}/resume`
    WorkflowResume { id: String, input: Value },
    /// `GET /workflows/awaiting`
    Awaiting,
    /// `GET /approvals` — pending HITL approvals (kova's curated projection, a
    /// distinct endpoint from `/workflows/awaiting`).
    Approvals,
    /// `GET /workflows/schedules`
    Schedules,
    /// `GET /workflows/schedules/{id}`
    ScheduleGet(String),
    /// `POST /workflows/schedules/{id}/pause`
    SchedulePause(String),
    /// `POST /workflows/schedules/{id}/resume`
    ScheduleResume(String),
    /// `DELETE /workflows/schedules/{id}` (destructive)
    ScheduleDelete(String),
    /// `GET /tools`
    Tools,
    /// `GET /queues`
    Queues,
    /// `GET /traces`
    Traces,
    /// `GET /traces/{id}`
    TraceGet(String),
    /// `DELETE /traces/{id}` (destructive)
    TraceDelete(String),
    /// `GET /llm/health`
    Llm,
}

/// The single REST request a [`KovaCommand`] maps to. Built only by
/// [`KovaCommand::to_request`] — the one place a path is constructed.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// HTTP method (`GET` / `POST` / `DELETE`).
    pub method: &'static str,
    /// Path under the Kova base URL (already segment-encoded).
    pub path: String,
    /// JSON request body, if any.
    pub body: Option<Value>,
    /// Whether the command changes server state (drives `✓` rendering).
    pub mutating: bool,
    /// Whether the command is irreversible (reset / terminate / delete) and
    /// therefore requires an explicit confirm before sending.
    pub destructive: bool,
    /// Human-readable one-line summary (used for `✓` ack and confirm prompts).
    pub summary: String,
}

impl Plan {
    /// A no-op plan for local commands (`help` / `clear`). Never sent — these are
    /// short-circuited in [`run_line`]; this exists only to keep
    /// [`KovaCommand::to_request`] total without a panic path.
    fn local() -> Self {
        Self {
            method: "",
            path: String::new(),
            body: None,
            mutating: false,
            destructive: false,
            summary: String::new(),
        }
    }

    fn get(path: String, summary: &str) -> Self {
        Self {
            method: "GET",
            path,
            body: None,
            mutating: false,
            destructive: false,
            summary: summary.to_string(),
        }
    }

    fn post(path: String, body: Option<Value>, summary: String) -> Self {
        Self {
            method: "POST",
            path,
            body,
            mutating: true,
            destructive: false,
            summary,
        }
    }

    fn destructive_post(path: String, body: Option<Value>, summary: String) -> Self {
        Self {
            method: "POST",
            path,
            body,
            mutating: true,
            destructive: true,
            summary,
        }
    }

    fn destructive_delete(path: String, summary: String) -> Self {
        Self {
            method: "DELETE",
            path,
            body: None,
            mutating: true,
            destructive: true,
            summary,
        }
    }
}

/// Percent-encode an id before it becomes a path segment so a hostile id can
/// never inject an extra path segment, a query string, or `..` traversal. Only
/// the unreserved set `[A-Za-z0-9._~-]` passes through unescaped.
fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_digit(b >> 4));
            out.push(hex_digit(b & 0x0f));
        }
    }
    out
}

fn hex_digit(n: u8) -> char {
    char::from_digit(u32::from(n), 16)
        .unwrap_or('0')
        .to_ascii_uppercase()
}

impl KovaCommand {
    /// Map this command to its single REST request. This is the only place a
    /// path is built, so the verb whitelist is the single chokepoint.
    #[must_use]
    pub fn to_request(&self) -> Plan {
        match self {
            Self::Help | Self::Clear => Plan::local(),
            Self::Status => Plan::get("/api/v1/analytics/overview".to_string(), "overview"),
            Self::Config => Plan::get("/api/v1/config".to_string(), "config"),
            Self::Agents => Plan::get("/api/v1/agents".to_string(), "agents"),
            Self::AgentGet(id) => {
                Plan::get(format!("/api/v1/agents/{}", encode_segment(id)), "agent")
            }
            Self::AgentTraces(id) => Plan::get(
                format!("/api/v1/agents/{}/traces", encode_segment(id)),
                "agent traces",
            ),
            Self::AgentHistory(id) => Plan::get(
                format!("/api/v1/agents/{}/history", encode_segment(id)),
                "agent history",
            ),
            Self::AgentStatus(id) => Plan::get(
                format!("/api/v1/agents/{}/status", encode_segment(id)),
                "agent status",
            ),
            Self::AgentRun { id, message } => Plan::post(
                format!("/api/v1/agents/{}/run", encode_segment(id)),
                Some(json!({ "message": message })),
                format!("dispatched run to agent `{id}`"),
            ),
            Self::AgentLifecycle { id, action } => Plan::post(
                format!("/api/v1/agents/{}/{}", encode_segment(id), action.segment()),
                None,
                format!("{} agent `{id}`", action.segment()),
            ),
            Self::AgentApproval {
                id,
                approve,
                comment,
            } => Plan::post(
                format!(
                    "/api/v1/agents/{}/{}",
                    encode_segment(id),
                    if *approve { "approve" } else { "deny" }
                ),
                comment.as_ref().map(|c| json!({ "comment": c })),
                format!(
                    "{} agent `{id}`",
                    if *approve { "approved" } else { "denied" }
                ),
            ),
            Self::AgentReset { id, reason } => Plan::destructive_post(
                format!("/api/v1/agents/{}/reset", encode_segment(id)),
                Some(json!({ "reason": reason })),
                format!("reset agent `{id}` (bumps generation, supersedes in-flight work)"),
            ),
            Self::AgentTerminate { id, reason } => Plan::destructive_post(
                format!("/api/v1/agents/{}/terminate", encode_segment(id)),
                Some(json!({ "reason": reason })),
                format!("terminate agent `{id}`"),
            ),
            Self::AgentDelete(id) => Plan::destructive_delete(
                format!("/api/v1/agents/{}", encode_segment(id)),
                format!("delete agent `{id}`"),
            ),
            Self::Workflows => Plan::get("/api/v1/workflows/runs".to_string(), "workflow runs"),
            Self::WorkflowGet(id) => Plan::get(
                format!("/api/v1/workflows/runs/{}", encode_segment(id)),
                "workflow run",
            ),
            Self::WorkflowCancel { id, reason } => Plan::post(
                format!("/api/v1/workflows/runs/{}/cancel", encode_segment(id)),
                reason.as_ref().map(|r| json!({ "reason": r })),
                format!("cancel workflow {id}"),
            ),
            Self::WorkflowResume { id, input } => Plan::post(
                format!("/api/v1/workflows/{}/resume", encode_segment(id)),
                Some(json!({ "input": input })),
                format!("resume workflow {id}"),
            ),
            Self::Awaiting => Plan::get(
                "/api/v1/workflows/awaiting".to_string(),
                "awaiting-input workflows",
            ),
            Self::Approvals => Plan::get("/api/v1/approvals".to_string(), "approvals"),
            Self::Schedules => Plan::get("/api/v1/workflows/schedules".to_string(), "schedules"),
            Self::ScheduleGet(id) => Plan::get(
                format!("/api/v1/workflows/schedules/{}", encode_segment(id)),
                "schedule",
            ),
            Self::SchedulePause(id) => Plan::post(
                format!("/api/v1/workflows/schedules/{}/pause", encode_segment(id)),
                None,
                format!("pause schedule `{id}`"),
            ),
            Self::ScheduleResume(id) => Plan::post(
                format!("/api/v1/workflows/schedules/{}/resume", encode_segment(id)),
                None,
                format!("resume schedule `{id}`"),
            ),
            Self::ScheduleDelete(id) => Plan::destructive_delete(
                format!("/api/v1/workflows/schedules/{}", encode_segment(id)),
                format!("delete schedule `{id}`"),
            ),
            Self::Tools => Plan::get("/api/v1/tools".to_string(), "tools"),
            Self::Queues => Plan::get("/api/v1/queues".to_string(), "queues"),
            Self::Traces => Plan::get("/api/v1/traces".to_string(), "traces"),
            Self::TraceGet(id) => {
                Plan::get(format!("/api/v1/traces/{}", encode_segment(id)), "trace")
            }
            Self::TraceDelete(id) => Plan::destructive_delete(
                format!("/api/v1/traces/{}", encode_segment(id)),
                format!("delete trace `{id}`"),
            ),
            Self::Llm => Plan::get("/api/v1/llm/health".to_string(), "llm health"),
        }
    }
}

/// Split a console line into tokens, honouring `"double"` and `'single'` quotes
/// (so `agent a run "hello world"` is three+ tokens with one message arg). Inside
/// double quotes, `\"` and `\\` escape; single quotes are literal.
fn tokenize(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut has_token = false;
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                cur.push(c);
            }
        } else if in_double {
            if c == '"' {
                in_double = false;
            } else if c == '\\' {
                match chars.peek() {
                    Some(&n) if n == '"' || n == '\\' => {
                        cur.push(n);
                        chars.next();
                    }
                    _ => cur.push(c),
                }
            } else {
                cur.push(c);
            }
        } else if c == '\'' {
            in_single = true;
            has_token = true;
        } else if c == '"' {
            in_double = true;
            has_token = true;
        } else if c.is_whitespace() {
            if has_token {
                out.push(std::mem::take(&mut cur));
                has_token = false;
            }
        } else {
            cur.push(c);
            has_token = true;
        }
    }
    if has_token {
        out.push(cur);
    }
    out
}

/// Parse a console line into one whitelisted [`KovaCommand`].
///
/// # Errors
/// Returns a [`ParseError`] for an empty line, an unknown verb, a missing
/// required argument, or an unknown subcommand.
pub fn parse_command(line: &str) -> Result<KovaCommand, ParseError> {
    let toks = tokenize(line);
    let Some(verb) = toks.first() else {
        return Err(ParseError::new("empty command; type `help`"));
    };
    let rest = &toks[1..];
    match verb.as_str() {
        "help" => Ok(KovaCommand::Help),
        "clear" => Ok(KovaCommand::Clear),
        "status" => no_args(rest, KovaCommand::Status, "status"),
        "config" => no_args(rest, KovaCommand::Config, "config"),
        "agents" => no_args(rest, KovaCommand::Agents, "agents"),
        "workflows" => no_args(rest, KovaCommand::Workflows, "workflows"),
        "awaiting" => no_args(rest, KovaCommand::Awaiting, "awaiting"),
        "approvals" => no_args(rest, KovaCommand::Approvals, "approvals"),
        "schedules" => no_args(rest, KovaCommand::Schedules, "schedules"),
        "tools" => no_args(rest, KovaCommand::Tools, "tools"),
        "queues" => no_args(rest, KovaCommand::Queues, "queues"),
        "traces" => no_args(rest, KovaCommand::Traces, "traces"),
        "llm" => no_args(rest, KovaCommand::Llm, "llm"),
        "agent" => parse_agent(rest),
        "workflow" => parse_workflow(rest, line),
        "schedule" => parse_schedule(rest),
        "trace" => parse_trace(rest),
        other => Err(ParseError::new(format!(
            "unknown command `{other}`; verbs: {}",
            KOVA_VERBS.join(" ")
        ))),
    }
}

fn no_args(rest: &[String], cmd: KovaCommand, verb: &str) -> Result<KovaCommand, ParseError> {
    if rest.is_empty() {
        Ok(cmd)
    } else {
        Err(ParseError::new(format!("`{verb}` takes no arguments")))
    }
}

fn parse_agent(rest: &[String]) -> Result<KovaCommand, ParseError> {
    let Some(id) = rest.first().cloned() else {
        return Err(ParseError::new(
            "usage: agent <id> [run <msg…>|stop|pause|resume|restart|approve|deny|reset|terminate|delete|traces|history|status]",
        ));
    };
    match rest.get(1).map(String::as_str) {
        None => Ok(KovaCommand::AgentGet(id)),
        Some("traces") => Ok(KovaCommand::AgentTraces(id)),
        Some("history") => Ok(KovaCommand::AgentHistory(id)),
        Some("status") => Ok(KovaCommand::AgentStatus(id)),
        Some("run") => {
            let message = rest[2..].join(" ");
            if message.trim().is_empty() {
                return Err(ParseError::new("usage: agent <id> run <message…>"));
            }
            Ok(KovaCommand::AgentRun { id, message })
        }
        Some("stop") => lifecycle(id, AgentAction::Stop),
        Some("pause") => lifecycle(id, AgentAction::Pause),
        Some("resume") => lifecycle(id, AgentAction::Resume),
        Some("restart") => lifecycle(id, AgentAction::Restart),
        Some("approve") => Ok(KovaCommand::AgentApproval {
            id,
            approve: true,
            comment: opt_join(&rest[2..]),
        }),
        Some("deny") => Ok(KovaCommand::AgentApproval {
            id,
            approve: false,
            comment: opt_join(&rest[2..]),
        }),
        Some("reset") => Ok(KovaCommand::AgentReset {
            id,
            reason: join_or(&rest[2..], "reset via lumen console"),
        }),
        Some("terminate") => Ok(KovaCommand::AgentTerminate {
            id,
            reason: join_or(&rest[2..], "terminate via lumen console"),
        }),
        Some("delete") => Ok(KovaCommand::AgentDelete(id)),
        Some(other) => Err(ParseError::new(format!(
            "unknown `agent` subcommand `{other}`"
        ))),
    }
}

fn lifecycle(id: String, action: AgentAction) -> Result<KovaCommand, ParseError> {
    Ok(KovaCommand::AgentLifecycle { id, action })
}

fn parse_workflow(rest: &[String], line: &str) -> Result<KovaCommand, ParseError> {
    let Some(id) = rest.first().cloned() else {
        return Err(ParseError::new(
            "usage: workflow <id> [cancel [reason…]|resume <input…>]",
        ));
    };
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ParseError::new(
            "workflow id must be a numeric run id (e.g. `workflow 12`)",
        ));
    }
    match rest.get(1).map(String::as_str) {
        None => Ok(KovaCommand::WorkflowGet(id)),
        Some("cancel") => Ok(KovaCommand::WorkflowCancel {
            id,
            reason: opt_join(&rest[2..]),
        }),
        Some("resume") => {
            // Take the raw remainder of the line (after `workflow <id> resume`)
            // verbatim, NOT the token-joined `rest[2..]`: `tokenize` strips the
            // `"` quotes, which would degrade `{"k":1}` to the plain string
            // `{k:1}`. The raw tail keeps the JSON intact for `parse_input_value`.
            let raw = raw_tail_after(line, 3);
            if raw.trim().is_empty() {
                return Err(ParseError::new("usage: workflow <id> resume <input…>"));
            }
            Ok(KovaCommand::WorkflowResume {
                id,
                input: parse_input_value(raw),
            })
        }
        Some(other) => Err(ParseError::new(format!(
            "unknown `workflow` subcommand `{other}`"
        ))),
    }
}

fn parse_schedule(rest: &[String]) -> Result<KovaCommand, ParseError> {
    let Some(id) = rest.first().cloned() else {
        return Err(ParseError::new(
            "usage: schedule <id> [pause|resume|delete]",
        ));
    };
    match rest.get(1).map(String::as_str) {
        None => Ok(KovaCommand::ScheduleGet(id)),
        Some("pause") => Ok(KovaCommand::SchedulePause(id)),
        Some("resume") => Ok(KovaCommand::ScheduleResume(id)),
        Some("delete") => Ok(KovaCommand::ScheduleDelete(id)),
        Some(other) => Err(ParseError::new(format!(
            "unknown `schedule` subcommand `{other}`"
        ))),
    }
}

fn parse_trace(rest: &[String]) -> Result<KovaCommand, ParseError> {
    let Some(id) = rest.first().cloned() else {
        return Err(ParseError::new("usage: trace <id> [delete]"));
    };
    match rest.get(1).map(String::as_str) {
        None => Ok(KovaCommand::TraceGet(id)),
        Some("delete") => Ok(KovaCommand::TraceDelete(id)),
        Some(other) => Err(ParseError::new(format!(
            "unknown `trace` subcommand `{other}`"
        ))),
    }
}

/// Join trailing tokens, or `None` when there are none (optional free text).
fn opt_join(toks: &[String]) -> Option<String> {
    if toks.is_empty() {
        None
    } else {
        Some(toks.join(" "))
    }
}

/// Join trailing tokens, or a default when there are none (required free text
/// with a sensible default, e.g. fence `reason`).
fn join_or(toks: &[String], default: &str) -> String {
    if toks.is_empty() {
        default.to_string()
    } else {
        toks.join(" ")
    }
}

/// Interpret resume-input text as JSON when it parses, else as a plain string.
/// So `workflow 5 resume {"k":1}` sends an object and `workflow 5 resume hi`
/// sends `"hi"`.
fn parse_input_value(raw: &str) -> Value {
    let raw = raw.trim();
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Return the raw remainder of `line` after the first `n` whitespace-separated
/// fields, verbatim (no quote processing). Used for `resume <input…>` so a JSON
/// object keeps its `"` quotes — `tokenize` would otherwise strip them.
fn raw_tail_after(line: &str, n: usize) -> &str {
    let mut rest = line.trim_start();
    for _ in 0..n {
        match rest.find(char::is_whitespace) {
            Some(i) => rest = rest[i..].trim_start(),
            None => return "",
        }
    }
    rest
}

/// Interpret one console line against `client`, enforcing the destructive-confirm
/// gate server-side. Local verbs (`help` / `clear`) never touch `client`; a
/// destructive command with `confirm == false` returns [`ConsoleOutcome::Confirm`]
/// and is **not** sent.
pub fn run_line(client: &dyn KovaControlClient, line: &str, confirm: bool) -> ConsoleOutcome {
    let cmd = match parse_command(line) {
        Ok(c) => c,
        Err(e) => return ConsoleOutcome::Error(e.to_string()),
    };
    match &cmd {
        KovaCommand::Help => return ConsoleOutcome::Output(help_text()),
        KovaCommand::Clear => return ConsoleOutcome::Output(String::new()),
        _ => {}
    }
    let plan = cmd.to_request();
    if plan.destructive && !confirm {
        return ConsoleOutcome::Confirm(format!(
            "⚠ {} — this is destructive and cannot be undone.",
            plan.summary
        ));
    }
    match client.send(plan.method, &plan.path, plan.body.as_ref()) {
        Ok((status, value)) => ConsoleOutcome::Output(render_result(&plan, status, &value)),
        Err(e) => ConsoleOutcome::Output(format!("✗ {e}")),
    }
}

/// Render a Kova response for the terminal: HTTP errors → `✗ <status> <msg>`,
/// mutations → `✓ <summary>`, reads → an aligned table (lists) or pretty JSON
/// (objects).
#[must_use]
pub fn render_result(plan: &Plan, status: u16, value: &Value) -> String {
    if status >= 400 {
        let msg = extract_error_message(value);
        return if msg.is_empty() {
            format!("✗ {status}")
        } else {
            format!("✗ {status} {msg}")
        };
    }
    if plan.mutating {
        let detail = mutation_detail(value);
        return if detail.is_empty() {
            format!("✓ {}", plan.summary)
        } else {
            format!("✓ {} ({detail})", plan.summary)
        };
    }
    render_value(value)
}

/// Help text listing the verb set, grouped by safety class.
#[must_use]
pub fn help_text() -> String {
    "Kova control console — type a verb. Read verbs are safe; mutations change \
     server state; destructive verbs ask to confirm.\n\
     \n\
     READ\n\
     \x20 status                      analytics overview\n\
     \x20 config                      resolved config + profile + validation\n\
     \x20 agents | agent <id>         list agents | one agent\n\
     \x20 agent <id> traces|history|status\n\
     \x20 workflows | workflow <id>   workflow runs | one run\n\
     \x20 awaiting                    workflows awaiting input\n\
     \x20 approvals                   pending HITL approvals\n\
     \x20 schedules | schedule <id>   recurring schedules\n\
     \x20 tools | queues | traces | trace <id> | llm\n\
     \n\
     SAFE MUTATIONS\n\
     \x20 agent <id> run <msg…>       dispatch a run\n\
     \x20 agent <id> stop|pause|resume|restart\n\
     \x20 agent <id> approve|deny [comment…]\n\
     \x20 workflow <id> cancel [reason…]\n\
     \x20 workflow <id> resume <input…>\n\
     \x20 schedule <id> pause|resume\n\
     \n\
     DESTRUCTIVE (confirm required)\n\
     \x20 agent <id> reset [reason…]  bump generation, supersede in-flight work\n\
     \x20 agent <id> terminate [reason…]\n\
     \x20 agent <id> delete | schedule <id> delete | trace <id> delete\n\
     \n\
     LOCAL\n\
     \x20 help                        this help\n\
     \x20 clear                       clear the scrollback"
        .to_string()
}

/// Pull a human-readable message out of a Kova error body (string body, or an
/// object with an `error`/`message`/`detail`/`reason` field), else compact JSON.
fn extract_error_message(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return truncate(s, 300);
    }
    if let Some(map) = value.as_object() {
        for k in ["error", "message", "detail", "reason"] {
            if let Some(s) = map.get(k).and_then(Value::as_str) {
                return truncate(s, 300);
            }
        }
    }
    if value.is_null() {
        return String::new();
    }
    truncate(&value.to_string(), 300)
}

/// A short `key=value` detail line for a mutation ack, from a small set of
/// interesting response fields. Empty when none are present (e.g. a 204).
fn mutation_detail(value: &Value) -> String {
    let Some(map) = value.as_object() else {
        return String::new();
    };
    const KEYS: &[&str] = &[
        "task_id",
        "new_gen",
        "outcome",
        "status",
        "schedule_id",
        "state",
        "workflow_id",
        "accepted",
    ];
    KEYS.iter()
        .filter_map(|k| map.get(*k).map(|v| format!("{k}={}", cell_str(v))))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render a read response. Arrays (and single-array envelopes) become aligned
/// tables; other objects become pretty JSON.
fn render_value(value: &Value) -> String {
    match value {
        Value::Array(a) => render_rows(a),
        Value::Object(map) => {
            if let Some((key, arr)) = map
                .iter()
                .find_map(|(k, v)| v.as_array().map(|a| (k.as_str(), a)))
            {
                let mut out = String::new();
                let meta: Vec<String> = map
                    .iter()
                    .filter(|(k, v)| k.as_str() != key && !v.is_array() && !v.is_object())
                    .map(|(k, v)| format!("{k}={}", cell_str(v)))
                    .collect();
                if !meta.is_empty() {
                    out.push_str(&meta.join("  "));
                    out.push('\n');
                }
                out.push_str(&format!("{key}:\n"));
                out.push_str(&render_rows(arr));
                out
            } else {
                pretty_json(value)
            }
        }
        _ => cell_str(value),
    }
}

/// Maximum columns and per-cell width in a rendered table.
const MAX_COLS: usize = 6;
const MAX_CELL: usize = 44;

/// Render a JSON array as an aligned table (when elements are objects) or one
/// value per line (scalars / mixed).
fn render_rows(arr: &[Value]) -> String {
    if arr.is_empty() {
        return "(empty)".to_string();
    }
    if !arr.iter().all(Value::is_object) {
        return arr
            .iter()
            .map(|v| truncate(&cell_str(v), 200))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // Column order = first-seen keys across all rows, capped.
    let mut cols: Vec<String> = Vec::new();
    for v in arr {
        if let Some(map) = v.as_object() {
            for k in map.keys() {
                if !cols.contains(k) {
                    cols.push(k.clone());
                }
            }
        }
    }
    cols.truncate(MAX_COLS);

    let mut widths: Vec<usize> = cols.iter().map(|c| c.chars().count()).collect();
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(arr.len());
    for v in arr {
        let map = v.as_object();
        let row: Vec<String> = cols
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let cell = map
                    .and_then(|m| m.get(c))
                    .map(|x| truncate(&cell_str(x), MAX_CELL))
                    .unwrap_or_default();
                widths[i] = widths[i].max(cell.chars().count());
                cell
            })
            .collect();
        rows.push(row);
    }

    let mut out = String::new();
    let header = cols
        .iter()
        .enumerate()
        .map(|(i, c)| pad(c, widths[i]))
        .collect::<Vec<_>>()
        .join("  ");
    out.push_str(header.trim_end());
    out.push('\n');
    for row in &rows {
        let line = row
            .iter()
            .enumerate()
            .map(|(i, c)| pad(c, widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out.push_str(&format!(
        "({} row{})",
        arr.len(),
        if arr.len() == 1 { "" } else { "s" }
    ));
    out
}

/// One scalar cell as a string; arrays/objects collapse to compact JSON.
fn cell_str(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn pretty_json(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

/// Right-pad `s` with spaces to `width` columns (char-count based).
fn pad(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - len))
    }
}

/// Truncate `s` to `max` chars with an ellipsis (char-boundary safe).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Abstraction over Kova's control-plane REST API — the single network seam.
///
/// Implemented by [`HttpKovaClient`] in production and a stub in tests so the
/// interpreter ([`run_line`]) is exercised without a real socket.
pub trait KovaControlClient {
    /// Issue one interpreter-built request. Returns `(http_status, body)` for any
    /// HTTP answer (including 4xx/5xx, so the caller renders `✗`); only a
    /// transport failure is an `Err`.
    ///
    /// # Errors
    /// Returns a message when the host is unreachable (connection refused,
    /// timeout, …).
    fn send(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value), String>;

    /// Cheap reachability probe (`GET /health`).
    fn status(&self) -> KovaStatus;
}

/// Production [`KovaControlClient`] backed by a synchronous `ureq` client.
#[derive(Clone)]
pub struct HttpKovaClient {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl HttpKovaClient {
    /// Build a client for a Kova base URL (e.g. `http://localhost:3010` or
    /// `http://100.122.83.20:3010` over Tailscale). A trailing slash is trimmed
    /// so path joining is predictable. `redirects(0)` blocks SSRF-via-redirect.
    #[must_use]
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(REQUEST_TIMEOUT)
            .redirects(0)
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            agent,
        }
    }
}

impl KovaControlClient for HttpKovaClient {
    fn send(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value), String> {
        let url = format!("{}{path}", self.base_url);
        let mut req = self.agent.request(method, &url);
        if let Some(ref key) = self.api_key {
            req = req.set(API_KEY_HEADER, key);
        }
        // Serialize the body ourselves (`send_string` is always available — no
        // need for ureq's optional `json` feature).
        let resp = match body {
            Some(b) => req
                .set("Content-Type", "application/json")
                .send_string(&b.to_string()),
            None => req.call(),
        };
        // Capture the status + body for any HTTP answer; only a transport error
        // (unreachable host) is fatal here.
        let (status, reader) = match resp {
            Ok(r) => (r.status(), r.into_reader()),
            Err(ureq::Error::Status(code, r)) => (code, r.into_reader()),
            Err(ureq::Error::Transport(t)) => return Err(format!("connection failed: {t}")),
        };
        let mut buf = Vec::new();
        reader
            .take(MAX_RESPONSE_BYTES)
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading kova response: {e}"))?;
        let value = if buf.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&buf)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&buf).into_owned()))
        };
        Ok((status, value))
    }

    fn status(&self) -> KovaStatus {
        // Dedicated short-timeout probe (not `send`'s 30 s budget) so a stalled
        // Kova doesn't freeze the dashboard's status check. Any HTTP answer
        // (incl. 3xx/4xx) means the host is up; only a transport error is down.
        let url = format!("{}/health", self.base_url);
        let mut req = self.agent.get(&url).timeout(PROBE_TIMEOUT);
        if let Some(ref key) = self.api_key {
            req = req.set(API_KEY_HEADER, key);
        }
        match req.call() {
            Ok(_) | Err(ureq::Error::Status(_, _)) => KovaStatus::Reachable,
            Err(ureq::Error::Transport(_)) => KovaStatus::Unreachable,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// In-memory stub client — records `(method, path)` calls and returns a
    /// canned `(status, body)` so the interpreter is tested without a socket.
    struct StubKova {
        resp: Result<(u16, Value), String>,
        calls: RefCell<Vec<(String, String)>>,
        reachable: bool,
    }

    impl StubKova {
        fn ok(status: u16, body: Value) -> Self {
            Self {
                resp: Ok((status, body)),
                calls: RefCell::new(Vec::new()),
                reachable: true,
            }
        }
    }

    impl KovaControlClient for StubKova {
        fn send(
            &self,
            method: &str,
            path: &str,
            _body: Option<&Value>,
        ) -> Result<(u16, Value), String> {
            self.calls
                .borrow_mut()
                .push((method.to_string(), path.to_string()));
            self.resp.clone()
        }
        fn status(&self) -> KovaStatus {
            if self.reachable {
                KovaStatus::Reachable
            } else {
                KovaStatus::Unreachable
            }
        }
    }

    // ---- tokenize / parse ----

    #[test]
    fn tokenize_respects_quoted_args() {
        assert_eq!(
            tokenize(r#"agent a run "hello world""#),
            vec!["agent", "a", "run", "hello world"]
        );
        assert_eq!(tokenize("agent  a   status"), vec!["agent", "a", "status"]);
        assert_eq!(tokenize("'single quoted'"), vec!["single quoted"]);
    }

    #[test]
    fn parse_read_verbs() {
        assert_eq!(parse_command("status").unwrap(), KovaCommand::Status);
        assert_eq!(parse_command("config").unwrap(), KovaCommand::Config);
        assert_eq!(parse_command("agents").unwrap(), KovaCommand::Agents);
        assert_eq!(parse_command("tools").unwrap(), KovaCommand::Tools);
        assert_eq!(parse_command("queues").unwrap(), KovaCommand::Queues);
        assert_eq!(parse_command("traces").unwrap(), KovaCommand::Traces);
        assert_eq!(parse_command("llm").unwrap(), KovaCommand::Llm);
        assert_eq!(parse_command("awaiting").unwrap(), KovaCommand::Awaiting);
        assert_eq!(parse_command("approvals").unwrap(), KovaCommand::Approvals);
        assert_eq!(
            parse_command("agent foo").unwrap(),
            KovaCommand::AgentGet("foo".into())
        );
        assert_eq!(
            parse_command("workflow 12").unwrap(),
            KovaCommand::WorkflowGet("12".into())
        );
    }

    #[test]
    fn parse_run_keeps_full_message() {
        assert_eq!(
            parse_command(r#"agent a run "summarize the doc""#).unwrap(),
            KovaCommand::AgentRun {
                id: "a".into(),
                message: "summarize the doc".into()
            }
        );
    }

    #[test]
    fn parse_run_requires_message() {
        assert!(parse_command("agent a run").is_err());
        assert!(parse_command("agent a run   ").is_err());
    }

    #[test]
    fn parse_workflow_id_must_be_numeric() {
        assert!(parse_command("workflow abc").is_err());
        assert!(parse_command("workflow 7 cancel").is_ok());
    }

    #[test]
    fn parse_unknown_verb_is_error() {
        assert!(parse_command("frobnicate").is_err());
        assert!(parse_command("").is_err());
        assert!(parse_command("agent a wat").is_err());
    }

    #[test]
    fn parse_no_args_rejects_extra() {
        assert!(parse_command("status extra").is_err());
    }

    // ---- to_request (verb → exact request + flags) ----

    #[test]
    fn to_request_read_paths() {
        assert_eq!(
            KovaCommand::Status.to_request().path,
            "/api/v1/analytics/overview"
        );
        let cfg = KovaCommand::Config.to_request();
        assert_eq!(cfg.path, "/api/v1/config");
        assert_eq!(cfg.method, "GET");
        assert!(!cfg.mutating && !cfg.destructive, "config is a read verb");
        assert_eq!(
            KovaCommand::AgentGet("a".into()).to_request().path,
            "/api/v1/agents/a"
        );
        assert_eq!(
            KovaCommand::WorkflowGet("9".into()).to_request().path,
            "/api/v1/workflows/runs/9"
        );
        let p = KovaCommand::Agents.to_request();
        assert_eq!(p.method, "GET");
        assert!(!p.mutating && !p.destructive);
    }

    #[test]
    fn to_request_run_builds_body() {
        let p = KovaCommand::AgentRun {
            id: "a".into(),
            message: "hi".into(),
        }
        .to_request();
        assert_eq!(p.method, "POST");
        assert_eq!(p.path, "/api/v1/agents/a/run");
        assert_eq!(p.body, Some(json!({"message": "hi"})));
        assert!(p.mutating && !p.destructive);
    }

    #[test]
    fn to_request_cancel_resume_paths() {
        let c = KovaCommand::WorkflowCancel {
            id: "5".into(),
            reason: Some("done".into()),
        }
        .to_request();
        assert_eq!(c.method, "POST");
        assert_eq!(c.path, "/api/v1/workflows/runs/5/cancel");
        assert_eq!(c.body, Some(json!({"reason": "done"})));

        let r = KovaCommand::WorkflowResume {
            id: "5".into(),
            input: json!({"k": 1}),
        }
        .to_request();
        assert_eq!(r.path, "/api/v1/workflows/5/resume");
        assert_eq!(r.body, Some(json!({"input": {"k": 1}})));
    }

    #[test]
    fn to_request_destructive_flags() {
        for cmd in [
            KovaCommand::AgentReset {
                id: "a".into(),
                reason: "r".into(),
            },
            KovaCommand::AgentTerminate {
                id: "a".into(),
                reason: "r".into(),
            },
            KovaCommand::AgentDelete("a".into()),
            KovaCommand::ScheduleDelete("s".into()),
            KovaCommand::TraceDelete("t".into()),
        ] {
            let p = cmd.to_request();
            assert!(p.destructive, "{cmd:?} must be destructive");
            assert!(p.mutating, "{cmd:?} must be mutating");
        }
        assert_eq!(
            KovaCommand::AgentDelete("a".into()).to_request().method,
            "DELETE"
        );
    }

    #[test]
    fn to_request_encodes_unsafe_ids() {
        // A slash in an id must not inject an extra path segment.
        let p = KovaCommand::AgentGet("../evil".into()).to_request();
        assert_eq!(p.path, "/api/v1/agents/..%2Fevil");
        assert!(!p.path.contains("/../"), "no traversal reaches the path");
    }

    // ---- render_result ----

    #[test]
    fn render_list_as_table() {
        let plan = KovaCommand::Agents.to_request();
        let out = render_result(
            &plan,
            200,
            &json!([{"id": "a", "status": "idle"}, {"id": "b", "status": "running"}]),
        );
        assert!(out.contains("id"));
        assert!(out.contains("running"));
        assert!(out.contains("(2 rows)"));
    }

    #[test]
    fn render_object_as_pretty_json() {
        let plan = KovaCommand::Status.to_request();
        let out = render_result(&plan, 200, &json!({"total_runs": 3, "total_cost_usd": 0.5}));
        assert!(out.contains("total_runs"));
        assert!(out.contains('\n'), "pretty json is multi-line");
    }

    #[test]
    fn render_mutation_ack() {
        let plan = KovaCommand::AgentRun {
            id: "a".into(),
            message: "hi".into(),
        }
        .to_request();
        let out = render_result(&plan, 200, &json!({"task_id": 7, "agent_id": "a"}));
        assert!(out.starts_with('✓'));
        assert!(out.contains("task_id=7"));
    }

    #[test]
    fn render_http_error() {
        let plan = KovaCommand::AgentGet("x".into()).to_request();
        let out = render_result(&plan, 404, &json!({"error": "agent not found"}));
        assert_eq!(out, "✗ 404 agent not found");
    }

    #[test]
    fn render_envelope_with_array_field() {
        let plan = KovaCommand::Workflows.to_request();
        let out = render_result(
            &plan,
            200,
            &json!({"total_filtered": 1, "runs": [{"workflow_id": 1, "status": "completed"}]}),
        );
        assert!(out.contains("total_filtered=1"));
        assert!(out.contains("runs:"));
        assert!(out.contains("workflow_id"));
    }

    // ---- run_line (interpreter + confirm gate) ----

    #[test]
    fn run_line_read_sends_and_renders() {
        let stub = StubKova::ok(200, json!([{"id": "a"}]));
        let out = run_line(&stub, "agents", false);
        assert!(matches!(out, ConsoleOutcome::Output(_)));
        assert_eq!(stub.calls.borrow().len(), 1);
        assert_eq!(
            stub.calls.borrow()[0],
            ("GET".into(), "/api/v1/agents".into())
        );
    }

    #[test]
    fn run_line_destructive_without_confirm_does_not_send() {
        let stub = StubKova::ok(204, Value::Null);
        let out = run_line(&stub, "agent foo delete", false);
        assert!(matches!(out, ConsoleOutcome::Confirm(_)));
        assert!(
            stub.calls.borrow().is_empty(),
            "destructive must not reach the client without confirm"
        );
    }

    #[test]
    fn run_line_destructive_with_confirm_sends() {
        let stub = StubKova::ok(204, Value::Null);
        let out = run_line(&stub, "agent foo delete", true);
        assert!(matches!(out, ConsoleOutcome::Output(_)));
        assert_eq!(stub.calls.borrow().len(), 1);
        assert_eq!(
            stub.calls.borrow()[0],
            ("DELETE".into(), "/api/v1/agents/foo".into())
        );
    }

    #[test]
    fn run_line_unknown_verb_is_error_without_send() {
        let stub = StubKova::ok(200, Value::Null);
        let out = run_line(&stub, "frobnicate", false);
        assert!(matches!(out, ConsoleOutcome::Error(_)));
        assert!(stub.calls.borrow().is_empty());
    }

    #[test]
    fn run_line_help_is_local() {
        let stub = StubKova::ok(200, Value::Null);
        let out = run_line(&stub, "help", false);
        match out {
            ConsoleOutcome::Output(t) => assert!(t.contains("READ")),
            _ => panic!("help should be Output"),
        }
        assert!(stub.calls.borrow().is_empty(), "help never hits the client");
    }

    #[test]
    fn run_line_transport_error_renders_cross() {
        struct DownStub;
        impl KovaControlClient for DownStub {
            fn send(&self, _m: &str, _p: &str, _b: Option<&Value>) -> Result<(u16, Value), String> {
                Err("connection failed: refused".into())
            }
            fn status(&self) -> KovaStatus {
                KovaStatus::Unreachable
            }
        }
        let out = run_line(&DownStub, "agents", false);
        match out {
            ConsoleOutcome::Output(t) => assert!(t.starts_with('✗')),
            _ => panic!("transport error should render as ✗ output"),
        }
    }

    #[test]
    fn verbs_list_is_self_consistent() {
        // Every parseable top-level verb appears in KOVA_VERBS (drives help +
        // completion); a typo here would silently drop completion for a verb.
        for v in ["status", "agent", "workflow", "schedule", "trace", "help"] {
            assert!(KOVA_VERBS.contains(&v), "{v} missing from KOVA_VERBS");
        }
    }

    #[test]
    fn approvals_is_distinct_from_awaiting() {
        // `approvals` must hit kova's dedicated `/approvals` projection, not the
        // raw awaiting-input list — the two are different endpoints in kova-rest.
        assert_eq!(parse_command("awaiting").unwrap(), KovaCommand::Awaiting);
        assert_eq!(parse_command("approvals").unwrap(), KovaCommand::Approvals);
        assert_eq!(
            KovaCommand::Approvals.to_request().path,
            "/api/v1/approvals"
        );
        assert_eq!(
            KovaCommand::Awaiting.to_request().path,
            "/api/v1/workflows/awaiting"
        );
    }

    #[test]
    fn parse_resume_keeps_json_object_through_tokenizer() {
        // Regression: the tokenizer strips `"`, so the resume payload must be
        // taken raw — `{"k":1}` stays an object, not the string `{k:1}`.
        match parse_command(r#"workflow 5 resume {"k":1,"s":"hi"}"#).unwrap() {
            KovaCommand::WorkflowResume { id, input } => {
                assert_eq!(id, "5");
                assert_eq!(input, json!({"k": 1, "s": "hi"}), "object preserved");
            }
            other => panic!("expected WorkflowResume, got {other:?}"),
        }
        // A non-JSON payload still degrades to a plain string.
        match parse_command("workflow 5 resume just words").unwrap() {
            KovaCommand::WorkflowResume { input, .. } => {
                assert_eq!(input, Value::String("just words".into()));
            }
            other => panic!("expected WorkflowResume, got {other:?}"),
        }
        assert!(parse_command("workflow 5 resume   ").is_err());
    }

    #[test]
    fn raw_tail_after_skips_fields_verbatim() {
        assert_eq!(
            raw_tail_after(r#"workflow 5 resume {"k":1}"#, 3),
            r#"{"k":1}"#
        );
        assert_eq!(raw_tail_after("a b  c   d e", 3), "d e");
        assert_eq!(raw_tail_after("too few", 3), "");
    }

    #[test]
    fn workflow_ids_are_segment_encoded_in_to_request() {
        // Numeric ids pass through unchanged, but encoding is applied at the
        // path-building chokepoint (consistent with agent/schedule/trace ids).
        assert_eq!(
            KovaCommand::WorkflowCancel {
                id: "12".into(),
                reason: None
            }
            .to_request()
            .path,
            "/api/v1/workflows/runs/12/cancel"
        );
    }
}
