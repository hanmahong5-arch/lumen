//! Local web dashboard — a Temporal-style timeline of agent traces, plus a
//! Metrics tab backed by Netdata (metrics + ML anomaly + alarms).
//!
//! Deliberately dependency-light: a tiny blocking HTTP/1.1 server over
//! `std::net` (no tokio, no axum) so the CLI binary stays small and the
//! dashboard works offline. It serves one self-contained HTML page plus JSON
//! endpoints:
//!   - `/api/traces`           — full per-step timeline read live from disk.
//!   - `/api/netdata-status`   — `{configured, reachable}` (drives the UI).
//!   - `/api/metrics`          — validated proxy of Netdata `/api/v1/data`.
//!   - `/api/anomalies`        — ML anomaly-rate for the kova charts + node.
//!   - `/api/alarms`           — proxy of Netdata `/api/v1/alarms`.
//!   - `/api/kova-status`      — `{configured, reachable}` for the Terminal tab.
//!   - `POST /api/console`     — run one whitelisted kova control verb (the
//!     Terminal tab); the kova API key is held here, server-side, and never
//!     reaches the browser.
//!
//! The browser only ever talks to this loopback origin; all Netdata + kova I/O is
//! a server-side `ureq` call, so the page stays single-origin (no CORS) and works
//! whether Lumen is co-located with the backends or one Tailscale hop away.

use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use lumen_core::lifecycle::Lifecycle;
use lumen_core::trace::TraceStore;

use crate::kova::{ConsoleOutcome, HttpKovaClient, KovaControlClient, KovaStatus, run_line};
use crate::netdata::{
    HttpNetdataClient, KOVA_CHARTS, NODE_ANOMALY_CHART, NetdataClient, NetdataStatus, clamp_points,
    valid_chart, valid_relative,
};

/// Per-connection read timeout. Each connection is handled on its own thread, so
/// a connection that opens but never sends a request frees its thread instead of
/// wedging the accept loop.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Cap on the request bytes read before the header terminator is seen, and on the
/// `POST /api/console` body. The console body is a tiny `{line, confirm}` JSON;
/// 64 KiB is generous headroom and bounds memory against a malformed request.
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Default relative window for the Metrics tab (seconds before now). 10 minutes
/// gives the anomaly ribbons enough history to be meaningful.
const DEFAULT_AFTER: i64 = -600;
/// Default number of points per metric/anomaly query.
const DEFAULT_POINTS: u32 = 60;

/// The single-page dashboard UI, embedded at compile time (offline-friendly,
/// no CDN dependency — important on restricted networks).
const INDEX_HTML: &str = include_str!("dashboard.html");

/// The self-contained lifecycle-export shell (placeholders filled at render
/// time). Embedded so `lumen export` produces a single offline `.html`.
const LIFECYCLE_HTML: &str = include_str!("lifecycle.html");

/// Markers bracketing the shared lifecycle-render JS in [`INDEX_HTML`]. The
/// export extracts the block between them and injects it into [`LIFECYCLE_HTML`]
/// so the live dashboard and the offline artifact render from ONE source.
const SHARED_START: &str = "<!-- SHARED_RENDER_START -->";
const SHARED_END: &str = "<!-- SHARED_RENDER_END -->";

/// Extract the shared render JS between the [`SHARED_START`]/[`SHARED_END`]
/// markers in `html` (exclusive of the markers themselves).
fn extract_shared_js(html: &str) -> Option<&str> {
    let start = html.find(SHARED_START)? + SHARED_START.len();
    let rest = &html[start..];
    let end = rest.find(SHARED_END)?;
    Some(rest[..end].trim())
}

/// Render the self-contained lifecycle HTML for `lc` (a single offline file).
///
/// Substitutes the three [`LIFECYCLE_HTML`] placeholders: `__RUN_ID__` (HTML
/// text), `// __SHARED_RENDER_BLOCK__` (the JS extracted from [`INDEX_HTML`]),
/// and `__LIFECYCLE_JSON__` (the serialized lifecycle). The JSON is substituted
/// LAST so any placeholder-looking text inside it stays literal, and `</` is
/// escaped so an embedded `</script>` can't break out of the script tag.
///
/// Returns `None` if the markers are missing (a drift-guard test prevents that)
/// or serialization fails.
pub fn render_lifecycle_export(lc: &Lifecycle, run_id: &str) -> Option<String> {
    let shared = extract_shared_js(INDEX_HTML)?;
    let json = serde_json::to_string(lc).ok()?.replace("</", "<\\/");
    let run = html_escape_min(run_id);
    // Single left-to-right pass (NOT chained .replace): a value that happens to
    // contain another placeholder's token — a run_id of "__LIFECYCLE_JSON__", or
    // a JSON summary containing "__RUN_ID__" — must never be re-substituted.
    let html = fill_placeholders(
        LIFECYCLE_HTML,
        &[
            ("__RUN_ID__", run.as_str()),
            ("// __SHARED_RENDER_BLOCK__", shared),
            ("__LIFECYCLE_JSON__", json.as_str()),
        ],
    );
    Some(html)
}

/// Fill template placeholders in ONE pass: at each step replace the earliest
/// remaining placeholder with its value and advance past it, so substituted
/// values are never re-scanned for other placeholders.
fn fill_placeholders(template: &str, subs: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len() + 4096);
    let mut rest = template;
    loop {
        let next = subs
            .iter()
            .filter_map(|(marker, value)| rest.find(marker).map(|idx| (idx, *marker, *value)))
            .min_by_key(|(idx, _, _)| *idx);
        match next {
            Some((idx, marker, value)) => {
                out.push_str(&rest[..idx]);
                out.push_str(value);
                rest = &rest[idx + marker.len()..];
            }
            None => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

/// Minimal HTML-text escaper for the run id placeholder.
fn html_escape_min(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `(status_line, content_type, body)` — the shape every route returns.
type Resp = (&'static str, &'static str, String);

const CT_HTML: &str = "text/html; charset=utf-8";
const CT_JSON: &str = "application/json";
const CT_TEXT: &str = "text/plain; charset=utf-8";

/// Serve the dashboard on `port`, reading traces from `trace_dir` live,
/// proxying Netdata at `netdata_url`, and proxying kova control commands at
/// `kova` (each when configured).
///
/// `kova` is `(base_url, api_key)`; the key is attached server-side to each
/// control request and never reaches the browser.
///
/// Blocks forever (until Ctrl-C). Binds loopback only — this is a local dev
/// tool, never an exposed service.
pub fn serve(
    port: u16,
    trace_dir: &str,
    netdata_url: Option<String>,
    kova: Option<(String, Option<String>)>,
) {
    let addr = format!("127.0.0.1:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("\x1b[31mError: cannot bind {addr}: {e}\x1b[0m");
            eprintln!("  (is another process already on port {port}? try --port <N>)");
            return;
        }
    };

    // Build the Netdata client once and share it across connection threads
    // (`ureq::Agent` is cheap to clone — an Arc internally). `None` ⇒ the
    // Metrics tab shows its config card instead of live panels.
    let netdata: Option<Arc<HttpNetdataClient>> = netdata_url
        .as_deref()
        .map(|u| Arc::new(HttpNetdataClient::new(u)));

    // Same pattern for the kova control client. `None` ⇒ the Terminal tab shows
    // its config card instead of a live prompt.
    let kova_client: Option<Arc<HttpKovaClient>> = kova
        .as_ref()
        .map(|(url, key)| Arc::new(HttpKovaClient::new(url, key.clone())));

    let url = format!("http://localhost:{port}");
    println!("\x1b[36m🌐 Lumen Dashboard\x1b[0m");
    println!("  \x1b[1m{url}\x1b[0m");
    println!("  trace dir: {trace_dir}");
    match netdata_url.as_deref() {
        Some(u) => println!("  netdata:   {u}"),
        None => println!("  netdata:   (unset — Metrics tab disabled; set LUMEN_NETDATA_URL)"),
    }
    match kova.as_ref() {
        Some((u, _)) => println!("  kova:      {u}"),
        None => println!("  kova:      (unset — Terminal tab disabled; set LUMEN_KOVA_URL)"),
    }
    println!("  press Ctrl-C to stop");
    open_browser(&url);

    let trace_dir = trace_dir.to_string();
    let allowed = Arc::new(allowed_origins(port));
    for stream in listener.incoming().flatten() {
        // Thread-per-connection so a slow (≤5 s) Netdata proxy call can't wedge
        // the UI or `/api/traces`. No pool: loopback-only dev tool, threads are
        // short-lived (read + netdata timeouts bound their lifetime).
        let trace_dir = trace_dir.clone();
        let netdata = netdata.clone();
        let kova_client = kova_client.clone();
        let allowed = allowed.clone();
        std::thread::spawn(move || {
            let nd = netdata.as_deref().map(|c| c as &dyn NetdataClient);
            let kv = kova_client.as_deref().map(|c| c as &dyn KovaControlClient);
            handle(stream, &trace_dir, nd, kv, &allowed);
        });
    }
}

/// Handle one connection: parse the request (line + headers + bounded body),
/// route, respond, close.
fn handle(
    mut stream: TcpStream,
    trace_dir: &str,
    netdata: Option<&dyn NetdataClient>,
    kova: Option<&dyn KovaControlClient>,
    allowed: &[String],
) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let Some((method, raw_path, body, origin)) = read_request(&mut stream) else {
        return;
    };
    // CSRF guard: reject a state-changing request carrying a foreign Origin
    // before it can reach the kova console (`POST /api/console` drives Kova with
    // the server-held key, so a cross-site POST must not borrow it).
    if is_cross_origin_write(&method, origin.as_deref(), allowed) {
        let _ = write_response(
            &mut stream,
            "403 Forbidden",
            CT_JSON,
            &json!({ "error": "cross-origin request rejected" }).to_string(),
        );
        return;
    }
    let (status, content_type, resp_body) =
        route(&method, &raw_path, &body, trace_dir, netdata, kova);
    let _ = write_response(&mut stream, status, content_type, &resp_body);
}

/// Parsed request prologue: method, raw target (path+query), declared body
/// length, the byte offset where the body begins, and the `Origin` header (used
/// by the CSRF guard).
struct RequestHead {
    method: String,
    raw_path: String,
    content_length: usize,
    header_len: usize,
    origin: Option<String>,
}

/// Locate `needle` within `hay`.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Parse the request line + headers from a buffer that contains at least the
/// full header block (terminated by `\r\n\r\n`). Pure — unit-tested without a
/// socket. Returns `None` until the header terminator is present or if the
/// request line is malformed.
fn parse_head(buf: &[u8]) -> Option<RequestHead> {
    let header_len = find_subslice(buf, b"\r\n\r\n")? + 4;
    let head = std::str::from_utf8(&buf[..header_len]).ok()?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let raw_path = parts.next()?.to_string();
    let mut content_length = 0usize;
    let mut origin = None;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if k.eq_ignore_ascii_case("origin") {
                origin = Some(v.trim().to_string());
            }
        }
    }
    Some(RequestHead {
        method,
        raw_path,
        content_length,
        header_len,
        origin,
    })
}

/// The loopback origins the dashboard page is served from; a same-origin browser
/// POST carries one of these as its `Origin`.
fn allowed_origins(port: u16) -> Vec<String> {
    vec![
        format!("http://localhost:{port}"),
        format!("http://127.0.0.1:{port}"),
        format!("http://[::1]:{port}"),
    ]
}

/// CSRF guard: a state-changing request (anything but GET/HEAD) carrying an
/// `Origin` header that is not one of the dashboard's own loopback origins is a
/// cross-site request and must be rejected. Without this, any web page the
/// operator happens to visit could `POST /api/console` and drive Kova with the
/// server-held key (a CORS-simple request reaches the server even though the
/// response is withheld). Same-origin browser POSTs send a matching `Origin`;
/// non-browser clients (curl, the `lumen` CLI) send none and are allowed.
fn is_cross_origin_write(method: &str, origin: Option<&str>, allowed: &[String]) -> bool {
    if matches!(method, "GET" | "HEAD") {
        return false;
    }
    match origin {
        Some(o) => !allowed.iter().any(|a| a == o),
        None => false,
    }
}

/// Read one HTTP/1.1 request: accumulate until the header terminator, then read
/// up to `Content-Length` body bytes (bounded), so a POST body survives TCP
/// segmentation. Returns `(method, raw_path, body)` or `None` on a malformed /
/// truncated request.
fn read_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>, Option<String>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let head = loop {
        if let Some(h) = parse_head(&buf) {
            break h;
        }
        if buf.len() > MAX_REQUEST_BYTES {
            return None;
        }
        match stream.read(&mut tmp) {
            Ok(0) => return None, // closed before a full header block arrived
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return None,
        }
    };

    let want = head.content_length.min(MAX_BODY_BYTES);
    let mut body = buf[head.header_len..].to_vec();
    while body.len() < want {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
            Err(_) => break,
        }
    }
    body.truncate(want);
    Some((head.method, head.raw_path, body, head.origin))
}

/// Split a raw request target into `(path, query)` (query empty if none).
fn split_path_query(raw: &str) -> (&str, &str) {
    match raw.split_once('?') {
        Some((p, q)) => (p, q),
        None => (raw, ""),
    }
}

/// Extract the first value of `key` from a `&`-separated query string. Values
/// are numbers or whitelisted chart ids (no `%`-encoding to decode).
fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == key).then_some(v)
    })
}

/// Map a method + raw request target to a response. Contains no socket I/O
/// itself — all Netdata + kova access goes through the injected clients, so this
/// is unit-tested with stubs.
fn route(
    method: &str,
    raw_path: &str,
    body: &[u8],
    trace_dir: &str,
    netdata: Option<&dyn NetdataClient>,
    kova: Option<&dyn KovaControlClient>,
) -> Resp {
    let (path, query) = split_path_query(raw_path);
    match (method, path) {
        ("GET", "/" | "/index.html") => ("200 OK", CT_HTML, INDEX_HTML.to_string()),
        ("GET", "/api/traces") => traces_response(trace_dir),
        ("GET", "/api/netdata-status") => netdata_status_response(netdata),
        ("GET", "/api/metrics") => metrics_response(netdata, query),
        ("GET", "/api/anomalies") => anomalies_response(netdata, query),
        ("GET", "/api/alarms") => alarms_response(netdata),
        ("GET", "/api/kova-status") => kova_status_response(kova),
        ("GET", "/export") => export_response(trace_dir, query),
        // `/api/traces/{id}/lifecycle` — the assembled lifecycle JSON for one run.
        ("GET", p) if p.starts_with("/api/traces/") && p.ends_with("/lifecycle") => {
            let id = &p["/api/traces/".len()..p.len() - "/lifecycle".len()];
            lifecycle_response(trace_dir, id)
        }
        ("POST", "/api/console") => console_response(kova, body),
        _ => ("404 Not Found", CT_TEXT, "not found".to_string()),
    }
}

/// `GET /api/traces/{id}/lifecycle` — assemble the lifecycle for one run from
/// disk (trace + recovery chain + sidecars) and return it as JSON. `404` with an
/// `{error}` envelope when nothing is on disk for `id` (the UI shows it inline).
fn lifecycle_response(trace_dir: &str, id: &str) -> Resp {
    match crate::lifecycle_load::build_from_dir(std::path::Path::new(trace_dir), id) {
        Some(lc) => match serde_json::to_string(&lc) {
            Ok(j) => ok_json(j),
            Err(e) => (
                "500 Internal Server Error",
                CT_TEXT,
                format!("serialize error: {e}"),
            ),
        },
        None => (
            "404 Not Found",
            CT_JSON,
            json!({ "error": format!("no lifecycle data on disk for run `{id}`") }).to_string(),
        ),
    }
}

/// `GET /export?id={id}` — the self-contained offline lifecycle HTML for one run.
fn export_response(trace_dir: &str, query: &str) -> Resp {
    let Some(id) = query_param(query, "id") else {
        return bad_request("id is required");
    };
    // `id` is not percent-decoded; the `safe_id` gate inside build_from_dir
    // rejects anything outside `[A-Za-z0-9_.-]` (and `..`), so a `%`-encoded or
    // traversal id simply 404s rather than escaping the trace dir.
    match crate::lifecycle_load::build_from_dir(std::path::Path::new(trace_dir), id) {
        Some(lc) => match render_lifecycle_export(&lc, &lc.run_id) {
            Some(html) => ("200 OK", CT_HTML, html),
            None => (
                "500 Internal Server Error",
                CT_TEXT,
                "lifecycle render failed (shared-render markers missing?)".to_string(),
            ),
        },
        None => (
            "404 Not Found",
            CT_TEXT,
            format!("no lifecycle data on disk for run `{id}`"),
        ),
    }
}

fn ok_json(body: String) -> Resp {
    ("200 OK", CT_JSON, body)
}

fn bad_request(msg: &str) -> Resp {
    (
        "400 Bad Request",
        CT_JSON,
        json!({ "error": msg }).to_string(),
    )
}

/// Clean, never-panic degradation when Netdata is unconfigured or unreachable.
fn unavailable(msg: &str) -> Resp {
    (
        "503 Service Unavailable",
        CT_JSON,
        json!({ "error": msg }).to_string(),
    )
}

/// Load every trace from disk on each request (so a fresh `lumen pull` shows up
/// on the next refresh).
fn traces_response(trace_dir: &str) -> Resp {
    match TraceStore::new(trace_dir).load_full() {
        Ok(traces) => match serde_json::to_string(&traces) {
            Ok(json) => ok_json(json),
            Err(e) => (
                "500 Internal Server Error",
                CT_TEXT,
                format!("serialize error: {e}"),
            ),
        },
        Err(e) => (
            "500 Internal Server Error",
            CT_TEXT,
            format!("trace read error: {e}"),
        ),
    }
}

/// `{configured, reachable}` — drives the UI's config-card-vs-live-panels and
/// whether to even attempt alarm/metric fetches. Always 200 (it *is* the probe).
fn netdata_status_response(netdata: Option<&dyn NetdataClient>) -> Resp {
    let reachable = netdata.is_some_and(|c| c.status() == NetdataStatus::Reachable);
    ok_json(
        json!({
            "configured": netdata.is_some(),
            "reachable": reachable,
        })
        .to_string(),
    )
}

/// `{configured, reachable}` for the Terminal tab — mirrors
/// `netdata_status_response`. Always 200 (it *is* the probe).
fn kova_status_response(kova: Option<&dyn KovaControlClient>) -> Resp {
    let reachable = kova.is_some_and(|c| c.status() == KovaStatus::Reachable);
    ok_json(
        json!({
            "configured": kova.is_some(),
            "reachable": reachable,
        })
        .to_string(),
    )
}

/// `POST /api/console` — interpret one whitelisted kova control verb. The body is
/// `{line, confirm}`. Always returns 200 with a `{kind, text}` envelope (never a
/// raw HTTP status for the command result): `output` (rendered result), `error`
/// (parse / unknown verb), `confirm` (destructive, not yet sent), or `config`
/// (kova unconfigured — drives the config card). Never panics or hangs.
fn console_response(kova: Option<&dyn KovaControlClient>, body: &[u8]) -> Resp {
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
    let line = parsed
        .get("line")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    let confirm = parsed
        .get("confirm")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    if line.is_empty() {
        return console_json("error", "empty command — type `help`");
    }
    let Some(client) = kova else {
        return console_json(
            "config",
            "Kova not configured. Restart with --kova-url (or set LUMEN_KOVA_URL).",
        );
    };
    match run_line(client, line, confirm) {
        ConsoleOutcome::Output(t) => console_json("output", &t),
        ConsoleOutcome::Error(t) => console_json("error", &t),
        ConsoleOutcome::Confirm(t) => console_json("confirm", &t),
    }
}

/// Build the `{kind, text}` console envelope (always 200).
fn console_json(kind: &str, text: &str) -> Resp {
    ok_json(json!({ "kind": kind, "text": text }).to_string())
}

/// Parse + validate the shared `after`/`before`/`points` window params (defaults
/// applied). Both `after` and `before` are relative (`<= 0`); `before` defaults
/// to `0` ("now") and is set by the dashboard's per-run deep-link to bound the
/// window's end. `Ok((after, before, points))` or `Err(bad_request_response)`.
fn parse_window(query: &str) -> Result<(i64, i64, u32), Resp> {
    let after = match query_param(query, "after") {
        Some(s) => s
            .parse::<i64>()
            .map_err(|_| bad_request("after must be an integer"))?,
        None => DEFAULT_AFTER,
    };
    if !valid_relative(after) {
        return Err(bad_request("after must be <= 0 (relative window)"));
    }
    let before = match query_param(query, "before") {
        Some(s) => s
            .parse::<i64>()
            .map_err(|_| bad_request("before must be an integer"))?,
        None => 0,
    };
    if !valid_relative(before) {
        return Err(bad_request("before must be <= 0 (relative window)"));
    }
    let points = match query_param(query, "points") {
        Some(s) => clamp_points(
            s.parse::<u32>()
                .map_err(|_| bad_request("points must be a positive integer"))?,
        ),
        None => DEFAULT_POINTS,
    };
    Ok((after, before, points))
}

/// `GET /api/metrics?chart=&after=&points=` — validated proxy of `/api/v1/data`.
fn metrics_response(netdata: Option<&dyn NetdataClient>, query: &str) -> Resp {
    let chart = match query_param(query, "chart") {
        Some(c) => c,
        None => return bad_request("chart is required"),
    };
    if !valid_chart(chart) {
        return bad_request("chart is not in the allowed set");
    }
    let (after, before, points) = match parse_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let Some(client) = netdata else {
        return unavailable("netdata not configured");
    };
    match client.query_data(chart, after, before, points) {
        Ok(v) => ok_json(v.to_string()),
        Err(e) => unavailable(&e),
    }
}

/// `GET /api/anomalies?after=&points=` — ML anomaly-rate for every kova chart
/// plus the node-level rate. Per-chart errors are reported inline; only a total
/// failure (no chart and no node) degrades to 503.
fn anomalies_response(netdata: Option<&dyn NetdataClient>, query: &str) -> Resp {
    let (after, before, points) = match parse_window(query) {
        Ok(w) => w,
        Err(resp) => return resp,
    };
    let Some(client) = netdata else {
        return unavailable("netdata not configured");
    };

    let mut charts = serde_json::Map::new();
    let mut any_ok = false;
    for &chart in KOVA_CHARTS {
        match client.anomaly_rate(chart, after, before, points) {
            Ok(v) => {
                any_ok = true;
                charts.insert(chart.to_string(), v);
            }
            Err(e) => {
                charts.insert(chart.to_string(), json!({ "error": e }));
            }
        }
    }
    let node = client
        .anomaly_rate(NODE_ANOMALY_CHART, after, before, points)
        .ok();
    any_ok |= node.is_some();

    if !any_ok {
        return unavailable("netdata unreachable");
    }
    ok_json(json!({ "node": node, "charts": charts }).to_string())
}

/// `GET /api/alarms` — proxy of Netdata `/api/v1/alarms`.
fn alarms_response(netdata: Option<&dyn NetdataClient>) -> Resp {
    let Some(client) = netdata else {
        return unavailable("netdata not configured");
    };
    match client.alarms() {
        Ok(v) => ok_json(v.to_string()),
        Err(e) => unavailable(&e),
    }
}

/// Write an HTTP/1.1 response with an explicit length and `Connection: close`.
/// Headers are fixed and safe; only the Content-Type varies (no passthrough of
/// any Netdata header in either direction).
fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// Best-effort open the system browser at `url`. Failure is ignored (headless).
///
/// Each arm is self-contained (`let _ = …spawn()`) so a target matching none of
/// the cfgs compiles to an empty body rather than referencing an unbound `cmd`.
fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/c", "start", "", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();

    #[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
    let _ = url; // unused on exotic targets with no opener
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::cell::RefCell;

    /// `route` a GET with no body — keeps the existing GET-route tests terse
    /// under the new `(method, body, kova)` signature.
    fn rget(path: &str, netdata: Option<&dyn NetdataClient>) -> Resp {
        route("GET", path, b"", "./traces", netdata, None)
    }

    /// In-memory stub kova control client — records `(method, path)` and returns
    /// a canned `(status, body)`; `reachable` drives `status()`.
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

    /// In-memory stub Netdata — no network. Each method returns a canned result;
    /// `reachable` drives `status()`.
    struct StubNetdata {
        data: Result<Value, String>,
        anomaly: Result<Value, String>,
        alarms: Result<Value, String>,
        reachable: bool,
    }

    impl StubNetdata {
        /// A reachable stub returning the same canned values everywhere.
        fn ok(value: Value) -> Self {
            Self {
                data: Ok(value.clone()),
                anomaly: Ok(value.clone()),
                alarms: Ok(value),
                reachable: true,
            }
        }

        /// An unreachable stub: every call is a transport error.
        fn down() -> Self {
            Self {
                data: Err("netdata unreachable: connection refused".to_string()),
                anomaly: Err("netdata unreachable: connection refused".to_string()),
                alarms: Err("netdata unreachable: connection refused".to_string()),
                reachable: false,
            }
        }
    }

    impl NetdataClient for StubNetdata {
        fn query_data(
            &self,
            _chart: &str,
            _after: i64,
            _before: i64,
            _points: u32,
        ) -> Result<Value, String> {
            self.data.clone()
        }
        fn anomaly_rate(
            &self,
            _chart: &str,
            _after: i64,
            _before: i64,
            _points: u32,
        ) -> Result<Value, String> {
            self.anomaly.clone()
        }
        fn alarms(&self) -> Result<Value, String> {
            self.alarms.clone()
        }
        fn status(&self) -> NetdataStatus {
            if self.reachable {
                NetdataStatus::Reachable
            } else {
                NetdataStatus::Unreachable
            }
        }
    }

    fn body_json(resp: &Resp) -> Value {
        serde_json::from_str(&resp.2).unwrap()
    }

    #[test]
    fn index_route_serves_html() {
        let (status, ct, body) = rget("/", None);
        assert_eq!(status, "200 OK");
        assert_eq!(ct, CT_HTML);
        assert!(body.contains("Lumen"));
    }

    #[test]
    fn unknown_route_is_404() {
        let (status, _, _) = rget("/nope", None);
        assert_eq!(status, "404 Not Found");
    }

    #[test]
    fn netdata_status_unconfigured() {
        let r = rget("/api/netdata-status", None);
        assert_eq!(r.0, "200 OK");
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(false));
        assert_eq!(v["reachable"], json!(false));
    }

    #[test]
    fn netdata_status_configured_and_reachable() {
        let stub = StubNetdata::ok(json!({"ok": true}));
        let r = rget("/api/netdata-status", Some(&stub));
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(true));
        assert_eq!(v["reachable"], json!(true));
    }

    #[test]
    fn netdata_status_configured_but_unreachable() {
        let stub = StubNetdata::down();
        let r = rget("/api/netdata-status", Some(&stub));
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(true));
        assert_eq!(v["reachable"], json!(false), "down stub ⇒ not reachable");
    }

    #[test]
    fn metrics_ok_with_valid_chart() {
        let stub = StubNetdata::ok(json!({"labels": ["time", "kova-rest-t1"], "data": [[1, 5]]}));
        let r = rget(
            "/api/metrics?chart=prometheus.kova_queue_depth&after=-60&points=30",
            Some(&stub),
        );
        assert_eq!(r.0, "200 OK");
        assert_eq!(r.1, CT_JSON);
        let v = body_json(&r);
        assert_eq!(v["data"][0][1], json!(5));
    }

    #[test]
    fn metrics_bad_chart_is_400_without_touching_client() {
        // An out-of-whitelist chart is rejected before any client call, even
        // with a reachable client present.
        let stub = StubNetdata::ok(json!({}));
        let r = rget("/api/metrics?chart=system.cpu", Some(&stub));
        assert_eq!(r.0, "400 Bad Request");
    }

    #[test]
    fn metrics_missing_chart_is_400() {
        let stub = StubNetdata::ok(json!({}));
        let r = rget("/api/metrics?after=-60", Some(&stub));
        assert_eq!(r.0, "400 Bad Request");
    }

    #[test]
    fn metrics_positive_after_is_400() {
        let stub = StubNetdata::ok(json!({}));
        let r = rget(
            "/api/metrics?chart=prometheus.kova_queue_depth&after=100",
            Some(&stub),
        );
        assert_eq!(
            r.0, "400 Bad Request",
            "absolute (positive) window rejected"
        );
    }

    #[test]
    fn metrics_unreachable_is_503() {
        let stub = StubNetdata::down();
        let r = rget(
            "/api/metrics?chart=prometheus.kova_queue_depth",
            Some(&stub),
        );
        assert_eq!(r.0, "503 Service Unavailable");
    }

    #[test]
    fn metrics_unconfigured_is_503() {
        let r = rget("/api/metrics?chart=prometheus.kova_queue_depth", None);
        assert_eq!(r.0, "503 Service Unavailable");
    }

    #[test]
    fn anomalies_ok_includes_all_charts_and_node() {
        let stub = StubNetdata::ok(json!({"labels": ["time", "d"], "data": [[1, 100]]}));
        let r = rget("/api/anomalies", Some(&stub));
        assert_eq!(r.0, "200 OK");
        let v = body_json(&r);
        assert!(v["node"].is_object(), "node anomaly present");
        for chart in KOVA_CHARTS {
            assert!(v["charts"][chart].is_object(), "{chart} present");
        }
    }

    #[test]
    fn anomalies_unreachable_is_503() {
        // Every chart + node errors ⇒ clean 503, no panic/hang.
        let stub = StubNetdata::down();
        let r = rget("/api/anomalies", Some(&stub));
        assert_eq!(r.0, "503 Service Unavailable");
    }

    #[test]
    fn alarms_proxy_and_degradation() {
        let stub = StubNetdata::ok(json!({"alarms": {}}));
        let r = rget("/api/alarms", Some(&stub));
        assert_eq!(r.0, "200 OK");

        let r = rget("/api/alarms", None);
        assert_eq!(r.0, "503 Service Unavailable");

        let down = StubNetdata::down();
        let r = rget("/api/alarms", Some(&down));
        assert_eq!(r.0, "503 Service Unavailable");
    }

    #[test]
    fn split_path_query_separates_params() {
        assert_eq!(
            split_path_query("/api/metrics?chart=x&after=-1"),
            ("/api/metrics", "chart=x&after=-1")
        );
        assert_eq!(split_path_query("/api/traces"), ("/api/traces", ""));
    }

    #[test]
    fn query_param_extracts_first_match() {
        let q = "chart=prometheus.kova_queue_depth&after=-60&points=30";
        assert_eq!(query_param(q, "chart"), Some("prometheus.kova_queue_depth"));
        assert_eq!(query_param(q, "after"), Some("-60"));
        assert_eq!(query_param(q, "missing"), None);
    }

    /// JS↔Rust drift guard: the dashboard JS replicates the per-run cost-outlier
    /// multiplier inline (`mean*2` in three spots). If lumen-core's
    /// `COST_ANOMALY_MULTIPLIER` ever changes, this forces the HTML to be updated
    /// in lockstep — precedent: the existing Rust↔Python drift guard.
    #[test]
    fn dashboard_js_mirrors_cost_outlier_multiplier() {
        let m = lumen_core::COST_ANOMALY_MULTIPLIER;
        assert!(
            (m - 2.0).abs() < f64::EPSILON,
            "update dashboard.html `mean*N` literals if this changes"
        );
        #[allow(clippy::cast_possible_truncation)]
        let needle = format!("mean*{}", m as i64);
        let count = INDEX_HTML.matches(&needle).count();
        assert!(
            count >= 3,
            "expected >=3 `{needle}` literals in dashboard JS, found {count}"
        );
    }

    /// JS↔Rust drift guard: the dashboard's client-side tab-completion mirrors
    /// `kova::KOVA_VERBS`. If a verb is added/removed in Rust without updating the
    /// HTML, completion silently diverges — this forces them in lockstep.
    #[test]
    fn dashboard_js_lists_all_kova_verbs() {
        for v in crate::kova::KOVA_VERBS {
            assert!(
                INDEX_HTML.contains(&format!("\"{v}\"")),
                "dashboard.html JS KOVA_VERBS is missing `{v}`"
            );
        }
    }

    // ---- kova control plane (console + status) ----

    #[test]
    fn kova_status_unconfigured() {
        let r = route("GET", "/api/kova-status", b"", "./traces", None, None);
        assert_eq!(r.0, "200 OK");
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(false));
        assert_eq!(v["reachable"], json!(false));
    }

    #[test]
    fn kova_status_configured_and_reachable() {
        let stub = StubKova::ok(200, json!({"ok": true}));
        let r = route(
            "GET",
            "/api/kova-status",
            b"",
            "./traces",
            None,
            Some(&stub),
        );
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(true));
        assert_eq!(v["reachable"], json!(true));
    }

    #[test]
    fn kova_status_configured_but_unreachable() {
        let stub = StubKova {
            resp: Ok((200, Value::Null)),
            calls: RefCell::new(Vec::new()),
            reachable: false,
        };
        let r = route(
            "GET",
            "/api/kova-status",
            b"",
            "./traces",
            None,
            Some(&stub),
        );
        let v = body_json(&r);
        assert_eq!(v["configured"], json!(true));
        assert_eq!(v["reachable"], json!(false));
    }

    /// `route` a `POST /api/console` with the given JSON body.
    fn console(body: &str, kova: Option<&dyn KovaControlClient>) -> Resp {
        route(
            "POST",
            "/api/console",
            body.as_bytes(),
            "./traces",
            None,
            kova,
        )
    }

    #[test]
    fn console_read_returns_output() {
        let stub = StubKova::ok(200, json!([{"agent_id": "a"}]));
        let r = console(r#"{"line":"agents","confirm":false}"#, Some(&stub));
        assert_eq!(r.0, "200 OK");
        assert_eq!(body_json(&r)["kind"], json!("output"));
        assert_eq!(
            stub.calls.borrow().len(),
            1,
            "read verb hits the client once"
        );
    }

    #[test]
    fn console_destructive_without_confirm_is_confirm_and_not_sent() {
        let stub = StubKova::ok(204, Value::Null);
        let r = console(
            r#"{"line":"agent foo delete","confirm":false}"#,
            Some(&stub),
        );
        assert_eq!(body_json(&r)["kind"], json!("confirm"));
        assert!(
            stub.calls.borrow().is_empty(),
            "destructive verb must not reach the client without confirm"
        );
    }

    #[test]
    fn console_destructive_with_confirm_is_sent() {
        let stub = StubKova::ok(204, Value::Null);
        let r = console(r#"{"line":"agent foo delete","confirm":true}"#, Some(&stub));
        assert_eq!(body_json(&r)["kind"], json!("output"));
        assert_eq!(stub.calls.borrow().len(), 1);
        assert_eq!(
            stub.calls.borrow()[0],
            ("DELETE".to_string(), "/api/v1/agents/foo".to_string())
        );
    }

    #[test]
    fn console_unknown_verb_is_error_and_not_sent() {
        let stub = StubKova::ok(200, Value::Null);
        let r = console(r#"{"line":"frobnicate","confirm":false}"#, Some(&stub));
        assert_eq!(body_json(&r)["kind"], json!("error"));
        assert!(stub.calls.borrow().is_empty());
    }

    #[test]
    fn console_unconfigured_is_config() {
        let r = console(r#"{"line":"agents","confirm":false}"#, None);
        assert_eq!(r.0, "200 OK");
        assert_eq!(body_json(&r)["kind"], json!("config"));
    }

    #[test]
    fn console_invalid_body_is_clean_error() {
        // A malformed body never panics; it degrades to a clean error envelope.
        let stub = StubKova::ok(200, Value::Null);
        let r = console("not json", Some(&stub));
        assert_eq!(r.0, "200 OK");
        assert_eq!(body_json(&r)["kind"], json!("error"));
        assert!(stub.calls.borrow().is_empty());
    }

    // ---- request body parsing (Content-Length) ----

    #[test]
    fn parse_head_extracts_method_path_and_body() {
        let raw = b"POST /api/console HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
        let h = parse_head(raw).expect("complete header block parses");
        assert_eq!(h.method, "POST");
        assert_eq!(h.raw_path, "/api/console");
        assert_eq!(h.content_length, 5);
        assert_eq!(&raw[h.header_len..], b"hello");
    }

    #[test]
    fn parse_head_none_until_terminator() {
        // No blank-line terminator yet ⇒ keep reading.
        assert!(parse_head(b"POST /api/console HTTP/1.1\r\nContent-Length: 5\r\n").is_none());
    }

    #[test]
    fn parse_head_captures_origin() {
        let raw = b"POST /api/console HTTP/1.1\r\nOrigin: http://evil.example\r\nContent-Length: 2\r\n\r\n{}";
        let h = parse_head(raw).expect("complete header block parses");
        assert_eq!(h.origin.as_deref(), Some("http://evil.example"));
        // Absent Origin ⇒ None (non-browser clients).
        let raw2 = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        assert_eq!(parse_head(raw2).unwrap().origin, None);
    }

    // ---- CSRF guard (cross-origin state-changing requests) ----

    #[test]
    fn cross_origin_write_is_rejected_same_origin_allowed() {
        let allowed = allowed_origins(9700);
        // A foreign Origin on a POST is a cross-site request ⇒ rejected.
        assert!(is_cross_origin_write(
            "POST",
            Some("http://evil.example"),
            &allowed
        ));
        // The dashboard's own loopback origins ⇒ allowed.
        assert!(!is_cross_origin_write(
            "POST",
            Some("http://localhost:9700"),
            &allowed
        ));
        assert!(!is_cross_origin_write(
            "POST",
            Some("http://127.0.0.1:9700"),
            &allowed
        ));
        // No Origin header (curl / the `lumen` CLI) ⇒ allowed.
        assert!(!is_cross_origin_write("POST", None, &allowed));
        // A sandboxed/opaque origin sends `Origin: null` ⇒ rejected.
        assert!(is_cross_origin_write("POST", Some("null"), &allowed));
        // GET/HEAD are read-only ⇒ never gated, even cross-origin.
        assert!(!is_cross_origin_write(
            "GET",
            Some("http://evil.example"),
            &allowed
        ));
    }

    // ---- lifecycle export drift guards (B7) ----

    /// The export shell must keep all three injection points, or
    /// `render_lifecycle_export` would silently emit an unrendered placeholder.
    #[test]
    fn lifecycle_html_has_all_three_placeholders() {
        for marker in [
            "__RUN_ID__",
            "__LIFECYCLE_JSON__",
            "// __SHARED_RENDER_BLOCK__",
        ] {
            assert!(
                LIFECYCLE_HTML.contains(marker),
                "lifecycle.html is missing placeholder `{marker}`"
            );
        }
    }

    /// The dashboard must keep both shared-render markers (the export extracts
    /// the JS between them — losing one breaks single-sourcing).
    #[test]
    fn dashboard_html_has_both_shared_markers() {
        assert!(
            INDEX_HTML.contains(SHARED_START),
            "missing SHARED_RENDER_START"
        );
        assert!(INDEX_HTML.contains(SHARED_END), "missing SHARED_RENDER_END");
    }

    /// The extracted shared block must define the three render functions the
    /// export/dashboard depend on. If one is renamed without updating the export,
    /// this fails instead of shipping a blank artifact.
    #[test]
    fn extract_shared_js_finds_render_functions() {
        let js = extract_shared_js(INDEX_HTML).expect("markers present → extractable");
        for needle in [
            "swimlane(",
            "causalDag(",
            "recoveryChain(",
            "swarmGraph(",
            "renderLifecycleInto(",
        ] {
            assert!(js.contains(needle), "shared block missing `{needle}`");
        }
        // The extracted block must NOT carry the marker comments themselves.
        assert!(!js.contains(SHARED_START));
        assert!(!js.contains(SHARED_END));
    }

    /// `render_lifecycle_export` must substitute EVERY placeholder — none may
    /// survive into the emitted artifact.
    #[test]
    fn render_lifecycle_export_substitutes_every_placeholder() {
        let lc = lumen_core::lifecycle_builder::build(&[], None, None, None);
        let html = render_lifecycle_export(&lc, &lc.run_id).expect("render succeeds");
        for marker in [
            "__RUN_ID__",
            "__LIFECYCLE_JSON__",
            "// __SHARED_RENDER_BLOCK__",
        ] {
            assert!(
                !html.contains(marker),
                "rendered export still contains placeholder `{marker}`"
            );
        }
        // The injected JS + JSON are present.
        assert!(html.contains("renderLifecycleInto("));
        assert!(html.contains("\"run_id\""));
        // No raw `</script>` from JSON content can break the page.
        assert!(!html.contains("</script></script>"));
    }

    #[test]
    fn fill_placeholders_does_not_resubstitute_values() {
        // A value for __A__ that contains __B__ must NOT be re-substituted by
        // __B__'s value — guards render_lifecycle_export against a run_id of
        // "__LIFECYCLE_JSON__" or a JSON summary containing "__RUN_ID__".
        let out = fill_placeholders(
            "title=__A__ data=__B__",
            &[("__A__", "x__B__x"), ("__B__", "DATA")],
        );
        assert_eq!(out, "title=x__B__x data=DATA");
    }

    /// A run that delegated to a swarm must carry its nodes/edges all the way
    /// into the offline export JSON, and the export must include the swarm-graph
    /// render path. The SVG is drawn client-side; this asserts the data reaches
    /// the artifact and the render function is single-sourced into it.
    #[test]
    fn swarm_lifecycle_export_embeds_graph_data() {
        use lumen_core::causal_types::{SwarmEdge, SwarmGraph, SwarmNode};
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
                    id: "researcher".into(),
                    agent_type: "worker".into(),
                    status: "completed".into(),
                    prompt_tokens: 120,
                    completion_tokens: 40,
                },
            ],
            edges: vec![SwarmEdge {
                from: "lead".into(),
                to: "researcher".into(),
                edge_type: "delegation".into(),
            }],
        };
        let lc = lumen_core::lifecycle_builder::build(&[], None, None, Some(&g));
        let html = render_lifecycle_export(&lc, &lc.run_id).expect("render");
        // Swarm data embedded verbatim in the lifecycle JSON.
        assert!(
            html.contains("\"researcher\""),
            "swarm node id in export JSON"
        );
        assert!(
            html.contains("\"delegation\""),
            "swarm edge type in export JSON"
        );
        // The graph render path is present (single-sourced shared block).
        assert!(
            html.contains("swarmGraph("),
            "swarm-graph render fn extracted into export"
        );
        assert!(
            html.contains("Swarm delegations"),
            "swarm section title present"
        );
    }

    #[test]
    fn render_lifecycle_export_run_id_collision_is_safe() {
        // A run_id equal to the JSON placeholder must not inject the blob into
        // the title (single-pass fill); the title shows the literal run_id.
        let mut lc = lumen_core::lifecycle_builder::build(&[], None, None, None);
        lc.run_id = "__LIFECYCLE_JSON__".to_string();
        let html = render_lifecycle_export(&lc, &lc.run_id).expect("render");
        assert!(
            html.contains("run __LIFECYCLE_JSON__"),
            "title shows literal id"
        );
        // The data block is still the real JSON, exactly once.
        assert_eq!(html.matches("const LIFECYCLE = {").count(), 1);
    }
}
