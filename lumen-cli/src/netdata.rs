//! Netdata REST client — the seam Lumen uses to consume Netdata's local JSON
//! API as the platform's metrics + ML-anomaly engine.
//!
//! The obs-pack deployment makes Netdata the observability engine (iron rule,
//! 2026-06-03): it scrapes kova-rest's Prometheus-format `/metrics` into its own
//! dbengine, runs built-in per-metric ML anomaly detection, and exposes both
//! over an unauthenticated **local** REST API on `:19999`. Lumen's dashboard
//! server proxies that API so the browser stays single-origin (no CORS, no
//! browser→Netdata calls).
//!
//! Iron-rule compliance: consuming Netdata's HTTP API is **not** linking netdata
//! (GPLv3) into a crate — it is a REST client call. No netdata code is compiled
//! into Lumen.
//!
//! Design mirrors [`crate::pull`]: the network seam is the [`NetdataClient`]
//! trait so the dashboard routes and validation are pure and unit-testable with
//! a stub. Only [`HttpNetdataClient`] touches the wire. The SSRF discipline
//! (fixed base URL, chart whitelist, relative-only windows, fixed `group`, no
//! redirects, bounded body) mirrors Kova's existing `http_call` guard.

use std::io::Read;
use std::time::Duration;

use serde_json::Value;

/// Per-request timeout. Netdata is local (loopback) or one Tailscale hop away;
/// fail fast so a slow proxy call can never wedge the dashboard's UI thread.
/// Deliberately shorter than [`crate::pull`]'s 30 s (interactive bulk fetch).
const NETDATA_TIMEOUT: Duration = Duration::from_secs(5);

/// Response body cap (~2 MiB). Bounds memory against a hostile or huge Netdata
/// reply; a single chart window is a few KiB, so this is generous headroom.
const MAX_RESPONSE_BYTES: u64 = 2 * 1024 * 1024;

/// Maximum data points Netdata may return per query. Caps response size and the
/// browser's render work; a relative window never needs sub-second resolution.
pub const MAX_POINTS: u32 = 1000;

/// Node-level ML anomaly chart: the fraction of all metrics currently anomalous.
/// Drives the dashboard's fleet "ML anomaly rate" KPI.
pub const NODE_ANOMALY_CHART: &str = "anomaly_detection.anomaly_rate";

/// The small, fixed set of kova charts the Metrics tab renders. `/api/v1/charts`
/// is deprecated in Netdata, so we hardcode known ids — no discovery needed.
/// Each metric becomes a `prometheus.<metric>` chart with one dimension per
/// tester job (`kova-rest-<tester>`). The cost chart is the planned
/// `kova_agent_run_total_cost_usd`; absent charts degrade gracefully to "no
/// data" rather than erroring the whole tab.
pub const KOVA_CHARTS: &[&str] = &[
    "prometheus.kova_queue_depth",
    "prometheus.kova_wal_bytes_total",
    "prometheus.kova_agent_iterations_total",
    "prometheus.kova_agent_tokens_completion",
    "prometheus.kova_agent_run_total_cost_usd",
];

/// Reachability of the configured Netdata endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetdataStatus {
    /// Netdata answered — the API is live.
    Reachable,
    /// Netdata is configured but did not answer (down / wrong URL / network).
    Unreachable,
}

/// Whether a chart id is allowed to be proxied.
///
/// Anchored prefix whitelist (kova's Prometheus charts + Netdata's anomaly
/// charts) plus a strict character class so a chart param can never inject extra
/// query keys, a path segment, or traversal. Anchored `starts_with` keeps us off
/// a `regex` dependency.
#[must_use]
pub fn valid_chart(chart: &str) -> bool {
    let allowed_prefix =
        chart.starts_with("prometheus.kova_") || chart.starts_with("anomaly_detection.");
    let safe_chars = !chart.is_empty()
        && chart.len() <= 128
        && chart
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
    allowed_prefix && safe_chars
}

/// Clamp a requested point count into `1..=MAX_POINTS`.
#[must_use]
pub fn clamp_points(points: u32) -> u32 {
    points.clamp(1, MAX_POINTS)
}

/// Whether a relative window bound is allowed.
///
/// Netdata relative windows are `<= 0` (`after=-3600` = the last hour). A
/// positive value would be an absolute epoch — we never proxy absolute windows,
/// so a browser cannot ask for arbitrary historical ranges.
#[must_use]
pub fn valid_relative(bound: i64) -> bool {
    bound <= 0
}

/// Abstraction over Netdata's REST API — the single network seam.
///
/// Implemented by [`HttpNetdataClient`] in production and by a stub in tests so
/// the dashboard routing logic is exercised without a real socket.
pub trait NetdataClient {
    /// Fetch a chart's values over a relative window (`/api/v1/data`,
    /// `group=average`, `format=json`). Both `after` and `before` are relative
    /// (`<= 0`); `before = 0` means "up to now", a negative `before` bounds the
    /// window's end (used by the dashboard's per-run deep-link).
    ///
    /// # Errors
    /// Returns a message when the chart id is invalid or Netdata is unreachable
    /// / returns non-JSON.
    fn query_data(
        &self,
        chart: &str,
        after: i64,
        before: i64,
        points: u32,
    ) -> Result<Value, String>;

    /// Fetch a chart's ML anomaly rate over a relative window (the same query
    /// plus `options=anomaly-bit`, so values are the 0/100 anomaly bit
    /// aggregated to a percentage).
    ///
    /// # Errors
    /// As [`NetdataClient::query_data`].
    fn anomaly_rate(
        &self,
        chart: &str,
        after: i64,
        before: i64,
        points: u32,
    ) -> Result<Value, String>;

    /// Fetch active alarms (`/api/v1/alarms`).
    ///
    /// # Errors
    /// Returns a message when Netdata is unreachable or returns non-JSON.
    fn alarms(&self) -> Result<Value, String>;

    /// Cheap reachability probe (a single `/api/v1/alarms` attempt).
    fn status(&self) -> NetdataStatus;
}

/// Production [`NetdataClient`] backed by a synchronous `ureq` client.
#[derive(Clone)]
pub struct HttpNetdataClient {
    base_url: String,
    agent: ureq::Agent,
}

impl HttpNetdataClient {
    /// Build a client for a Netdata base URL (e.g. `http://localhost:19999` or
    /// `http://100.122.83.20:19999` over Tailscale). A trailing slash is trimmed
    /// so path joining is predictable.
    ///
    /// `redirects(0)` blocks SSRF-via-redirect (a hostile Netdata reply can't
    /// bounce us to cloud metadata or an internal host).
    #[must_use]
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(NETDATA_TIMEOUT)
            .redirects(0)
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            agent,
        }
    }

    /// GET a URL and parse the body as JSON, with a bounded read.
    fn get_json(&self, url: &str) -> Result<Value, String> {
        let resp = self
            .agent
            .get(url)
            .call()
            .map_err(|e| describe_ureq_error(&e))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .take(MAX_RESPONSE_BYTES)
            .read_to_end(&mut buf)
            .map_err(|e| format!("reading netdata response: {e}"))?;
        serde_json::from_slice(&buf).map_err(|e| format!("invalid netdata JSON: {e}"))
    }

    /// Build a validated `/api/v1/data` URL. The chart is re-checked here — this
    /// is the single URL-construction chokepoint, so no caller can build a URL
    /// for a chart outside the whitelist. `group` is fixed to `average`; only
    /// relative (`<= 0`) `after`/`before` windows are ever issued (a positive
    /// bound is clamped to `0` defensively).
    fn data_url(
        &self,
        chart: &str,
        after: i64,
        before: i64,
        points: u32,
        anomaly: bool,
    ) -> Result<String, String> {
        if !valid_chart(chart) {
            return Err(format!("chart `{chart}` is not in the allowed set"));
        }
        let after = if valid_relative(after) { after } else { 0 };
        let before = if valid_relative(before) { before } else { 0 };
        let points = clamp_points(points);
        let opts = if anomaly { "&options=anomaly-bit" } else { "" };
        Ok(format!(
            "{}/api/v1/data?chart={chart}&after={after}&before={before}&points={points}&group=average&format=json{opts}",
            self.base_url
        ))
    }
}

impl NetdataClient for HttpNetdataClient {
    fn query_data(
        &self,
        chart: &str,
        after: i64,
        before: i64,
        points: u32,
    ) -> Result<Value, String> {
        let url = self.data_url(chart, after, before, points, false)?;
        self.get_json(&url)
    }

    fn anomaly_rate(
        &self,
        chart: &str,
        after: i64,
        before: i64,
        points: u32,
    ) -> Result<Value, String> {
        let url = self.data_url(chart, after, before, points, true)?;
        self.get_json(&url)
    }

    fn alarms(&self) -> Result<Value, String> {
        self.get_json(&format!("{}/api/v1/alarms", self.base_url))
    }

    fn status(&self) -> NetdataStatus {
        match self.alarms() {
            Ok(_) => NetdataStatus::Reachable,
            Err(_) => NetdataStatus::Unreachable,
        }
    }
}

/// Turn a ureq error into a single human-readable line (HTTP status vs
/// transport failure).
fn describe_ureq_error(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, _) => format!("netdata HTTP {code}"),
        ureq::Error::Transport(t) => format!("netdata unreachable: {t}"),
    }
}

/// Aggregate a Netdata `options=anomaly-bit` `/api/v1/data` body to a single
/// 0–100 anomaly-rate number: the mean, across all dimensions, of the most
/// recent data row's values. Defensive against missing fields (returns `None`),
/// so a schema drift degrades to "no data" rather than panicking.
///
/// Netdata's `format=json` shape is `{ "labels": ["time", dim, …],
/// "data": [[t, v, …], …] }`; each `v` under `anomaly-bit` is the 0/100 bit
/// (or a percentage when aggregated). We average the latest row's dimension
/// values — a stable, dimension-count-independent fleet figure for that chart.
#[must_use]
pub fn anomaly_rate_from_data(body: &Value) -> Option<f64> {
    let rows = body.get("data")?.as_array()?;
    let last = rows.last()?.as_array()?;
    // Column 0 is the timestamp; the rest are per-dimension values.
    let dims = last.get(1..)?;
    let mut sum = 0.0_f64;
    let mut n = 0_u32;
    for v in dims {
        if let Some(x) = v.as_f64() {
            sum += x;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / f64::from(n))
    }
}

/// Extract the most recent per-dimension values from a Netdata `/api/v1/data`
/// `format=json` body as `(dimension, value)` pairs (the dimension is the tester
/// job, e.g. `kova-rest-t1`). Empty when the body has no rows; defensive against
/// schema drift (skips non-numeric / unlabeled columns).
#[must_use]
pub fn latest_dimensions(body: &Value) -> Vec<(String, f64)> {
    let labels = body.get("labels").and_then(Value::as_array);
    let last = body
        .get("data")
        .and_then(Value::as_array)
        .and_then(|rows| rows.last())
        .and_then(Value::as_array);
    let (Some(labels), Some(last)) = (labels, last) else {
        return Vec::new();
    };
    // Column 0 is the timestamp; pair each later column with its label.
    let mut out = Vec::new();
    for i in 1..last.len() {
        let name = labels
            .get(i)
            .and_then(Value::as_str)
            .map_or_else(|| format!("dim{i}"), str::to_string);
        if let Some(v) = last.get(i).and_then(Value::as_f64) {
            out.push((name, v));
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_chart_accepts_kova_prometheus_charts() {
        assert!(valid_chart("prometheus.kova_queue_depth"));
        assert!(valid_chart("prometheus.kova_wal_bytes_total"));
        assert!(valid_chart("anomaly_detection.anomaly_rate"));
    }

    #[test]
    fn valid_chart_rejects_out_of_whitelist_and_injection() {
        assert!(!valid_chart("system.cpu"), "non-kova system chart");
        assert!(!valid_chart("prometheus.evil"), "kova_ prefix required");
        assert!(!valid_chart("../etc/passwd"), "path traversal");
        assert!(
            !valid_chart("prometheus.kova_q&options=unaligned"),
            "query-injection chars must be rejected"
        );
        assert!(!valid_chart("prometheus.kova_q/extra"), "slash rejected");
        assert!(!valid_chart(""), "empty rejected");
        assert!(
            !valid_chart(&format!("prometheus.kova_{}", "x".repeat(200))),
            "over-long rejected"
        );
    }

    #[test]
    fn clamp_points_bounds_to_range() {
        assert_eq!(clamp_points(0), 1, "zero clamps up to 1");
        assert_eq!(clamp_points(60), 60);
        assert_eq!(clamp_points(5000), MAX_POINTS, "over-max clamps down");
    }

    #[test]
    fn valid_relative_rejects_positive_absolute_windows() {
        assert!(valid_relative(0));
        assert!(valid_relative(-3600));
        assert!(!valid_relative(1), "positive (absolute epoch) rejected");
        assert!(!valid_relative(i64::MAX));
    }

    #[test]
    fn anomaly_rate_aggregates_latest_row_mean() {
        // labels: time, dimA, dimB; latest row values 0 and 100 → mean 50.
        let body = json!({
            "labels": ["time", "kova-rest-t1", "kova-rest-t2"],
            "data": [[1000, 0, 0], [2000, 0, 100]]
        });
        let r = anomaly_rate_from_data(&body).unwrap();
        assert!((r - 50.0).abs() < 1e-9, "mean of [0,100] is 50, got {r}");
    }

    #[test]
    fn anomaly_rate_handles_missing_or_empty_data() {
        assert_eq!(anomaly_rate_from_data(&json!({})), None);
        assert_eq!(anomaly_rate_from_data(&json!({"data": []})), None);
        // A row with only a timestamp (no dimensions) yields None, not a panic.
        assert_eq!(
            anomaly_rate_from_data(&json!({"labels": ["time"], "data": [[1000]]})),
            None
        );
    }

    #[test]
    fn latest_dimensions_pairs_labels_with_last_row() {
        let body = json!({
            "labels": ["time", "kova-rest-t1", "kova-rest-t2"],
            "data": [[1000, 3, 4], [2000, 7, 9]]
        });
        let dims = latest_dimensions(&body);
        assert_eq!(
            dims,
            vec![
                ("kova-rest-t1".to_string(), 7.0),
                ("kova-rest-t2".to_string(), 9.0)
            ]
        );
        assert!(latest_dimensions(&json!({})).is_empty());
        assert!(latest_dimensions(&json!({"data": []})).is_empty());
    }

    #[test]
    fn kova_charts_are_all_valid() {
        for c in KOVA_CHARTS {
            assert!(valid_chart(c), "{c} must pass valid_chart");
        }
        assert!(valid_chart(NODE_ANOMALY_CHART));
    }
}
