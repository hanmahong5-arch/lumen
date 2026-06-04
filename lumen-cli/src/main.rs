//! Lumen CLI — illuminate your AI agents.

mod dashboard;
mod kova;
mod netdata;
mod pull;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lumen", version, about = "Illuminate your AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Replay an agent run from trace files (zero LLM cost).
    Replay {
        /// Trace ID to replay.
        trace_id: String,
        /// Trace directory.
        #[arg(long, default_value = "./traces", alias = "wal-dir")]
        trace_dir: String,
        /// Start replay from this step.
        #[arg(long)]
        from_step: Option<u32>,
    },
    /// Show LLM cost report.
    Cost {
        /// Time window (e.g., "24h", "7d").
        #[arg(long, default_value = "24h")]
        last: String,
        /// Trace directory.
        #[arg(long, default_value = "./traces", alias = "wal-dir")]
        trace_dir: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// List all agent traces.
    Traces {
        /// Trace directory.
        #[arg(long, default_value = "./traces", alias = "wal-dir")]
        trace_dir: String,
        /// Maximum traces to show.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Open the web dashboard — a Temporal-style timeline of every agent run,
    /// plus a Netdata-backed Metrics tab (metrics + ML anomaly + alarms).
    Dashboard {
        /// Port to serve on.
        #[arg(long, default_value = "9700")]
        port: u16,
        /// Trace directory to visualize.
        #[arg(long, default_value = "./traces", alias = "wal-dir")]
        trace_dir: String,
        /// Netdata base URL for the Metrics tab (e.g. http://localhost:19999).
        /// Falls back to LUMEN_NETDATA_URL. Unset ⇒ Metrics tab shows a config card.
        #[arg(long)]
        netdata_url: Option<String>,
        /// Kova base URL for the Terminal tab (e.g. http://localhost:3010).
        /// Falls back to LUMEN_KOVA_URL. Unset ⇒ Terminal tab shows a config card.
        #[arg(long)]
        kova_url: Option<String>,
        /// Kova API key for the Terminal tab (`X-API-Key`), held server-side.
        /// Falls back to LUMEN_KOVA_API_KEY then KOVA_API_KEY.
        #[arg(long)]
        api_key: Option<String>,
    },
    /// Snapshot kova metrics + ML anomaly rates from Netdata (headless).
    ///
    /// Proxies Netdata `/api/v1/data` for the fixed kova chart set and prints
    /// the latest per-tester values alongside each chart's ML anomaly rate —
    /// useful for stress-campaign anomaly snapshotting without the dashboard.
    Metrics {
        /// Netdata base URL (e.g. http://localhost:19999). Falls back to LUMEN_NETDATA_URL.
        #[arg(long)]
        netdata_url: Option<String>,
        /// Relative time window (e.g. "10m", "1h", "24h").
        #[arg(long, default_value = "10m")]
        last: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Pull traces from a running Kova into a local trace directory.
    ///
    /// Lists traces via `GET /api/v1/traces?limit=N`, fetches each via
    /// `GET /api/v1/traces/{id}`, and writes `{trace-dir}/{id}.json` so the
    /// replay/cost/traces commands then work against pulled data.
    Pull {
        /// Base URL of the running Kova (e.g. http://100.122.83.20:3010).
        #[arg(long)]
        kova_url: String,
        /// API key (`X-API-Key`). Falls back to LUMEN_KOVA_API_KEY then KOVA_API_KEY.
        #[arg(long)]
        api_key: Option<String>,
        /// Directory to write pulled trace JSON files into.
        #[arg(long, default_value = "./traces", alias = "wal-dir")]
        trace_dir: String,
        /// Maximum number of traces to fetch (default 200, max 10000).
        #[arg(long, default_value_t = pull::DEFAULT_PULL_LIMIT)]
        limit: usize,
    },
    /// Run a single Kova control command (headless console).
    ///
    /// Parses one whitelisted verb (same set as the dashboard's Terminal tab),
    /// sends it to a running Kova, and prints the result. e.g.
    /// `lumen kova "agents"` or `lumen kova "agent foo run hello"`.
    Kova {
        /// The command line, e.g. "agents" or "agent foo run hello world".
        command: String,
        /// Base URL of the running Kova (e.g. http://localhost:3010).
        /// Falls back to LUMEN_KOVA_URL.
        #[arg(long)]
        kova_url: Option<String>,
        /// API key (`X-API-Key`). Falls back to LUMEN_KOVA_API_KEY then KOVA_API_KEY.
        #[arg(long)]
        api_key: Option<String>,
        /// Confirm destructive verbs (reset/terminate/delete) non-interactively.
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Replay {
            trace_id,
            trace_dir,
            from_step,
        } => cmd_replay(&trace_id, &trace_dir, from_step),
        Commands::Cost {
            last,
            trace_dir,
            format,
        } => cmd_cost(&last, &trace_dir, &format),
        Commands::Traces { trace_dir, limit } => cmd_traces(&trace_dir, limit),
        Commands::Dashboard {
            port,
            trace_dir,
            netdata_url,
            kova_url,
            api_key,
        } => {
            let kova = resolve_kova_url(kova_url).map(|u| (u, resolve_api_key(api_key)));
            dashboard::serve(port, &trace_dir, resolve_netdata_url(netdata_url), kova);
        }
        Commands::Metrics {
            netdata_url,
            last,
            format,
        } => return cmd_metrics(resolve_netdata_url(netdata_url), &last, &format),
        Commands::Pull {
            kova_url,
            api_key,
            trace_dir,
            limit,
        } => return cmd_pull(&kova_url, api_key, &trace_dir, limit),
        Commands::Kova {
            command,
            kova_url,
            api_key,
            yes,
        } => return cmd_kova(kova_url, api_key, &command, yes),
    }
    std::process::ExitCode::SUCCESS
}

/// Resolve the Netdata base URL: explicit `--netdata-url` wins, then
/// `LUMEN_NETDATA_URL`. `None` ⇒ Netdata features are disabled.
fn resolve_netdata_url(explicit: Option<String>) -> Option<String> {
    explicit
        .or_else(|| std::env::var("LUMEN_NETDATA_URL").ok())
        .filter(|u| !u.is_empty())
}

/// Resolve the API key: explicit `--api-key` wins, then `LUMEN_KOVA_API_KEY`,
/// then `KOVA_API_KEY`. Returns `None` if none set (unauthenticated request).
fn resolve_api_key(explicit: Option<String>) -> Option<String> {
    explicit
        .or_else(|| std::env::var("LUMEN_KOVA_API_KEY").ok())
        .or_else(|| std::env::var("KOVA_API_KEY").ok())
        .filter(|k| !k.is_empty())
}

/// Resolve the Kova base URL: explicit `--kova-url` wins, then `LUMEN_KOVA_URL`.
/// `None` ⇒ the Terminal tab / `lumen kova` are disabled.
fn resolve_kova_url(explicit: Option<String>) -> Option<String> {
    explicit
        .or_else(|| std::env::var("LUMEN_KOVA_URL").ok())
        .filter(|u| !u.is_empty())
}

/// `lumen kova "<command>"` — one-shot headless console. Parses one whitelisted
/// verb, sends it to Kova, prints the result. Destructive verbs require `--yes`.
fn cmd_kova(
    kova_url: Option<String>,
    api_key: Option<String>,
    command: &str,
    yes: bool,
) -> std::process::ExitCode {
    let Some(url) = resolve_kova_url(kova_url) else {
        eprintln!(
            "\x1b[31mError: no Kova URL.\x1b[0m set --kova-url or LUMEN_KOVA_URL \
             (e.g. http://localhost:3010)"
        );
        return std::process::ExitCode::FAILURE;
    };
    let client = kova::HttpKovaClient::new(&url, resolve_api_key(api_key));
    match kova::run_line(&client, command, yes) {
        kova::ConsoleOutcome::Output(text) => {
            println!("{text}");
            std::process::ExitCode::SUCCESS
        }
        kova::ConsoleOutcome::Confirm(text) => {
            eprintln!("\x1b[33m{text}\x1b[0m");
            eprintln!("  (re-run with --yes to confirm)");
            std::process::ExitCode::FAILURE
        }
        kova::ConsoleOutcome::Error(text) => {
            eprintln!("\x1b[31mError: {text}\x1b[0m");
            std::process::ExitCode::FAILURE
        }
    }
}

fn cmd_pull(
    kova_url: &str,
    api_key: Option<String>,
    trace_dir: &str,
    limit: usize,
) -> std::process::ExitCode {
    let limit = limit.min(pull::MAX_PULL_LIMIT);
    let key = resolve_api_key(api_key);
    let fetcher = pull::HttpFetcher::new(kova_url, key);
    let dir = std::path::Path::new(trace_dir);

    println!("\x1b[36m⇣ Pulling up to {limit} traces from {kova_url}\x1b[0m");
    match pull::pull_into(&fetcher, dir, limit) {
        Ok(summary) => {
            if summary.listed == 0 {
                println!("  No traces on the server. Nothing to pull.");
            } else {
                println!(
                    "  \x1b[32m✓ {} written\x1b[0m of {} listed → {}",
                    summary.written,
                    summary.listed,
                    dir.display()
                );
                if !summary.skipped.is_empty() {
                    println!(
                        "  \x1b[33m⚠ {} skipped: {}\x1b[0m",
                        summary.skipped.len(),
                        summary.skipped.join(", ")
                    );
                }
            }
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\x1b[31mError: {e}\x1b[0m");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Points requested per chart for the headless metrics snapshot.
const METRICS_POINTS: u32 = 60;

fn cmd_metrics(netdata_url: Option<String>, last: &str, format: &str) -> std::process::ExitCode {
    let Some(url) = netdata_url else {
        eprintln!(
            "\x1b[31mError: no Netdata URL.\x1b[0m set --netdata-url or LUMEN_NETDATA_URL \
             (e.g. http://localhost:19999)"
        );
        return std::process::ExitCode::FAILURE;
    };
    let client = netdata::HttpNetdataClient::new(&url);
    let after = parse_relative_secs(last);
    let snapshot = metrics_snapshot(&client, after, METRICS_POINTS);

    if format == "json" {
        if let Ok(json) = serde_json::to_string_pretty(&snapshot) {
            println!("{json}");
        }
    } else {
        print_metrics_snapshot(&snapshot, &url, last);
    }
    std::process::ExitCode::SUCCESS
}

/// Collect a metrics + ML-anomaly snapshot for the fixed kova chart set into a
/// JSON value. Per-chart failures are recorded as `{ "error": … }` so a missing
/// chart (e.g. the planned cost metric) never sinks the whole snapshot.
fn metrics_snapshot(
    client: &dyn netdata::NetdataClient,
    after: i64,
    points: u32,
) -> serde_json::Value {
    use serde_json::json;

    // Headless snapshot always uses an up-to-now window (before = 0).
    const NOW: i64 = 0;
    let mut charts = serde_json::Map::new();
    for &chart in netdata::KOVA_CHARTS {
        let entry = match client.query_data(chart, after, NOW, points) {
            Ok(body) => {
                let dims: serde_json::Map<String, serde_json::Value> =
                    netdata::latest_dimensions(&body)
                        .into_iter()
                        .map(|(k, v)| (k, json!(v)))
                        .collect();
                let anomaly = client
                    .anomaly_rate(chart, after, NOW, points)
                    .ok()
                    .as_ref()
                    .and_then(netdata::anomaly_rate_from_data);
                json!({ "dimensions": dims, "anomaly_rate": anomaly })
            }
            Err(e) => json!({ "error": e }),
        };
        charts.insert(chart.to_string(), entry);
    }
    let node = client
        .anomaly_rate(netdata::NODE_ANOMALY_CHART, after, NOW, points)
        .ok()
        .as_ref()
        .and_then(netdata::anomaly_rate_from_data);
    json!({ "node_anomaly_rate": node, "charts": charts })
}

fn print_metrics_snapshot(snapshot: &serde_json::Value, url: &str, window: &str) {
    println!("\x1b[36m📈 Kova Metrics via Netdata (last {window})\x1b[0m");
    println!("  source: {url}");
    match snapshot
        .get("node_anomaly_rate")
        .and_then(serde_json::Value::as_f64)
    {
        Some(rate) => println!("  \x1b[1mfleet ML anomaly rate: {rate:.1}%\x1b[0m\n"),
        None => println!("  fleet ML anomaly rate: n/a (node anomaly chart unavailable)\n"),
    }

    let Some(charts) = snapshot
        .get("charts")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    for (chart, entry) in charts {
        let short = chart.strip_prefix("prometheus.").unwrap_or(chart);
        if let Some(err) = entry.get("error").and_then(serde_json::Value::as_str) {
            println!("  {short:38} \x1b[33m(no data: {err})\x1b[0m");
            continue;
        }
        let anomaly = entry
            .get("anomaly_rate")
            .and_then(serde_json::Value::as_f64)
            .map_or_else(|| "—".to_string(), |a| format!("{a:.1}% anomaly"));
        println!("  \x1b[1m{short}\x1b[0m  ({anomaly})");
        match entry
            .get("dimensions")
            .and_then(serde_json::Value::as_object)
        {
            Some(dims) if !dims.is_empty() => {
                for (dim, val) in dims {
                    let v = val.as_f64().unwrap_or(0.0);
                    println!("      {dim:30} {v}");
                }
            }
            _ => println!("      (no current samples)"),
        }
    }
}

/// Parse a relative time window (e.g. "10m", "1h", "24h", "7d", "30s") into a
/// negative seconds offset for Netdata's relative `after` param. Falls back to
/// 10 minutes on an unrecognized value.
fn parse_relative_secs(s: &str) -> i64 {
    let s = s.trim();
    let neg = |body: &str, mult: i64| body.parse::<i64>().ok().map(|n| -(n.saturating_mul(mult)));
    let parsed = if let Some(b) = s.strip_suffix('s') {
        neg(b, 1)
    } else if let Some(b) = s.strip_suffix('m') {
        neg(b, 60)
    } else if let Some(b) = s.strip_suffix('h') {
        neg(b, 3_600)
    } else if let Some(b) = s.strip_suffix('d') {
        neg(b, 86_400)
    } else {
        None
    };
    parsed.unwrap_or(-600)
}

fn cmd_replay(trace_id: &str, trace_dir: &str, from_step: Option<u32>) {
    let engine = lumen_core::replay::ReplayEngine::new(trace_dir);
    match engine.replay(trace_id) {
        Ok(trace) => {
            let start = from_step.unwrap_or(1);
            println!(
                "\x1b[36m🔄 Replaying trace {} ({} iterations, ${:.2} original cost)\x1b[0m",
                trace.trace_id, trace.total_iterations, trace.original_cost_usd
            );
            if let Some(ref r) = trace.recovery {
                match r.resumed_at_iteration {
                    Some(n) => println!(
                        "\x1b[35m⟳ resumed from {} at iteration {n}\x1b[0m",
                        r.parent_trace_id
                    ),
                    None => println!("\x1b[35m⟳ resumed from {}\x1b[0m", r.parent_trace_id),
                }
            }
            println!();
            for step in &trace.steps {
                if step.step < start {
                    continue;
                }
                print_replay_step(step);
            }
            if trace.success {
                println!("\n\x1b[32m✅ Run completed successfully\x1b[0m");
            } else {
                println!("\n\x1b[31m❌ Run failed\x1b[0m");
            }
        }
        Err(e) => {
            eprintln!("\x1b[31mError: {e}\x1b[0m");
        }
    }
}

fn print_replay_step(step: &lumen_core::replay::ReplayStep) {
    match &step.decision {
        lumen_core::replay::ReplayDecision::ToolCalls { calls } => {
            for call in calls {
                let status = if call.success { "✓" } else { "✗" };
                println!(
                    "  Step {}: {status} {}({}) → {}",
                    step.step, call.tool_name, call.call_id, call.output_preview
                );
            }
        }
        lumen_core::replay::ReplayDecision::FinalAnswer { content } => {
            let preview = if content.len() > 100 {
                format!("{}...", &content[..100])
            } else {
                content.clone()
            };
            println!("  Step {}: 💡 {preview}", step.step);
        }
    }
}

fn cmd_cost(last: &str, trace_dir: &str, format: &str) {
    let tracker = lumen_core::cost::CostTracker::new(trace_dir);
    let since_ms = parse_duration(last);

    match tracker.report(since_ms) {
        Ok(report) => {
            if format == "json" {
                if let Ok(json) = serde_json::to_string_pretty(&report) {
                    println!("{json}");
                }
            } else {
                print_cost_report(&report, last);
            }
        }
        Err(e) => {
            eprintln!("\x1b[31mError: {e}\x1b[0m");
        }
    }
}

fn print_cost_report(report: &lumen_core::cost::CostReport, window: &str) {
    println!("\x1b[36m📊 Cost Report (last {window})\x1b[0m");
    println!(
        "  Total: \x1b[1m${:.2}\x1b[0m across {} runs",
        report.total_usd, report.total_runs
    );
    println!(
        "  Tokens: {} prompt + {} completion\n",
        report.total_prompt_tokens, report.total_completion_tokens
    );

    if !report.by_agent.is_empty() {
        println!("  By Agent:");
        let mut agents: Vec<_> = report.by_agent.iter().collect();
        agents.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (agent, cost) in &agents {
            let pct = if report.total_usd > 0.0 {
                *cost / report.total_usd * 100.0
            } else {
                0.0
            };
            println!("    {agent:24} ${cost:.2} ({pct:.0}%)");
        }
    }

    if !report.by_model.is_empty() {
        println!("\n  By Model:");
        for (model, cost) in &report.by_model {
            println!("    {model:24} ${cost:.2}");
        }
    }

    if !report.anomalies.is_empty() {
        println!("\n  \x1b[33m⚠️  Anomalies:\x1b[0m");
        for a in &report.anomalies {
            println!(
                "    {} ({}): ${:.2} ({:.1}x avg) — {}",
                a.trace_id, a.agent_name, a.cost_usd, a.multiplier, a.reason
            );
        }
    }
}

fn cmd_traces(trace_dir: &str, limit: usize) {
    let store = lumen_core::trace::TraceStore::new(trace_dir);
    match store.list() {
        Ok(traces) => {
            if traces.is_empty() {
                println!("No traces found. Run an agent first.");
                return;
            }
            println!(
                "\x1b[36m📋 Agent Traces (showing {})\x1b[0m\n",
                traces.len().min(limit)
            );
            for trace in traces.iter().take(limit) {
                let status = if trace.success { "✅" } else { "❌" };
                println!(
                    "  {status} {} | {} | {} iters | ${:.2} | {:.1}s",
                    trace.trace_id,
                    trace.agent_name,
                    trace.iterations,
                    trace.cost_usd,
                    trace.duration_ms as f64 / 1000.0
                );
            }
        }
        Err(e) => {
            eprintln!("\x1b[31mError: {e}\x1b[0m");
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn parse_duration(s: &str) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let s = s.trim();
    if let Some(hours) = s.strip_suffix('h')
        && let Ok(h) = hours.parse::<u64>()
    {
        return now.saturating_sub(h * 3_600_000);
    }
    if let Some(days) = s.strip_suffix('d')
        && let Ok(d) = days.parse::<u64>()
    {
        return now.saturating_sub(d * 86_400_000);
    }
    // Default: 24 hours ago
    now.saturating_sub(86_400_000)
}
