//! Lumen CLI — illuminate your AI agents.

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
    /// Open the web dashboard.
    Dashboard {
        /// Port to serve on.
        #[arg(long, default_value = "9700")]
        port: u16,
    },
    /// Pull traces from a running Kova into a local trace directory.
    ///
    /// Lists traces via `GET /api/v1/traces`, fetches each via
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
        Commands::Dashboard { port } => cmd_dashboard(port),
        Commands::Pull {
            kova_url,
            api_key,
            trace_dir,
        } => return cmd_pull(&kova_url, api_key, &trace_dir),
    }
    std::process::ExitCode::SUCCESS
}

/// Resolve the API key: explicit `--api-key` wins, then `LUMEN_KOVA_API_KEY`,
/// then `KOVA_API_KEY`. Returns `None` if none set (unauthenticated request).
fn resolve_api_key(explicit: Option<String>) -> Option<String> {
    explicit
        .or_else(|| std::env::var("LUMEN_KOVA_API_KEY").ok())
        .or_else(|| std::env::var("KOVA_API_KEY").ok())
        .filter(|k| !k.is_empty())
}

fn cmd_pull(kova_url: &str, api_key: Option<String>, trace_dir: &str) -> std::process::ExitCode {
    let key = resolve_api_key(api_key);
    let fetcher = pull::HttpFetcher::new(kova_url, key);
    let dir = std::path::Path::new(trace_dir);

    println!("\x1b[36m⇣ Pulling traces from {kova_url}\x1b[0m");
    match pull::pull_into(&fetcher, dir) {
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

fn cmd_replay(trace_id: &str, trace_dir: &str, from_step: Option<u32>) {
    let engine = lumen_core::replay::ReplayEngine::new(trace_dir);
    match engine.replay(trace_id) {
        Ok(trace) => {
            let start = from_step.unwrap_or(1);
            println!(
                "\x1b[36m🔄 Replaying trace {} ({} iterations, ${:.2} original cost)\x1b[0m\n",
                trace.trace_id, trace.total_iterations, trace.original_cost_usd
            );
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

fn cmd_dashboard(port: u16) {
    println!("\x1b[36m🌐 Lumen Dashboard\x1b[0m");
    println!("  http://localhost:{port}");
    println!("  (coming soon)");
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
