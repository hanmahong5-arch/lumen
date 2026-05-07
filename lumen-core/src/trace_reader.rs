//! Shared trace file reader.
//!
//! Loads [`AgentTrace`] JSON files from a directory.
//! Used by replay, cost, and trace modules.

use std::path::Path;

use kova_types::trace::AgentTrace;

use crate::LumenError;

/// Load all agent traces from a directory containing `{trace_id}.json` files.
///
/// Returns traces sorted newest-first by `started_at_ms`.
/// Files that fail to parse are silently skipped.
/// If the directory does not exist, returns an empty vec.
pub(crate) fn load_traces(dir: &Path) -> Result<Vec<AgentTrace>, LumenError> {
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut traces = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Only read .json files, skip dotfiles (temp files from atomic writes)
        let dominated_by_dot = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        let is_json = path.extension().is_some_and(|ext| ext == "json");

        if !is_json || dominated_by_dot {
            continue;
        }

        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        match serde_json::from_slice::<AgentTrace>(&data) {
            Ok(trace) => traces.push(trace),
            Err(_) => {
                tracing::debug!("skipped non-trace file: {}", path.display());
            }
        }
    }

    // Sort newest first
    traces.sort_by(|a, b| b.started_at_ms.cmp(&a.started_at_ms));

    Ok(traces)
}
