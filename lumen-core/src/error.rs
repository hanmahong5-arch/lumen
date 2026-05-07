//! Lumen error types.

/// Lumen error.
#[derive(Debug, thiserror::Error)]
pub enum LumenError {
    /// WAL read/write error.
    #[error("WAL error: {0}")]
    Wal(String),

    /// Trace not found.
    #[error("trace not found: {0}")]
    TraceNotFound(String),

    /// Agent execution error.
    #[error("agent error: {0}")]
    Agent(String),

    /// Cost limit exceeded.
    #[error("cost limit exceeded: spent ${spent:.2}, limit ${limit:.2}")]
    CostLimitExceeded {
        /// Amount spent.
        spent: f64,
        /// Budget limit.
        limit: f64,
    },

    /// JSON parse/serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
