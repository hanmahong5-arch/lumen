//! Lumen Core — AI agent observability and reliability engine.
//!
//! Wraps Kova's durable execution primitives into a developer-friendly
//! API focused on three capabilities:
//!
//! 1. **Replay** — Deterministic replay of any agent run from WAL
//! 2. **Cost** — Real-time LLM token/USD tracking and aggregation
//! 3. **Guard** — Loop detection and cost limits

pub mod causal_types;
pub mod cost;
pub mod error;
pub mod flow_types;
pub mod lifecycle;
pub mod lifecycle_builder;
pub mod pricing;
pub mod replay;
pub mod trace;
pub mod trace_types;

mod trace_reader;

pub use cost::{COST_ANOMALY_MIN_USD, COST_ANOMALY_MULTIPLIER};
pub use error::LumenError;
