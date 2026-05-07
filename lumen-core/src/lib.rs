//! Lumen Core — AI agent observability and reliability engine.
//!
//! Wraps Kova's durable execution primitives into a developer-friendly
//! API focused on three capabilities:
//!
//! 1. **Replay** — Deterministic replay of any agent run from WAL
//! 2. **Cost** — Real-time LLM token/USD tracking and aggregation
//! 3. **Guard** — Loop detection and cost limits

pub mod cost;
pub mod error;
pub mod pricing;
pub mod replay;
pub mod trace;

mod trace_reader;

pub use error::LumenError;
