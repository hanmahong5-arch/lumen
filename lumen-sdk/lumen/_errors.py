"""Unified error hierarchy and warning infrastructure for Lumen.

Design contract:
  - Lumen NEVER crashes the user's application. All internal errors
    are caught and reported via warnings or logging.
  - Exceptions are raised ONLY for explicit contract violations that
    the user needs to handle (e.g., BudgetExceededError).
  - All warnings include: what happened, why it matters, what to do.

Warning categories:
  - LumenWarning: Base for all Lumen warnings (filterable with warnings.filterwarnings)
  - TraceWriteWarning: Trace data could not be persisted
  - ConfigWarning: Configuration is invalid or partially loaded
  - IntegrationWarning: SDK integration issue (patching failure, missing dependency)
  - RedactionWarning: Redaction pattern is invalid (data may leak)
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger("lumen")


# ── Warning categories ───────────────────────────────────────────────────
# Users can filter these with: warnings.filterwarnings("ignore", category=LumenWarning)

class LumenWarning(UserWarning):
    """Base warning for all Lumen issues. Filter with warnings.filterwarnings."""


class TraceWriteWarning(LumenWarning):
    """Trace data could not be written to disk — data may be lost."""


class ConfigWarning(LumenWarning):
    """Configuration is invalid or could not be fully loaded."""


class IntegrationWarning(LumenWarning):
    """SDK integration issue — patching or dependency problem."""


class RedactionWarning(LumenWarning):
    """Redaction pattern is invalid — sensitive data may not be filtered."""


# ── Helper functions ─────────────────────────────────────────────────────

def warn_trace_write(message: str, *, trace_id: str = "") -> None:
    """Emit a trace write warning. Logged + warned."""
    prefix = f"[trace:{trace_id}] " if trace_id else ""
    full = f"{prefix}{message}"
    logger.warning(full)
    warnings.warn(full, TraceWriteWarning, stacklevel=3)


def warn_config(message: str) -> None:
    """Emit a configuration warning."""
    logger.warning(message)
    warnings.warn(message, ConfigWarning, stacklevel=3)


def warn_integration(message: str) -> None:
    """Emit an integration warning."""
    logger.warning(message)
    warnings.warn(message, IntegrationWarning, stacklevel=3)


def warn_redaction(message: str) -> None:
    """Emit a redaction warning."""
    logger.warning(message)
    warnings.warn(message, RedactionWarning, stacklevel=3)
