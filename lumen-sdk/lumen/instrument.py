"""One-line instrumentation for AI agent observability.

Auto-detects installed LLM SDKs and patches them to emit Lumen traces.

Usage::

    from lumen import instrument

    instrument()  # auto-detect all installed SDKs

    # Or specify frameworks explicitly:
    instrument("openai", "anthropic")

    # With config:
    from lumen import LumenConfig
    config = LumenConfig.auto()
    instrument(config=config)

    # Group calls into a single trace:
    from lumen import trace
    with trace("research-task"):
        response1 = client.chat.completions.create(...)
        response2 = client.chat.completions.create(...)

    # Decorator form:
    from lumen import traced_fn

    @traced_fn("research-task")
    def do_research():
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Generator, TypeVar

from lumen._budget import BudgetTracker
from lumen._context import TraceSession, get_active_session, set_active_session, should_sample

if TYPE_CHECKING:
    from lumen.config import LumenConfig

logger = logging.getLogger("lumen")

_config: LumenConfig | None = None
_instrumented = False
_budget_tracker: BudgetTracker | None = None
_writer: Any = None
_instrument_lock = threading.Lock()

F = TypeVar("F", bound=Callable[..., Any])


def instrument(
    *frameworks: str,
    config: LumenConfig | None = None,
) -> dict[str, bool]:
    """Enable automatic tracing for LLM SDK calls.

    When called with no arguments, auto-detects all installed SDKs.
    When called with framework names, patches only those.

    Args:
        *frameworks: Framework names to patch ("openai", "anthropic",
                     "langgraph"). Empty = auto-detect all.
        config: Lumen configuration. If None, auto-discovers from
                lumen.toml / env vars / defaults.

    Returns:
        Dict mapping framework name → whether it was patched successfully.

    Example::

        instrument()                            # auto-detect
        instrument("openai")                    # only openai
        instrument("openai", "anthropic")       # both
        instrument(config=LumenConfig.auto())   # with config
    """
    global _config, _instrumented, _budget_tracker, _writer  # noqa: PLW0603

    with _instrument_lock:
        if config is None:
            from lumen.config import LumenConfig as _LC
            config = _LC.auto()
        _config = config

        # Initialize budget tracker from config
        _budget_tracker = BudgetTracker.from_config(config)

        # Register custom pricing if configured
        if config.cost.custom_pricing:
            from lumen.pricing import register_custom_pricing
            register_custom_pricing(config.cost.custom_pricing)

        # Initialize trace writer pipeline (wires up buffer_size + flush_interval_ms)
        _writer = _create_writer(config)

        # Initialize runtime metrics (wires up MonitoringConfig)
        _init_metrics(config)

    if not config.tracing.enabled:
        logger.info("Lumen tracing is disabled by configuration")
        return {}

    # Determine which frameworks to patch
    targets = set(frameworks) if frameworks else {"openai", "anthropic", "langgraph"}

    results: dict[str, bool] = {}

    if "openai" in targets:
        try:
            from lumen.integrations.openai_tracer import patch_openai
            results["openai"] = patch_openai(config)
            if results["openai"]:
                logger.info("Lumen: OpenAI SDK patched for tracing")
        except Exception as exc:
            results["openai"] = False
            logger.warning("Lumen: Failed to patch OpenAI SDK: %s", exc)

    if "anthropic" in targets:
        try:
            from lumen.integrations.anthropic_tracer import patch_anthropic
            results["anthropic"] = patch_anthropic(config)
            if results["anthropic"]:
                logger.info("Lumen: Anthropic SDK patched for tracing")
        except Exception as exc:
            results["anthropic"] = False
            logger.warning("Lumen: Failed to patch Anthropic SDK: %s", exc)

    if "langgraph" in targets:
        try:
            from lumen.integrations.langgraph_tracer import LumenTracer  # noqa: F401
            results["langgraph"] = True
            logger.info("Lumen: LangGraph integration available (use LumenTracer callback)")
        except ImportError:
            results["langgraph"] = False

    _instrumented = True
    patched = [k for k, v in results.items() if v]
    if patched:
        logger.info("Lumen instrumented: %s", ", ".join(patched))

    return results


def uninstrument() -> None:
    """Remove all Lumen patches (for testing)."""
    global _config, _instrumented, _budget_tracker, _writer  # noqa: PLW0603

    try:
        from lumen.integrations.openai_tracer import unpatch_openai
        unpatch_openai()
    except Exception:
        pass
    try:
        from lumen.integrations.anthropic_tracer import unpatch_anthropic
        unpatch_anthropic()
    except Exception:
        pass

    # Flush and shutdown writer
    if _writer is not None:
        try:
            from lumen._batch_writer import BatchTraceWriter
            if isinstance(_writer, BatchTraceWriter):
                _writer.shutdown()
        except Exception:
            pass
    _writer = None

    # Clear runtime metrics
    from lumen._context import set_runtime_metrics
    set_runtime_metrics(None)

    _config = None
    _instrumented = False
    _budget_tracker = None


@contextlib.contextmanager
def trace(
    name: str = "agent",
    *,
    user_id: str = "",
    session_id: str = "",
    config: LumenConfig | None = None,
) -> Generator[TraceSession, None, None]:
    """Context manager to group multiple LLM calls into a single trace.

    All calls within the block are recorded as steps in one trace,
    instead of creating separate traces per call.

    Args:
        name: Agent or task name for this trace.
        user_id: Optional user identifier for tracking.
        session_id: Optional session identifier for multi-turn grouping.
        config: Lumen config override.

    Yields:
        TraceSession that steps are recorded into.

    Example::

        from lumen import trace

        with trace("research-agent", user_id="u-123") as t:
            resp1 = client.chat.completions.create(model="gpt-4o", ...)
            resp2 = client.chat.completions.create(model="gpt-4o", ...)
            print(f"Trace: {t.trace_id}, cost: ${t.total_cost_usd:.4f}")
    """
    effective_config = config or _config

    # Check sampling
    if not should_sample(effective_config):
        # Yield a dummy session that won't be written
        dummy = TraceSession(name=name, trace_dir="/dev/null", config=effective_config)
        yield dummy
        return

    trace_dir = "./traces"
    if effective_config is not None:
        trace_dir = effective_config.tracing.trace_dir

    # Nested trace support — record parent trace_id
    previous = get_active_session()
    parent_trace_id = previous.trace_id if previous is not None else ""

    session = TraceSession(
        name=name,
        trace_dir=trace_dir,
        config=effective_config,
        user_id=user_id,
        session_id=session_id,
        budget_tracker=_budget_tracker,
        writer=_writer,
        parent_trace_id=parent_trace_id,
    )

    set_active_session(session)

    try:
        yield session
    except Exception as exc:
        session.add_error_step(str(exc))
        raise
    finally:
        # Finalize and restore previous session
        session.finalize()
        set_active_session(previous)


@contextlib.asynccontextmanager
async def atrace(
    name: str = "agent",
    *,
    user_id: str = "",
    session_id: str = "",
    config: LumenConfig | None = None,
) -> AsyncGenerator[TraceSession, None]:
    """Async context manager to group multiple LLM calls into a single trace.

    Works identically to ``trace()`` but for async code.

    Example::

        from lumen import atrace

        async with atrace("research-agent") as t:
            resp = await aclient.chat.completions.create(model="gpt-4o", ...)
    """
    effective_config = config or _config

    if not should_sample(effective_config):
        dummy = TraceSession(name=name, trace_dir="/dev/null", config=effective_config)
        yield dummy
        return

    trace_dir = "./traces"
    if effective_config is not None:
        trace_dir = effective_config.tracing.trace_dir

    # Nested trace support
    previous = get_active_session()
    parent_trace_id = previous.trace_id if previous is not None else ""

    session = TraceSession(
        name=name,
        trace_dir=trace_dir,
        config=effective_config,
        user_id=user_id,
        session_id=session_id,
        budget_tracker=_budget_tracker,
        writer=_writer,
        parent_trace_id=parent_trace_id,
    )

    set_active_session(session)

    try:
        yield session
    except Exception as exc:
        session.add_error_step(str(exc))
        raise
    finally:
        session.finalize()
        set_active_session(previous)


def traced_fn(
    name: str = "agent",
    *,
    user_id: str = "",
    session_id: str = "",
    config: LumenConfig | None = None,
) -> Callable[[F], F]:
    """Decorator to wrap a function in a Lumen trace.

    Supports both sync and async functions. The decorated function
    automatically records all LLM calls as steps in one trace.

    Example::

        @traced_fn("summarizer")
        def summarize(text):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Summarize: {text}"}],
            )

        @traced_fn("async-agent")
        async def agent_loop():
            ...
    """
    def decorator(fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with atrace(name, user_id=user_id, session_id=session_id, config=config):
                    return await fn(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with trace(name, user_id=user_id, session_id=session_id, config=config):
                    return fn(*args, **kwargs)
            return sync_wrapper  # type: ignore[return-value]
    return decorator


def get_config() -> LumenConfig | None:
    """Return the currently active Lumen config (set by instrument())."""
    return _config


def get_budget_tracker() -> BudgetTracker | None:
    """Return the currently active budget tracker (set by instrument())."""
    return _budget_tracker


def get_writer() -> Any:
    """Return the currently active trace writer (set by instrument())."""
    return _writer


def get_metrics() -> Any:
    """Return the runtime metrics collector, or None if not initialized.

    Example::

        from lumen.instrument import get_metrics
        m = get_metrics()
        if m:
            print(m.summary())
    """
    from lumen._context import get_runtime_metrics
    return get_runtime_metrics()


# ── Internal initialization ───────────────────────────────────────────────

def _create_writer(config: Any) -> Any:
    """Build the trace writer pipeline from config.

    If buffer_size > 1, wraps the file writer in a BatchTraceWriter.
    Otherwise returns a direct FileTraceWriter.
    """
    from lumen._writers import FileTraceWriter

    base_writer = FileTraceWriter(config.tracing.trace_dir)

    if config.tracing.buffer_size > 1:
        from lumen._batch_writer import BatchTraceWriter
        return BatchTraceWriter(
            base_writer,
            buffer_size=config.tracing.buffer_size,
            flush_interval_sec=config.tracing.flush_interval_ms / 1000.0,
        )

    return base_writer


def _init_metrics(config: Any) -> None:
    """Initialize the runtime metrics pipeline if monitoring is enabled."""
    from lumen._context import set_runtime_metrics
    from lumen._metrics import RuntimeMetrics

    metrics = RuntimeMetrics()
    set_runtime_metrics(metrics)
