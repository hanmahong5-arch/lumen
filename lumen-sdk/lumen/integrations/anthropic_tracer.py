"""Anthropic SDK integration — automatic tracing for messages API.

Patches ``anthropic.messages.create()`` (sync and async) to record
every LLM call as a Lumen trace step. Works with Anthropic SDK v0.40+.

Usage::

    from lumen import instrument
    instrument("anthropic")

    client = anthropic.Anthropic()
    message = client.messages.create(model="claude-sonnet-4-6", ...)

Install with: ``pip install "lumen-ai[anthropic]"``
"""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any

from lumen._context import (
    TraceSession,
    get_active_session,
    should_sample,
)
from lumen.pricing import estimate_cost

logger = logging.getLogger("lumen")

if TYPE_CHECKING:
    from lumen.config import LumenConfig

try:
    from anthropic.resources.messages import (
        AsyncMessages,
        Messages,
    )

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

_patched = False


def patch_anthropic(config: LumenConfig | None = None) -> bool:
    """Monkey-patch Anthropic SDK to enable automatic tracing.

    Safe to call multiple times — only patches once.

    Args:
        config: Lumen config. If None, uses defaults.

    Returns:
        True if patched successfully, False if anthropic not installed.
    """
    global _patched  # noqa: PLW0603
    if not _HAS_ANTHROPIC:
        return False
    if _patched:
        return True

    _patch_sync(config)
    _patch_async(config)
    _patched = True
    return True


def unpatch_anthropic() -> None:
    """Remove Lumen patches from Anthropic SDK (for testing)."""
    global _patched  # noqa: PLW0603
    if not _HAS_ANTHROPIC or not _patched:
        return
    if hasattr(Messages.create, "_lumen_original"):
        Messages.create = Messages.create._lumen_original  # type: ignore[attr-defined]
    if hasattr(AsyncMessages.create, "_lumen_original"):
        AsyncMessages.create = AsyncMessages.create._lumen_original  # type: ignore[attr-defined]
    _patched = False


# ── Sync patch ────────────────────────────────────────────────────────────

def _patch_sync(config: LumenConfig | None) -> None:
    original = Messages.create

    @functools.wraps(original)
    def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_sample(config):
            return original(self, *args, **kwargs)

        is_stream = kwargs.get("stream", False)
        if is_stream:
            return _handle_sync_stream(original, self, config, args, kwargs)

        return _handle_sync_call(original, self, config, args, kwargs)

    traced_create._lumen_original = original  # type: ignore[attr-defined]
    Messages.create = traced_create  # type: ignore[assignment]


def _handle_sync_call(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Handle a non-streaming sync messages.create call."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        response = original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    duration_ms = _now_ms() - start_ms
    try:
        _record_message(config, response, model, start_ms, duration_ms, kwargs)
    except Exception:
        logger.debug("Lumen: failed to record Anthropic message", exc_info=True)
    return response


def _handle_sync_stream(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Wrap a streaming response to capture trace data."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        stream = original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    return _SyncStreamWrapper(stream, config, model, start_ms, kwargs)


class _SyncStreamWrapper:
    """Wraps an Anthropic sync stream to capture trace data."""

    def __init__(
        self,
        stream: Any,
        config: LumenConfig | None,
        model: str,
        start_ms: int,
        kwargs: dict[str, Any],
    ) -> None:
        self._stream = stream
        self._config = config
        self._model = model
        self._start_ms = start_ms
        self._kwargs = kwargs
        self._stop_reason = "end_turn"
        self._has_tool_use = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._finalized = False

    def __iter__(self) -> _SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._stream)
            self._inspect_event(event)
            return event
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self) -> _SyncStreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)
        self._finalize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _inspect_event(self, event: Any) -> None:
        # message_start has model and usage
        etype = getattr(event, "type", "")
        if etype == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                self._model = getattr(msg, "model", self._model) or self._model
                usage = getattr(msg, "usage", None)
                if usage:
                    self._input_tokens = getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                sr = getattr(delta, "stop_reason", None)
                if sr:
                    self._stop_reason = sr
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = getattr(usage, "output_tokens", 0) or 0
        elif etype == "content_block_start":
            cb = getattr(event, "content_block", None)
            if cb and getattr(cb, "type", "") == "tool_use":
                self._has_tool_use = True

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            duration_ms = _now_ms() - self._start_ms

            finish_reason = _map_stop_reason(self._stop_reason, self._has_tool_use)
            _record_step(
                config=self._config,
                model=self._model,
                prompt_tokens=self._input_tokens,
                completion_tokens=self._output_tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                start_ms=self._start_ms,
                agent_name=self._kwargs.get("_lumen_agent", "anthropic"),
            )
        except Exception:
            logger.debug("Lumen: failed to finalize Anthropic sync stream trace", exc_info=True)


# ── Async patch ───────────────────────────────────────────────────────────

def _patch_async(config: LumenConfig | None) -> None:
    original = AsyncMessages.create

    @functools.wraps(original)
    async def traced_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_sample(config):
            return await original(self, *args, **kwargs)

        is_stream = kwargs.get("stream", False)
        if is_stream:
            return await _handle_async_stream(original, self, config, args, kwargs)

        return await _handle_async_call(original, self, config, args, kwargs)

    traced_acreate._lumen_original = original  # type: ignore[attr-defined]
    AsyncMessages.create = traced_acreate  # type: ignore[assignment]


async def _handle_async_call(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Handle a non-streaming async messages.create call."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        response = await original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    duration_ms = _now_ms() - start_ms
    try:
        _record_message(config, response, model, start_ms, duration_ms, kwargs)
    except Exception:
        logger.debug("Lumen: failed to record async Anthropic message", exc_info=True)
    return response


async def _handle_async_stream(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Wrap an async streaming response to capture trace data."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        stream = await original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    return _AsyncStreamWrapper(stream, config, model, start_ms, kwargs)


class _AsyncStreamWrapper:
    """Wraps an Anthropic async stream to capture trace data."""

    def __init__(
        self,
        stream: Any,
        config: LumenConfig | None,
        model: str,
        start_ms: int,
        kwargs: dict[str, Any],
    ) -> None:
        self._stream = stream
        self._config = config
        self._model = model
        self._start_ms = start_ms
        self._kwargs = kwargs
        self._stop_reason = "end_turn"
        self._has_tool_use = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._finalized = False

    def __aiter__(self) -> _AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._stream.__anext__()
            self._inspect_event(event)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise

    async def __aenter__(self) -> _AsyncStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
        self._finalize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _inspect_event(self, event: Any) -> None:
        etype = getattr(event, "type", "")
        if etype == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                self._model = getattr(msg, "model", self._model) or self._model
                usage = getattr(msg, "usage", None)
                if usage:
                    self._input_tokens = getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            delta = getattr(event, "delta", None)
            if delta:
                sr = getattr(delta, "stop_reason", None)
                if sr:
                    self._stop_reason = sr
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = getattr(usage, "output_tokens", 0) or 0
        elif etype == "content_block_start":
            cb = getattr(event, "content_block", None)
            if cb and getattr(cb, "type", "") == "tool_use":
                self._has_tool_use = True

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            duration_ms = _now_ms() - self._start_ms

            finish_reason = _map_stop_reason(self._stop_reason, self._has_tool_use)
            _record_step(
                config=self._config,
                model=self._model,
                prompt_tokens=self._input_tokens,
                completion_tokens=self._output_tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                start_ms=self._start_ms,
                agent_name=self._kwargs.get("_lumen_agent", "anthropic"),
            )
        except Exception:
            logger.debug("Lumen: failed to finalize Anthropic async stream trace", exc_info=True)


# ── Shared recording logic ────────────────────────────────────────────────

def _map_stop_reason(stop_reason: str, has_tool_use: bool) -> str:
    """Map Anthropic stop_reason to Lumen finish_reason."""
    if has_tool_use or stop_reason == "tool_use":
        return "tool_use"
    if stop_reason in ("end_turn", "stop_sequence"):
        return "stop"
    if stop_reason == "max_tokens":
        return "length"
    return stop_reason


def _record_message(
    config: LumenConfig | None,
    response: Any,
    model: str,
    start_ms: int,
    duration_ms: int,
    kwargs: dict[str, Any],
) -> None:
    """Extract data from an Anthropic Message and record a trace step."""
    actual_model = getattr(response, "model", model) or model
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    has_tool_use = False
    output_content: str | None = None

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

    sr = getattr(response, "stop_reason", None)
    if sr:
        stop_reason = sr

    # Check for tool_use in content blocks and extract text content
    content = getattr(response, "content", [])
    text_parts: list[str] = []
    if content:
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use":
                has_tool_use = True
            elif block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    text_parts.append(text)

    if text_parts:
        output_content = "\n".join(text_parts)

    finish_reason = _map_stop_reason(stop_reason, has_tool_use)

    # Capture input messages if content_capture is "full"
    input_messages: list[dict[str, Any]] | None = None
    if config and config.tracing.content_capture == "full":
        raw_msgs = kwargs.get("messages")
        if raw_msgs and isinstance(raw_msgs, list):
            input_messages = raw_msgs

    _record_step(
        config=config,
        model=actual_model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        finish_reason=finish_reason,
        duration_ms=duration_ms,
        start_ms=start_ms,
        agent_name=kwargs.get("_lumen_agent", "anthropic"),
        input_messages=input_messages,
        output_content=output_content if config and config.tracing.content_capture == "full" else None,
    )


def _record_error(
    config: LumenConfig | None,
    model: str,
    start_ms: int,
    error_msg: str,
    kwargs: dict[str, Any],
) -> None:
    """Record an error for a failed LLM call."""
    session = get_active_session()
    if session is not None:
        session.add_error_step(error_msg)
        return

    trace_dir = _get_trace_dir(config)
    sess = TraceSession(
        name=kwargs.get("_lumen_agent", "anthropic"),
        trace_dir=trace_dir,
        config=config,
    )
    sess.add_llm_step(
        model=model,
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason="error",
        duration_ms=_now_ms() - start_ms,
        started_at_ms=start_ms,
    )
    sess.add_error_step(error_msg)
    sess.finalize()


def _record_step(
    *,
    config: LumenConfig | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    duration_ms: int,
    start_ms: int,
    agent_name: str,
    input_messages: list[dict[str, Any]] | None = None,
    output_content: str | None = None,
) -> None:
    """Record an LLM step into active session or standalone trace."""
    cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

    session = get_active_session()
    if session is not None:
        session.add_llm_step(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            duration_ms=duration_ms,
            started_at_ms=start_ms,
            cost_usd=cost_usd,
            input_messages=input_messages,
            output_content=output_content,
        )
        return

    trace_dir = _get_trace_dir(config)
    sess = TraceSession(name=agent_name, trace_dir=trace_dir, config=config)
    sess.add_llm_step(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        duration_ms=duration_ms,
        started_at_ms=start_ms,
        cost_usd=cost_usd,
        input_messages=input_messages,
        output_content=output_content,
    )
    sess.finalize()


def _get_trace_dir(config: LumenConfig | None) -> str:
    if config is not None:
        return config.tracing.trace_dir
    return "./traces"


def _now_ms() -> int:
    return int(time.time() * 1000)
