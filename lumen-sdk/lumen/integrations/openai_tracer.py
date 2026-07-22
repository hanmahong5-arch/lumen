"""OpenAI SDK integration — automatic tracing for chat completions.

Patches ``openai.chat.completions.create()`` (sync and async) to record
every LLM call as a Lumen trace step. Works with OpenAI SDK v1.0+.

Usage::

    from lumen import instrument
    instrument("openai")

    # All subsequent calls are traced automatically
    client = openai.OpenAI()
    response = client.chat.completions.create(model="gpt-4o", messages=[...])

Install from the ``lumen-sdk`` source directory: ``pip install -e ".[openai]"``
(the package is not published on PyPI).
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
    import openai
    from openai.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

_patched = False


def patch_openai(config: LumenConfig | None = None) -> bool:
    """Monkey-patch OpenAI SDK to enable automatic tracing.

    Safe to call multiple times — only patches once.

    Args:
        config: Lumen config. If None, uses defaults.

    Returns:
        True if patched successfully, False if openai not installed.
    """
    global _patched  # noqa: PLW0603
    if not _HAS_OPENAI:
        return False
    if _patched:
        return True

    _patch_sync(config)
    _patch_async(config)
    _patched = True
    return True


def unpatch_openai() -> None:
    """Remove Lumen patches from OpenAI SDK (for testing)."""
    global _patched  # noqa: PLW0603
    if not _HAS_OPENAI or not _patched:
        return
    if hasattr(Completions.create, "_lumen_original"):
        Completions.create = Completions.create._lumen_original  # type: ignore[attr-defined]
    if hasattr(AsyncCompletions.create, "_lumen_original"):
        AsyncCompletions.create = AsyncCompletions.create._lumen_original  # type: ignore[attr-defined]
    _patched = False


# ── Sync patch ────────────────────────────────────────────────────────────

def _patch_sync(config: LumenConfig | None) -> None:
    original = Completions.create

    @functools.wraps(original)
    def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Check if tracing is enabled and should sample
        if not should_sample(config):
            return original(self, *args, **kwargs)

        # Check if streaming
        is_stream = kwargs.get("stream", False)
        if is_stream:
            return _handle_sync_stream(original, self, config, args, kwargs)

        return _handle_sync_call(original, self, config, args, kwargs)

    traced_create._lumen_original = original  # type: ignore[attr-defined]
    Completions.create = traced_create  # type: ignore[assignment]


def _handle_sync_call(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Handle a non-streaming sync chat completion call."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        response = original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    duration_ms = _now_ms() - start_ms
    try:
        _record_completion(config, response, model, start_ms, duration_ms, kwargs)
    except Exception:
        logger.debug("Lumen: failed to record OpenAI completion", exc_info=True)
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
    """Wraps an OpenAI sync stream to capture trace data on completion."""

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
        self._finish_reason = "stop"
        self._has_tool_calls = False
        self._usage: Any = None
        self._finalized = False

    def __iter__(self) -> _SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._inspect_chunk(chunk)
            return chunk
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

    def _inspect_chunk(self, chunk: Any) -> None:
        if hasattr(chunk, "model") and chunk.model:
            self._model = chunk.model
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                self._finish_reason = choice.finish_reason
            if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                if choice.delta.tool_calls:
                    self._has_tool_calls = True
        if hasattr(chunk, "usage") and chunk.usage:
            self._usage = chunk.usage

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            duration_ms = _now_ms() - self._start_ms

            prompt_tokens = 0
            completion_tokens = 0
            if self._usage:
                prompt_tokens = getattr(self._usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(self._usage, "completion_tokens", 0) or 0

            finish_reason = self._finish_reason
            if self._has_tool_calls:
                finish_reason = "tool_use"

            _record_step(
                config=self._config,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                start_ms=self._start_ms,
                agent_name=self._kwargs.get("_lumen_agent", "openai"),
            )
        except Exception:
            logger.debug("Lumen: failed to finalize OpenAI sync stream trace", exc_info=True)


# ── Async patch ───────────────────────────────────────────────────────────

def _patch_async(config: LumenConfig | None) -> None:
    original = AsyncCompletions.create

    @functools.wraps(original)
    async def traced_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not should_sample(config):
            return await original(self, *args, **kwargs)

        is_stream = kwargs.get("stream", False)
        if is_stream:
            return await _handle_async_stream(original, self, config, args, kwargs)

        return await _handle_async_call(original, self, config, args, kwargs)

    traced_acreate._lumen_original = original  # type: ignore[attr-defined]
    AsyncCompletions.create = traced_acreate  # type: ignore[assignment]


async def _handle_async_call(
    original: Any,
    self_obj: Any,
    config: LumenConfig | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Handle a non-streaming async chat completion call."""
    model = kwargs.get("model", "unknown")
    start_ms = _now_ms()

    try:
        response = await original(self_obj, *args, **kwargs)
    except Exception as exc:
        _record_error(config, model, start_ms, str(exc), kwargs)
        raise

    duration_ms = _now_ms() - start_ms
    try:
        _record_completion(config, response, model, start_ms, duration_ms, kwargs)
    except Exception:
        logger.debug("Lumen: failed to record async OpenAI completion", exc_info=True)
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
    """Wraps an OpenAI async stream to capture trace data on completion."""

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
        self._finish_reason = "stop"
        self._has_tool_calls = False
        self._usage: Any = None
        self._finalized = False

    def __aiter__(self) -> _AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._inspect_chunk(chunk)
            return chunk
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

    def _inspect_chunk(self, chunk: Any) -> None:
        if hasattr(chunk, "model") and chunk.model:
            self._model = chunk.model
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                self._finish_reason = choice.finish_reason
            if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                if choice.delta.tool_calls:
                    self._has_tool_calls = True
        if hasattr(chunk, "usage") and chunk.usage:
            self._usage = chunk.usage

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            duration_ms = _now_ms() - self._start_ms

            prompt_tokens = 0
            completion_tokens = 0
            if self._usage:
                prompt_tokens = getattr(self._usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(self._usage, "completion_tokens", 0) or 0

            finish_reason = self._finish_reason
            if self._has_tool_calls:
                finish_reason = "tool_use"

            _record_step(
                config=self._config,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                start_ms=self._start_ms,
                agent_name=self._kwargs.get("_lumen_agent", "openai"),
            )
        except Exception:
            logger.debug("Lumen: failed to finalize OpenAI async stream trace", exc_info=True)


# ── Shared recording logic ────────────────────────────────────────────────

def _record_completion(
    config: LumenConfig | None,
    response: Any,
    model: str,
    start_ms: int,
    duration_ms: int,
    kwargs: dict[str, Any],
) -> None:
    """Extract data from a ChatCompletion response and record a trace step."""
    # Extract response metadata
    actual_model = getattr(response, "model", model) or model
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    choices = getattr(response, "choices", [])
    output_content: str | None = None
    if choices:
        choice = choices[0]
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr
        # Detect tool calls
        msg = getattr(choice, "message", None)
        if msg and getattr(msg, "tool_calls", None):
            finish_reason = "tool_use"
        # Capture output content
        if msg and hasattr(msg, "content") and msg.content:
            output_content = str(msg.content)

    # Capture input messages if content_capture is "full"
    input_messages: list[dict[str, Any]] | None = None
    if config and config.tracing.content_capture == "full":
        raw_msgs = kwargs.get("messages")
        if raw_msgs and isinstance(raw_msgs, list):
            input_messages = raw_msgs

    _record_step(
        config=config,
        model=actual_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        duration_ms=duration_ms,
        start_ms=start_ms,
        agent_name=kwargs.get("_lumen_agent", "openai"),
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

    # Standalone trace for the error
    trace_dir = _get_trace_dir(config)
    sess = TraceSession(
        name=kwargs.get("_lumen_agent", "openai"),
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
    """Record an LLM step — either into active session or as standalone trace."""
    cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

    session = get_active_session()
    if session is not None:
        # Add step to existing session
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

    # No active session — create standalone trace
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
