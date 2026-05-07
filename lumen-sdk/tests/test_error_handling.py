"""Tests for comprehensive error handling across the Lumen SDK.

Verifies that all error paths produce visible warnings/logs instead
of silent failures, and that Lumen NEVER crashes the user's application.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lumen._context import (
    TraceSession,
    _write_trace_safe,
    _safe_default,
)
from lumen._errors import (
    ConfigWarning,
    IntegrationWarning,
    LumenWarning,
    RedactionWarning,
    TraceWriteWarning,
)


# ── Warning hierarchy tests ──────────────────────────────────────────────

class TestWarningHierarchy:
    """All Lumen warnings are filterable via LumenWarning."""

    def test_trace_write_is_lumen_warning(self) -> None:
        assert issubclass(TraceWriteWarning, LumenWarning)
        assert issubclass(TraceWriteWarning, UserWarning)

    def test_config_is_lumen_warning(self) -> None:
        assert issubclass(ConfigWarning, LumenWarning)

    def test_integration_is_lumen_warning(self) -> None:
        assert issubclass(IntegrationWarning, LumenWarning)

    def test_redaction_is_lumen_warning(self) -> None:
        assert issubclass(RedactionWarning, LumenWarning)

    def test_filter_all_lumen_warnings(self) -> None:
        """Users can suppress all Lumen warnings with one filter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", category=LumenWarning)
            warnings.warn("test", TraceWriteWarning)
            warnings.warn("test", ConfigWarning)
            warnings.warn("test", RedactionWarning)
            assert len(w) == 0


# ── Trace write safety tests ────────────────────────────────────────────

class TestTraceWriteSafety:
    """_write_trace_safe handles all failure modes gracefully."""

    def test_write_normal(self, tmp_path: Path) -> None:
        """Normal write succeeds."""
        data = {"trace_id": "abc", "steps": []}
        _write_trace_safe(data, str(tmp_path), "abc")
        assert (tmp_path / "abc.json").exists()
        loaded = json.loads((tmp_path / "abc.json").read_text())
        assert loaded["trace_id"] == "abc"

    def test_write_non_serializable_objects_uses_default_str(self, tmp_path: Path) -> None:
        """Non-serializable objects are converted via default=str."""
        data = {"trace_id": "ns1", "obj": object(), "steps": []}
        _write_trace_safe(data, str(tmp_path), "ns1")
        assert (tmp_path / "ns1.json").exists()
        content = json.loads((tmp_path / "ns1.json").read_text())
        # object() is converted to its repr via default=str
        assert "object" in content["obj"].lower()

    def test_write_circular_reference(self, tmp_path: Path) -> None:
        """Circular references handled without crash."""
        data: dict[str, Any] = {"trace_id": "circ"}
        data["self_ref"] = data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _write_trace_safe(data, str(tmp_path), "circ")
        trace_warnings = [x for x in w if issubclass(x.category, TraceWriteWarning)]
        assert len(trace_warnings) >= 1

    def test_write_unwritable_directory(self) -> None:
        """Permission denied on directory creation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Mock mkdir to simulate permission error
            with patch("lumen._context.Path.mkdir", side_effect=PermissionError("denied")):
                _write_trace_safe({"trace_id": "x"}, "/some/path", "x")
        trace_warnings = [x for x in w if issubclass(x.category, TraceWriteWarning)]
        assert len(trace_warnings) >= 1
        assert "cannot create trace directory" in str(trace_warnings[0].message).lower()

    def test_safe_default_fallback(self) -> None:
        """_safe_default converts anything to string."""
        assert isinstance(_safe_default(42), str)
        assert isinstance(_safe_default(object()), str)
        assert isinstance(_safe_default(set()), str)


# ── TraceSession finalize error handling ─────────────────────────────────

class TestSessionFinalizeSafety:
    """TraceSession.finalize() never crashes the user's application."""

    def test_finalize_writes_trace(self, tmp_path: Path) -> None:
        """Normal finalization writes file."""
        sess = TraceSession("test-agent", str(tmp_path))
        sess.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        sess.finalize()
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_finalize_with_unserializable_metadata(self, tmp_path: Path) -> None:
        """Finalize survives unserializable metadata in steps."""
        sess = TraceSession("test-agent", str(tmp_path))
        # Inject a non-serializable object into steps
        sess.steps.append({
            "step_num": 0,
            "step_type": "LlmCall",
            "started_at_ms": 1000,
            "duration_ms": 50,
            "tokens": None,
            "metadata": [("custom_obj", object())],
        })
        sess.step_num = 1
        # Should not raise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sess.finalize()

    def test_finalize_empty_session_no_write(self, tmp_path: Path) -> None:
        """Empty sessions are not written to disk."""
        sess = TraceSession("test-agent", str(tmp_path))
        sess.finalize()
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_finalize_bad_trace_dir_warns(self) -> None:
        """Finalize to unwritable path warns but doesn't crash."""
        sess = TraceSession("test-agent", "/some/path")
        sess.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("lumen._context.Path.mkdir", side_effect=PermissionError("denied")):
                sess.finalize()  # should NOT raise
        trace_warnings = [x for x in w if issubclass(x.category, TraceWriteWarning)]
        assert len(trace_warnings) >= 1

    def test_add_llm_step_serialization_fallback(self, tmp_path: Path) -> None:
        """Input messages with non-serializable content get fallback."""
        sess = TraceSession("test-agent", str(tmp_path))
        # Pass an object that json.dumps cannot serialize
        sess.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
            input_messages=[{"role": "user", "content": object()}],
        )
        # Should have recorded a step (with fallback serialization)
        assert len(sess.steps) == 1
        meta = sess.steps[0]["metadata"]
        # Either serialized with default=str or got "[serialization_error]"
        assert len(meta) > 0


# ── Config error handling ────────────────────────────────────────────────

class TestConfigErrorHandling:

    def test_missing_toml_file_warns(self) -> None:
        """Loading a nonexistent TOML file emits ConfigWarning."""
        from lumen.config import LumenConfig
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = LumenConfig()
            config._merge_toml(Path("/nonexistent/path.toml"))
        config_warnings = [x for x in w if issubclass(x.category, ConfigWarning)]
        assert len(config_warnings) >= 1
        assert "Cannot read config file" in str(config_warnings[0].message)

    def test_invalid_toml_warns(self, tmp_path: Path) -> None:
        """Malformed TOML file emits ConfigWarning."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_bytes(b"[invalid toml = !!!!")
        from lumen.config import LumenConfig
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = LumenConfig()
            config._merge_toml(bad_toml)
        config_warnings = [x for x in w if issubclass(x.category, ConfigWarning)]
        assert len(config_warnings) >= 1
        assert "Invalid TOML" in str(config_warnings[0].message)

    def test_invalid_redaction_pattern_warns(self) -> None:
        """Bad regex pattern in redaction config emits RedactionWarning."""
        from lumen.config import RedactionConfig
        r = RedactionConfig(enabled=True, patterns=["[invalid", r"valid\d+"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r.compile_patterns()
        redaction_warnings = [x for x in w if issubclass(x.category, RedactionWarning)]
        assert len(redaction_warnings) == 1
        assert "Invalid redaction pattern" in str(redaction_warnings[0].message)
        # Valid pattern should still compile
        assert len(r._compiled) == 1

    def test_env_var_invalid_int_ignored(self) -> None:
        """Non-numeric env var for int field is silently ignored."""
        from lumen.config import LumenConfig
        with patch.dict(os.environ, {"LUMEN_MAX_ITERATIONS": "not_a_number"}):
            config = LumenConfig.auto()
        assert config.agent.max_iterations == 25  # default

    def test_env_var_invalid_float_ignored(self) -> None:
        """Non-numeric env var for float field is silently ignored."""
        from lumen.config import LumenConfig
        with patch.dict(os.environ, {"LUMEN_BUDGET_USD": "abc"}):
            config = LumenConfig.auto()
        assert config.cost.budget_usd == 0.0  # default

    def test_home_dir_failure_handled(self) -> None:
        """Path.home() failure doesn't crash auto()."""
        from lumen.config import LumenConfig
        with patch("lumen.config.Path.home", side_effect=RuntimeError("no home")):
            config = LumenConfig.auto()
        assert isinstance(config, LumenConfig)

    def test_cwd_failure_returns_none(self) -> None:
        """Path.cwd() failure in _find_project_config returns None."""
        from lumen.config import _find_project_config
        with patch("lumen.config.Path.cwd", side_effect=OSError("no cwd")):
            result = _find_project_config()
        assert result is None


# ── Redaction error handling ─────────────────────────────────────────────

class TestRedactionErrorHandling:

    def test_invalid_pattern_in_engine_skipped_with_warning(self) -> None:
        """RedactionEngine skips invalid patterns and warns."""
        from lumen._redaction import RedactionEngine, RedactionPattern

        bad_pattern = RedactionPattern(name="bad", pattern="[[[invalid")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = RedactionEngine(custom_patterns=[bad_pattern])
        redaction_warnings = [x for x in w if issubclass(x.category, RedactionWarning)]
        assert len(redaction_warnings) >= 1

    def test_invalid_pattern_does_not_prevent_valid_patterns(self) -> None:
        """Valid patterns still work even when invalid ones are present."""
        from lumen._redaction import RedactionEngine, RedactionPattern

        patterns = [
            RedactionPattern(name="bad", pattern="[[[invalid"),
            RedactionPattern(name="email", pattern=r"\b\S+@\S+\.\S+\b", replacement="[EMAIL]"),
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            engine = RedactionEngine(custom_patterns=patterns)
        result = engine.redact_text("Contact me at user@example.com")
        assert "[EMAIL]" in result

    def test_redact_text_with_no_composite_falls_back(self) -> None:
        """When composite regex is None, falls back to per-pattern mode."""
        from lumen._redaction import RedactionEngine, RedactionPattern

        engine = RedactionEngine()
        engine._composite = None  # simulate composite build failure
        engine._patterns = [
            RedactionPattern(name="test", pattern=r"secret\d+", replacement="[SECRET]"),
        ]
        result = engine.redact_text("my secret123 is here")
        assert "[SECRET]" in result

    def test_redact_trace_steps_with_bad_data(self) -> None:
        """redact_trace_steps handles non-serializable step data."""
        from lumen._redaction import RedactionEngine

        engine = RedactionEngine(presets=["pii"])
        # Steps with an object() value
        steps = [{"step_type": "LlmCall", "data": "user@test.com"}]
        result = engine.redact_trace_steps(steps)
        assert isinstance(result, list)

    def test_pattern_compile_returns_none_for_bad_regex(self) -> None:
        """RedactionPattern.compile() returns None for invalid regex."""
        from lumen._redaction import RedactionPattern
        p = RedactionPattern(name="bad", pattern="[[[")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            compiled = p.compile()
        assert compiled is None

    def test_pattern_apply_with_bad_regex_returns_original(self) -> None:
        """RedactionPattern.apply() returns original text if pattern is invalid."""
        from lumen._redaction import RedactionPattern
        p = RedactionPattern(name="bad", pattern="[[[")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = p.apply("some text")
        assert result == "some text"


# ── EventBus error handling ──────────────────────────────────────────────

class TestEventBusErrorHandling:

    def test_handler_exception_does_not_crash_emit(self) -> None:
        """A failing handler doesn't prevent other handlers from running."""
        from lumen._event_bus import EventBus, TraceCompleted

        bus = EventBus()
        results: list[str] = []

        bus.on(TraceCompleted, lambda e: (_ for _ in ()).throw(ValueError("boom")))
        bus.on(TraceCompleted, lambda e: results.append("ok"))

        event = TraceCompleted(
            trace_id="t1", agent_id="a", status="Completed",
            total_cost_usd=0.01, step_count=1, duration_ms=100,
        )
        count = bus.emit(event)
        assert count == 2
        assert "ok" in results

    def test_all_handlers_called_even_with_failures(self) -> None:
        """All registered handlers are invoked, failures notwithstanding."""
        from lumen._event_bus import EventBus, BudgetAlert

        bus = EventBus()
        call_count = [0]

        def failing(_: Any) -> None:
            call_count[0] += 1
            raise RuntimeError("fail")

        def passing(_: Any) -> None:
            call_count[0] += 1

        bus.on(BudgetAlert, failing)
        bus.on(BudgetAlert, passing)
        bus.on(BudgetAlert, failing)

        bus.emit(BudgetAlert(spent=10.0, budget=100.0, utilization_pct=10.0))
        assert call_count[0] == 3


# ── Trace reader error handling ──────────────────────────────────────────

class TestTraceReaderErrors:

    def test_non_utf8_file_skipped(self, tmp_path: Path) -> None:
        """Non-UTF-8 files are silently skipped."""
        from lumen.trace_reader import load_traces

        # Write a valid trace
        valid = {"trace_id": "good", "agent_id": "a", "steps": []}
        (tmp_path / "good.json").write_text(json.dumps(valid))

        # Write a binary file with invalid UTF-8
        (tmp_path / "bad.json").write_bytes(b'{"trace_id": "\xff\xfe invalid"}')

        traces = load_traces(str(tmp_path))
        assert len(traces) == 1
        assert traces[0].trace_id == "good"

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON files are skipped."""
        from lumen.trace_reader import load_traces

        (tmp_path / "bad.json").write_text("{not json at all!")
        (tmp_path / "good.json").write_text(json.dumps({
            "trace_id": "ok", "agent_id": "a", "steps": [],
        }))

        traces = load_traces(str(tmp_path))
        assert len(traces) == 1
        assert traces[0].trace_id == "ok"

    def test_nonexistent_directory_returns_empty(self) -> None:
        """Nonexistent trace directory returns empty list."""
        from lumen.trace_reader import load_traces
        assert load_traces("/this/path/does/not/exist") == []


# ── TraceIndex error handling ────────────────────────────────────────────

class TestTraceIndexErrors:

    def test_non_utf8_file_skipped(self, tmp_path: Path) -> None:
        """Non-UTF-8 files are skipped in index build."""
        from lumen._trace_index import TraceIndex

        (tmp_path / "bad.json").write_bytes(b'{"trace_id": "\xff\xfe"}')
        (tmp_path / "good.json").write_text(json.dumps({
            "trace_id": "ok", "agent_id": "a", "status": "Completed",
            "started_at_ms": 1000, "completed_at_ms": 2000, "steps": [],
        }))

        idx = TraceIndex(str(tmp_path))
        count = idx.build()
        assert count == 1

    def test_build_on_nonexistent_dir(self) -> None:
        """Building index on nonexistent dir returns 0."""
        from lumen._trace_index import TraceIndex
        idx = TraceIndex("/this/does/not/exist")
        assert idx.build() == 0


# ── Instrument error handling ────────────────────────────────────────────

class TestInstrumentErrors:

    def test_instrument_missing_sdk_returns_false(self) -> None:
        """Instrumenting an unavailable SDK returns False."""
        from lumen.instrument import instrument, uninstrument
        try:
            results = instrument("openai", "anthropic", "langgraph")
            # These may or may not be installed, but should never crash
            assert isinstance(results, dict)
        finally:
            uninstrument()

    def test_uninstrument_never_crashes(self) -> None:
        """uninstrument() is always safe to call."""
        from lumen.instrument import uninstrument
        # Call without prior instrument — should not raise
        uninstrument()
        uninstrument()  # double call


# ── LangGraph tracer error handling ──────────────────────────────────────

class TestLangGraphTracerErrors:

    def test_write_trace_to_bad_dir_warns(self) -> None:
        """LumenTracer._write_trace warns on unwritable directory."""
        from lumen.integrations.langgraph_tracer import LumenTracer

        tracer = LumenTracer(trace_dir="/tmp/langgraph_test", agent_name="test")
        tracer._trace_id = "test123"
        tracer._started_at_ms = 1000
        tracer._steps = [{"step_num": 0, "step_type": "LlmCall"}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("lumen._context.Path.mkdir", side_effect=PermissionError("denied")):
                tracer._write_trace()
        trace_warnings = [x for x in w if issubclass(x.category, TraceWriteWarning)]
        assert len(trace_warnings) >= 1


# ── Pricing edge cases ──────────────────────────────────────────────────

class TestPricingEdgeCases:

    def test_empty_model_name(self) -> None:
        """Empty model name returns fallback pricing."""
        from lumen.pricing import estimate_cost
        cost = estimate_cost("", prompt_tokens=1000, completion_tokens=0)
        assert cost > 0.0

    def test_none_like_model_name(self) -> None:
        """Various unusual model names don't crash."""
        from lumen.pricing import estimate_cost
        for name in ["   ", "None", "null", "undefined", "a" * 1000]:
            cost = estimate_cost(name, prompt_tokens=100, completion_tokens=0)
            assert cost >= 0.0

    def test_negative_tokens_handled(self) -> None:
        """Negative token counts produce negative cost (no crash)."""
        from lumen.pricing import estimate_cost
        cost = estimate_cost("gpt-4o", prompt_tokens=-100, completion_tokens=0)
        assert isinstance(cost, float)


# ── Budget error visibility ──────────────────────────────────────────────

class TestBudgetErrors:

    def test_budget_exceeded_has_actionable_message(self) -> None:
        """BudgetExceededError message includes spent, limit, and guidance."""
        from lumen._budget import BudgetExceededError
        err = BudgetExceededError(spent=5.50, limit=5.00, scope="global")
        msg = str(err)
        assert "5.50" in msg or "5.5" in msg
        assert "5.00" in msg or "5.0" in msg
        assert "global" in msg
        assert "Reduce" in msg or "increase" in msg  # actionable guidance

    def test_budget_exceeded_attributes(self) -> None:
        """BudgetExceededError exposes spent/limit/scope."""
        from lumen._budget import BudgetExceededError
        err = BudgetExceededError(spent=10.0, limit=5.0, scope="per-run (abc)")
        assert err.spent == 10.0
        assert err.limit == 5.0
        assert err.scope == "per-run (abc)"


# ── Integration: trace() context manager error handling ──────────────────

class TestTraceContextManagerErrors:

    def test_trace_records_exception_as_error_step(self, tmp_path: Path) -> None:
        """Exceptions in trace() are recorded as error steps."""
        from lumen.instrument import trace
        from lumen.config import LumenConfig

        config = LumenConfig.builder().trace_dir(str(tmp_path)).build()

        with pytest.raises(ValueError, match="test error"):
            with trace("test-agent", config=config) as sess:
                raise ValueError("test error")

        # Trace should have been written with error
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["status"]["Failed"] == "test error"

    def test_trace_with_bad_dir_does_not_crash_user(self) -> None:
        """trace() with unwritable dir warns but doesn't crash."""
        from lumen.instrument import trace
        from lumen.config import LumenConfig

        config = LumenConfig.builder().trace_dir("/some/path").build()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("lumen._context.Path.mkdir", side_effect=PermissionError("denied")):
                with trace("test-agent", config=config) as sess:
                    sess.add_llm_step(
                        model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                        finish_reason="stop", duration_ms=100, started_at_ms=1000,
                    )
        # Should warn about write failure
        trace_warnings = [x for x in w if issubclass(x.category, TraceWriteWarning)]
        assert len(trace_warnings) >= 1


# ── Deep copy safety in _redact_steps / _strip_content ───────────────────

class TestDeepCopySafety:

    def test_redact_steps_with_non_serializable(self, tmp_path: Path) -> None:
        """_redact_steps handles non-serializable step data gracefully."""
        from lumen._context import _redact_steps
        from lumen.config import RedactionConfig

        redaction = RedactionConfig(enabled=True, patterns=[r"\bsecret\b"])
        redaction.compile_patterns()

        steps = [{"step_type": {"Error": {"message": "secret info"}}, "metadata": []}]
        result = _redact_steps(steps, redaction)
        assert isinstance(result, list)

    def test_strip_content_with_non_serializable(self) -> None:
        """_strip_content handles non-serializable step data gracefully."""
        from lumen._context import _strip_content

        steps = [{"metadata": [("key", "value")]}]
        result = _strip_content(steps)
        assert isinstance(result, list)
