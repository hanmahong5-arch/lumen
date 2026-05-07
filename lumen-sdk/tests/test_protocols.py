"""Tests for Protocol interfaces — structural typing contracts.

Covers: runtime_checkable isinstance checks, conformance verification,
non-conformance detection, duck typing validation for all 5 protocols.
"""

from __future__ import annotations

from typing import Any

import pytest

from lumen._protocols import (
    MetricsCollector,
    PricingProvider,
    RedactionProvider,
    TraceReader,
    TraceWriter,
)


# ── Conforming implementations ─────────────────────────────────────────

class _FakeWriter:
    """Conforms to TraceWriter."""
    def __init__(self):
        self.written = []

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        self.written.append(trace_data)


class _FakeReader:
    """Conforms to TraceReader."""
    def __init__(self, traces=None):
        self._traces = traces or []

    def load_traces(self) -> list[Any]:
        return self._traces

    def get_trace(self, trace_id: str) -> Any | None:
        for t in self._traces:
            if t.get("trace_id") == trace_id:
                return t
        return None


class _FakePricing:
    """Conforms to PricingProvider."""
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens + completion_tokens) * 0.001


class _FakeRedactor:
    """Conforms to RedactionProvider."""
    def redact_text(self, text: str) -> str:
        return text.replace("secret", "***")


class _FakeMetrics:
    """Conforms to MetricsCollector."""
    def __init__(self):
        self.latencies = []
        self.costs = []
        self.tokens = []

    def record_latency(self, duration_ms: float) -> None:
        self.latencies.append(duration_ms)

    def record_cost(self, cost_usd: float) -> None:
        self.costs.append(cost_usd)

    def record_tokens(self, prompt: int, completion: int) -> None:
        self.tokens.append((prompt, completion))


# ── Non-conforming implementations ─────────────────────────────────────

class _BadWriter:
    """Missing write_trace method."""
    def save(self, data):
        pass


class _BadReader:
    """Has load_traces but missing get_trace."""
    def load_traces(self) -> list:
        return []


class _BadPricing:
    """Wrong method signature (missing params)."""
    def estimate_cost(self) -> float:
        return 0.0


class _BadRedactor:
    """Wrong method name."""
    def sanitize(self, text: str) -> str:
        return text


class _BadMetrics:
    """Only has record_latency — missing record_cost and record_tokens."""
    def record_latency(self, duration_ms: float) -> None:
        pass


# ── Tests ───────────────────────────────────────────────────────────────

class TestTraceWriterProtocol:
    """TraceWriter structural typing."""

    def test_conforming_isinstance(self):
        writer = _FakeWriter()
        assert isinstance(writer, TraceWriter)

    def test_non_conforming_isinstance(self):
        bad = _BadWriter()
        assert not isinstance(bad, TraceWriter)

    def test_functional_write(self):
        writer = _FakeWriter()
        writer.write_trace({"trace_id": "abc", "steps": []})
        assert len(writer.written) == 1
        assert writer.written[0]["trace_id"] == "abc"

    def test_plain_object_not_writer(self):
        assert not isinstance(object(), TraceWriter)
        assert not isinstance("string", TraceWriter)


class TestTraceReaderProtocol:
    """TraceReader structural typing."""

    def test_conforming_isinstance(self):
        reader = _FakeReader()
        assert isinstance(reader, TraceReader)

    def test_non_conforming_missing_method(self):
        bad = _BadReader()
        assert not isinstance(bad, TraceReader)

    def test_functional_read(self):
        traces = [{"trace_id": "t1"}, {"trace_id": "t2"}]
        reader = _FakeReader(traces)
        assert len(reader.load_traces()) == 2
        assert reader.get_trace("t1") == {"trace_id": "t1"}
        assert reader.get_trace("t3") is None


class TestPricingProviderProtocol:
    """PricingProvider structural typing."""

    def test_conforming_isinstance(self):
        pricing = _FakePricing()
        assert isinstance(pricing, PricingProvider)

    def test_non_conforming_wrong_signature(self):
        bad = _BadPricing()
        # runtime_checkable only checks method name existence, not signature
        # So this actually passes isinstance — it's a known limitation
        # We test functional correctness separately
        assert isinstance(bad, PricingProvider)

    def test_functional_cost(self):
        pricing = _FakePricing()
        cost = pricing.estimate_cost("gpt-4o", 1000, 500)
        assert cost == pytest.approx(1.5)


class TestRedactionProviderProtocol:
    """RedactionProvider structural typing."""

    def test_conforming_isinstance(self):
        redactor = _FakeRedactor()
        assert isinstance(redactor, RedactionProvider)

    def test_non_conforming_wrong_method(self):
        bad = _BadRedactor()
        assert not isinstance(bad, RedactionProvider)

    def test_functional_redact(self):
        redactor = _FakeRedactor()
        result = redactor.redact_text("my secret key")
        assert result == "my *** key"


class TestMetricsCollectorProtocol:
    """MetricsCollector structural typing."""

    def test_conforming_isinstance(self):
        metrics = _FakeMetrics()
        assert isinstance(metrics, MetricsCollector)

    def test_non_conforming_partial(self):
        bad = _BadMetrics()
        assert not isinstance(bad, MetricsCollector)

    def test_functional_record(self):
        m = _FakeMetrics()
        m.record_latency(150.0)
        m.record_cost(0.05)
        m.record_tokens(1000, 500)
        assert m.latencies == [150.0]
        assert m.costs == [0.05]
        assert m.tokens == [(1000, 500)]


class TestProtocolDuckTyping:
    """Verify that any object with the right shape works — no inheritance needed."""

    def test_lambda_based_writer(self):
        """A class created entirely from scratch conforms."""
        class InlineWriter:
            write_trace = lambda self, data: None  # noqa: E731
        assert isinstance(InlineWriter(), TraceWriter)

    def test_dict_not_protocol(self):
        """Dicts don't accidentally match protocols."""
        d = {"write_trace": lambda data: None}
        assert not isinstance(d, TraceWriter)

    def test_multiple_protocol_conformance(self):
        """One object can conform to multiple protocols."""
        class MultiTool:
            def write_trace(self, trace_data): pass
            def redact_text(self, text): return text

        obj = MultiTool()
        assert isinstance(obj, TraceWriter)
        assert isinstance(obj, RedactionProvider)
        assert not isinstance(obj, PricingProvider)
