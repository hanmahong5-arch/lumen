"""Tests for AnomalyDetector — real-time cost anomaly detection.

Covers: detection threshold, minimum cost gate, warm-up period,
event emission via EventBus, reset, and integration with MetricsWindow.
"""

from __future__ import annotations

import pytest

from lumen._anomaly import AnomalyDetector
from lumen._event_bus import CostAnomaly, EventBus, get_default_bus


class TestAnomalyDetectorBasics:
    """Core anomaly detection logic."""

    def test_no_anomaly_during_warmup(self):
        d = AnomalyDetector(multiplier=2.0, min_cost_usd=0.01, window_size=100)
        # Need 5+ data points before detection kicks in
        for _ in range(4):
            assert d.check("t", "a", 0.10) is False

    def test_normal_cost_not_anomalous(self):
        d = AnomalyDetector(multiplier=3.0)
        # Build baseline
        for i in range(10):
            d.check(f"t{i}", "agent", 1.0)
        # Cost at 2x mean — below 3x threshold
        assert d.check("t-test", "agent", 2.0) is False

    def test_high_cost_detected_as_anomaly(self):
        d = AnomalyDetector(multiplier=2.0)
        # Build baseline of $1.00 each
        for i in range(10):
            d.check(f"t{i}", "agent", 1.0)
        # $5.00 is 5x the mean → anomaly
        assert d.check("t-expensive", "agent", 5.0) is True

    def test_min_cost_gate(self):
        d = AnomalyDetector(multiplier=2.0, min_cost_usd=1.0)
        # Build baseline of $0.001 each
        for i in range(10):
            d.check(f"t{i}", "agent", 0.001)
        # $0.01 is 10x mean but below min_cost_usd=1.0
        assert d.check("t-test", "agent", 0.01) is False


class TestAnomalyDetectorEventEmission:
    """Verify CostAnomaly events are emitted via EventBus."""

    def test_anomaly_emits_event(self):
        bus = get_default_bus()
        received: list[CostAnomaly] = []
        handler = lambda e: received.append(e)
        bus.on(CostAnomaly, handler)

        try:
            d = AnomalyDetector(multiplier=2.0, min_cost_usd=0.01)
            for i in range(10):
                d.check(f"t{i}", "agent-x", 0.10)
            d.check("t-spike", "agent-x", 5.0)

            assert len(received) == 1
            e = received[0]
            assert e.trace_id == "t-spike"
            assert e.agent_id == "agent-x"
            assert e.cost_usd == 5.0
            assert e.average_usd == pytest.approx(0.10, rel=0.1)
            assert e.multiplier > 2.0
        finally:
            bus.off(CostAnomaly, handler)

    def test_no_event_for_normal_cost(self):
        bus = get_default_bus()
        received: list[CostAnomaly] = []
        handler = lambda e: received.append(e)
        bus.on(CostAnomaly, handler)

        try:
            d = AnomalyDetector(multiplier=2.0)
            for i in range(10):
                d.check(f"t{i}", "a", 1.0)
            d.check("t-normal", "a", 1.5)  # 1.5x, below 2.0 threshold
            assert len(received) == 0
        finally:
            bus.off(CostAnomaly, handler)


class TestAnomalyDetectorProperties:
    """mean_cost, sample_count, reset."""

    def test_mean_cost(self):
        d = AnomalyDetector()
        d.check("t1", "a", 1.0)
        d.check("t2", "a", 3.0)
        assert d.mean_cost == pytest.approx(2.0)

    def test_sample_count(self):
        d = AnomalyDetector()
        assert d.sample_count == 0
        for i in range(5):
            d.check(f"t{i}", "a", 1.0)
        assert d.sample_count == 5

    def test_reset(self):
        d = AnomalyDetector()
        for i in range(10):
            d.check(f"t{i}", "a", 1.0)
        d.reset()
        assert d.sample_count == 0
        assert d.mean_cost == 0.0


class TestAnomalyDetectorEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_cost_never_anomalous(self):
        d = AnomalyDetector(multiplier=2.0, min_cost_usd=0.0)
        for i in range(10):
            d.check(f"t{i}", "a", 1.0)
        # Zero cost can't exceed multiplier * mean in a meaningful way
        assert d.check("t-zero", "a", 0.0) is False

    def test_all_identical_costs(self):
        d = AnomalyDetector(multiplier=2.0, min_cost_usd=0.01)
        for i in range(20):
            assert d.check(f"t{i}", "a", 1.0) is False

    def test_gradually_increasing_costs(self):
        d = AnomalyDetector(multiplier=3.0, min_cost_usd=0.01, window_size=10)
        results = []
        for i in range(20):
            # Gradually increase: 0.1, 0.2, ..., 2.0
            cost = (i + 1) * 0.1
            results.append(d.check(f"t{i}", "a", cost))
        # Gradual increase shouldn't trigger (window adapts)
        # The later costs are within 3x of the sliding mean
        anomaly_count = sum(results)
        # At most a few triggers during the ramp-up
        assert anomaly_count <= 5

    def test_spike_after_low_baseline(self):
        d = AnomalyDetector(multiplier=2.0, min_cost_usd=0.01)
        # Low baseline
        for i in range(10):
            d.check(f"t{i}", "a", 0.05)
        # Spike
        assert d.check("t-spike", "a", 0.50) is True
