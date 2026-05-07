"""Real-time cost anomaly detection using sliding window statistics.

Monitors per-trace costs via a MetricsWindow and emits CostAnomaly events
when a trace's cost exceeds a configurable multiplier above the running
average. Uses the EventBus for decoupled notification.

Detection algorithm:
  1. Maintain a sliding window of recent trace costs (default: last 100)
  2. On each new trace, compute: multiplier = cost / mean
  3. If multiplier > threshold AND cost > minimum → emit CostAnomaly event

The minimum cost threshold prevents false positives on cheap traces
(e.g., $0.001 that is 10x the average of $0.0001).

Usage::

    from lumen._anomaly import AnomalyDetector

    detector = AnomalyDetector(multiplier=3.0, min_cost_usd=0.01)
    detector.check(trace_id="abc", agent_id="research", cost_usd=5.00)
    # → emits CostAnomaly if 5.00 > 3.0 * running_mean AND 5.00 > 0.01
"""

from __future__ import annotations

from lumen._event_bus import CostAnomaly, get_default_bus
from lumen._ring_buffer import MetricsWindow


class AnomalyDetector:
    """Sliding-window cost anomaly detector.

    Tracks the last ``window_size`` trace costs and fires CostAnomaly
    events via the default EventBus when anomalies are detected.
    """

    __slots__ = ("_window", "_multiplier", "_min_cost", "_bus")

    def __init__(
        self,
        multiplier: float = 2.0,
        min_cost_usd: float = 0.01,
        window_size: int = 100,
    ) -> None:
        """
        Args:
            multiplier: Cost must exceed mean * multiplier to be anomalous.
            min_cost_usd: Minimum absolute cost to trigger an anomaly.
            window_size: Number of recent costs to track.
        """
        self._window = MetricsWindow(window_size)
        self._multiplier = multiplier
        self._min_cost = min_cost_usd
        self._bus = get_default_bus()

    def check(
        self,
        trace_id: str,
        agent_id: str,
        cost_usd: float,
    ) -> bool:
        """Check if a trace cost is anomalous and record it.

        Always records the cost into the sliding window. Returns True
        if an anomaly was detected (and a CostAnomaly event emitted).

        Args:
            trace_id: The trace that incurred this cost.
            agent_id: The agent that ran.
            cost_usd: The cost to check.

        Returns:
            True if anomaly detected, False otherwise.
        """
        mean = self._window.mean
        is_anomaly = False

        # Need at least 5 data points for a meaningful average
        if (
            self._window.count >= 5
            and mean > 0
            and cost_usd >= self._min_cost
        ):
            actual_multiplier = cost_usd / mean
            if actual_multiplier > self._multiplier:
                is_anomaly = True
                self._bus.emit(CostAnomaly(
                    trace_id=trace_id,
                    agent_id=agent_id,
                    cost_usd=cost_usd,
                    average_usd=mean,
                    multiplier=actual_multiplier,
                ))

        # Always record — even anomalous costs enter the window
        self._window.record(cost_usd)
        return is_anomaly

    @property
    def mean_cost(self) -> float:
        """Current running average cost."""
        return self._window.mean

    @property
    def sample_count(self) -> int:
        """Number of costs in the sliding window."""
        return self._window.count

    def reset(self) -> None:
        """Clear all recorded costs."""
        self._window.clear()
