"""Thread-safe budget tracking and enforcement.

Tracks cumulative cost across all trace sessions and enforces:
  - Per-run cost limits (max_cost_per_run_usd)
  - Global budget limits (budget_usd with kill_on_budget_exceeded)

Usage::

    from lumen._budget import BudgetTracker, BudgetExceededError

    tracker = BudgetTracker(max_per_run=5.0, budget=100.0, kill=True)
    tracker.record(0.05)        # OK
    tracker.check_per_run(4.99) # OK
    tracker.check_per_run(5.01) # raises BudgetExceededError
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumen.config import LumenConfig


class BudgetExceededError(Exception):
    """Raised when a cost operation would exceed the configured budget."""

    def __init__(self, spent: float, limit: float, scope: str) -> None:
        self.spent = spent
        self.limit = limit
        self.scope = scope
        super().__init__(
            f"Budget exceeded ({scope}): spent ${spent:.4f}, "
            f"limit ${limit:.2f}. Reduce usage or increase budget."
        )


@dataclass
class BudgetSnapshot:
    """Point-in-time budget status."""

    total_spent_usd: float
    budget_usd: float
    remaining_usd: float
    utilization_pct: float
    period: str
    period_start_ms: int
    run_count: int
    alert_triggered: bool


class BudgetTracker:
    """Thread-safe cumulative cost tracker with budget enforcement.

    Tracks spending across all trace sessions in the current process.
    Optionally kills (raises) when limits are exceeded.
    """

    def __init__(
        self,
        max_per_run: float = 0.0,
        budget: float = 0.0,
        budget_period: str = "monthly",
        alert_threshold_pct: int = 80,
        kill_on_exceeded: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._max_per_run = max_per_run
        self._budget = budget
        self._budget_period = budget_period
        self._alert_threshold_pct = alert_threshold_pct
        self._kill_on_exceeded = kill_on_exceeded

        self._total_spent = 0.0
        self._run_count = 0
        self._period_start_ms = _now_ms()
        self._run_costs: dict[str, float] = {}  # trace_id → accumulated cost
        self._alert_fired = False

    @classmethod
    def from_config(cls, config: LumenConfig) -> BudgetTracker:
        """Create a BudgetTracker from LumenConfig."""
        return cls(
            max_per_run=config.agent.max_cost_per_run_usd,
            budget=config.cost.budget_usd,
            budget_period=config.cost.budget_period,
            alert_threshold_pct=config.cost.alert_threshold_pct,
            kill_on_exceeded=config.cost.kill_on_budget_exceeded,
        )

    def record(self, cost_usd: float, trace_id: str = "") -> None:
        """Record a cost increment. Enforces global budget if kill_on_exceeded.

        Args:
            cost_usd: Cost to add.
            trace_id: Optional trace ID for per-run tracking.

        Raises:
            BudgetExceededError: If kill_on_exceeded and global budget is blown.
        """
        if cost_usd <= 0:
            return

        with self._lock:
            self._total_spent += cost_usd

            if trace_id:
                self._run_costs[trace_id] = self._run_costs.get(trace_id, 0.0) + cost_usd

            # Check global budget
            if self._budget > 0 and self._kill_on_exceeded:
                if self._total_spent > self._budget:
                    raise BudgetExceededError(
                        self._total_spent, self._budget, "global",
                    )

    def check_per_run(self, cost_usd: float, trace_id: str = "") -> None:
        """Check if adding cost_usd would exceed per-run limit.

        Args:
            cost_usd: Proposed cost increment.
            trace_id: Trace ID for per-run accounting.

        Raises:
            BudgetExceededError: If the run's total would exceed max_per_run.
        """
        if self._max_per_run <= 0:
            return

        with self._lock:
            current = self._run_costs.get(trace_id, 0.0)
            projected = current + cost_usd
            if projected > self._max_per_run:
                raise BudgetExceededError(
                    projected, self._max_per_run, f"per-run ({trace_id})",
                )

    def finish_run(self, trace_id: str) -> float:
        """Mark a run as finished and return its total cost.

        Args:
            trace_id: The trace ID of the completed run.

        Returns:
            Total cost for this run.
        """
        with self._lock:
            cost = self._run_costs.pop(trace_id, 0.0)
            self._run_count += 1
            return cost

    def snapshot(self) -> BudgetSnapshot:
        """Get a point-in-time budget status snapshot."""
        with self._lock:
            remaining = max(0.0, self._budget - self._total_spent) if self._budget > 0 else float("inf")
            utilization = (self._total_spent / self._budget * 100) if self._budget > 0 else 0.0
            alert = (
                self._budget > 0
                and utilization >= self._alert_threshold_pct
            )

            return BudgetSnapshot(
                total_spent_usd=self._total_spent,
                budget_usd=self._budget,
                remaining_usd=remaining,
                utilization_pct=min(utilization, 100.0),
                period=self._budget_period,
                period_start_ms=self._period_start_ms,
                run_count=self._run_count,
                alert_triggered=alert,
            )

    def reset(self) -> None:
        """Reset all counters (e.g., at period rollover)."""
        with self._lock:
            self._total_spent = 0.0
            self._run_count = 0
            self._run_costs.clear()
            self._period_start_ms = _now_ms()
            self._alert_fired = False

    @property
    def total_spent(self) -> float:
        with self._lock:
            return self._total_spent

    @property
    def run_count(self) -> int:
        with self._lock:
            return self._run_count

    def get_run_cost(self, trace_id: str) -> float:
        with self._lock:
            return self._run_costs.get(trace_id, 0.0)


def _now_ms() -> int:
    return int(time.time() * 1000)
