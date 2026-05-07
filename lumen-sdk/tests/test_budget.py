"""Comprehensive tests for the budget tracking and enforcement engine."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from lumen._budget import BudgetExceededError, BudgetSnapshot, BudgetTracker
from lumen.config import LumenConfig


class TestBudgetExceededError:
    """Test the BudgetExceededError exception."""

    def test_message_format(self) -> None:
        err = BudgetExceededError(spent=1.5, limit=1.0, scope="global")
        assert "1.5" in str(err)
        assert "1.0" in str(err)
        assert "global" in str(err)

    def test_attributes(self) -> None:
        err = BudgetExceededError(spent=5.0, limit=3.0, scope="per-run (abc)")
        assert err.spent == 5.0
        assert err.limit == 3.0
        assert err.scope == "per-run (abc)"

    def test_is_exception(self) -> None:
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError(1.0, 0.5, "test")


class TestBudgetTrackerDefaults:
    """Test BudgetTracker with default settings (no limits)."""

    def test_initial_state(self) -> None:
        bt = BudgetTracker()
        assert bt.total_spent == 0.0
        assert bt.run_count == 0

    def test_record_zero_cost(self) -> None:
        bt = BudgetTracker()
        bt.record(0.0)
        bt.record(-1.0)
        assert bt.total_spent == 0.0

    def test_record_accumulates(self) -> None:
        bt = BudgetTracker()
        bt.record(0.05)
        bt.record(0.10)
        bt.record(0.03)
        assert bt.total_spent == pytest.approx(0.18)

    def test_no_limit_never_raises(self) -> None:
        bt = BudgetTracker()
        for _ in range(1000):
            bt.record(100.0)
        assert bt.total_spent == pytest.approx(100_000.0)

    def test_check_per_run_no_limit(self) -> None:
        bt = BudgetTracker()
        bt.check_per_run(999_999.0, "t1")  # no exception


class TestBudgetTrackerPerRun:
    """Test per-run budget enforcement."""

    def test_within_limit(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.record(2.0, trace_id="t1")
        bt.check_per_run(2.0, "t1")  # 2+2=4 < 5, OK

    def test_exceeds_limit(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.record(3.0, trace_id="t1")
        with pytest.raises(BudgetExceededError, match="per-run"):
            bt.check_per_run(3.0, "t1")  # 3+3=6 > 5

    def test_exact_limit(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.record(2.5, trace_id="t1")
        bt.check_per_run(2.5, "t1")  # 2.5+2.5=5.0, exactly at limit

    def test_exceeds_by_tiny_amount(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.record(2.5, trace_id="t1")
        with pytest.raises(BudgetExceededError):
            bt.check_per_run(2.51, "t1")  # 5.01 > 5.0

    def test_separate_traces_independent(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.record(4.0, trace_id="t1")
        bt.record(4.0, trace_id="t2")
        bt.check_per_run(0.9, "t1")  # t1: 4+0.9=4.9, OK
        bt.check_per_run(0.9, "t2")  # t2: 4+0.9=4.9, OK

    def test_unknown_trace_id(self) -> None:
        bt = BudgetTracker(max_per_run=5.0)
        bt.check_per_run(4.99, "unknown")  # 0+4.99 < 5, OK


class TestBudgetTrackerGlobal:
    """Test global budget enforcement."""

    def test_within_budget(self) -> None:
        bt = BudgetTracker(budget=10.0, kill_on_exceeded=True)
        bt.record(5.0)
        bt.record(4.99)
        assert bt.total_spent == pytest.approx(9.99)

    def test_exceeds_budget_raises(self) -> None:
        bt = BudgetTracker(budget=10.0, kill_on_exceeded=True)
        bt.record(9.0)
        with pytest.raises(BudgetExceededError, match="global"):
            bt.record(2.0)  # 11 > 10

    def test_exceeds_budget_no_kill(self) -> None:
        bt = BudgetTracker(budget=10.0, kill_on_exceeded=False)
        bt.record(9.0)
        bt.record(2.0)  # no exception, just tracking
        assert bt.total_spent == pytest.approx(11.0)

    def test_exact_budget_ok(self) -> None:
        bt = BudgetTracker(budget=10.0, kill_on_exceeded=True)
        bt.record(10.0)  # exactly at budget, no exception

    def test_zero_budget_means_unlimited(self) -> None:
        bt = BudgetTracker(budget=0.0, kill_on_exceeded=True)
        bt.record(999.0)  # 0 budget means no limit


class TestBudgetTrackerFinishRun:
    """Test run completion tracking."""

    def test_finish_returns_cost(self) -> None:
        bt = BudgetTracker()
        bt.record(1.0, trace_id="t1")
        bt.record(2.0, trace_id="t1")
        cost = bt.finish_run("t1")
        assert cost == pytest.approx(3.0)

    def test_finish_increments_run_count(self) -> None:
        bt = BudgetTracker()
        bt.record(1.0, trace_id="t1")
        bt.record(1.0, trace_id="t2")
        bt.finish_run("t1")
        assert bt.run_count == 1
        bt.finish_run("t2")
        assert bt.run_count == 2

    def test_finish_removes_from_tracking(self) -> None:
        bt = BudgetTracker()
        bt.record(5.0, trace_id="t1")
        bt.finish_run("t1")
        assert bt.get_run_cost("t1") == 0.0

    def test_finish_unknown_trace(self) -> None:
        bt = BudgetTracker()
        cost = bt.finish_run("nonexistent")
        assert cost == 0.0
        assert bt.run_count == 1


class TestBudgetSnapshot:
    """Test budget status snapshots."""

    def test_snapshot_no_budget(self) -> None:
        bt = BudgetTracker()
        bt.record(5.0)
        snap = bt.snapshot()
        assert snap.total_spent_usd == pytest.approx(5.0)
        assert snap.budget_usd == 0.0
        assert snap.remaining_usd == float("inf")
        assert snap.utilization_pct == 0.0
        assert snap.alert_triggered is False

    def test_snapshot_with_budget(self) -> None:
        bt = BudgetTracker(budget=100.0)
        bt.record(75.0)
        snap = bt.snapshot()
        assert snap.total_spent_usd == pytest.approx(75.0)
        assert snap.budget_usd == 100.0
        assert snap.remaining_usd == pytest.approx(25.0)
        assert snap.utilization_pct == pytest.approx(75.0)
        assert snap.alert_triggered is False

    def test_snapshot_alert_threshold(self) -> None:
        bt = BudgetTracker(budget=100.0, alert_threshold_pct=80)
        bt.record(80.0)
        snap = bt.snapshot()
        assert snap.alert_triggered is True

    def test_snapshot_below_threshold(self) -> None:
        bt = BudgetTracker(budget=100.0, alert_threshold_pct=80)
        bt.record(79.0)
        snap = bt.snapshot()
        assert snap.alert_triggered is False

    def test_snapshot_over_budget(self) -> None:
        bt = BudgetTracker(budget=100.0, kill_on_exceeded=False)
        bt.record(120.0)
        snap = bt.snapshot()
        assert snap.remaining_usd == 0.0
        assert snap.utilization_pct == 100.0  # capped at 100
        assert snap.alert_triggered is True

    def test_snapshot_period(self) -> None:
        bt = BudgetTracker(budget_period="weekly")
        snap = bt.snapshot()
        assert snap.period == "weekly"

    def test_snapshot_run_count(self) -> None:
        bt = BudgetTracker()
        bt.record(1.0, "t1")
        bt.finish_run("t1")
        bt.record(2.0, "t2")
        bt.finish_run("t2")
        snap = bt.snapshot()
        assert snap.run_count == 2


class TestBudgetTrackerReset:
    """Test budget reset (period rollover)."""

    def test_reset_clears_all(self) -> None:
        bt = BudgetTracker(budget=100.0)
        bt.record(50.0, trace_id="t1")
        bt.finish_run("t1")
        bt.reset()
        assert bt.total_spent == 0.0
        assert bt.run_count == 0
        assert bt.get_run_cost("t1") == 0.0

    def test_reset_updates_period_start(self) -> None:
        bt = BudgetTracker()
        old_snap = bt.snapshot()
        time.sleep(0.01)
        bt.reset()
        new_snap = bt.snapshot()
        assert new_snap.period_start_ms >= old_snap.period_start_ms


class TestBudgetTrackerFromConfig:
    """Test creating BudgetTracker from LumenConfig."""

    def test_from_config(self) -> None:
        config = LumenConfig.from_dict({
            "agent": {"max_cost_per_run_usd": 5.0},
            "cost": {
                "budget_usd": 100.0,
                "budget_period": "weekly",
                "alert_threshold_pct": 90,
                "kill_on_budget_exceeded": True,
            },
        })
        bt = BudgetTracker.from_config(config)
        bt.record(99.0)
        snap = bt.snapshot()
        assert snap.budget_usd == 100.0
        assert snap.period == "weekly"
        assert snap.alert_triggered is True  # 99% > 90%

    def test_from_config_defaults(self) -> None:
        config = LumenConfig()
        bt = BudgetTracker.from_config(config)
        bt.record(999.0)  # no exception, no limits


class TestBudgetTrackerThreadSafety:
    """Test thread safety of budget tracking."""

    def test_concurrent_records(self) -> None:
        bt = BudgetTracker()
        n_threads = 10
        increments_per_thread = 100
        amount = 0.01

        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                bt.record(amount)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * increments_per_thread * amount
        assert bt.total_spent == pytest.approx(expected, rel=1e-6)

    def test_concurrent_finish_runs(self) -> None:
        bt = BudgetTracker()
        n = 50

        for i in range(n):
            bt.record(1.0, trace_id=f"t{i}")

        barrier = threading.Barrier(n)

        def finish_worker(idx: int) -> None:
            barrier.wait()
            bt.finish_run(f"t{idx}")

        threads = [threading.Thread(target=finish_worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bt.run_count == n

    def test_concurrent_snapshots(self) -> None:
        bt = BudgetTracker(budget=1000.0)
        bt.record(500.0)

        results: list[BudgetSnapshot] = []
        lock = threading.Lock()

        def snapshot_worker() -> None:
            snap = bt.snapshot()
            with lock:
                results.append(snap)

        threads = [threading.Thread(target=snapshot_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(s.total_spent_usd == pytest.approx(500.0) for s in results)


class TestBudgetTrackerGetRunCost:
    """Test per-run cost querying."""

    def test_get_existing(self) -> None:
        bt = BudgetTracker()
        bt.record(3.5, trace_id="t1")
        assert bt.get_run_cost("t1") == pytest.approx(3.5)

    def test_get_nonexistent(self) -> None:
        bt = BudgetTracker()
        assert bt.get_run_cost("t1") == 0.0

    def test_accumulates_across_records(self) -> None:
        bt = BudgetTracker()
        bt.record(1.0, trace_id="t1")
        bt.record(2.0, trace_id="t1")
        bt.record(0.5, trace_id="t1")
        assert bt.get_run_cost("t1") == pytest.approx(3.5)
