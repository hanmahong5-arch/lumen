"""Tests for EventBus — type-safe publish-subscribe event system.

Covers: on/off/once/emit, decorator mode, event isolation (no subclass dispatch),
thread safety, handler ordering, clear, handler_count, built-in event types.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from lumen._event_bus import (
    BudgetAlert,
    BudgetExceeded,
    CostAnomaly,
    EventBus,
    TraceSampled,
    TraceCompleted,
    get_default_bus,
)


@dataclass(frozen=True, slots=True)
class _TestEvent:
    """Custom event for testing."""
    value: int


@dataclass(frozen=True, slots=True)
class _OtherEvent:
    """A different event type for isolation tests."""
    data: str


class TestEventBusOn:
    """Handler registration and dispatch."""

    def test_basic_emit_and_receive(self):
        bus = EventBus()
        received = []
        bus.on(_TestEvent, lambda e: received.append(e.value))
        bus.emit(_TestEvent(value=42))
        assert received == [42]

    def test_multiple_handlers(self):
        bus = EventBus()
        a, b = [], []
        bus.on(_TestEvent, lambda e: a.append(e.value))
        bus.on(_TestEvent, lambda e: b.append(e.value))
        bus.emit(_TestEvent(value=7))
        assert a == [7]
        assert b == [7]

    def test_emit_returns_handler_count(self):
        bus = EventBus()
        bus.on(_TestEvent, lambda e: None)
        bus.on(_TestEvent, lambda e: None)
        assert bus.emit(_TestEvent(value=1)) == 2

    def test_emit_no_handlers_returns_zero(self):
        bus = EventBus()
        assert bus.emit(_TestEvent(value=1)) == 0

    def test_handler_ordering(self):
        """Handlers called in registration order."""
        bus = EventBus()
        order = []
        bus.on(_TestEvent, lambda e: order.append("first"))
        bus.on(_TestEvent, lambda e: order.append("second"))
        bus.on(_TestEvent, lambda e: order.append("third"))
        bus.emit(_TestEvent(value=0))
        assert order == ["first", "second", "third"]


class TestEventBusDecorator:
    """on() used as a decorator."""

    def test_decorator_mode(self):
        bus = EventBus()
        received = []

        @bus.on(_TestEvent)
        def handler(event: _TestEvent):
            received.append(event.value)

        bus.emit(_TestEvent(value=99))
        assert received == [99]

    def test_decorator_returns_original_function(self):
        bus = EventBus()

        @bus.on(_TestEvent)
        def handler(event: _TestEvent):
            return event.value * 2

        # The function itself is still usable
        assert handler(_TestEvent(value=5)) == 10


class TestEventBusOnce:
    """One-shot handler auto-removal."""

    def test_once_fires_once(self):
        bus = EventBus()
        received = []
        bus.once(_TestEvent, lambda e: received.append(e.value))

        bus.emit(_TestEvent(value=1))
        bus.emit(_TestEvent(value=2))
        assert received == [1]

    def test_once_removed_from_count(self):
        bus = EventBus()
        bus.once(_TestEvent, lambda e: None)
        assert bus.handler_count(_TestEvent) == 1
        bus.emit(_TestEvent(value=0))
        assert bus.handler_count(_TestEvent) == 0

    def test_once_mixed_with_persistent(self):
        bus = EventBus()
        persistent = []
        oneshot = []
        bus.on(_TestEvent, lambda e: persistent.append(e.value))
        bus.once(_TestEvent, lambda e: oneshot.append(e.value))

        bus.emit(_TestEvent(value=1))
        bus.emit(_TestEvent(value=2))

        assert persistent == [1, 2]
        assert oneshot == [1]


class TestEventBusOff:
    """Handler unregistration."""

    def test_off_removes_handler(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e.value)
        bus.on(_TestEvent, handler)
        assert bus.off(_TestEvent, handler) is True
        bus.emit(_TestEvent(value=42))
        assert received == []

    def test_off_nonexistent_returns_false(self):
        bus = EventBus()
        assert bus.off(_TestEvent, lambda e: None) is False

    def test_off_wrong_type_returns_false(self):
        bus = EventBus()
        handler = lambda e: None
        bus.on(_TestEvent, handler)
        assert bus.off(_OtherEvent, handler) is False


class TestEventBusTypeIsolation:
    """Events dispatched by exact type — no inheritance matching."""

    def test_different_types_isolated(self):
        bus = EventBus()
        test_events = []
        other_events = []
        bus.on(_TestEvent, lambda e: test_events.append(e))
        bus.on(_OtherEvent, lambda e: other_events.append(e))

        bus.emit(_TestEvent(value=1))
        bus.emit(_OtherEvent(data="hello"))

        assert len(test_events) == 1
        assert len(other_events) == 1

    def test_subclass_not_dispatched_to_parent(self):
        @dataclass(frozen=True)
        class Parent:
            x: int

        @dataclass(frozen=True)
        class Child(Parent):
            y: int

        bus = EventBus()
        parent_received = []
        bus.on(Parent, lambda e: parent_received.append(e))

        bus.emit(Child(x=1, y=2))
        # Exact type dispatch — parent handler NOT called for Child
        assert parent_received == []


class TestEventBusClear:
    """Clear all or type-specific handlers."""

    def test_clear_all(self):
        bus = EventBus()
        bus.on(_TestEvent, lambda e: None)
        bus.on(_OtherEvent, lambda e: None)
        bus.clear()
        assert bus.handler_count() == 0

    def test_clear_specific_type(self):
        bus = EventBus()
        bus.on(_TestEvent, lambda e: None)
        bus.on(_OtherEvent, lambda e: None)
        bus.clear(_TestEvent)
        assert bus.handler_count(_TestEvent) == 0
        assert bus.handler_count(_OtherEvent) == 1


class TestEventBusHandlerCount:
    """handler_count with and without type filter."""

    def test_count_empty(self):
        bus = EventBus()
        assert bus.handler_count() == 0
        assert bus.handler_count(_TestEvent) == 0

    def test_count_by_type(self):
        bus = EventBus()
        bus.on(_TestEvent, lambda e: None)
        bus.on(_TestEvent, lambda e: None)
        bus.on(_OtherEvent, lambda e: None)
        assert bus.handler_count(_TestEvent) == 2
        assert bus.handler_count(_OtherEvent) == 1
        assert bus.handler_count() == 3


class TestEventBusThreadSafety:
    """Concurrent registration and emission."""

    def test_concurrent_emit(self):
        bus = EventBus()
        results = []
        lock = threading.Lock()

        bus.on(_TestEvent, lambda e: _thread_safe_append(results, lock, e.value))

        threads = []
        for i in range(20):
            t = threading.Thread(target=bus.emit, args=(_TestEvent(value=i),))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(20))

    def test_concurrent_register_and_emit(self):
        bus = EventBus()
        counts = {"emitted": 0}
        lock = threading.Lock()

        def register_and_emit(i: int):
            handler = lambda e: None
            bus.on(_TestEvent, handler)
            bus.emit(_TestEvent(value=i))
            with lock:
                counts["emitted"] += 1

        threads = [
            threading.Thread(target=register_and_emit, args=(i,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counts["emitted"] == 10


def _thread_safe_append(lst, lock, value):
    with lock:
        lst.append(value)


class TestBuiltInEvents:
    """Built-in event types are properly constructed."""

    def test_trace_completed(self):
        e = TraceCompleted(
            trace_id="abc",
            agent_id="agent-1",
            status="Completed",
            total_cost_usd=0.05,
            step_count=3,
            duration_ms=1500,
        )
        assert e.trace_id == "abc"
        assert e.total_cost_usd == 0.05

    def test_budget_alert(self):
        e = BudgetAlert(spent=9.5, budget=10.0, utilization_pct=95.0)
        assert e.utilization_pct == 95.0

    def test_budget_exceeded(self):
        e = BudgetExceeded(spent=10.5, budget=10.0, scope="global")
        assert e.spent > e.budget

    def test_cost_anomaly(self):
        e = CostAnomaly(
            trace_id="t1",
            agent_id="a1",
            cost_usd=5.0,
            average_usd=0.50,
            multiplier=10.0,
        )
        assert e.multiplier == 10.0

    def test_trace_sampled(self):
        e = TraceSampled(agent_id="a1", sampling_rate=0.1)
        assert e.sampling_rate == 0.1

    def test_events_are_frozen(self):
        e = TraceCompleted(
            trace_id="x", agent_id="a", status="Completed",
            total_cost_usd=0.0, step_count=0, duration_ms=0,
        )
        with pytest.raises(AttributeError):
            e.trace_id = "y"  # type: ignore[misc]


class TestDefaultBus:
    """Module-level default bus."""

    def test_get_default_bus_returns_same_instance(self):
        a = get_default_bus()
        b = get_default_bus()
        assert a is b

    def test_default_bus_is_event_bus(self):
        bus = get_default_bus()
        assert isinstance(bus, EventBus)

    def test_default_bus_functional(self):
        bus = get_default_bus()
        received = []

        @bus.on(_TestEvent)
        def h(e):
            received.append(e.value)

        bus.emit(_TestEvent(value=777))
        assert 777 in received

        # Cleanup
        bus.off(_TestEvent, h)
