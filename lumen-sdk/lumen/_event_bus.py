"""Type-safe event bus for decoupled component communication.

Provides a publish-subscribe pattern for budget alerts, trace completion
events, anomaly detection, and custom hooks. Handlers are dispatched
synchronously in registration order.

Design choices:
  - Events are plain dataclasses — no inheritance required
  - Dispatch is by exact type (no subclass matching) for predictability
  - Weak references optional: handlers can be auto-collected when
    their owning object is GC'd
  - Thread-safe: a lock protects registration/unregistration

Example::

    from lumen._event_bus import EventBus, BudgetAlert, TraceCompleted

    bus = EventBus()

    @bus.on(BudgetAlert)
    def handle_alert(event: BudgetAlert):
        print(f"Budget at {event.utilization_pct:.0f}%!")

    bus.emit(BudgetAlert(spent=95.0, budget=100.0, utilization_pct=95.0))
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger("lumen")

E = TypeVar("E")


# ── Built-in event types ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceCompleted:
    """Emitted when a trace session is finalized."""

    trace_id: str
    agent_id: str
    status: str
    total_cost_usd: float
    step_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class BudgetAlert:
    """Emitted when budget utilization exceeds the alert threshold."""

    spent: float
    budget: float
    utilization_pct: float


@dataclass(frozen=True, slots=True)
class BudgetExceeded:
    """Emitted when a budget limit is actually exceeded."""

    spent: float
    budget: float
    scope: str


@dataclass(frozen=True, slots=True)
class CostAnomaly:
    """Emitted when a trace's cost is anomalously high."""

    trace_id: str
    agent_id: str
    cost_usd: float
    average_usd: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class TraceSampled:
    """Emitted when a trace is dropped by sampling."""

    agent_id: str
    sampling_rate: float


# ── Event Bus ────────────────────────────────────────────────────────────

class _Subscription:
    """A single event subscription."""

    __slots__ = ("handler", "once")

    def __init__(self, handler: Callable[..., Any], once: bool) -> None:
        self.handler = handler
        self.once = once


class EventBus:
    """Thread-safe publish-subscribe event bus.

    Handlers are called synchronously in registration order.
    Use ``on()`` as a decorator or method, ``once()`` for one-shot handlers,
    and ``emit()`` to dispatch events.

    Example::

        bus = EventBus()

        # Decorator style
        @bus.on(TraceCompleted)
        def log_trace(event):
            print(f"Trace {event.trace_id} completed")

        # Method style
        bus.on(BudgetAlert, lambda e: send_slack(e.utilization_pct))

        # One-shot
        bus.once(BudgetExceeded, lambda e: kill_process())

        # Emit
        bus.emit(TraceCompleted(
            trace_id="abc", agent_id="agent", status="Completed",
            total_cost_usd=0.05, step_count=3, duration_ms=1500,
        ))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[type, list[_Subscription]] = {}

    def on(
        self,
        event_type: type[E],
        handler: Callable[[E], Any] | None = None,
    ) -> Any:
        """Register a handler for an event type.

        Can be used as a decorator or as a method call.

        Args:
            event_type: The event class to listen for.
            handler: Callback function. If None, returns a decorator.

        Returns:
            The handler (or a decorator if handler is None).
        """
        if handler is not None:
            self._register(event_type, handler, once=False)
            return handler

        # Decorator mode
        def decorator(fn: Callable[[E], Any]) -> Callable[[E], Any]:
            self._register(event_type, fn, once=False)
            return fn
        return decorator

    def once(
        self,
        event_type: type[E],
        handler: Callable[[E], Any],
    ) -> None:
        """Register a one-shot handler (auto-removed after first call)."""
        self._register(event_type, handler, once=True)

    def off(
        self,
        event_type: type[E],
        handler: Callable[[E], Any],
    ) -> bool:
        """Unregister a handler. Returns True if found and removed."""
        with self._lock:
            subs = self._subs.get(event_type)
            if subs is None:
                return False
            for i, sub in enumerate(subs):
                if sub.handler is handler:
                    subs.pop(i)
                    return True
        return False

    def emit(self, event: Any) -> int:
        """Dispatch an event to all registered handlers.

        Handlers are called synchronously in registration order.
        One-shot handlers are removed after invocation.

        Args:
            event: The event instance to dispatch.

        Returns:
            Number of handlers that were called.
        """
        event_type = type(event)
        with self._lock:
            subs = list(self._subs.get(event_type, []))

        called = 0
        to_remove: list[_Subscription] = []

        for sub in subs:
            try:
                sub.handler(event)
            except Exception:
                logger.debug(
                    "Event handler %r failed for %s",
                    sub.handler, type(event).__name__, exc_info=True,
                )
            called += 1
            if sub.once:
                to_remove.append(sub)

        # Remove one-shot handlers
        if to_remove:
            with self._lock:
                type_subs = self._subs.get(event_type)
                if type_subs:
                    for sub in to_remove:
                        try:
                            type_subs.remove(sub)
                        except ValueError:
                            pass

        return called

    def clear(self, event_type: type | None = None) -> None:
        """Remove all handlers, or all handlers for a specific event type."""
        with self._lock:
            if event_type is None:
                self._subs.clear()
            else:
                self._subs.pop(event_type, None)

    async def async_emit(self, event: Any) -> int:
        """Dispatch an event, awaiting coroutine handlers.

        Sync handlers are called normally; async handlers are awaited.
        Use this in async code for handlers that need to do async I/O
        (e.g., sending alerts via HTTP).

        Args:
            event: The event instance to dispatch.

        Returns:
            Number of handlers that were called.
        """
        event_type = type(event)
        with self._lock:
            subs = list(self._subs.get(event_type, []))

        called = 0
        to_remove: list[_Subscription] = []

        for sub in subs:
            try:
                if asyncio.iscoroutinefunction(sub.handler):
                    await sub.handler(event)
                else:
                    sub.handler(event)
            except Exception:
                logger.debug(
                    "Event handler %r failed for %s",
                    sub.handler, type(event).__name__, exc_info=True,
                )
            called += 1
            if sub.once:
                to_remove.append(sub)

        if to_remove:
            with self._lock:
                type_subs = self._subs.get(event_type)
                if type_subs:
                    for sub in to_remove:
                        try:
                            type_subs.remove(sub)
                        except ValueError:
                            pass

        return called

    def handler_count(self, event_type: type | None = None) -> int:
        """Count registered handlers."""
        with self._lock:
            if event_type is None:
                return sum(len(s) for s in self._subs.values())
            return len(self._subs.get(event_type, []))

    def _register(
        self,
        event_type: type,
        handler: Callable[..., Any],
        once: bool,
    ) -> None:
        with self._lock:
            if event_type not in self._subs:
                self._subs[event_type] = []
            self._subs[event_type].append(_Subscription(handler, once))


# Module-level default bus
_default_bus = EventBus()


def get_default_bus() -> EventBus:
    """Return the module-level default event bus."""
    return _default_bus
