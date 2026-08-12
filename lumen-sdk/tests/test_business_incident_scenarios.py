"""Sharp business-incident regression tests for the Lumen SDK.

Each test here targets a concrete, customer-visible failure mode rather
than a code path for its own sake:

  - Tenant tag bleed: two customers' agents run concurrently and their
    billing/cost trace files get each other's `user_id` or content —
    breaks per-tenant cost attribution and can leak one customer's prompts
    into another customer's trace history.
  - Observability taking down business logic: Lumen wraps a real business
    function; if something inside the tracing/finalize internals throws,
    the customer's actual result (e.g. "payment confirmed") must still
    reach the caller instead of being replaced by an unrelated
    instrumentation exception.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lumen._context import set_active_session
from lumen.config import LumenConfig
from lumen.instrument import get_active_session, traced_fn


def _run_isolated(coro: object) -> None:
    """Run *coro* on a private event loop without touching the process-wide
    "current event loop" pointer or leaking a stray active trace session.

    ``asyncio.run()`` clears the global current-loop pointer on exit
    (``set_event_loop(None)``), which breaks other test modules still using
    the legacy ``asyncio.get_event_loop().run_until_complete(...)`` idiom
    depending on collection order. Using a scratch loop directly avoids
    that side effect; the explicit ``set_active_session(None)`` afterwards
    is a defensive belt-and-braces reset so this test can never leave a
    dangling active session for whichever test runs next.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)  # type: ignore[arg-type]
    finally:
        loop.close()
        set_active_session(None)


class TestTenantTagIsolation:
    """Concurrent traces from different tenants must never cross-label."""

    def test_user_id_tag_never_bleeds_across_concurrent_traces(
        self, tmp_path: Path,
    ) -> None:
        """Business accident this guards against: tenant A's LLM spend gets
        billed to tenant B (or vice versa) because concurrent traces shared
        state and one trace file ended up tagged with the wrong `user_id` —
        a cost-attribution and data-isolation failure a paying customer
        would notice on their invoice.
        """
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(tmp_path)},
        })

        async def run_for_tenant(tenant: str, secret_marker: str) -> None:
            @traced_fn(f"agent-for-{tenant}", user_id=tenant, config=config)
            async def handle_request() -> str:
                session = get_active_session()
                assert session is not None
                # Simulate work interleaving with other concurrent tenants.
                await asyncio.sleep(0.001)
                session.add_llm_step(
                    model="gpt-4o",
                    prompt_tokens=10,
                    completion_tokens=5,
                    finish_reason="stop",
                    duration_ms=50,
                    started_at_ms=1_000_000,
                    output_content=secret_marker,
                )
                await asyncio.sleep(0.001)
                return secret_marker

            result = await handle_request()
            assert result == secret_marker

        async def main() -> None:
            await asyncio.gather(
                *(
                    run_for_tenant(f"tenant-{i}", f"tenant-{i}-secret-payload")
                    for i in range(8)
                )
            )

        _run_isolated(main())

        trace_files = list(tmp_path.glob("*.json"))
        assert len(trace_files) == 8, "one trace file per concurrent tenant call"

        for path in trace_files:
            data = json.loads(path.read_text())
            tagged_user = data["user_id"]
            trace_blob = json.dumps(data)
            for i in range(8):
                other_marker = f"tenant-{i}-secret-payload"
                if f"tenant-{i}" == tagged_user:
                    continue
                assert other_marker not in trace_blob, (
                    f"trace tagged user_id={tagged_user!r} contains "
                    f"tenant-{i}'s payload — tenant tag bled across "
                    f"concurrent traces"
                )


class TestObservabilityNeverBlocksBusiness:
    """Instrumentation internals failing must not eat the business result."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "CONFIRMED real defect (found by this test, not a test bug): "
            "TraceSession.finalize() (lumen/_context.py) calls "
            "self._budget_tracker.finish_run(self.trace_id) with no "
            "try/except. traced_fn()'s sync_wrapper does "
            "`with trace(...): return fn(...)`; when finalize() raises "
            "from inside that context manager's __exit__, Python replaces "
            "the pending `return` value with the new exception per normal "
            "with-statement semantics — so a successful business result "
            "(e.g. a confirmed order) is silently discarded and the caller "
            "sees an unrelated internal exception instead. Every other "
            "finalize() step (writer, event emission, anomaly detection) "
            "is wrapped in try/except specifically to prevent this; the "
            "budget-tracker call is the one gap. A second-order effect of "
            "the same gap: trace()'s finally block is "
            "`session.finalize(); set_active_session(previous)` — when "
            "finalize() raises, set_active_session(previous) never runs "
            "either, so the broken session stays the process's 'active "
            "session' indefinitely (this test cleans that up itself in "
            "its own finally so it doesn't corrupt other tests). "
            "xfail(strict=True) so this stays visible until fixed, and "
            "fails loudly (informing whoever removes the gap) if "
            "finalize() is ever hardened. Not fixed here per this "
            "review's scope (tests only)."
        ),
    )
    def test_traced_fn_return_value_survives_finalize_internal_failure(
        self, tmp_path: Path,
    ) -> None:
        """Business accident this guards against: a customer's order/payment
        function runs and succeeds, but an unrelated bug inside Lumen's own
        bookkeeping (budget tracker, in this case) throws while finalizing
        the trace — and the caller receives that internal exception instead
        of the real "order confirmed" result. Observability must be able to
        fail without silently discarding a successful business outcome.
        """
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(tmp_path)},
        })

        broken_budget_tracker = MagicMock()
        broken_budget_tracker.finish_run.side_effect = RuntimeError(
            "unrelated internal bookkeeping bug"
        )

        @traced_fn("checkout-agent", config=config)
        def process_order() -> dict[str, str]:
            session = get_active_session()
            assert session is not None
            # Simulate an instrumentation-internal component breaking,
            # independent of anything the business logic below does.
            session._budget_tracker = broken_budget_tracker
            return {"order_id": "ORD-42", "status": "confirmed"}

        try:
            result = process_order()
            assert result == {"order_id": "ORD-42", "status": "confirmed"}, (
                "the real business result must reach the caller even "
                "though an internal Lumen component raised while "
                "finalizing the trace"
            )
        finally:
            # See xfail reason: finalize() raising also skips
            # `set_active_session(previous)` in trace()'s finally block,
            # so this cleans up the process-wide active-session state
            # this test's forced failure would otherwise leave dangling.
            set_active_session(None)
