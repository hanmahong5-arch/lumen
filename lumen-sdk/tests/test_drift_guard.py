"""Rust<->Python drift guard.

Parses the SAME shared fixture that the Rust test suite parses
(lumen-core/tests/fixtures/trace_examples.json) and asserts the same key
fields. If the vendored Rust AgentTrace and this Python dataclass diverge in
how they read the on-disk Kova trace shape, one of the two suites breaks here.

The fixture is canonical and lives in lumen-core; this test reaches it via a
relative path so there is exactly one source of truth for both languages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.trace_reader import _parse_trace

# lumen-sdk/tests/ -> repo root -> lumen-core/tests/fixtures/trace_examples.json
FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "lumen-core"
    / "tests"
    / "fixtures"
    / "trace_examples.json"
)


@pytest.fixture(scope="module")
def examples() -> dict:
    """Load the shared fixture, skipping the leading _comment key."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {name: _parse_trace(body) for name, body in raw.items() if not name.startswith("_")}


def test_fixture_exists() -> None:
    assert FIXTURE.is_file(), f"shared fixture missing: {FIXTURE}"


def test_normal_multi_step_parses(examples: dict) -> None:
    t = examples["normal_multi_step"]
    assert t.trace_id == "trace_normal_001"
    assert t.agent_id == "research-agent"
    assert t.status == "Completed"
    assert len(t.steps) == 3
    assert t.total_tokens.total() == 1550
    # A non-zero total_cost_usd must survive verbatim.
    assert abs(t.total_cost_usd - 0.0123) < 1e-12
    # Fresh run: no parent.
    assert t.parent_trace_id is None
    # Current trace carries the on-disk schema version.
    assert t.schema_version == 1
    # First step metadata preserved (this is the bug the fix addresses).
    assert t.steps[0].metadata == [("phase", "plan")]


def test_recovered_run_has_parent_and_recovery_marker(examples: dict) -> None:
    t = examples["recovered_with_parent"]
    assert t.trace_id == "trace_recovered_002"
    # parent_trace_id links back to the crashed run.
    assert t.parent_trace_id == "trace_normal_001"
    # Current trace carries the on-disk schema version.
    assert t.schema_version == 1

    # First step is a recovery Checkpoint carrying the recovery markers.
    first = t.steps[0]
    assert first.step_type == "Checkpoint"
    assert first.checkpoint_iteration == 4
    assert first.metadata == [("recovery", "true"), ("resumed_at_iteration", "4")]


def test_legacy_trace_without_parent_field_still_loads(examples: dict) -> None:
    t = examples["legacy_no_parent_field"]
    assert t.trace_id == "trace_legacy_003"
    # JSON has no parent_trace_id key at all -> defaults to None.
    assert t.parent_trace_id is None
    # No schema_version key either -> defaults to 0 (pre-versioning trace).
    assert t.schema_version == 0
    assert t.status == "Completed"


def test_failed_trace_parses_status_with_reason(examples: dict) -> None:
    # The {"Failed": "tool timeout"} status shape is the most consumer-breaking
    # variant: a consumer that only handles plain-string statuses will silently
    # misread it. Both the Rust and Python parsers must surface the reason.
    t = examples["failed_trace"]
    assert t.trace_id == "trace_failed_004"
    # _parse_trace maps {"Failed": reason} -> status="Failed", failure_reason=reason.
    assert t.status == "Failed", f"expected status 'Failed', got {t.status!r}"
    assert t.failure_reason == "tool timeout", (
        f"expected failure_reason 'tool timeout', got {t.failure_reason!r}"
    )
    assert t.schema_version == 1
    assert len(t.steps) == 3
    # Last step is an Error step.
    last = t.steps[-1]
    assert last.step_type == "Error"
    assert last.error_message == "tool timeout"
