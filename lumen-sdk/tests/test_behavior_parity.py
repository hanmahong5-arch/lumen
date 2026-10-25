"""Rust↔Python behavioral parity for cost + replay.

Materializes shared fixtures, runs CostTracker / ReplayEngine, and asserts
against lumen-core/tests/fixtures/behavior_expectations.json — the same file
the Rust suite (`behavior_parity.rs`) asserts. Catches algorithm drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.cost import CostTracker, _trace_cost
from lumen.replay import ReplayEngine
from lumen.trace_reader import _parse_trace

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "lumen-core" / "tests" / "fixtures"
EXAMPLES = FIXTURES / "trace_examples.json"
EXPECTATIONS = FIXTURES / "behavior_expectations.json"


@pytest.fixture(scope="module")
def examples() -> dict:
    raw = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    return {n: body for n, body in raw.items() if not n.startswith("_")}


@pytest.fixture(scope="module")
def expectations() -> dict:
    raw = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    return {n: body for n, body in raw.items() if not n.startswith("_")}


def test_expectation_coverage(examples: dict, expectations: dict) -> None:
    assert set(examples) == set(expectations)


@pytest.mark.parametrize(
    "name",
    [
        "normal_multi_step",
        "recovered_with_parent",
        "legacy_no_parent_field",
        "failed_trace",
    ],
)
def test_cost_and_replay_match_shared_expectations(
    name: str, examples: dict, expectations: dict, tmp_path: Path
) -> None:
    body = examples[name]
    exp = expectations[name]
    trace = _parse_trace(body)
    assert trace.trace_id == exp["trace_id"]

    cost = _trace_cost(trace)
    assert abs(cost - exp["cost_usd"]) < 1e-9

    trace_path = tmp_path / f"{trace.trace_id}.json"
    trace_path.write_text(json.dumps(body), encoding="utf-8")

    # Wide window so historical fixture timestamps are not filtered out.
    report = CostTracker(trace_dir=str(tmp_path)).report(last="10000d")
    assert report.total_runs == 1
    assert abs(report.total_usd - exp["cost_usd"]) < 1e-9

    replay = ReplayEngine(trace_dir=str(tmp_path)).replay(trace.trace_id)
    assert replay.success is exp["replay_success"]
    assert replay.total_iterations == exp["replay_iterations"]
    assert abs(replay.original_cost_usd - exp["replay_original_cost_usd"]) < 1e-9

    parent = None if replay.recovery is None else replay.recovery.parent_trace_id
    resumed = None if replay.recovery is None else replay.recovery.resumed_at_iteration
    assert parent == exp["replay_parent"]
    assert resumed == exp["replay_resumed_at"]
