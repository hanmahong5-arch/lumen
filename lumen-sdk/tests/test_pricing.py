"""Tests for pricing module."""

from __future__ import annotations

import pytest

from lumen.pricing import PRICING, estimate_cost, register_custom_pricing, _CUSTOM_TRIE


def test_exact_match() -> None:
    # claude-sonnet-4-6: $3/$15 per 1K
    cost = estimate_cost("claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=100)
    expected = (1000 * 3.0 + 100 * 15.0) / 1000.0  # $3.00 + $1.50 = $4.50
    assert abs(cost - expected) < 0.001


def test_prefix_match() -> None:
    # "claude-sonnet-4-6-20260301" should match "claude-sonnet-4-6"
    cost = estimate_cost("claude-sonnet-4-6-20260301", prompt_tokens=1000, completion_tokens=0)
    expected = 1000 * 3.0 / 1000.0  # $3.00
    assert abs(cost - expected) < 0.001


def test_case_insensitive() -> None:
    cost = estimate_cost("Claude-Sonnet-4-6", prompt_tokens=1000, completion_tokens=0)
    expected = 1000 * 3.0 / 1000.0
    assert abs(cost - expected) < 0.001


def test_gpt4o_pricing() -> None:
    # gpt-4o: $2.50/$10 per 1K
    cost = estimate_cost("gpt-4o", prompt_tokens=500, completion_tokens=200)
    expected = (500 * 2.5 + 200 * 10.0) / 1000.0
    assert abs(cost - expected) < 0.001


def test_fallback_for_unknown_model() -> None:
    cost = estimate_cost("totally-unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
    # Fallback: $1.0/$3.0 per 1K
    expected = (1000 * 1.0 + 500 * 3.0) / 1000.0
    assert abs(cost - expected) < 0.001


def test_zero_tokens() -> None:
    cost = estimate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
    assert cost == 0.0


def test_family_match_in_path() -> None:
    # Model names with provider prefix
    cost = estimate_cost("accounts/fireworks/models/llama-3.1-70b", prompt_tokens=1000, completion_tokens=0)
    expected = 1000 * 0.88 / 1000.0
    assert abs(cost - expected) < 0.001


def test_pricing_table_has_entries() -> None:
    assert len(PRICING) >= 20


def test_prefix_match_with_date_suffix() -> None:
    """claude-sonnet-4-6-20260301 should match claude-sonnet-4-6 pricing."""
    base_cost = estimate_cost("claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=100)
    dated_cost = estimate_cost("claude-sonnet-4-6-20260301", prompt_tokens=1000, completion_tokens=100)
    assert abs(base_cost - dated_cost) < 0.0001


def test_prefix_match_opus() -> None:
    """claude-opus-4-6-20260415 should match claude-opus-4-6."""
    cost = estimate_cost("claude-opus-4-6-20260415", prompt_tokens=1000, completion_tokens=0)
    expected = 1000 * 15.0 / 1000.0  # $15/1K
    assert abs(cost - expected) < 0.001


def test_deepseek_models() -> None:
    cost_v3 = estimate_cost("deepseek-v3", prompt_tokens=1000, completion_tokens=1000)
    expected_v3 = (1000 * 0.27 + 1000 * 1.10) / 1000.0
    assert abs(cost_v3 - expected_v3) < 0.001

    cost_r1 = estimate_cost("deepseek-r1", prompt_tokens=1000, completion_tokens=1000)
    expected_r1 = (1000 * 0.55 + 1000 * 2.19) / 1000.0
    assert abs(cost_r1 - expected_r1) < 0.001


def test_gemini_models() -> None:
    cost = estimate_cost("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 * 1.25 + 500 * 10.0) / 1000.0
    assert abs(cost - expected) < 0.001


def test_unknown_model_returns_fallback_not_zero() -> None:
    """Unknown model uses fallback pricing ($1/$3 per 1K), not 0.0."""
    cost = estimate_cost("nonexistent-model-v99", prompt_tokens=100, completion_tokens=0)
    assert cost > 0.0  # Fallback, not zero
    expected = 100 * 1.0 / 1000.0  # $0.10
    assert abs(cost - expected) < 0.001


def test_whitespace_stripped() -> None:
    cost = estimate_cost("  gpt-4o  ", prompt_tokens=1000, completion_tokens=0)
    expected = 1000 * 2.50 / 1000.0
    assert abs(cost - expected) < 0.001


def test_every_model_in_table_has_valid_prices() -> None:
    """All entries in PRICING table must have non-negative prices."""
    for model, (prompt_price, completion_price) in PRICING.items():
        assert prompt_price >= 0.0, f"{model} has negative prompt price"
        assert completion_price >= 0.0, f"{model} has negative completion price"


class TestCustomPricing:
    """Custom pricing overrides built-in rates."""

    def setup_method(self):
        # Clear custom pricing before each test
        register_custom_pricing({})

    def teardown_method(self):
        register_custom_pricing({})

    def test_custom_exact_match_overrides_builtin(self):
        register_custom_pricing({"gpt-4o": (0.50, 2.00)})
        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        expected = (1000 * 0.50 + 1000 * 2.00) / 1000.0
        assert cost == pytest.approx(expected)

    def test_custom_model_not_in_builtin(self):
        register_custom_pricing({"my-local-llama": (0.0, 0.0)})
        cost = estimate_cost("my-local-llama", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0

    def test_custom_prefix_match(self):
        register_custom_pricing({"qwen-72b": (0.10, 0.30)})
        cost = estimate_cost("qwen-72b-instruct-v2", prompt_tokens=1000, completion_tokens=0)
        expected = 1000 * 0.10 / 1000.0
        assert cost == pytest.approx(expected)

    def test_custom_takes_priority_over_builtin(self):
        """Custom pricing checked first, even if builtin has the key."""
        register_custom_pricing({"claude-sonnet-4-6": (0.01, 0.02)})
        cost = estimate_cost("claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=1000)
        expected = (1000 * 0.01 + 1000 * 0.02) / 1000.0
        assert cost == pytest.approx(expected)

    def test_clear_custom_falls_back_to_builtin(self):
        register_custom_pricing({"gpt-4o": (0.01, 0.01)})
        register_custom_pricing({})  # clear
        cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=0)
        expected = 1000 * 2.50 / 1000.0  # built-in rate
        assert cost == pytest.approx(expected)
