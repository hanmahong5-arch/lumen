"""Tests for PrefixTrie — O(K) model name lookup data structure.

Covers: insert, get (exact), longest_prefix, starts_with, edge cases,
capacity, duplicate inserts, unicode, and performance characteristics.
"""

from __future__ import annotations

import pytest

from lumen._trie import PrefixTrie


class TestTrieInsertAndGet:
    """Basic insert + exact match."""

    def test_insert_and_retrieve(self):
        t = PrefixTrie[float]()
        t.insert("gpt-4o", 2.50)
        assert t.get("gpt-4o") == 2.50

    def test_get_missing_returns_none(self):
        t = PrefixTrie[str]()
        t.insert("a", "val")
        assert t.get("b") is None

    def test_empty_trie_get(self):
        t = PrefixTrie[int]()
        assert t.get("anything") is None

    def test_insert_overwrites_value(self):
        t = PrefixTrie[int]()
        t.insert("key", 1)
        t.insert("key", 2)
        assert t.get("key") == 2
        assert len(t) == 1  # size unchanged on overwrite

    def test_multiple_keys(self):
        t = PrefixTrie[tuple[float, float]]()
        t.insert("gpt-4o", (2.50, 10.0))
        t.insert("gpt-4o-mini", (0.15, 0.60))
        t.insert("claude-3-opus", (15.0, 75.0))
        assert t.get("gpt-4o") == (2.50, 10.0)
        assert t.get("gpt-4o-mini") == (0.15, 0.60)
        assert t.get("claude-3-opus") == (15.0, 75.0)
        assert len(t) == 3

    def test_empty_key(self):
        t = PrefixTrie[str]()
        t.insert("", "empty")
        assert t.get("") == "empty"
        assert t.get("a") is None

    def test_single_char_keys(self):
        t = PrefixTrie[int]()
        for i, ch in enumerate("abcdefghij"):
            t.insert(ch, i)
        assert len(t) == 10
        assert t.get("e") == 4


class TestTrieLongestPrefix:
    """Longest prefix match — the core pricing lookup pattern."""

    def test_exact_is_longest_prefix(self):
        t = PrefixTrie[float]()
        t.insert("gpt-4o", 2.50)
        result = t.longest_prefix("gpt-4o")
        assert result == ("gpt-4o", 2.50)

    def test_prefix_match_with_suffix(self):
        t = PrefixTrie[float]()
        t.insert("gpt-4o", 2.50)
        result = t.longest_prefix("gpt-4o-20260301")
        assert result == ("gpt-4o", 2.50)

    def test_longest_wins(self):
        t = PrefixTrie[str]()
        t.insert("gpt", "family")
        t.insert("gpt-4", "model")
        t.insert("gpt-4o", "variant")
        result = t.longest_prefix("gpt-4o-mini-preview")
        assert result == ("gpt-4o", "variant")

    def test_no_prefix_match(self):
        t = PrefixTrie[int]()
        t.insert("gpt-4o", 1)
        assert t.longest_prefix("claude-3") is None

    def test_partial_key_no_match(self):
        """'gpt' is NOT a stored key, so no prefix match for 'gpt-5'."""
        t = PrefixTrie[int]()
        t.insert("gpt-4o", 1)
        assert t.longest_prefix("gpt-5") is None

    def test_empty_key_prefix(self):
        t = PrefixTrie[str]()
        t.insert("", "root")
        result = t.longest_prefix("anything")
        assert result == ("", "root")

    def test_claude_version_matching(self):
        """Real-world scenario: claude-sonnet-4-6-20260301 → claude-sonnet-4-6."""
        t = PrefixTrie[tuple[float, float]]()
        t.insert("claude-sonnet-4-6", (3.0, 15.0))
        t.insert("claude-opus-4-6", (15.0, 75.0))
        t.insert("claude-haiku-4-5", (0.80, 4.0))

        r = t.longest_prefix("claude-sonnet-4-6-20260301")
        assert r is not None
        assert r[0] == "claude-sonnet-4-6"
        assert r[1] == (3.0, 15.0)


class TestTrieStartsWith:
    """Subtree collection via prefix scan."""

    def test_starts_with_exact(self):
        t = PrefixTrie[int]()
        t.insert("abc", 1)
        results = t.starts_with("abc")
        assert len(results) == 1
        assert results[0] == ("abc", 1)

    def test_starts_with_prefix(self):
        t = PrefixTrie[int]()
        t.insert("gpt-4o", 1)
        t.insert("gpt-4o-mini", 2)
        t.insert("gpt-4-turbo", 3)
        t.insert("claude-3", 4)

        results = t.starts_with("gpt-4o")
        keys = {k for k, v in results}
        assert keys == {"gpt-4o", "gpt-4o-mini"}

    def test_starts_with_no_match(self):
        t = PrefixTrie[int]()
        t.insert("abc", 1)
        assert t.starts_with("xyz") == []

    def test_starts_with_empty_prefix(self):
        t = PrefixTrie[int]()
        t.insert("a", 1)
        t.insert("b", 2)
        results = t.starts_with("")
        assert len(results) == 2


class TestTrieContains:
    """__contains__ and __bool__ protocol."""

    def test_contains_existing_key(self):
        t = PrefixTrie[int]()
        t.insert("key", 42)
        assert "key" in t

    def test_not_contains_missing(self):
        t = PrefixTrie[int]()
        t.insert("key", 42)
        assert "other" not in t

    def test_bool_empty(self):
        t = PrefixTrie[int]()
        assert not t

    def test_bool_non_empty(self):
        t = PrefixTrie[int]()
        t.insert("x", 1)
        assert t


class TestTrieKeys:
    """Keys tracking in insertion order."""

    def test_keys_insertion_order(self):
        t = PrefixTrie[int]()
        t.insert("c", 3)
        t.insert("a", 1)
        t.insert("b", 2)
        assert t.keys == ["c", "a", "b"]

    def test_keys_returns_copy(self):
        t = PrefixTrie[int]()
        t.insert("x", 1)
        keys = t.keys
        keys.append("hacked")
        assert t.keys == ["x"]

    def test_overwrite_does_not_duplicate_key(self):
        t = PrefixTrie[int]()
        t.insert("k", 1)
        t.insert("k", 2)
        assert t.keys == ["k"]


class TestTrieEdgeCases:
    """Edge cases: unicode, long keys, special characters."""

    def test_unicode_keys(self):
        t = PrefixTrie[str]()
        t.insert("模型-alpha", "chinese")
        assert t.get("模型-alpha") == "chinese"
        result = t.longest_prefix("模型-alpha-v2")
        assert result == ("模型-alpha", "chinese")

    def test_long_key(self):
        key = "a" * 10_000
        t = PrefixTrie[int]()
        t.insert(key, 42)
        assert t.get(key) == 42

    def test_keys_with_special_chars(self):
        t = PrefixTrie[int]()
        t.insert("model/v1.0", 1)
        t.insert("model:latest", 2)
        t.insert("accounts/fireworks/models/llama", 3)
        assert t.get("model/v1.0") == 1
        assert t.get("model:latest") == 2
        assert t.get("accounts/fireworks/models/llama") == 3

    def test_shared_prefix_independence(self):
        """Keys sharing a prefix are independent — deleting one doesn't affect the other."""
        t = PrefixTrie[int]()
        t.insert("ab", 1)
        t.insert("abc", 2)
        t.insert("abd", 3)
        assert t.get("ab") == 1
        assert t.get("abc") == 2
        assert t.get("abd") == 3
        assert t.get("a") is None


class TestTriePricingLookupScenario:
    """End-to-end test mirroring the pricing.py use case."""

    def _build_pricing_trie(self) -> PrefixTrie[tuple[float, float]]:
        t = PrefixTrie[tuple[float, float]]()
        t.insert("claude-opus-4-6", (15.0, 75.0))
        t.insert("claude-sonnet-4-6", (3.0, 15.0))
        t.insert("claude-haiku-4-5", (0.80, 4.0))
        t.insert("gpt-4o", (2.50, 10.0))
        t.insert("gpt-4o-mini", (0.15, 0.60))
        t.insert("gpt-4-turbo", (10.0, 30.0))
        t.insert("gpt-4", (30.0, 60.0))
        t.insert("o1", (15.0, 60.0))
        t.insert("o1-mini", (3.0, 12.0))
        t.insert("deepseek-v3", (0.27, 1.10))
        t.insert("deepseek-r1", (0.55, 2.19))
        return t

    def test_exact_lookup(self):
        t = self._build_pricing_trie()
        assert t.get("gpt-4o") == (2.50, 10.0)
        assert t.get("deepseek-r1") == (0.55, 2.19)

    def test_versioned_model_prefix_match(self):
        """gpt-4o-20260301 should match gpt-4o, not gpt-4o-mini."""
        t = self._build_pricing_trie()
        result = t.longest_prefix("gpt-4o-20260301")
        assert result is not None
        assert result[0] == "gpt-4o"

    def test_mini_exact_over_prefix(self):
        """gpt-4o-mini should match exactly, not prefix gpt-4o."""
        t = self._build_pricing_trie()
        assert t.get("gpt-4o-mini") == (0.15, 0.60)

    def test_mini_versioned(self):
        """gpt-4o-mini-20260301 should prefix-match gpt-4o-mini."""
        t = self._build_pricing_trie()
        result = t.longest_prefix("gpt-4o-mini-20260301")
        assert result is not None
        assert result[0] == "gpt-4o-mini"
        assert result[1] == (0.15, 0.60)

    def test_unknown_model_no_match(self):
        t = self._build_pricing_trie()
        assert t.get("llama-3.3-70b") is None
        assert t.longest_prefix("llama-3.3-70b") is None

    def test_gpt4_vs_gpt4_turbo(self):
        """gpt-4 and gpt-4-turbo are distinct; gpt-4-turbo-preview → gpt-4-turbo."""
        t = self._build_pricing_trie()
        assert t.get("gpt-4") == (30.0, 60.0)
        result = t.longest_prefix("gpt-4-turbo-preview")
        assert result is not None
        assert result[0] == "gpt-4-turbo"

    def test_all_models_retrievable(self):
        t = self._build_pricing_trie()
        assert len(t) == 11
        for key in t.keys:
            assert t.get(key) is not None
