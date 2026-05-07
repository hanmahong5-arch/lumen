"""Tests for LRUCache — O(1) eviction cache with doubly-linked list + hash map.

Covers: put/get/peek/delete, eviction policy, capacity enforcement,
MRU/LRU access, iteration order, edge cases (capacity=1, overwrite).
"""

from __future__ import annotations

import pytest

from lumen._lru_cache import LRUCache


class TestLRUCacheBasics:
    """Core get/put operations."""

    def test_put_and_get(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_returns_none(self):
        cache = LRUCache[str, int](5)
        assert cache.get("missing") is None

    def test_empty_cache(self):
        cache = LRUCache[str, int](5)
        assert len(cache) == 0
        assert not cache
        assert cache.capacity == 5
        assert not cache.is_full

    def test_update_existing_key(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert len(cache) == 1

    def test_multiple_keys(self):
        cache = LRUCache[str, int](10)
        for i in range(5):
            cache.put(f"k{i}", i)
        assert len(cache) == 5
        for i in range(5):
            assert cache.get(f"k{i}") == i

    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            LRUCache[str, int](0)


class TestLRUCacheEviction:
    """LRU eviction policy."""

    def test_evicts_lru_on_overflow(self):
        cache = LRUCache[str, int](3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_access_promotes_and_changes_eviction(self):
        cache = LRUCache[str, int](3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")       # promote "a" to MRU
        cache.put("d", 4)    # now evicts "b" (LRU), not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_update_promotes(self):
        cache = LRUCache[str, int](3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("a", 10)   # update "a", promotes to MRU
        cache.put("d", 4)    # evicts "b"
        assert cache.get("a") == 10
        assert cache.get("b") is None

    def test_capacity_one(self):
        cache = LRUCache[str, int](1)
        cache.put("a", 1)
        assert cache.get("a") == 1
        cache.put("b", 2)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_eviction_chain(self):
        cache = LRUCache[str, int](2)
        cache.put("a", 1)
        cache.put("b", 2)  # full
        cache.put("c", 3)  # evicts a
        cache.put("d", 4)  # evicts b
        assert "a" not in cache
        assert "b" not in cache
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestLRUCachePeek:
    """peek() reads without promoting."""

    def test_peek_returns_value(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 42)
        assert cache.peek("a") == 42

    def test_peek_missing_returns_none(self):
        cache = LRUCache[str, int](5)
        assert cache.peek("x") is None

    def test_peek_does_not_promote(self):
        cache = LRUCache[str, int](3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.peek("a")      # does NOT promote "a"
        cache.put("d", 4)    # still evicts "a" (LRU)
        assert cache.get("a") is None


class TestLRUCacheDelete:
    """Explicit deletion."""

    def test_delete_existing(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert len(cache) == 0

    def test_delete_missing(self):
        cache = LRUCache[str, int](5)
        assert cache.delete("nonexistent") is False

    def test_delete_frees_slot(self):
        cache = LRUCache[str, int](2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.delete("a")
        cache.put("c", 3)  # no eviction needed
        assert cache.get("b") == 2
        assert cache.get("c") == 3


class TestLRUCacheIteration:
    """Iteration in MRU-to-LRU order."""

    def test_iteration_order(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        items = list(cache)
        # MRU first: c was last put
        assert items == [("c", 3), ("b", 2), ("a", 1)]

    def test_iteration_after_access(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # promote "a" to MRU
        keys = cache.keys()
        assert keys[0] == "a"  # MRU
        assert keys[-1] == "b"  # LRU

    def test_empty_iteration(self):
        cache = LRUCache[str, int](5)
        assert list(cache) == []

    def test_values(self):
        cache = LRUCache[str, int](3)
        cache.put("x", 10)
        cache.put("y", 20)
        assert cache.values() == [20, 10]


class TestLRUCacheMRULRU:
    """mru() and lru() accessors."""

    def test_mru_and_lru(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.mru() == ("c", 3)
        assert cache.lru() == ("a", 1)

    def test_mru_lru_empty(self):
        cache = LRUCache[str, int](5)
        assert cache.mru() is None
        assert cache.lru() is None

    def test_single_element(self):
        cache = LRUCache[str, int](5)
        cache.put("only", 42)
        assert cache.mru() == ("only", 42)
        assert cache.lru() == ("only", 42)


class TestLRUCacheClear:
    """Clear all entries."""

    def test_clear(self):
        cache = LRUCache[str, int](5)
        for i in range(5):
            cache.put(f"k{i}", i)
        cache.clear()
        assert len(cache) == 0
        assert cache.mru() is None
        for i in range(5):
            assert cache.get(f"k{i}") is None

    def test_reuse_after_clear(self):
        cache = LRUCache[str, int](2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        cache.put("c", 3)
        assert cache.get("c") == 3
        assert len(cache) == 1


class TestLRUCacheContains:
    """__contains__ protocol."""

    def test_contains(self):
        cache = LRUCache[str, int](5)
        cache.put("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_is_full(self):
        cache = LRUCache[str, int](2)
        assert not cache.is_full
        cache.put("a", 1)
        assert not cache.is_full
        cache.put("b", 2)
        assert cache.is_full


class TestLRUCacheTraceCaching:
    """Simulate the trace-loading cache use case."""

    def test_trace_load_caching(self):
        cache = LRUCache[str, dict](3)
        traces = {
            "t1": {"trace_id": "t1", "steps": [1, 2, 3]},
            "t2": {"trace_id": "t2", "steps": [4, 5]},
            "t3": {"trace_id": "t3", "steps": [6]},
        }
        for tid, data in traces.items():
            cache.put(tid, data)

        # Simulate repeated access pattern
        for _ in range(100):
            hit = cache.get("t1")
            assert hit is not None
            assert hit["trace_id"] == "t1"

        # t1 is now MRU, adding t4 evicts t2 (LRU)
        cache.put("t4", {"trace_id": "t4", "steps": []})
        assert cache.get("t2") is None
        assert cache.get("t1") is not None
