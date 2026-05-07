"""O(1) LRU cache using doubly-linked list + hash map.

A classic data structure for bounded caching. Both get() and put() are
O(1) amortized. When the cache is full, the least-recently-used entry
is evicted.

Internal structure:
  - Hash map: key → Node  (O(1) lookup)
  - Doubly-linked list: tracks access order (head = MRU, tail = LRU)
  - Sentinel head/tail nodes eliminate null-check edge cases

Memory: ~100 bytes per entry overhead (2 pointers + dict entry).

Usage::

    cache = LRUCache(capacity=100)
    cache.put("trace-abc", expensive_load("trace-abc"))
    data = cache.get("trace-abc")  # O(1), moves to front
"""

from __future__ import annotations

from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class _Node(Generic[K, V]):
    """Doubly-linked list node."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.prev: _Node[K, V] | None = None
        self.next: _Node[K, V] | None = None


class LRUCache(Generic[K, V]):
    """Bounded LRU cache with O(1) get, put, and eviction.

    When capacity is exceeded, the least-recently-used entry is evicted.
    Supports ``get()``, ``put()``, ``peek()`` (no promotion), ``delete()``,
    ``clear()``, and iteration from MRU to LRU.

    Example::

        cache = LRUCache(3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")       # promotes "a" to MRU
        cache.put("d", 4)    # evicts "b" (LRU)
        list(cache)           # [("a", 1), ("d", 4), ... ] — MRU order
    """

    __slots__ = ("_capacity", "_map", "_head", "_tail")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._map: dict[K, _Node[K, V]] = {}

        # Sentinel nodes — simplify insert/remove (no null checks)
        self._head: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._tail: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head

    def get(self, key: K) -> V | None:
        """Retrieve value by key. Promotes to MRU. O(1).

        Returns None if key is not in cache.
        """
        node = self._map.get(key)
        if node is None:
            return None
        self._move_to_front(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        """Insert or update a key-value pair. O(1).

        If key exists, updates value and promotes to MRU.
        If cache is full, evicts the LRU entry first.
        """
        node = self._map.get(key)
        if node is not None:
            # Update existing
            node.value = value
            self._move_to_front(node)
            return

        # New entry
        if len(self._map) >= self._capacity:
            self._evict_lru()

        new_node = _Node(key, value)
        self._map[key] = new_node
        self._add_after_head(new_node)

    def peek(self, key: K) -> V | None:
        """Retrieve value without promoting to MRU. O(1)."""
        node = self._map.get(key)
        return node.value if node is not None else None

    def delete(self, key: K) -> bool:
        """Remove a key from the cache. Returns True if found. O(1)."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._remove(node)
        return True

    def clear(self) -> None:
        """Remove all entries. O(1)."""
        self._map.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: K) -> bool:
        return key in self._map

    def __bool__(self) -> bool:
        return len(self._map) > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return len(self._map) >= self._capacity

    def keys(self) -> list[K]:
        """All keys in MRU-to-LRU order. O(N)."""
        return [k for k, _ in self]

    def values(self) -> list[V]:
        """All values in MRU-to-LRU order. O(N)."""
        return [v for _, v in self]

    def __iter__(self):
        """Iterate (key, value) pairs from MRU to LRU. O(N)."""
        node = self._head.next
        while node is not self._tail:
            yield (node.key, node.value)
            node = node.next

    def mru(self) -> tuple[K, V] | None:
        """Return the most recently used entry. O(1)."""
        if self._head.next is self._tail:
            return None
        node = self._head.next
        return (node.key, node.value)

    def lru(self) -> tuple[K, V] | None:
        """Return the least recently used entry. O(1)."""
        if self._tail.prev is self._head:
            return None
        node = self._tail.prev
        return (node.key, node.value)

    # ── Internal linked-list operations ─────────────────────────────────

    def _add_after_head(self, node: _Node[K, V]) -> None:
        """Insert node right after the head sentinel (MRU position)."""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore[union-attr]
        self._head.next = node

    def _remove(self, node: _Node[K, V]) -> None:
        """Unlink node from the list."""
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]

    def _move_to_front(self, node: _Node[K, V]) -> None:
        """Move existing node to MRU position."""
        self._remove(node)
        self._add_after_head(node)

    def _evict_lru(self) -> None:
        """Remove the LRU node (just before tail sentinel)."""
        lru_node = self._tail.prev
        if lru_node is self._head:
            return  # empty
        self._remove(lru_node)  # type: ignore[arg-type]
        del self._map[lru_node.key]  # type: ignore[union-attr]
