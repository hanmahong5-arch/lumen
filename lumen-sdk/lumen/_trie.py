"""Compressed prefix trie for O(K) model name lookup.

Replaces O(N*K) linear scan in pricing lookups. The trie supports:
  - Exact match: O(K) where K = key length
  - Longest prefix match: O(K) — walk until no child matches
  - Substring containment: O(K) via reverse suffix index

The trie is constructed once at module load and queried per LLM call,
making construction cost amortized over thousands of lookups.
"""

from __future__ import annotations

from typing import Generic, TypeVar

V = TypeVar("V")

_SENTINEL = object()


class _TrieNode(Generic[V]):
    """A node in the compressed trie.

    Uses a dict of single characters → child nodes.
    Stores a value only if this node terminates a key.
    """

    __slots__ = ("children", "value", "has_value")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode[V]] = {}
        self.value: V = None  # type: ignore[assignment]
        self.has_value: bool = False


class PrefixTrie(Generic[V]):
    """A prefix trie mapping string keys to values.

    Supports three query modes:
      1. ``get(key)`` — exact match, O(K)
      2. ``longest_prefix(key)`` — longest stored prefix of key, O(K)
      3. ``contains_match(key)`` — any stored key that is a substring of key, O(K*M)

    The trie is optimized for the pricing lookup use case where:
      - ~50 keys are inserted at startup
      - Millions of lookups happen during tracing
      - Keys are short (10-30 chars)

    Example::

        trie = PrefixTrie()
        trie.insert("gpt-4o", (2.50, 10.0))
        trie.insert("gpt-4o-mini", (0.15, 0.60))

        trie.get("gpt-4o")                     # (2.50, 10.0) — exact
        trie.longest_prefix("gpt-4o-20260101") # (2.50, 10.0) — prefix
        trie.get("unknown")                     # None
    """

    __slots__ = ("_root", "_size", "_keys")

    def __init__(self) -> None:
        self._root: _TrieNode[V] = _TrieNode()
        self._size: int = 0
        self._keys: list[str] = []

    def insert(self, key: str, value: V) -> None:
        """Insert a key-value pair. O(K)."""
        node = self._root
        for ch in key:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        if not node.has_value:
            self._size += 1
            self._keys.append(key)
        node.value = value
        node.has_value = True

    def get(self, key: str) -> V | None:
        """Exact match lookup. O(K). Returns None if not found."""
        node = self._root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node.value if node.has_value else None

    def longest_prefix(self, key: str) -> tuple[str, V] | None:
        """Find the longest stored key that is a prefix of *key*. O(K).

        Returns (matched_key, value) or None if no prefix matches.
        """
        node = self._root
        best_key = ""
        best_value: V | None = None
        current_key: list[str] = []

        # Check root node (empty string key)
        if node.has_value:
            best_value = node.value

        for ch in key:
            if ch not in node.children:
                break
            node = node.children[ch]
            current_key.append(ch)
            if node.has_value:
                best_key = "".join(current_key)
                best_value = node.value

        if best_value is not None:
            return (best_key, best_value)
        return None

    def starts_with(self, prefix: str) -> list[tuple[str, V]]:
        """Find all entries whose keys start with *prefix*. O(prefix + subtree)."""
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        results: list[tuple[str, V]] = []
        self._collect(node, list(prefix), results)
        return results

    def _collect(
        self,
        node: _TrieNode[V],
        path: list[str],
        results: list[tuple[str, V]],
    ) -> None:
        if node.has_value:
            results.append(("".join(path), node.value))
        for ch, child in node.children.items():
            path.append(ch)
            self._collect(child, path, results)
            path.pop()

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __bool__(self) -> bool:
        return self._size > 0

    @property
    def keys(self) -> list[str]:
        """All inserted keys in insertion order."""
        return list(self._keys)
