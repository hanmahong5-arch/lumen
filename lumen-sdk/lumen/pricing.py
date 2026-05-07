"""LLM model pricing table and cost estimation.

Provides per-token pricing for popular models. Uses a PrefixTrie for
O(K) lookup (K = key length) instead of O(N*K) linear scan over all models.

Lookup strategy:
  1. Exact match via trie — O(K)
  2. Longest prefix match via trie — O(K), handles versioned model names
  3. Substring containment scan — O(N*K) fallback for provider-prefixed paths
  4. Fallback pricing — $1.00/$3.00 per 1K tokens
"""

from __future__ import annotations

from lumen._trie import PrefixTrie

# (prompt_price_per_1k_tokens, completion_price_per_1k_tokens) in USD
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude 4.x
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    # Anthropic Claude 3.x
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI GPT-4o family
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-audio": (2.50, 10.0),
    # OpenAI GPT-4
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    # OpenAI GPT-3.5
    "gpt-3.5-turbo": (0.50, 1.50),
    # OpenAI o-series
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o1-preview": (15.0, 60.0),
    "o3": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Google Gemini
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    # Meta Llama
    "llama-3.1-70b": (0.88, 0.88),
    "llama-3.1-8b": (0.18, 0.18),
    "llama-3.3-70b": (0.88, 0.88),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.20, 0.60),
    # DeepSeek
    "deepseek-v3": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
}

# Default pricing for unknown models (conservative estimate)
_FALLBACK_PRICING: tuple[float, float] = (1.0, 3.0)

# Build trie once at module load — amortized over millions of lookups
_PRICING_TRIE: PrefixTrie[tuple[float, float]] = PrefixTrie()
for _model, _prices in PRICING.items():
    _PRICING_TRIE.insert(_model, _prices)


# Custom pricing overlay — set at runtime via register_custom_pricing()
_CUSTOM_TRIE: PrefixTrie[tuple[float, float]] = PrefixTrie()


def register_custom_pricing(pricing: dict[str, tuple[float, float]]) -> None:
    """Register custom model pricing that overrides built-in rates.

    Used by CostConfig.custom_pricing to let users define their own rates
    for self-hosted or fine-tuned models.

    Args:
        pricing: Dict mapping model name → (prompt_per_1k, completion_per_1k).
    """
    global _CUSTOM_TRIE  # noqa: PLW0603
    _CUSTOM_TRIE = PrefixTrie()
    for model, prices in pricing.items():
        _CUSTOM_TRIE.insert(model.lower().strip(), prices)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts.

    Lookup order:
      1. Custom pricing (user-defined) — exact then prefix
      2. Built-in pricing — exact, prefix, then substring
      3. Fallback — $1.00/$3.00 per 1K tokens

    Args:
        model: Model identifier string.
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD.
    """
    prompt_price, completion_price = _lookup_pricing(model)
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000.0


def _lookup_pricing(model: str) -> tuple[float, float]:
    """Look up pricing for a model using trie-based O(K) matching."""
    model_lower = model.lower().strip()

    # 0. Custom pricing — user overrides take priority
    if _CUSTOM_TRIE:
        custom_exact = _CUSTOM_TRIE.get(model_lower)
        if custom_exact is not None:
            return custom_exact
        custom_prefix = _CUSTOM_TRIE.longest_prefix(model_lower)
        if custom_prefix is not None:
            return custom_prefix[1]

    # 1. Exact match — O(K)
    exact = _PRICING_TRIE.get(model_lower)
    if exact is not None:
        return exact

    # 2. Longest prefix match — O(K)
    # e.g. "claude-sonnet-4-6-20260301" → "claude-sonnet-4-6"
    prefix_result = _PRICING_TRIE.longest_prefix(model_lower)
    if prefix_result is not None:
        return prefix_result[1]

    # 3. Substring containment — O(N*K) fallback for provider-prefixed paths
    # e.g. "accounts/fireworks/models/llama-3.1-70b" → "llama-3.1-70b"
    for key in PRICING:
        if key in model_lower:
            return PRICING[key]

    return _FALLBACK_PRICING
