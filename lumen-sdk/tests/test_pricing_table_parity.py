"""Rust pricing.rs is the authority — Python PRICING must contain the same keys."""

from __future__ import annotations

import re
from pathlib import Path

from lumen.pricing import PRICING, estimate_cost, is_fallback_pricing

RUST_PRICING = (
    Path(__file__).resolve().parent.parent.parent
    / "lumen-core"
    / "src"
    / "pricing.rs"
)


def _rust_prefixes() -> set[str]:
    text = RUST_PRICING.read_text(encoding="utf-8")
    # Only the static PRICING_TABLE — ignore #[cfg(test)] examples.
    table = text.split("const PRICING_TABLE")[1].split("];")[0]
    return set(re.findall(r'\("([a-z0-9.\-]+)",\s*[0-9.]+,\s*[0-9.]+\)', table))


def test_python_pricing_covers_rust_table() -> None:
    rust = _rust_prefixes()
    missing = sorted(rust - set(PRICING))
    assert not missing, f"Python PRICING missing Rust keys: {missing}"


def test_deepseek_chat_not_fallback() -> None:
    assert not is_fallback_pricing("deepseek-chat")
    assert abs(estimate_cost("deepseek-chat", 1000, 0) - 0.27) < 1e-9


def test_unknown_model_is_fallback() -> None:
    assert is_fallback_pricing("totally-unknown-xyz-99")
