"""Built-in redaction patterns and field-level redaction engine.

Provides a library of pre-built regex patterns for common sensitive data
types (PII, credentials, financial) and a pipeline for applying them
to trace data.

Usage::

    from lumen._redaction import RedactionEngine, PRESET_PII, PRESET_CREDENTIALS

    engine = RedactionEngine(presets=["pii", "credentials"])
    clean = engine.redact_text("My email is foo@bar.com and key is sk-abc123")
    # "My email is [EMAIL] and key is [API_KEY]"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lumen")


# ── Built-in pattern library ──────────────────────────────────────────────

@dataclass
class RedactionPattern:
    """A named redaction pattern with replacement text."""

    name: str
    pattern: str
    replacement: str = "[REDACTED]"
    description: str = ""

    _compiled: re.Pattern[str] | None = field(
        default=None, repr=False, compare=False,
    )

    def compile(self) -> re.Pattern[str] | None:
        if self._compiled is None:
            try:
                self._compiled = re.compile(self.pattern, re.IGNORECASE)
            except re.error as exc:
                from lumen._errors import warn_redaction
                warn_redaction(
                    f"Invalid redaction pattern '{self.name}': {exc}. "
                    f"This pattern will not match anything."
                )
                return None
        return self._compiled

    def apply(self, text: str) -> str:
        compiled = self.compile()
        if compiled is None:
            return text
        return compiled.sub(self.replacement, text)


# ── Pattern presets ───────────────────────────────────────────────────────

PATTERNS_PII: list[RedactionPattern] = [
    RedactionPattern(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        replacement="[EMAIL]",
        description="Email addresses",
    ),
    RedactionPattern(
        name="phone_us",
        pattern=r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        replacement="[PHONE]",
        description="US phone numbers",
    ),
    RedactionPattern(
        name="phone_intl",
        pattern=r"\b\+\d{1,3}[-.\s]?\d{4,14}\b",
        replacement="[PHONE]",
        description="International phone numbers",
    ),
    RedactionPattern(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        replacement="[SSN]",
        description="US Social Security Numbers",
    ),
    RedactionPattern(
        name="ip_address",
        pattern=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        replacement="[IP]",
        description="IPv4 addresses",
    ),
]

PATTERNS_CREDENTIALS: list[RedactionPattern] = [
    RedactionPattern(
        name="openai_key",
        pattern=r"\bsk-[A-Za-z0-9]{20,}\b",
        replacement="[OPENAI_KEY]",
        description="OpenAI API keys",
    ),
    RedactionPattern(
        name="anthropic_key",
        pattern=r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
        replacement="[ANTHROPIC_KEY]",
        description="Anthropic API keys",
    ),
    RedactionPattern(
        name="bearer_token",
        pattern=r"(?i)bearer\s+[A-Za-z0-9_.~+/=-]{20,}",
        replacement="Bearer [TOKEN]",
        description="Bearer tokens in headers",
    ),
    RedactionPattern(
        name="generic_secret",
        pattern=r"(?i)(?:api[_-]?key|secret|token|password|passwd|credential)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?",
        replacement="[SECRET]",
        description="Generic key=value secrets",
    ),
    RedactionPattern(
        name="aws_key",
        pattern=r"\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
        replacement="[AWS_KEY]",
        description="AWS access key IDs",
    ),
    RedactionPattern(
        name="github_token",
        pattern=r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b",
        replacement="[GITHUB_TOKEN]",
        description="GitHub personal access tokens",
    ),
]

PATTERNS_FINANCIAL: list[RedactionPattern] = [
    RedactionPattern(
        name="credit_card",
        pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        replacement="[CARD]",
        description="Credit card numbers (16 digits)",
    ),
    RedactionPattern(
        name="iban",
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?\d{0,16})?\b",
        replacement="[IBAN]",
        description="International Bank Account Numbers",
    ),
]

# Preset name → pattern list mapping
PRESETS: dict[str, list[RedactionPattern]] = {
    "pii": PATTERNS_PII,
    "credentials": PATTERNS_CREDENTIALS,
    "financial": PATTERNS_FINANCIAL,
    "all": PATTERNS_PII + PATTERNS_CREDENTIALS + PATTERNS_FINANCIAL,
}


# ── Redaction engine ──────────────────────────────────────────────────────

class RedactionEngine:
    """Configurable redaction engine with preset and custom patterns.

    Example::

        engine = RedactionEngine(presets=["pii", "credentials"])
        engine.add_pattern(RedactionPattern("custom", r"PROJECT-\\d+", "[PROJ_ID]"))

        clean = engine.redact_text("Email: a@b.com, key: sk-abc123")
        # "Email: [EMAIL], key: [OPENAI_KEY]"

        clean_trace = engine.redact_trace_data(trace_dict)
    """

    def __init__(
        self,
        presets: list[str] | None = None,
        custom_patterns: list[RedactionPattern] | None = None,
        redact_fields: list[str] | None = None,
    ) -> None:
        self._patterns: list[RedactionPattern] = []
        self._redact_fields: set[str] = set(redact_fields or [])

        # Load presets
        if presets:
            for name in presets:
                ps = PRESETS.get(name, [])
                self._patterns.extend(ps)

        # Add custom patterns
        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Pre-compile all
        for p in self._patterns:
            p.compile()

        # Build composite regex for single-pass O(N) redaction
        self._composite: re.Pattern[str] | None = None
        self._group_map: dict[str, str] = {}
        self._rebuild_composite()

    def add_pattern(self, pattern: RedactionPattern) -> None:
        """Add a custom pattern at runtime."""
        pattern.compile()
        self._patterns.append(pattern)
        self._rebuild_composite()

    def _rebuild_composite(self) -> None:
        """Build a single combined regex from all patterns.

        Each pattern becomes a named group: (?P<name>pattern).
        On match, the group name identifies which replacement to use.
        Single re.sub pass → O(N) instead of O(N*M) for M patterns.

        If any individual pattern is invalid, it is skipped and a warning
        is emitted. If the combined regex fails, falls back to per-pattern mode.
        """
        if not self._patterns:
            self._composite = None
            self._group_map.clear()
            return

        parts: list[str] = []
        self._group_map.clear()

        for i, p in enumerate(self._patterns):
            # Sanitize group name: only alphanumeric + underscore
            group_name = f"_p{i}_{re.sub(r'[^a-zA-Z0-9]', '_', p.name)}"
            # Strip inline case-insensitive flags — the composite uses re.IGNORECASE
            raw_pattern = re.sub(r"\(\?i\)", "", p.pattern)
            # Validate individual pattern before adding to composite
            try:
                re.compile(raw_pattern, re.IGNORECASE)
            except re.error as exc:
                from lumen._errors import warn_redaction
                warn_redaction(
                    f"Skipping invalid redaction pattern '{p.name}': {exc}. "
                    f"Sensitive data matching this pattern may not be filtered."
                )
                continue
            parts.append(f"(?P<{group_name}>{raw_pattern})")
            self._group_map[group_name] = p.replacement

        if not parts:
            self._composite = None
            return

        combined = "|".join(parts)
        try:
            self._composite = re.compile(combined, re.IGNORECASE)
        except re.error as exc:
            from lumen._errors import warn_redaction
            warn_redaction(
                f"Combined redaction regex failed to compile: {exc}. "
                f"Falling back to per-pattern mode (slower but safe)."
            )
            self._composite = None

    def _composite_replace(self, match: re.Match[str]) -> str:
        """Replacement function for composite regex."""
        for group_name, replacement in self._group_map.items():
            if match.group(group_name) is not None:
                return replacement
        return match.group(0)  # fallback: no change

    def redact_text(self, text: str) -> str:
        """Apply all patterns in a single pass. O(N) where N = text length.

        Falls back to per-pattern mode if the composite regex is unavailable.
        """
        if not text:
            return text
        if self._composite is not None:
            return self._composite.sub(self._composite_replace, text)
        # Fallback: per-pattern mode (slower but always works)
        for p in self._patterns:
            compiled = p.compile()
            if compiled is not None:
                text = compiled.sub(p.replacement, text)
        return text

    def redact_dict(self, data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """Recursively redact string values in a dict.

        Redacts:
          - All string values matching any pattern
          - All values under keys in redact_fields (replaced entirely)
        """
        if depth > 20:  # safety against circular refs
            return data

        result: dict[str, Any] = {}
        for key, val in data.items():
            # Field-level redaction: replace entire value
            if key in self._redact_fields:
                result[key] = "[REDACTED_FIELD]"
                continue

            if isinstance(val, str):
                result[key] = self.redact_text(val)
            elif isinstance(val, dict):
                result[key] = self.redact_dict(val, depth + 1)
            elif isinstance(val, list):
                result[key] = self._redact_list(val, depth + 1)
            else:
                result[key] = val
        return result

    def _redact_list(self, items: list[Any], depth: int) -> list[Any]:
        result: list[Any] = []
        for item in items:
            if isinstance(item, str):
                result.append(self.redact_text(item))
            elif isinstance(item, dict):
                result.append(self.redact_dict(item, depth))
            elif isinstance(item, list):
                result.append(self._redact_list(item, depth + 1))
            else:
                result.append(item)
        return result

    def redact_trace_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Redact sensitive data in trace steps (deep copy first)."""
        try:
            deep = json.loads(json.dumps(steps, default=str))
        except (TypeError, ValueError):
            logger.debug("Cannot deep-copy trace steps for redaction, operating on originals")
            deep = steps
        result = []
        for step in deep:
            result.append(self.redact_dict(step))
        return result

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def pattern_names(self) -> list[str]:
        return [p.name for p in self._patterns]
