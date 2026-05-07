"""Comprehensive tests for the built-in redaction engine."""

from __future__ import annotations

import json
from typing import Any

import pytest

from lumen._redaction import (
    PATTERNS_CREDENTIALS,
    PATTERNS_FINANCIAL,
    PATTERNS_PII,
    PRESETS,
    RedactionEngine,
    RedactionPattern,
)


class TestRedactionPattern:
    """Test individual RedactionPattern behavior."""

    def test_basic_apply(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"\d{4}", replacement="[NUM]")
        assert pat.apply("code 1234 done") == "code [NUM] done"

    def test_compile_caches(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"\d+")
        c1 = pat.compile()
        c2 = pat.compile()
        assert c1 is c2

    def test_case_insensitive(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"secret", replacement="[X]")
        assert pat.apply("my SECRET is here") == "my [X] is here"

    def test_no_match(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"zzzzz", replacement="[X]")
        assert pat.apply("hello world") == "hello world"

    def test_multiple_matches(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"\d+", replacement="[N]")
        assert pat.apply("a=1, b=22, c=333") == "a=[N], b=[N], c=[N]"

    def test_default_replacement(self) -> None:
        pat = RedactionPattern(name="test", pattern=r"foo")
        assert pat.replacement == "[REDACTED]"
        assert pat.apply("foo bar foo") == "[REDACTED] bar [REDACTED]"


class TestPIIPatterns:
    """Test built-in PII redaction patterns."""

    def test_email(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        assert "[EMAIL]" in engine.redact_text("contact user@example.com for info")

    def test_email_preserves_non_email(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        assert engine.redact_text("no email here") == "no email here"

    def test_us_phone(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_text("Call (555) 123-4567 for support")
        assert "[PHONE]" in result
        assert "555" not in result

    def test_us_phone_with_country_code(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_text("Dial +1-555-123-4567")
        assert "[PHONE]" in result

    def test_intl_phone(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_text("UK: +44-7700900000")
        assert "[PHONE]" in result

    def test_ssn(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_text("SSN: 123-45-6789")
        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_ip_address(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_text("Server at 192.168.1.100")
        assert "[IP]" in result
        assert "192.168.1.100" not in result

    def test_multiple_pii_in_one_string(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        text = "Email: test@a.com, SSN: 111-22-3333, IP: 10.0.0.1"
        result = engine.redact_text(text)
        assert "[EMAIL]" in result
        assert "[SSN]" in result
        assert "[IP]" in result


class TestCredentialPatterns:
    """Test built-in credential redaction patterns."""

    def test_openai_key(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        result = engine.redact_text("Key: sk-abc123def456ghi789jkl012mno345pqr678")
        assert "[OPENAI_KEY]" in result
        assert "sk-abc123" not in result

    def test_anthropic_key(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        result = engine.redact_text("API: sk-ant-api03-abcdefghijk123456789")
        assert "[ANTHROPIC_KEY]" in result

    def test_bearer_token(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        result = engine.redact_text("Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "Bearer [TOKEN]" in result
        assert "eyJ" not in result

    def test_generic_secret(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        result = engine.redact_text('api_key="sk_live_1234567890abcdef"')
        assert "[SECRET]" in result

    def test_aws_key(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        result = engine.redact_text("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "[AWS_KEY]" in result

    def test_github_token(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        # Use a context that won't match generic_secret first
        result = engine.redact_text("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789")
        assert "[GITHUB_TOKEN]" in result

    def test_no_false_positive_on_normal_text(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        text = "This is a normal sentence about API design."
        assert engine.redact_text(text) == text


class TestFinancialPatterns:
    """Test built-in financial data redaction patterns."""

    def test_credit_card(self) -> None:
        engine = RedactionEngine(presets=["financial"])
        result = engine.redact_text("Card: 4111-1111-1111-1111")
        assert "[CARD]" in result
        assert "4111" not in result

    def test_credit_card_no_dashes(self) -> None:
        engine = RedactionEngine(presets=["financial"])
        result = engine.redact_text("Card: 4111111111111111")
        assert "[CARD]" in result

    def test_iban(self) -> None:
        engine = RedactionEngine(presets=["financial"])
        result = engine.redact_text("IBAN: DE89370400440532013000")
        assert "[IBAN]" in result


class TestRedactionPresets:
    """Test preset loading and combination."""

    def test_pii_preset_exists(self) -> None:
        assert "pii" in PRESETS
        assert len(PRESETS["pii"]) == len(PATTERNS_PII)

    def test_credentials_preset_exists(self) -> None:
        assert "credentials" in PRESETS
        assert len(PRESETS["credentials"]) == len(PATTERNS_CREDENTIALS)

    def test_financial_preset_exists(self) -> None:
        assert "financial" in PRESETS
        assert len(PRESETS["financial"]) == len(PATTERNS_FINANCIAL)

    def test_all_preset_combines(self) -> None:
        total = len(PATTERNS_PII) + len(PATTERNS_CREDENTIALS) + len(PATTERNS_FINANCIAL)
        assert len(PRESETS["all"]) == total

    def test_unknown_preset_ignored(self) -> None:
        engine = RedactionEngine(presets=["nonexistent"])
        assert engine.pattern_count == 0

    def test_multiple_presets(self) -> None:
        engine = RedactionEngine(presets=["pii", "credentials"])
        assert engine.pattern_count == len(PATTERNS_PII) + len(PATTERNS_CREDENTIALS)


class TestRedactionEngine:
    """Test the RedactionEngine class."""

    def test_empty_engine(self) -> None:
        engine = RedactionEngine()
        assert engine.pattern_count == 0
        assert engine.redact_text("hello") == "hello"

    def test_all_preset(self) -> None:
        engine = RedactionEngine(presets=["all"])
        text = "Email: a@b.com, Key: sk-abc123def456ghi789jkl012"
        result = engine.redact_text(text)
        assert "[EMAIL]" in result
        assert "[OPENAI_KEY]" in result

    def test_custom_patterns_only(self) -> None:
        custom = [
            RedactionPattern(name="project_id", pattern=r"PROJ-\d+", replacement="[PROJ]"),
        ]
        engine = RedactionEngine(custom_patterns=custom)
        assert engine.redact_text("Working on PROJ-123 and PROJ-456") == "Working on [PROJ] and [PROJ]"

    def test_add_pattern_at_runtime(self) -> None:
        engine = RedactionEngine()
        engine.add_pattern(
            RedactionPattern(name="custom", pattern=r"CUSTOM-\d+", replacement="[C]"),
        )
        assert engine.pattern_count == 1
        assert engine.redact_text("ref CUSTOM-99") == "ref [C]"

    def test_pattern_names(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        names = engine.pattern_names
        assert "email" in names
        assert "ssn" in names

    def test_redact_empty_string(self) -> None:
        engine = RedactionEngine(presets=["all"])
        assert engine.redact_text("") == ""

    def test_redact_none_patterns(self) -> None:
        engine = RedactionEngine()
        assert engine.redact_text("anything") == "anything"


class TestRedactionEngineDict:
    """Test recursive dict redaction."""

    def test_simple_dict(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        data = {"email": "user@test.com", "name": "Alice"}
        result = engine.redact_dict(data)
        assert "[EMAIL]" in result["email"]
        assert result["name"] == "Alice"

    def test_nested_dict(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        data = {"user": {"contact": {"email": "a@b.com"}}}
        result = engine.redact_dict(data)
        assert "[EMAIL]" in result["user"]["contact"]["email"]

    def test_dict_with_list(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        data = {"emails": ["one@x.com", "two@y.org"]}
        result = engine.redact_dict(data)
        assert all("[EMAIL]" in e for e in result["emails"])

    def test_non_string_values_preserved(self) -> None:
        engine = RedactionEngine(presets=["all"])
        data = {"count": 42, "active": True, "ratio": 3.14, "nothing": None}
        result = engine.redact_dict(data)
        assert result["count"] == 42
        assert result["active"] is True
        assert result["ratio"] == 3.14
        assert result["nothing"] is None

    def test_field_level_redaction(self) -> None:
        engine = RedactionEngine(
            presets=["pii"],
            redact_fields=["password", "secret"],
        )
        data = {"username": "alice", "password": "hunter2", "secret": "xyz"}
        result = engine.redact_dict(data)
        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED_FIELD]"
        assert result["secret"] == "[REDACTED_FIELD]"

    def test_depth_limit(self) -> None:
        # Build deeply nested dict
        data: dict[str, Any] = {"val": "a@b.com"}
        current = data
        for i in range(25):
            current["nested"] = {"val": f"email{i}@test.com"}
            current = current["nested"]

        engine = RedactionEngine(presets=["pii"])
        result = engine.redact_dict(data)
        # Should not crash, depth limit is 20
        assert "[EMAIL]" in result["val"]

    def test_empty_dict(self) -> None:
        engine = RedactionEngine(presets=["all"])
        assert engine.redact_dict({}) == {}


class TestRedactionEngineList:
    """Test list redaction."""

    def test_list_of_strings(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        items = ["user@a.com", "normal text", "admin@b.com"]
        result = engine._redact_list(items, 0)
        assert "[EMAIL]" in result[0]
        assert result[1] == "normal text"
        assert "[EMAIL]" in result[2]

    def test_list_of_dicts(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        items = [
            {"email": "a@b.com"},
            {"email": "c@d.com"},
        ]
        result = engine._redact_list(items, 0)
        assert all("[EMAIL]" in d["email"] for d in result)

    def test_nested_lists(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        items: list[Any] = [["a@b.com", "hello"], ["c@d.com"]]
        result = engine._redact_list(items, 0)
        assert "[EMAIL]" in result[0][0]
        assert result[0][1] == "hello"


class TestRedactionEngineTraceSteps:
    """Test trace step redaction (the primary use case)."""

    def test_redact_error_message(self) -> None:
        engine = RedactionEngine(presets=["credentials"])
        steps = [
            {
                "step_type": {"Error": {"message": "Invalid key: sk-abc123def456ghi789jkl012mno345pqr678"}},
                "metadata": [],
            },
        ]
        result = engine.redact_trace_steps(steps)
        msg = result[0]["step_type"]["Error"]["message"]
        assert "sk-abc123" not in msg
        assert "[OPENAI_KEY]" in msg

    def test_redact_llm_call_metadata(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        steps = [
            {
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
                "metadata": [["prompt", "Send email to user@test.com"]],
            },
        ]
        result = engine.redact_trace_steps(steps)
        # metadata values are lists, not strings, so the engine redacts recursively
        assert "user@test.com" not in json.dumps(result)

    def test_original_not_modified(self) -> None:
        engine = RedactionEngine(presets=["pii"])
        original_steps = [
            {"step_type": {"Error": {"message": "a@b.com"}}, "metadata": []},
        ]
        engine.redact_trace_steps(original_steps)
        # Original should be untouched
        assert original_steps[0]["step_type"]["Error"]["message"] == "a@b.com"

    def test_empty_steps(self) -> None:
        engine = RedactionEngine(presets=["all"])
        assert engine.redact_trace_steps([]) == []

    def test_multiple_steps(self) -> None:
        engine = RedactionEngine(presets=["pii", "credentials"])
        steps = [
            {"step_type": {"Error": {"message": "Bad key sk-abc123def456ghi789jkl012mno345pqr678"}}, "metadata": []},
            {"step_type": {"Error": {"message": "Contact admin@test.com"}}, "metadata": []},
        ]
        result = engine.redact_trace_steps(steps)
        assert "[OPENAI_KEY]" in result[0]["step_type"]["Error"]["message"]
        assert "[EMAIL]" in result[1]["step_type"]["Error"]["message"]


class TestRedactionEnginePresetCombinations:
    """Test combining presets and custom patterns."""

    def test_preset_plus_custom(self) -> None:
        custom = [
            RedactionPattern(name="order_id", pattern=r"ORD-\d+", replacement="[ORDER]"),
        ]
        engine = RedactionEngine(presets=["pii"], custom_patterns=custom)
        text = "Order ORD-12345 for user@test.com"
        result = engine.redact_text(text)
        assert "[ORDER]" in result
        assert "[EMAIL]" in result

    def test_overlapping_patterns(self) -> None:
        # Both patterns could match the same text
        engine = RedactionEngine(presets=["pii", "credentials"])
        text = "key=sk-abc123def456ghi789jkl012mno345pqr678"
        result = engine.redact_text(text)
        assert "sk-abc123" not in result
