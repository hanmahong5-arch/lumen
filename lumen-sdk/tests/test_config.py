"""Tests for the Lumen configuration system."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from lumen.config import LumenConfig, RedactionConfig


class TestDefaults:
    """Verify default values for all config sections."""

    def test_default_config(self) -> None:
        cfg = LumenConfig()
        assert cfg.project_name == ""
        assert cfg.environment == "development"
        assert cfg.tracing.enabled is True
        assert cfg.tracing.trace_dir == "./traces"
        assert cfg.tracing.sampling_rate == 1.0
        assert cfg.tracing.content_capture == "full"
        assert cfg.cost.budget_usd == 0.0
        assert cfg.cost.anomaly_multiplier == 2.0
        assert cfg.agent.default_model == "claude-sonnet-4-6"
        assert cfg.agent.max_iterations == 25
        assert cfg.monitoring.enabled is False
        assert cfg.export.otlp_enabled is False
        assert cfg.retention.trace_max_age_days == 90


class TestFromDict:
    """Test config creation from dictionaries."""

    def test_project_fields(self) -> None:
        cfg = LumenConfig.from_dict({
            "project": {"name": "test-agent", "environment": "staging", "version": "1.0"},
        })
        assert cfg.project_name == "test-agent"
        assert cfg.environment == "staging"
        assert cfg.version == "1.0"

    def test_tracing_fields(self) -> None:
        cfg = LumenConfig.from_dict({
            "tracing": {
                "enabled": False,
                "trace_dir": "/tmp/traces",
                "sampling_rate": 0.5,
                "content_capture": "metadata_only",
            },
        })
        assert cfg.tracing.enabled is False
        assert cfg.tracing.trace_dir == "/tmp/traces"
        assert cfg.tracing.sampling_rate == 0.5
        assert cfg.tracing.content_capture == "metadata_only"

    def test_cost_fields(self) -> None:
        cfg = LumenConfig.from_dict({
            "cost": {
                "budget_usd": 500.0,
                "kill_on_budget_exceeded": True,
                "anomaly_multiplier": 3.0,
                "anomaly_min_usd": 0.05,
            },
        })
        assert cfg.cost.budget_usd == 500.0
        assert cfg.cost.kill_on_budget_exceeded is True
        assert cfg.cost.anomaly_multiplier == 3.0
        assert cfg.cost.anomaly_min_usd == 0.05

    def test_agent_fields(self) -> None:
        cfg = LumenConfig.from_dict({
            "agent": {
                "default_model": "gpt-4o",
                "max_iterations": 10,
                "max_cost_per_run_usd": 2.0,
                "tool_timeout_secs": 60,
                "tools": {
                    "allowlist": ["search"],
                    "denylist": ["shell_exec"],
                },
            },
        })
        assert cfg.agent.default_model == "gpt-4o"
        assert cfg.agent.max_iterations == 10
        assert cfg.agent.max_cost_per_run_usd == 2.0
        assert cfg.agent.tool_allowlist == ["search"]
        assert cfg.agent.tool_denylist == ["shell_exec"]

    def test_custom_pricing(self) -> None:
        cfg = LumenConfig.from_dict({
            "cost": {
                "custom_pricing": {
                    "local-llama": {"prompt": 0.0, "completion": 0.0},
                    "my-model": {"prompt": 5.0, "completion": 15.0},
                },
            },
        })
        assert cfg.cost.custom_pricing["local-llama"] == (0.0, 0.0)
        assert cfg.cost.custom_pricing["my-model"] == (5.0, 15.0)

    def test_monitoring_dashboard(self) -> None:
        cfg = LumenConfig.from_dict({
            "monitoring": {
                "enabled": True,
                "metrics_port": 8888,
                "dashboard": {"port": 9999, "host": "0.0.0.0"},
            },
        })
        assert cfg.monitoring.enabled is True
        assert cfg.monitoring.metrics_port == 8888
        assert cfg.monitoring.dashboard_port == 9999
        assert cfg.monitoring.dashboard_host == "0.0.0.0"

    def test_export_otlp(self) -> None:
        cfg = LumenConfig.from_dict({
            "export": {
                "otlp": {
                    "enabled": True,
                    "endpoint": "http://otel:4318",
                    "headers": {"Authorization": "Bearer xxx"},
                },
            },
        })
        assert cfg.export.otlp_enabled is True
        assert cfg.export.otlp_endpoint == "http://otel:4318"
        assert cfg.export.otlp_headers["Authorization"] == "Bearer xxx"

    def test_empty_dict(self) -> None:
        cfg = LumenConfig.from_dict({})
        assert cfg.tracing.sampling_rate == 1.0  # defaults preserved

    def test_invalid_values_ignored(self) -> None:
        cfg = LumenConfig.from_dict({
            "tracing": {"sampling_rate": "not-a-number"},
        })
        assert cfg.tracing.sampling_rate == 1.0  # default preserved


class TestTomlLoading:
    """Test loading from TOML files."""

    def test_load_from_file(self, tmp_path: Path) -> None:
        toml_content = textwrap.dedent("""\
            [project]
            name = "toml-test"
            environment = "production"

            [tracing]
            sampling_rate = 0.3
            trace_dir = "./my-traces"

            [cost]
            budget_usd = 1000.0
        """)
        config_file = tmp_path / "lumen.toml"
        config_file.write_text(toml_content)

        cfg = LumenConfig.from_file(config_file)
        assert cfg.project_name == "toml-test"
        assert cfg.environment == "production"
        assert cfg.tracing.sampling_rate == 0.3
        assert cfg.tracing.trace_dir == "./my-traces"
        assert cfg.cost.budget_usd == 1000.0

    def test_nonexistent_file(self) -> None:
        cfg = LumenConfig.from_file("/nonexistent/path.toml")
        assert cfg.tracing.sampling_rate == 1.0  # defaults

    def test_redaction_from_toml(self, tmp_path: Path) -> None:
        toml_content = textwrap.dedent("""\
            [tracing.redaction]
            enabled = true
            patterns = ['\\\\b\\\\d{16}\\\\b', 'secret_\\\\w+']
            custom_fields = ["ssn", "phone"]
        """)
        config_file = tmp_path / "lumen.toml"
        config_file.write_text(toml_content)

        cfg = LumenConfig.from_file(config_file)
        assert cfg.tracing.redaction.enabled is True
        assert len(cfg.tracing.redaction.patterns) == 2
        assert cfg.tracing.redaction.custom_fields == ["ssn", "phone"]


class TestEnvVars:
    """Test environment variable overlay."""

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LUMEN_PROJECT", "env-project")
        monkeypatch.setenv("LUMEN_ENVIRONMENT", "staging")
        monkeypatch.setenv("LUMEN_SAMPLING_RATE", "0.1")
        monkeypatch.setenv("LUMEN_TRACE_DIR", "/tmp/env-traces")
        monkeypatch.setenv("LUMEN_BUDGET_USD", "999.99")
        monkeypatch.setenv("LUMEN_MAX_ITERATIONS", "50")
        monkeypatch.setenv("LUMEN_ENABLED", "false")

        cfg = LumenConfig.from_dict({})  # env is applied in from_dict
        assert cfg.project_name == "env-project"
        assert cfg.environment == "staging"
        assert cfg.tracing.sampling_rate == 0.1
        assert cfg.tracing.trace_dir == "/tmp/env-traces"
        assert cfg.cost.budget_usd == 999.99
        assert cfg.agent.max_iterations == 50
        assert cfg.tracing.enabled is False

    def test_env_bool_true_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("true", "True", "1", "yes"):
            monkeypatch.setenv("LUMEN_KILL_ON_BUDGET", val)
            cfg = LumenConfig.from_dict({})
            assert cfg.cost.kill_on_budget_exceeded is True, f"Failed for {val}"

    def test_env_bool_false_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("false", "False", "0", "no"):
            monkeypatch.setenv("LUMEN_KILL_ON_BUDGET", val)
            cfg = LumenConfig.from_dict({})
            assert cfg.cost.kill_on_budget_exceeded is False, f"Failed for {val}"

    def test_invalid_env_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LUMEN_SAMPLING_RATE", "not-a-float")
        cfg = LumenConfig.from_dict({})
        assert cfg.tracing.sampling_rate == 1.0  # default preserved

    def test_env_overrides_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toml_content = "[tracing]\nsampling_rate = 0.5\n"
        config_file = tmp_path / "lumen.toml"
        config_file.write_text(toml_content)

        monkeypatch.setenv("LUMEN_SAMPLING_RATE", "0.1")
        cfg = LumenConfig.from_file(config_file)
        assert cfg.tracing.sampling_rate == 0.1  # env wins


class TestBuilder:
    """Test the fluent builder API."""

    def test_builder_basic(self) -> None:
        cfg = (
            LumenConfig.builder()
            .project("builder-test")
            .environment("staging")
            .sampling_rate(0.5)
            .trace_dir("/tmp/builder")
            .budget_usd(200.0)
            .max_iterations(30)
            .build()
        )
        assert cfg.project_name == "builder-test"
        assert cfg.environment == "staging"
        assert cfg.tracing.sampling_rate == 0.5
        assert cfg.tracing.trace_dir == "/tmp/builder"
        assert cfg.cost.budget_usd == 200.0
        assert cfg.agent.max_iterations == 30

    def test_builder_redaction(self) -> None:
        cfg = (
            LumenConfig.builder()
            .redaction(enabled=True, patterns=[r"\d{4}-\d{4}"])
            .build()
        )
        assert cfg.tracing.redaction.enabled is True
        assert len(cfg.tracing.redaction.patterns) == 1

    def test_builder_otlp(self) -> None:
        cfg = (
            LumenConfig.builder()
            .otlp("http://collector:4318", enabled=True)
            .build()
        )
        assert cfg.export.otlp_enabled is True
        assert cfg.export.otlp_endpoint == "http://collector:4318"

    def test_builder_tool_denylist(self) -> None:
        cfg = (
            LumenConfig.builder()
            .tool_denylist(["shell_exec", "rm_rf"])
            .build()
        )
        assert cfg.agent.tool_denylist == ["shell_exec", "rm_rf"]

    def test_builder_kill_on_budget(self) -> None:
        cfg = (
            LumenConfig.builder()
            .budget_usd(100.0)
            .kill_on_budget(True)
            .build()
        )
        assert cfg.cost.kill_on_budget_exceeded is True


class TestRedaction:
    """Test the redaction engine."""

    def test_redact_disabled(self) -> None:
        r = RedactionConfig(enabled=False, patterns=[r"\d+"])
        assert r.redact("abc123") == "abc123"

    def test_redact_enabled(self) -> None:
        r = RedactionConfig(enabled=True, patterns=[r"\b\d{4}\b"])
        r.compile_patterns()
        assert r.redact("card: 1234 is valid") == "card: [REDACTED] is valid"

    def test_redact_multiple_patterns(self) -> None:
        r = RedactionConfig(
            enabled=True,
            patterns=[r"\b\d{4}\b", r"secret_\w+"],
        )
        r.compile_patterns()
        result = r.redact("code 1234 and secret_key123")
        assert "1234" not in result
        assert "secret_key123" not in result
        assert "[REDACTED]" in result

    def test_redact_invalid_pattern_skipped(self) -> None:
        r = RedactionConfig(
            enabled=True,
            patterns=["[invalid", r"\d+"],
        )
        r.compile_patterns()
        assert len(r._compiled) == 1  # only valid pattern compiled
        assert r.redact("abc123") == "abc[REDACTED]"

    def test_redact_empty_text(self) -> None:
        r = RedactionConfig(enabled=True, patterns=[r"\d+"])
        r.compile_patterns()
        assert r.redact("") == ""


class TestAutoDiscovery:
    """Test LumenConfig.auto() project config discovery."""

    def test_auto_with_project_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toml_content = textwrap.dedent("""\
            [project]
            name = "auto-test"

            [tracing]
            sampling_rate = 0.42
        """)
        (tmp_path / "lumen.toml").write_text(toml_content)
        monkeypatch.chdir(tmp_path)

        cfg = LumenConfig.auto()
        assert cfg.project_name == "auto-test"
        assert cfg.tracing.sampling_rate == 0.42

    def test_auto_no_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = LumenConfig.auto()
        assert cfg.tracing.sampling_rate == 1.0  # defaults

    def test_auto_walks_up_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toml_content = '[project]\nname = "parent-config"\n'
        (tmp_path / "lumen.toml").write_text(toml_content)
        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        cfg = LumenConfig.auto()
        assert cfg.project_name == "parent-config"
