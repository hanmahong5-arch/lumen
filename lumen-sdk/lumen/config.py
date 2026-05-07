"""Hierarchical configuration system for Lumen.

Load order (later wins):
  1. Hardcoded defaults
  2. Global config:  ~/.lumen/config.toml
  3. Project config:  ./lumen.toml  (walks up directories to find it)
  4. Environment variables:  LUMEN_*
  5. Explicit overrides via builder / from_dict
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("lumen")

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


# ── Sub-configs ───────────────────────────────────────────────────────────

@dataclass
class RedactionConfig:
    """Sensitive data filtering rules."""

    enabled: bool = False
    patterns: list[str] = field(default_factory=list)
    custom_fields: list[str] = field(default_factory=list)

    _compiled: list[re.Pattern[str]] = field(
        default_factory=list, repr=False, compare=False,
    )

    def compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        from lumen._errors import warn_redaction

        self._compiled = []
        for p in self.patterns:
            try:
                self._compiled.append(re.compile(p))
            except re.error as exc:
                warn_redaction(
                    f"Invalid redaction pattern '{p}': {exc}. "
                    f"This pattern will be skipped — sensitive data may not be filtered."
                )

    def redact(self, text: str) -> str:
        """Replace all matching patterns in *text* with [REDACTED]."""
        if not self.enabled:
            return text
        if not self._compiled:
            self.compile_patterns()
        for pat in self._compiled:
            text = pat.sub("[REDACTED]", text)
        return text


@dataclass
class TracingConfig:
    """Trace capture settings."""

    enabled: bool = True
    trace_dir: str = "./traces"
    sampling_rate: float = 1.0
    content_capture: str = "full"  # full | metadata_only | off
    max_trace_size_mb: int = 50
    buffer_size: int = 100
    flush_interval_ms: int = 5000
    redaction: RedactionConfig = field(default_factory=RedactionConfig)


@dataclass
class CostConfig:
    """Cost tracking and budget settings."""

    budget_usd: float = 0.0
    budget_period: str = "monthly"
    alert_threshold_pct: int = 80
    kill_on_budget_exceeded: bool = False
    anomaly_multiplier: float = 2.0
    anomaly_min_usd: float = 0.01
    custom_pricing: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class AgentSettings:
    """Default agent execution parameters."""

    default_model: str = "claude-sonnet-4-6"
    max_iterations: int = 25
    max_cost_per_run_usd: float = 0.0
    tool_timeout_secs: int = 120
    tool_allowlist: list[str] = field(default_factory=list)
    tool_denylist: list[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Metrics and dashboard settings."""

    enabled: bool = False
    metrics_port: int = 9701
    dashboard_port: int = 9700
    dashboard_host: str = "127.0.0.1"


@dataclass
class ExportConfig:
    """Trace export settings."""

    otlp_enabled: bool = False
    otlp_endpoint: str = ""
    otlp_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class RetentionConfig:
    """Data retention policy."""

    trace_max_age_days: int = 90
    checkpoint_max_age_days: int = 30
    auto_cleanup: bool = False


# ── Main config ───────────────────────────────────────────────────────────

@dataclass
class LumenConfig:
    """Root configuration for all Lumen components.

    Example::

        # Auto-discover config (recommended)
        config = LumenConfig.auto()

        # Or build programmatically
        config = (
            LumenConfig.builder()
            .project("my-agent")
            .environment("production")
            .sampling_rate(0.3)
            .budget_usd(100.0)
            .build()
        )
    """

    project_name: str = ""
    environment: str = "development"
    version: str = ""

    tracing: TracingConfig = field(default_factory=TracingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    agent: AgentSettings = field(default_factory=AgentSettings)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def auto(cls) -> LumenConfig:
        """Auto-discover config from files and environment.

        Searches (in order, later wins):
          1. ``~/.lumen/config.toml``
          2. ``./lumen.toml`` (walks up to repo root)
          3. ``LUMEN_*`` environment variables
        """
        config = cls()

        # Layer 1: global config — Path.home() can fail on some systems
        try:
            global_path = Path.home() / ".lumen" / "config.toml"
            if global_path.is_file():
                config._merge_toml(global_path)
        except (RuntimeError, OSError):
            logger.debug("Could not resolve home directory for global config")

        # Layer 2: project config (walk up)
        project_path = _find_project_config()
        if project_path is not None:
            config._merge_toml(project_path)

        # Layer 3: environment variables
        config._merge_env()

        # Compile redaction patterns
        config.tracing.redaction.compile_patterns()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> LumenConfig:
        """Load config from a specific TOML file."""
        config = cls()
        config._merge_toml(Path(path))
        config._merge_env()
        config.tracing.redaction.compile_patterns()
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LumenConfig:
        """Create config from a nested dictionary."""
        config = cls()
        config._merge_dict(data)
        config._merge_env()
        config.tracing.redaction.compile_patterns()
        return config

    @classmethod
    def builder(cls) -> ConfigBuilder:
        """Return a fluent builder for constructing config."""
        return ConfigBuilder()

    # ── Internal merge logic ──────────────────────────────────────────

    def _merge_toml(self, path: Path) -> None:
        """Merge a TOML file into this config."""
        from lumen._errors import warn_config

        if tomllib is None:
            warn_config(
                f"Cannot load '{path}': no TOML parser available. "
                f"Install tomli (`pip install tomli`) for Python < 3.11."
            )
            return
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except OSError as exc:
            warn_config(f"Cannot read config file '{path}': {exc}. Using defaults.")
            return
        except ValueError as exc:
            warn_config(f"Invalid TOML in '{path}': {exc}. Using defaults.")
            return
        self._merge_dict(data)

    def _merge_dict(self, data: dict[str, Any]) -> None:
        """Merge a nested dict into this config (non-default values only)."""
        # Top-level project fields
        project = data.get("project", {})
        if isinstance(project, dict):
            if "name" in project:
                self.project_name = str(project["name"])
            if "environment" in project:
                self.environment = str(project["environment"])
            if "version" in project:
                self.version = str(project["version"])

        # Tracing
        tracing = data.get("tracing", {})
        if isinstance(tracing, dict):
            _set_if(tracing, "enabled", self.tracing, "enabled", bool)
            _set_if(tracing, "trace_dir", self.tracing, "trace_dir", str)
            _set_if(tracing, "sampling_rate", self.tracing, "sampling_rate", float)
            _set_if(tracing, "content_capture", self.tracing, "content_capture", str)
            _set_if(tracing, "max_trace_size_mb", self.tracing, "max_trace_size_mb", int)
            _set_if(tracing, "buffer_size", self.tracing, "buffer_size", int)
            _set_if(tracing, "flush_interval_ms", self.tracing, "flush_interval_ms", int)

            # Nested redaction
            redaction = tracing.get("redaction", {})
            if isinstance(redaction, dict):
                _set_if(redaction, "enabled", self.tracing.redaction, "enabled", bool)
                if "patterns" in redaction and isinstance(redaction["patterns"], list):
                    self.tracing.redaction.patterns = [
                        str(p) for p in redaction["patterns"]
                    ]
                if "custom_fields" in redaction and isinstance(redaction["custom_fields"], list):
                    self.tracing.redaction.custom_fields = [
                        str(f) for f in redaction["custom_fields"]
                    ]

        # Cost
        cost = data.get("cost", {})
        if isinstance(cost, dict):
            _set_if(cost, "budget_usd", self.cost, "budget_usd", float)
            _set_if(cost, "budget_period", self.cost, "budget_period", str)
            _set_if(cost, "alert_threshold_pct", self.cost, "alert_threshold_pct", int)
            _set_if(cost, "kill_on_budget_exceeded", self.cost, "kill_on_budget_exceeded", bool)
            _set_if(cost, "anomaly_multiplier", self.cost, "anomaly_multiplier", float)
            _set_if(cost, "anomaly_min_usd", self.cost, "anomaly_min_usd", float)

            custom = cost.get("custom_pricing", {})
            if isinstance(custom, dict):
                for model, prices in custom.items():
                    if isinstance(prices, dict):
                        p = float(prices.get("prompt", 0.0))
                        c = float(prices.get("completion", 0.0))
                        self.cost.custom_pricing[str(model)] = (p, c)

        # Agent
        agent = data.get("agent", {})
        if isinstance(agent, dict):
            _set_if(agent, "default_model", self.agent, "default_model", str)
            _set_if(agent, "max_iterations", self.agent, "max_iterations", int)
            _set_if(agent, "max_cost_per_run_usd", self.agent, "max_cost_per_run_usd", float)
            _set_if(agent, "tool_timeout_secs", self.agent, "tool_timeout_secs", int)
            if "tools" in agent and isinstance(agent["tools"], dict):
                tools = agent["tools"]
                if "allowlist" in tools and isinstance(tools["allowlist"], list):
                    self.agent.tool_allowlist = [str(t) for t in tools["allowlist"]]
                if "denylist" in tools and isinstance(tools["denylist"], list):
                    self.agent.tool_denylist = [str(t) for t in tools["denylist"]]

        # Monitoring
        monitoring = data.get("monitoring", {})
        if isinstance(monitoring, dict):
            _set_if(monitoring, "enabled", self.monitoring, "enabled", bool)
            _set_if(monitoring, "metrics_port", self.monitoring, "metrics_port", int)
            dashboard = monitoring.get("dashboard", {})
            if isinstance(dashboard, dict):
                _set_if(dashboard, "port", self.monitoring, "dashboard_port", int)
                _set_if(dashboard, "host", self.monitoring, "dashboard_host", str)

        # Export
        export = data.get("export", {})
        if isinstance(export, dict):
            otlp = export.get("otlp", {})
            if isinstance(otlp, dict):
                _set_if(otlp, "enabled", self.export, "otlp_enabled", bool)
                _set_if(otlp, "endpoint", self.export, "otlp_endpoint", str)
                if "headers" in otlp and isinstance(otlp["headers"], dict):
                    self.export.otlp_headers = {
                        str(k): str(v) for k, v in otlp["headers"].items()
                    }

        # Retention
        retention = data.get("retention", {})
        if isinstance(retention, dict):
            _set_if(retention, "trace_max_age_days", self.retention, "trace_max_age_days", int)
            _set_if(retention, "checkpoint_max_age_days", self.retention, "checkpoint_max_age_days", int)
            _set_if(retention, "auto_cleanup", self.retention, "auto_cleanup", bool)

    def _merge_env(self) -> None:
        """Overlay LUMEN_* environment variables."""
        _env_str("LUMEN_PROJECT", self, "project_name")
        _env_str("LUMEN_ENVIRONMENT", self, "environment")

        _env_bool("LUMEN_ENABLED", self.tracing, "enabled")
        _env_str("LUMEN_TRACE_DIR", self.tracing, "trace_dir")
        _env_float("LUMEN_SAMPLING_RATE", self.tracing, "sampling_rate")
        _env_str("LUMEN_CONTENT_CAPTURE", self.tracing, "content_capture")
        _env_bool("LUMEN_REDACTION_ENABLED", self.tracing.redaction, "enabled")

        _env_float("LUMEN_BUDGET_USD", self.cost, "budget_usd")
        _env_bool("LUMEN_KILL_ON_BUDGET", self.cost, "kill_on_budget_exceeded")
        _env_float("LUMEN_ANOMALY_MULTIPLIER", self.cost, "anomaly_multiplier")

        _env_int("LUMEN_MAX_ITERATIONS", self.agent, "max_iterations")
        _env_float("LUMEN_MAX_COST_PER_RUN", self.agent, "max_cost_per_run_usd")
        _env_int("LUMEN_TOOL_TIMEOUT", self.agent, "tool_timeout_secs")

        _env_int("LUMEN_METRICS_PORT", self.monitoring, "metrics_port")
        _env_int("LUMEN_DASHBOARD_PORT", self.monitoring, "dashboard_port")

        _env_bool("LUMEN_OTLP_ENABLED", self.export, "otlp_enabled")
        _env_str("LUMEN_OTLP_ENDPOINT", self.export, "otlp_endpoint")


# ── Builder ───────────────────────────────────────────────────────────────

class ConfigBuilder:
    """Fluent builder for LumenConfig.

    Example::

        config = (
            LumenConfig.builder()
            .project("my-agent")
            .environment("production")
            .sampling_rate(0.3)
            .budget_usd(100.0)
            .build()
        )
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def project(self, name: str) -> ConfigBuilder:
        self._data.setdefault("project", {})["name"] = name
        return self

    def environment(self, env: str) -> ConfigBuilder:
        self._data.setdefault("project", {})["environment"] = env
        return self

    def trace_dir(self, path: str) -> ConfigBuilder:
        self._data.setdefault("tracing", {})["trace_dir"] = path
        return self

    def sampling_rate(self, rate: float) -> ConfigBuilder:
        self._data.setdefault("tracing", {})["sampling_rate"] = rate
        return self

    def content_capture(self, mode: str) -> ConfigBuilder:
        self._data.setdefault("tracing", {})["content_capture"] = mode
        return self

    def redaction(
        self,
        *,
        enabled: bool = True,
        patterns: list[str] | None = None,
        custom_fields: list[str] | None = None,
    ) -> ConfigBuilder:
        r = self._data.setdefault("tracing", {}).setdefault("redaction", {})
        r["enabled"] = enabled
        if patterns is not None:
            r["patterns"] = patterns
        if custom_fields is not None:
            r["custom_fields"] = custom_fields
        return self

    def budget_usd(self, amount: float) -> ConfigBuilder:
        self._data.setdefault("cost", {})["budget_usd"] = amount
        return self

    def kill_on_budget(self, enabled: bool = True) -> ConfigBuilder:
        self._data.setdefault("cost", {})["kill_on_budget_exceeded"] = enabled
        return self

    def max_iterations(self, n: int) -> ConfigBuilder:
        self._data.setdefault("agent", {})["max_iterations"] = n
        return self

    def max_cost_per_run(self, usd: float) -> ConfigBuilder:
        self._data.setdefault("agent", {})["max_cost_per_run_usd"] = usd
        return self

    def tool_denylist(self, tools: list[str]) -> ConfigBuilder:
        self._data.setdefault("agent", {}).setdefault("tools", {})["denylist"] = tools
        return self

    def otlp(self, endpoint: str, *, enabled: bool = True) -> ConfigBuilder:
        o = self._data.setdefault("export", {}).setdefault("otlp", {})
        o["enabled"] = enabled
        o["endpoint"] = endpoint
        return self

    def build(self) -> LumenConfig:
        """Build config, applying env vars on top."""
        config = LumenConfig()
        config._merge_dict(self._data)
        config._merge_env()
        config.tracing.redaction.compile_patterns()
        return config


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_project_config() -> Path | None:
    """Walk up from cwd looking for lumen.toml."""
    try:
        current = Path.cwd()
    except OSError:
        return None
    for _ in range(20):  # safety bound
        candidate = current / "lumen.toml"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _set_if(src: dict, key: str, target: Any, attr: str, typ: type) -> None:
    """Set target.attr from src[key] if present, coercing to typ."""
    if key in src:
        try:
            setattr(target, attr, typ(src[key]))
        except (ValueError, TypeError):
            logger.debug(
                "Config key '%s' has invalid value %r for type %s, using default",
                key, src[key], typ.__name__,
            )


def _env_str(var: str, target: Any, attr: str) -> None:
    val = os.environ.get(var)
    if val is not None:
        setattr(target, attr, val)


def _env_int(var: str, target: Any, attr: str) -> None:
    val = os.environ.get(var)
    if val is not None:
        try:
            setattr(target, attr, int(val))
        except ValueError:
            logger.debug("Env var %s='%s' is not a valid integer, ignoring", var, val)


def _env_float(var: str, target: Any, attr: str) -> None:
    val = os.environ.get(var)
    if val is not None:
        try:
            setattr(target, attr, float(val))
        except ValueError:
            logger.debug("Env var %s='%s' is not a valid float, ignoring", var, val)


def _env_bool(var: str, target: Any, attr: str) -> None:
    val = os.environ.get(var)
    if val is not None:
        setattr(target, attr, val.lower() in ("true", "1", "yes"))
