"""OTLP/HTTP JSON exporter for Lumen traces (stdlib only).

Maps a completed AgentTrace dict to an OTLP ExportTraceServiceRequest and
POSTs it to the collector. Never raises from ``write_trace`` — failures
are logged so user code stays uninterrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("lumen")

_DEFAULT_TIMEOUT_SEC = 5.0


def normalize_otlp_endpoint(endpoint: str) -> str:
    """Ensure the URL targets the OTLP/HTTP traces path."""
    ep = endpoint.strip().rstrip("/")
    if not ep:
        return ep
    if ep.endswith("/v1/traces"):
        return ep
    return f"{ep}/v1/traces"


def _hex_trace_id(trace_id: str) -> str:
    """32-char hex (16 bytes) derived from the Lumen trace id."""
    return hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()[:32]


def _hex_span_id(seed: str) -> str:
    """16-char hex (8 bytes) derived from a stable seed."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _ms_to_nano_str(ms: Any) -> str:
    try:
        return str(int(ms) * 1_000_000)
    except (TypeError, ValueError):
        return "0"


def _attr_str(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _attr_double(key: str, value: float) -> dict[str, Any]:
    return {"key": key, "value": {"doubleValue": float(value)}}


def _attr_int(key: str, value: int) -> dict[str, Any]:
    # OTLP JSON encodes int64 as a string.
    return {"key": key, "value": {"intValue": str(int(value))}}


def _step_identity(step: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Return (kind, model, tool_name) from flat or kova step_type shapes."""
    model = step.get("model")
    tool_name = step.get("tool_name")
    kind_val = step.get("kind") or step.get("step_type")

    if isinstance(kind_val, dict) and kind_val:
        kind = next(iter(kind_val.keys()))
        detail = kind_val.get(kind)
        if isinstance(detail, dict):
            model = model or detail.get("model")
            tool_name = tool_name or detail.get("tool_name")
    elif kind_val is not None:
        kind = str(kind_val)
    else:
        kind = "step"

    return (
        str(kind),
        str(model) if model else None,
        str(tool_name) if tool_name else None,
    )


def lumen_trace_to_otlp(
    trace_data: dict[str, Any],
    *,
    service_name: str = "lumen",
) -> dict[str, Any]:
    """Convert a Lumen AgentTrace dict to OTLP ExportTraceServiceRequest JSON."""
    raw_tid = str(trace_data.get("trace_id", "unknown"))
    otlp_tid = _hex_trace_id(raw_tid)
    root_span_id = _hex_span_id(f"{raw_tid}:root")

    started_ms = trace_data.get("started_at_ms", 0)
    completed_ms = trace_data.get("completed_at_ms")
    if completed_ms is None:
        steps = trace_data.get("steps") or []
        end = int(started_ms) if started_ms else 0
        for step in steps:
            try:
                end = max(
                    end,
                    int(step.get("started_at_ms", 0))
                    + int(step.get("duration_ms", 0)),
                )
            except (TypeError, ValueError):
                continue
        completed_ms = end or started_ms

    status_raw = str(trace_data.get("status", "unknown")).lower()
    root_status: dict[str, Any]
    if status_raw in ("failed", "error", "crashed"):
        root_status = {"code": 2, "message": status_raw}
    else:
        root_status = {"code": 1}

    root_attrs: list[dict[str, Any]] = [
        _attr_str("lumen.trace_id", raw_tid),
        _attr_str("lumen.status", status_raw),
    ]
    if "total_cost_usd" in trace_data and trace_data["total_cost_usd"] is not None:
        try:
            root_attrs.append(
                _attr_double("lumen.total_cost_usd", float(trace_data["total_cost_usd"]))
            )
        except (TypeError, ValueError):
            pass
    if trace_data.get("agent_id"):
        root_attrs.append(_attr_str("lumen.agent_id", str(trace_data["agent_id"])))
    if trace_data.get("run_id"):
        root_attrs.append(_attr_str("lumen.run_id", str(trace_data["run_id"])))

    spans: list[dict[str, Any]] = [
        {
            "traceId": otlp_tid,
            "spanId": root_span_id,
            "name": f"lumen.trace:{raw_tid[:12]}",
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": _ms_to_nano_str(started_ms),
            "endTimeUnixNano": _ms_to_nano_str(completed_ms),
            "attributes": root_attrs,
            "status": root_status,
        }
    ]

    for idx, step in enumerate(trace_data.get("steps") or []):
        if not isinstance(step, dict):
            continue
        step_start = step.get("started_at_ms", started_ms)
        try:
            step_end = int(step_start) + int(step.get("duration_ms", 0))
        except (TypeError, ValueError):
            step_end = step_start

        kind, model, tool_name = _step_identity(step)
        name = str(
            step.get("name")
            or step.get("label")
            or tool_name
            or model
            or kind
            or "step"
        )
        step_attrs: list[dict[str, Any]] = [
            _attr_str("lumen.step.kind", kind),
            _attr_int("lumen.step.index", idx),
        ]
        if model:
            step_attrs.append(_attr_str("gen_ai.request.model", model))
        if tool_name:
            step_attrs.append(_attr_str("lumen.tool.name", tool_name))

        tokens = step.get("tokens") if isinstance(step.get("tokens"), dict) else {}
        tokens_in = (
            step.get("tokens_in")
            or step.get("input_tokens")
            or tokens.get("prompt_tokens")
        )
        tokens_out = (
            step.get("tokens_out")
            or step.get("output_tokens")
            or tokens.get("completion_tokens")
        )
        if tokens_in is not None:
            try:
                step_attrs.append(_attr_int("gen_ai.usage.input_tokens", int(tokens_in)))
            except (TypeError, ValueError):
                pass
        if tokens_out is not None:
            try:
                step_attrs.append(_attr_int("gen_ai.usage.output_tokens", int(tokens_out)))
            except (TypeError, ValueError):
                pass

        spans.append(
            {
                "traceId": otlp_tid,
                "spanId": _hex_span_id(f"{raw_tid}:step:{idx}"),
                "parentSpanId": root_span_id,
                "name": name[:128],
                "kind": 1,
                "startTimeUnixNano": _ms_to_nano_str(step_start),
                "endTimeUnixNano": _ms_to_nano_str(step_end),
                "attributes": step_attrs,
                "status": {"code": 1},
            }
        )

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attr_str("service.name", service_name),
                        _attr_str("telemetry.sdk.name", "lumen"),
                        _attr_str("telemetry.sdk.language", "python"),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "lumen", "version": "0.1.0"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


class OtlpHttpTraceWriter:
    """POST each completed trace as OTLP/HTTP JSON to a collector.

    Implements the TraceWriter protocol. Uses only the stdlib.
    """

    __slots__ = ("_endpoint", "_headers", "_timeout_sec", "_service_name")

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        service_name: str = "lumen",
    ) -> None:
        self._endpoint = normalize_otlp_endpoint(endpoint)
        self._headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self._timeout_sec = timeout_sec
        self._service_name = service_name

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        from lumen._errors import warn_trace_write

        trace_id = trace_data.get("trace_id", "unknown")
        if not self._endpoint:
            warn_trace_write(
                "OTLP endpoint is empty — skipping export.",
                trace_id=trace_id,
            )
            return

        try:
            payload = lumen_trace_to_otlp(
                trace_data, service_name=self._service_name
            )
            body = json.dumps(payload, default=str).encode("utf-8")
        except Exception as exc:
            warn_trace_write(
                f"OTLP payload build failed: {exc}",
                trace_id=trace_id,
            )
            return

        req_headers = {
            "Content-Type": "application/json",
            **self._headers,
        }
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as resp:
                # 2xx is success; collectors may return 200 with partial failures.
                if getattr(resp, "status", 200) >= 300:
                    warn_trace_write(
                        f"OTLP export HTTP {resp.status} from {self._endpoint}",
                        trace_id=trace_id,
                    )
        except urllib.error.HTTPError as exc:
            warn_trace_write(
                f"OTLP export HTTP {exc.code} from {self._endpoint}: {exc.reason}",
                trace_id=trace_id,
            )
        except Exception as exc:
            warn_trace_write(
                f"OTLP export failed ({type(exc).__name__}): {exc}",
                trace_id=trace_id,
            )
