"""OTLP/HTTP JSON exporter tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from lumen._otlp import (
    OtlpHttpTraceWriter,
    lumen_trace_to_otlp,
    normalize_otlp_endpoint,
)
from lumen.config import LumenConfig
from lumen.instrument import _create_writer
from lumen._writers import CompositeWriter, FileTraceWriter


SAMPLE_TRACE: dict[str, Any] = {
    "trace_id": "trace_abc123",
    "agent_id": "research-agent",
    "status": "Completed",
    "total_cost_usd": 0.012,
    "started_at_ms": 1710000000000,
    "completed_at_ms": 1710000004900,
    "steps": [
        {
            "step_num": 0,
            "step_type": {
                "LlmCall": {"model": "claude-sonnet-4-6", "finish_reason": "tool_use"}
            },
            "started_at_ms": 1710000000000,
            "duration_ms": 1200,
            "tokens": {"prompt_tokens": 500, "completion_tokens": 100},
        },
        {
            "step_num": 1,
            "step_type": {"ToolCall": {"tool_name": "search_web", "success": True}},
            "started_at_ms": 1710000001200,
            "duration_ms": 800,
            "tokens": None,
        },
    ],
}


class TestNormalizeEndpoint:
    def test_appends_v1_traces(self) -> None:
        assert (
            normalize_otlp_endpoint("http://localhost:4318")
            == "http://localhost:4318/v1/traces"
        )

    def test_keeps_existing_path(self) -> None:
        assert (
            normalize_otlp_endpoint("http://otel:4318/v1/traces/")
            == "http://otel:4318/v1/traces"
        )


class TestLumenTraceToOtlp:
    def test_root_and_step_spans(self) -> None:
        payload = lumen_trace_to_otlp(SAMPLE_TRACE)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 3  # root + 2 steps
        root, llm, tool = spans
        assert len(root["traceId"]) == 32
        assert len(root["spanId"]) == 16
        assert llm["parentSpanId"] == root["spanId"]
        assert tool["parentSpanId"] == root["spanId"]
        assert llm["traceId"] == root["traceId"]

        llm_attrs = {a["key"]: a["value"] for a in llm["attributes"]}
        assert llm_attrs["lumen.step.kind"]["stringValue"] == "LlmCall"
        assert llm_attrs["gen_ai.request.model"]["stringValue"] == "claude-sonnet-4-6"
        assert llm_attrs["gen_ai.usage.input_tokens"]["intValue"] == "500"

        tool_attrs = {a["key"]: a["value"] for a in tool["attributes"]}
        assert tool_attrs["lumen.tool.name"]["stringValue"] == "search_web"

    def test_failed_status_maps_to_error(self) -> None:
        failed = {**SAMPLE_TRACE, "status": "Failed"}
        spans = lumen_trace_to_otlp(failed)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert spans[0]["status"]["code"] == 2


class _CaptureHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        _CaptureHandler.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class TestOtlpHttpTraceWriter:
    def test_posts_json_to_collector(self) -> None:
        _CaptureHandler.requests = []
        server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            writer = OtlpHttpTraceWriter(
                f"http://127.0.0.1:{port}",
                headers={"Authorization": "Bearer test"},
            )
            writer.write_trace(SAMPLE_TRACE)
        finally:
            server.shutdown()

        assert len(_CaptureHandler.requests) == 1
        req = _CaptureHandler.requests[0]
        assert req["path"] == "/v1/traces"
        assert req["headers"].get("Authorization") == "Bearer test"
        assert "resourceSpans" in req["body"]
        spans = req["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 3

    def test_http_error_does_not_raise(self) -> None:
        class _FailHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.send_response(503)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        server = HTTPServer(("127.0.0.1", 0), _FailHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.warns(Warning, match="OTLP export HTTP 503"):
                OtlpHttpTraceWriter(f"http://127.0.0.1:{port}").write_trace(
                    SAMPLE_TRACE
                )
        finally:
            server.shutdown()


def _unwrap_base(writer: Any) -> Any:
    """Peel BatchTraceWriter if present to inspect the storage backend."""
    inner = getattr(writer, "_writer", None) or getattr(writer, "writer", None)
    return inner if inner is not None else writer


class TestCreateWriterWiring:
    def test_composite_when_otlp_enabled(self, tmp_path: Any) -> None:
        cfg = LumenConfig.from_dict(
            {
                "tracing": {"trace_dir": str(tmp_path), "buffer_size": 1},
                "export": {
                    "otlp": {
                        "enabled": True,
                        "endpoint": "http://127.0.0.1:9",
                    }
                },
            }
        )
        writer = _unwrap_base(_create_writer(cfg))
        assert isinstance(writer, CompositeWriter)
        kinds = {type(w).__name__ for w in writer.writers}
        assert kinds == {"FileTraceWriter", "OtlpHttpTraceWriter"}

    def test_file_only_by_default(self, tmp_path: Any) -> None:
        cfg = LumenConfig.from_dict(
            {"tracing": {"trace_dir": str(tmp_path), "buffer_size": 1}}
        )
        writer = _unwrap_base(_create_writer(cfg))
        assert isinstance(writer, FileTraceWriter)
