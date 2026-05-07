"""Lumen integrations with AI agent frameworks."""

try:
    from lumen.integrations.langgraph_tracer import LumenTracer as LumenTracer
except ImportError:
    pass

try:
    from lumen.integrations.openai_tracer import patch_openai as patch_openai
except ImportError:
    pass

try:
    from lumen.integrations.anthropic_tracer import patch_anthropic as patch_anthropic
except ImportError:
    pass
