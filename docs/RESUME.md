# Crash → resume with LangGraph + Lumen

Lumen does **not** re-run your agent for you. After a process crash you keep the
same `thread_id`, load the last checkpoint, and invoke again. This page is the
exact ritual.

## One-time setup

```bash
pip install -e "./lumen-sdk[langgraph]"
```

```python
from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
from langgraph.prebuilt import create_react_agent

checkpointer = LumenCheckpointer(storage_dir="./checkpoints")
tracer = LumenTracer(trace_dir="./traces")
graph = create_react_agent(model, tools, checkpointer=checkpointer)

THREAD = "research-1"
config = {
    "configurable": {"thread_id": THREAD},
    "callbacks": [tracer],
}
graph.invoke({"messages": [("user", "…")]}, config=config)
```

## After a crash

```python
checkpointer = LumenCheckpointer(storage_dir="./checkpoints")  # same dir
print(checkpointer.list_threads())          # ["research-1", …]
print(checkpointer.resume_hint("research-1"))

cfg = checkpointer.latest_config("research-1")
graph.invoke(
    None,  # continue; do not re-send the original user turn
    config={
        "configurable": cfg["configurable"],
        "callbacks": [LumenTracer(trace_dir="./traces")],
    },
)
```

Then inspect / share:

```bash
lumen traces
lumen export <trace-id>          # lifecycle HTML
lumen cost --format md           # pasteable spend summary
```

## Rules of thumb

- **Same `storage_dir` + same `thread_id`** or you start a new run.
- Prefer `resume_hint` / `latest_config` over guessing checkpoint ids.
- Connected Kova WAL recovery is a deeper path — see [CONNECTED.md](CONNECTED.md).
  LangGraph + `LumenCheckpointer` is the Core (no Kova) resume story.
