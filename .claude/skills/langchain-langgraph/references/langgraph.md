# LangGraph — orchestration runtime

LangGraph is a low-level orchestration framework + runtime for **stateful, long-running** agents
and workflows. It does not abstract prompts or architecture — it gives you durable execution,
streaming, human-in-the-loop, persistence, and memory. Use it when control flow exceeds a simple
tool-calling loop, or when you need to mix deterministic and agentic steps. It is inspired by
Pregel / Apache Beam (message-passing over a graph) and can be used without LangChain.

Table of contents:
1. Graph API basics (StateGraph)
2. State, channels, and reducers
3. Edges: normal, conditional, cycles
4. Compiling and running
5. Persistence & checkpointers
6. Human-in-the-loop (interrupts)
7. Memory: short-term vs long-term
8. Streaming
9. Functional API (`@entrypoint` / `@task`)

---

## 1. Graph API basics (StateGraph)

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

builder = StateGraph(MessagesState)
builder.add_node(mock_llm)            # node name defaults to the function name
builder.add_edge(START, "mock_llm")   # entry
builder.add_edge("mock_llm", END)     # exit
graph = builder.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

A **node** is a function `state -> partial-state-update` (returns only the keys it changes).
`START` and `END` are sentinel nodes for the entry/exit points. You wire nodes with edges.

---

## 2. State, channels, and reducers

State is a typed dict (a `TypedDict`, dataclass, or Pydantic model). `MessagesState` is the
built-in convenience state with a `messages` channel that uses an **add-messages reducer** (node
updates are appended, not overwritten). Define custom state when you need extra channels:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: append
    retrieved_docs: list                      # no reducer: overwrite
    step_count: int
```

A **reducer** (`Annotated[type, reducer_fn]`) controls how concurrent/successive updates to a
channel combine. Without one, the latest write replaces the value. This is the core of how
parallel branches merge cleanly.

---

## 3. Edges: normal, conditional, cycles

- **Normal edge** — always go A → B: `builder.add_edge("a", "b")`.
- **Conditional edge** — branch on state via a router function returning the next node name(s):

```python
def route(state: State) -> str:
    return "tools" if needs_tool(state) else END

builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
```

- **Cycles** — loop by edging back (e.g. `tools` → `agent`). Cycles are first-class; this is how
  tool-calling loops and reflection/retry patterns are built. Always ensure a path to `END`.

---

## 4. Compiling and running

`builder.compile(checkpointer=..., store=..., interrupt_before=[...], interrupt_after=[...])`
returns a runnable graph exposing `invoke`, `ainvoke`, `stream`, `astream`, `get_state`,
`update_state`, and `get_graph().draw_mermaid()` for visualization. Pass runtime config such as
`thread_id` via the second arg: `{"configurable": {"thread_id": "..."}}`.

---

## 5. Persistence & checkpointers

A **checkpointer** snapshots state after every super-step, keyed by `thread_id`. This gives you,
for free: durable execution (resume after a crash), time-travel (`get_state_history`), and
multi-turn memory. Without it, runs are stateless.

```python
from langgraph.checkpoint.memory import InMemorySaver   # dev only
graph = builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "abc"}}
graph.invoke({"messages": [...]}, cfg)   # state for thread "abc" persists
```

Production checkpointers: `langgraph-checkpoint-postgres`, `langgraph-checkpoint-sqlite`, plus
Redis/MongoDB backends (see `providers.md`). Set up the backing table once (each saver has a
`.setup()` / migration step).

---

## 6. Human-in-the-loop (interrupts)

Pause the graph, surface state to a human, then resume — for approvals, edits, or tool review.
Two mechanisms:

- **Static**: `compile(interrupt_before=["sensitive_node"])` to always pause before a node.
- **Dynamic**: call `interrupt(payload)` inside a node to pause and emit data for the human.

```python
from langgraph.types import interrupt, Command

def review(state):
    decision = interrupt({"proposed_action": state["draft"]})  # pauses here
    return {"approved": decision}

# ... later, resume with the human's input:
graph.invoke(Command(resume="approve"), cfg)
```

Resuming requires a checkpointer (the paused state must be persisted). Inspect/modify a paused
run with `get_state` / `update_state`.

---

## 7. Memory: short-term vs long-term

- **Short-term (working) memory** = the thread's state, kept by the checkpointer and scoped to a
  `thread_id`. This is conversation history within one session/thread.
- **Long-term memory** = a **Store**, shared *across* threads, for facts/preferences you want to
  recall later. Compile with `store=...` and read/write namespaced keys inside nodes.

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
graph = builder.compile(checkpointer=InMemorySaver(), store=store)
# in a node: store.put((user_id, "facts"), key, value) / store.search(...)
```

Production stores include Postgres-backed variants. For semantic recall, configure the store with
an embeddings index.

---

## 8. Streaming

```python
for chunk in graph.stream(inputs, cfg, stream_mode="updates"):
    ...
```

Modes: `values` (full state each step), `updates` (per-node deltas), `messages` (LLM tokens),
`custom` (data you emit via `get_stream_writer()`), and `debug`. Pass a list to combine modes,
e.g. `stream_mode=["updates", "messages"]`. Use `astream` for async; `astream_events` for a
fine-grained event stream suitable for rich UIs.

---

## 9. Functional API (`@entrypoint` / `@task`)

When a graph would be overkill, build the agent as a single function. `@task` marks a unit of
work (memoized per checkpoint); `@entrypoint(checkpointer=...)` defines the durable top-level
function. You keep persistence, streaming, and human-in-the-loop without declaring nodes/edges.

```python
from langgraph.func import entrypoint, task

@task
def step(x): ...

@entrypoint(checkpointer=InMemorySaver())
def workflow(inputs):
    a = step(inputs).result()
    return a
```

Choose Graph API for explicit, inspectable topology; Functional API for control flow that's
naturally expressed as ordinary Python. Both run on the same runtime. Current details:
`https://docs.langchain.com/oss/python/langgraph/functional-api.md` and
`https://docs.langchain.com/oss/python/langgraph/graph-api.md`.
