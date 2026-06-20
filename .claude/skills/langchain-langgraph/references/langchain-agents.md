# LangChain v1 agents — `create_agent` in depth

Table of contents:
1. Mental model: Agent = Model + Harness
2. Tools
3. Models and `content_blocks`
4. Structured output
5. Middleware (the configuration mechanism)
6. Streaming
7. Memory (conversation threads)
8. Multi-agent patterns
9. Common mistakes

---

## 1. Mental model: Agent = Model + Harness

`create_agent` is a minimal, highly configurable **harness**: everything around the model loop —
the system prompt, the tools, and the middleware that shapes behavior. You start from primitives
and compose exactly what the use case needs. Under the hood it is a LangGraph graph, so it
inherits durable execution, persistence, and human-in-the-loop.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    system_prompt="You are a helpful assistant",
    # middleware=[...],            # optional, see §5
    # response_format=MySchema,    # optional structured output, see §4
    # checkpointer=...,            # optional memory/persistence, see §7
)
```

Invoke with a messages dict and read the last message:

```python
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
final = result["messages"][-1]
print(final.content_blocks)   # structured; or final.content for plain text
```

`ainvoke` is the async equivalent. For token/step streaming, see §6.

---

## 2. Tools

A tool is a plain Python function. The **docstring becomes the description the model reads**, and
type hints become the argument schema — so write both for the model, not just for humans.

```python
def search_orders(customer_id: str, status: str = "open") -> str:
    """Look up a customer's orders. status is one of: open, shipped, all."""
    ...
    return result_text
```

Guidelines:
- Keep one clear responsibility per tool; name it as an action.
- Return strings or JSON-serializable data the model can reason over.
- For richer control (custom args schema, injected state, error handling), use the `@tool`
  decorator from `langchain_core.tools` and/or accept injected state. Fetch
  `https://docs.langchain.com/oss/python/langchain/tools.md` for the current decorator options.
- Prebuilt/third-party tools come from provider packages and `langchain-community`
  (e.g. `langchain-tavily` for web search). See `providers.md`.

---

## 3. Models and `content_blocks`

Specify the model as a `"provider:model"` string (see `providers.md` for the full list and the
`model_provider=` fallback). v1 responses expose **`content_blocks`**: a typed list of parts
(text, tool calls, reasoning/thinking, citations, etc.). Prefer it over the flat `.content`
string whenever you need to inspect or route on the structure:

```python
for block in result["messages"][-1].content_blocks:
    # block has a "type" (e.g. "text", "tool_call", "reasoning") and payload
    ...
```

To configure the model object directly (temperature, max tokens, etc.) instead of a string,
build it with `init_chat_model("provider:model", temperature=...)` from `langchain.chat_models`
and pass that object as `model=`.

---

## 4. Structured output

Pass a Pydantic model (or TypedDict / JSON schema) as `response_format` to force a typed result:

```python
from pydantic import BaseModel

class Triage(BaseModel):
    category: str
    urgency: int
    summary: str

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    response_format=Triage,
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
result["structured_response"]   # -> Triage instance
```

This is the v1 replacement for hand-rolled output parsers. For schema-only extraction without
tools, you can also call a model's `.with_structured_output(Schema)` directly.

---

## 5. Middleware (the configuration mechanism)

Middleware is how you add capability to the harness **incrementally** — the v1 replacement for
subclassing `AgentExecutor`. Each middleware can hook into the loop (before/after model, before/
after tools, on error) to implement guardrails, retries, summarization, routing, dynamic
tool-selection, PII redaction, human approval, and more.

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    middleware=[
        # e.g. summarization, human-in-the-loop, guardrails, retry policies
    ],
)
```

Because middleware composes, you only pay for what you add. For the current catalogue of
built-in middleware and the hook API to write your own, fetch
`https://docs.langchain.com/oss/python/langchain/middleware.md`. Provider/3rd-party middleware
is listed at `https://docs.langchain.com/oss/python/integrations/middleware.md`.

---

## 6. Streaming

Stream steps and/or tokens instead of waiting for the full result:

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "..."}]},
    stream_mode="values",       # "values" | "updates" | "messages" | "custom"
):
    print(chunk)
```

- `values` — full state after each step.
- `updates` — only what each node changed.
- `messages` — token-level streaming of LLM output (use for chat UIs).
Async: `astream`. Streaming is a LangGraph capability; see `langgraph.md` for mixing modes.

---

## 7. Memory (conversation threads)

For multi-turn memory, attach a checkpointer and pass a `thread_id`. The agent then persists and
reloads state per thread automatically — no `ConversationBufferMemory`.

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model="anthropic:claude-sonnet-4-6", tools=[...],
                     checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm José"}]}, cfg)
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, cfg)  # remembers
```

Use a durable checkpointer in production (`langgraph-checkpoint-postgres`,
`langgraph-checkpoint-sqlite`, Redis, MongoDB — see `providers.md`). For knowledge that should
persist *across* threads (long-term memory), use a Store; details in `langgraph.md`.

---

## 8. Multi-agent patterns

Blend LangChain agents with LangGraph orchestration. The four documented patterns:

- **Subagents** — a coordinator agent delegates sub-tasks to specialized agents (e.g. a personal
  assistant). Good for decomposition.
- **Handoffs** — a single workflow transitions control between specialized states/agents
  (e.g. customer support tiers).
- **Router** — classify the query, then dispatch to the right specialized agent
  (e.g. multi-source knowledge base).
- **Skills** — load specialized capabilities on demand via progressive context loading
  (e.g. a SQL assistant that pulls in only the schema/tools it needs).

For non-trivial multi-agent topology, implement the orchestration in LangGraph (nodes = agents,
edges = handoff logic) and keep each agent a `create_agent`. Recipes in `recipes.md`; current
walkthroughs at `https://docs.langchain.com/oss/python/langchain/multi-agent.md`.

---

## 9. Common mistakes

- Passing a bare model object when a `"provider:model"` string would do (both work, but the
  string is the documented default and keeps code portable).
- Reading `.content` and missing tool calls / reasoning that live in `.content_blocks`.
- Re-implementing memory or output parsing by hand instead of using `checkpointer` /
  `response_format`.
- Forgetting that tool **docstrings are prompts** — vague docstrings cause bad tool use.
- Reaching for `AgentExecutor`/`initialize_agent` — that's the legacy API; use `create_agent`.
