---
name: langchain-langgraph
description: >-
  Write correct, current LangChain v1 and LangGraph (Python) code for building AI agents,
  tool-calling loops, RAG pipelines, SQL agents, multi-agent systems, and stateful/durable
  workflows. Use this skill WHENEVER the task involves LangChain, LangGraph, `create_agent`,
  `StateGraph`, agent middleware, checkpointers/persistence, human-in-the-loop interrupts,
  LangChain provider/model strings (e.g. "anthropic:claude-...", "openai:gpt-..."), Deep Agents,
  or LangSmith tracing — even if the user only says "agent framework", "build an agent in Python",
  "RAG with LangChain", or pastes LangChain/LangGraph code to debug. The LangChain v1 API
  (`create_agent`, middleware, `content_blocks`) is DIFFERENT from older patterns
  (`AgentExecutor`, `initialize_agent`, `LLMChain`, `RunnableSequence`) that appear in stale
  training data — consult this skill instead of relying on memory.
---

# LangChain v1 + LangGraph (Python)

Authoritative patterns for the **current** LangChain / LangGraph stack. The single most
important reason this skill exists: the v1 API diverged sharply from what most pretraining
data contains. If you reach for `AgentExecutor`, `initialize_agent`, `LLMChain`,
`ConversationChain`, `RetrievalQA`, or `Runnable` pipe-chains, **stop** — those are legacy.
The current way to build an agent is `create_agent` (LangChain) over `StateGraph` (LangGraph).

## How the pieces fit (pick the right layer)

Decide which layer the task needs before writing code. From highest-level to lowest:

- **Deep Agents** — batteries-included harness: planning, subagents, a virtual filesystem,
  automatic context compression. Reach for it when the user wants a research/analysis agent
  "that just works" without wiring internals.
- **LangChain (`create_agent`)** — a minimal, highly configurable agent harness =
  model + tools + prompt + middleware. This is the **default** answer for "build an agent".
- **LangGraph (`StateGraph`)** — low-level orchestration runtime: durable execution, custom
  graph topology, persistence, human-in-the-loop, streaming. Use when the control flow is more
  than a tool-calling loop (branching, cycles, mixing deterministic + agentic steps), or when
  `create_agent` can't express the needed customization.
- **LangSmith** — observability/eval/deployment layer, framework-agnostic. Always available
  as the tracing+eval backend regardless of which of the above you chose.

Rule of thumb: **start as high as possible.** Drop to LangGraph only when you hit a wall with
`create_agent`. You do not need LangChain to use LangGraph, and vice versa, but LangChain's
agents are built on LangGraph, so you can always reach the lower layer when needed.

## Install

```bash
# LangChain agents (pick the provider extra you need)
pip install -qU langchain "langchain[anthropic]"     # or [openai], [google-genai], [huggingface]
# LangGraph (orchestration runtime) — only if you drop to the graph layer
pip install -U langgraph
# Provider-specific packages when there is no extra (see references/providers.md)
pip install -qU langchain-aws langchain-ollama langchain-groq   # examples
```

For tracing, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in the environment — no code
changes required; LangChain/LangGraph auto-instrument.

## The 30-second agent (LangChain)

This is the canonical shape. Tools are plain Python functions; their **docstring is the tool
description the model sees**, so write it for the model.

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",   # "provider:model" string
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)   # v1: read structured content_blocks
```

Key conventions, all easy to get wrong from stale memory:
- **Model is a string** in `"provider:model"` form (e.g. `"openai:gpt-5.5"`,
  `"google_genai:gemini-2.5-flash"`, `"anthropic:claude-sonnet-4-6"`). For providers without a
  prefix shortcut, pass `model=...` plus `model_provider=...`. See `references/providers.md`.
- **Input/output are `messages`.** Invoke with `{"messages": [...]}`; read the last message.
  Prefer `.content_blocks` (v1 structured output) over `.content` when you need typed parts
  (text, tool calls, reasoning, citations).
- **Behavior is shaped by middleware**, not by subclassing executors. Guardrails, retries,
  summarization, routing, human-in-the-loop, tool-call policies are all middleware. See
  `references/langchain-agents.md`.

## The 30-second graph (LangGraph)

When you need explicit topology instead of a loop:

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def my_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

builder = StateGraph(MessagesState)
builder.add_node(my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

Nodes are functions `state -> partial state update`. Edges define flow; use **conditional
edges** for branching and **cycles** for loops. Add a **checkpointer** at `compile()` to get
persistence, memory, and human-in-the-loop for free. Full API, conditional edges, the
Functional API (`@entrypoint`/`@task`), persistence, interrupts, and streaming are in
`references/langgraph.md`.

## Reference files — read the one that matches the task

Load only what you need (progressive disclosure). Each file has a table of contents.

- **`references/langchain-agents.md`** — `create_agent` in depth: tools, structured output,
  middleware catalogue, streaming, memory, multi-agent (subagents/handoffs/router), models &
  `content_blocks`. Read for anything built on the LangChain harness.
- **`references/langgraph.md`** — `StateGraph` Graph API + Functional API, state & reducers,
  conditional edges/cycles, checkpointers & persistence, human-in-the-loop interrupts,
  short/long-term memory, streaming modes. Read when dropping to the orchestration layer.
- **`references/providers.md`** — provider packages, the `"provider:model"` string format,
  install extras, picking chat models / embeddings / vector stores / checkpointers, and how to
  swap providers without touching app code.
- **`references/recipes.md`** — copy-adaptable end-to-end recipes: RAG agent, SQL agent (with
  HITL review), semantic search, multi-agent patterns, plus the legacy→v1 migration cheat sheet.

## Staying current (do this when unsure)

The docs are kept in sync and are machine-readable. When an API detail is uncertain or the
user is on a newer version than this skill assumes, **fetch the live docs** rather than guessing:

- Append `.md` to any docs page to get clean markdown, e.g.
  `https://docs.langchain.com/oss/python/langchain/agents.md`.
- The full page index is at `https://docs.langchain.com/llms.txt` — fetch it to discover the
  exact path for a topic before exploring.
- The API reference lives at `https://reference.langchain.com/python/`.

Prefer this over inventing signatures. LangChain ships frequently; a 30-second fetch beats a
plausible-but-wrong import.

## Anti-patterns (legacy — do not emit)

If you catch yourself writing any of these, switch to the v1 equivalent (cheat sheet in
`references/recipes.md`):

- `AgentExecutor`, `initialize_agent`, `AgentType.*` → `create_agent`
- `LLMChain`, `ConversationChain`, `SequentialChain` → compose with `create_agent` / LangGraph
- `RetrievalQA`, `ConversationalRetrievalChain` → a RAG agent with a retriever tool
  (`references/recipes.md`)
- `ConversationBufferMemory` and friends → LangGraph checkpointer + thread_id, or middleware
- `from langchain.chat_models import ChatOpenAI` ad-hoc wiring → `model="openai:..."` string,
  or `init_chat_model` / the provider package (`references/providers.md`)
