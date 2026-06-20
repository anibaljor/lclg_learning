# Recipes & migration

Copy-adaptable starting points for the most common tasks, plus a legacy→v1 cheat sheet. These
follow the documented use cases (semantic search, RAG, SQL, multi-agent). For full tutorials,
fetch the matching `.md` page (links at the end).

Table of contents:
1. RAG agent (retriever as a tool)
2. SQL agent with human-in-the-loop review
3. Semantic search over documents
4. Multi-agent (router) skeleton
5. Custom RAG/SQL in pure LangGraph (when to)
6. Legacy → v1 cheat sheet

---

## 1. RAG agent (retriever as a tool)

The v1 way to do RAG is **not** `RetrievalQA` — it's an agent with a retriever exposed as a tool,
so the model decides when/what to retrieve.

```python
from langchain.agents import create_agent
# 1) Build a vector store + retriever (provider of your choice; see providers.md)
#    e.g. Chroma/Pinecone/pgvector + an embeddings model. Index your docs once.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2) Wrap retrieval as a tool — docstring tells the model when to use it
def search_docs(query: str) -> str:
    """Search the internal knowledge base for relevant passages. Use for any
    question about company policies, products, or documentation."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

# 3) The agent loop handles retrieve→answer, multi-hop, and "no retrieval needed"
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_docs],
    system_prompt="Answer using the knowledge base. Cite what you used; if unsure, say so.",
)
agent.invoke({"messages": [{"role": "user", "content": "What's our refund window?"}]})
```

For fine-grained control (grade retrieved docs, rewrite the query, decide retrieve-vs-answer as
explicit graph nodes) drop to LangGraph — see §5 and the Agentic RAG tutorial.

---

## 2. SQL agent with human-in-the-loop review

Give the agent DB tools, but gate write/expensive queries behind an approval interrupt.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# DB tools: list_tables, get_schema, run_query (read-only by default).
# Put run_query behind human approval via HITL middleware or a LangGraph interrupt
# (see langgraph.md §6) so a person confirms before execution.
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[list_tables, get_schema, run_query],
    system_prompt=("You are a careful SQL analyst. Inspect schema before querying. "
                   "Never modify data. Show the query you intend to run."),
    checkpointer=InMemorySaver(),   # required so the run can pause/resume for review
)
cfg = {"configurable": {"thread_id": "sql-session-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Top 5 customers by revenue last quarter"}]}, cfg)
```

---

## 3. Semantic search over documents

Pipeline: load → split → embed → store → query. Use a document loader (e.g.
`langchain-unstructured`, PDF loaders), a text splitter, an embeddings model, and a vector store.
Then `vectorstore.similarity_search(query, k=...)`. This is the building block under §1; expose
it as a tool to turn search into an agent.

---

## 4. Multi-agent (router) skeleton

Classify, then dispatch to a specialized `create_agent`. Implement the routing in LangGraph so
the topology is explicit (see `langgraph.md` §3).

```python
from langgraph.graph import StateGraph, MessagesState, START, END

billing = create_agent(model="anthropic:claude-sonnet-4-6", tools=[...], system_prompt="Billing")
tech    = create_agent(model="anthropic:claude-sonnet-4-6", tools=[...], system_prompt="Tech")

def classify(state): ...        # returns "billing" or "tech"
def run_billing(state): return billing.invoke({"messages": state["messages"]})
def run_tech(state):    return tech.invoke({"messages": state["messages"]})

b = StateGraph(MessagesState)
b.add_node(run_billing); b.add_node(run_tech)
b.add_conditional_edges(START, classify, {"billing": "run_billing", "tech": "run_tech"})
b.add_edge("run_billing", END); b.add_edge("run_tech", END)
graph = b.compile()
```

Other documented patterns: **subagents** (coordinator delegates), **handoffs** (one workflow
shifts between states), **skills** (load capabilities on demand). See
`https://docs.langchain.com/oss/python/langchain/multi-agent.md`.

---

## 5. Custom RAG/SQL in pure LangGraph (when to)

Use raw LangGraph instead of `create_agent` when you need explicit control the loop can't give:
grade-and-retry retrieval, query rewriting as its own node, deterministic guardrail steps mixed
with agentic ones, or fan-out/fan-in over multiple sources. Model each stage as a node, branch
with conditional edges, loop with cycles, and add a checkpointer for durability. Tutorials:
`https://docs.langchain.com/oss/python/langgraph/agentic-rag.md` and `.../sql-agent.md`.

---

## 6. Legacy → v1 cheat sheet

| Legacy (do NOT emit)                              | v1 replacement                                              |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `AgentExecutor`, `initialize_agent`, `AgentType`  | `create_agent(model=..., tools=..., system_prompt=...)`     |
| `LLMChain`, `ConversationChain`                    | call the model directly / `create_agent`                    |
| `SequentialChain`, `SimpleSequentialChain`        | LangGraph `StateGraph` with edges, or compose functions     |
| `RetrievalQA`, `ConversationalRetrievalChain`     | RAG agent with a retriever tool (§1) or LangGraph RAG (§5)  |
| `ConversationBufferMemory` / `*Memory`            | LangGraph checkpointer + `thread_id` (short-term); Store (long-term) |
| output parsers (`PydanticOutputParser`, …)        | `response_format=Schema` or `.with_structured_output()`     |
| `chain = prompt | llm | parser` (LCEL pipes)      | still valid for simple chains, but prefer `create_agent`/graph for agents |
| `from langchain.chat_models import ChatOpenAI`    | `model="openai:gpt-..."` string or `init_chat_model(...)`   |
| `Tool(...)` boilerplate                           | plain function + docstring, or `@tool` decorator            |

When unsure whether something is current, fetch the `.md` doc — appending `.md` to any
`docs.langchain.com` page returns clean markdown, and `https://docs.langchain.com/llms.txt`
indexes every page.

---

## Tutorial index (fetch `.md` for full walkthroughs)

- Semantic search: `/oss/python/langchain/knowledge-base.md`
- RAG agent: `/oss/python/langchain/rag.md`
- SQL agent: `/oss/python/langchain/sql-agent.md`
- Voice agent: `/oss/python/langchain/voice-agent.md`
- Custom RAG (LangGraph): `/oss/python/langgraph/agentic-rag.md`
- Custom SQL (LangGraph): `/oss/python/langgraph/sql-agent.md`
- Multi-agent: `/oss/python/langchain/multi-agent.md`
- Memory & context: `/oss/python/concepts/memory.md`, `/oss/python/concepts/context.md`
- Deep Agents: `/oss/python/deepagents/overview.md`

(prefix all with `https://docs.langchain.com`)
