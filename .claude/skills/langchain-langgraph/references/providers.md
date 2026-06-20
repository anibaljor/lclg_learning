# Providers, models, and integrations

LangChain has 1000+ integrations across chat models, embeddings, tools/toolkits, document
loaders, vector stores, checkpointers, and sandboxes. A **provider** is a platform that hosts
models behind an API (OpenAI, Anthropic, Google, …). Most have a dedicated `langchain-<provider>`
package implementing LangChain's standard interfaces, so you can swap providers without changing
app code: install the package, pick a model name, done.

Table of contents:
1. The `"provider:model"` string
2. Install extras vs standalone packages
3. Provider quick reference
4. Embeddings, vector stores, checkpointers
5. Swapping providers / routing
6. Finding the exact current name

---

## 1. The `"provider:model"` string

`create_agent(model="provider:model", ...)` is the portable default. Examples seen in the docs:

| Provider        | Example model string                         |
| --------------- | -------------------------------------------- |
| Anthropic       | `anthropic:claude-sonnet-4-6` (or bare `claude-sonnet-4-6`) |
| OpenAI          | `openai:gpt-5.5`                             |
| Google GenAI    | `google_genai:gemini-2.5-flash-lite`        |
| Azure OpenAI    | `azure_openai:gpt-5.5` (+ `azure_deployment=`) |
| OpenRouter      | `openrouter:anthropic/claude-sonnet-4-6`    |
| Fireworks       | `fireworks:accounts/fireworks/models/...`   |
| Ollama (local)  | `ollama:devstral-2`                         |
| Baseten         | `baseten:zai-org/GLM-5`                      |

When a provider has **no prefix shortcut**, pass the raw model id plus `model_provider=`:

```python
agent = create_agent(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",        # AWS Bedrock
    tools=[...], system_prompt="...",
)
# HuggingFace example:
create_agent(model="microsoft/Phi-3-mini-4k-instruct", model_provider="huggingface",
             tools=[...], temperature=0.7, max_tokens=1024)
```

To configure a model object directly, use `init_chat_model("provider:model", temperature=...)`
from `langchain.chat_models` and pass it as `model=`.

> Model **names** go stale fast. Don't hardcode a version from memory for the user — confirm the
> current one (see §6), or use the name the user already specified.

---

## 2. Install extras vs standalone packages

- **Extras** bundle the common providers into LangChain:
  `pip install -qU langchain "langchain[anthropic]"` (also `[openai]`, `[google-genai]`,
  `[huggingface]`).
- **Standalone packages** for everything else: `pip install -qU langchain-aws`,
  `langchain-ollama`, `langchain-groq`, `langchain-mistralai`, `langchain-cohere`,
  `langchain-openrouter`, `langchain-fireworks`, `langchain-deepseek`, `langchain-xai`, …
- Community/unmaintained integrations live in `langchain-community`.

---

## 3. Provider quick reference (popular packages)

Chat/model providers: `langchain-openai`, `langchain-anthropic`, `langchain-google-vertexai`,
`langchain-google-genai`, `langchain-aws`, `langchain-ollama`, `langchain-groq`,
`langchain-huggingface`, `langchain-mistralai`, `langchain-cohere`, `langchain-deepseek`,
`langchain-xai`, `langchain-perplexity`, `langchain-together`, `langchain-fireworks`,
`langchain-cerebras`, `langchain-nvidia-ai-endpoints`, `langchain-ibm`, `langchain-azure-ai`,
`databricks-langchain`, `langchain-litellm` (multi-provider router), `langchain-nebius`,
`langchain-sambanova`, `langchain-openrouter`.

Tools/search: `langchain-tavily` (web search). Sandboxes for code execution: see
`https://docs.langchain.com/oss/python/integrations/sandboxes.md`.

---

## 4. Embeddings, vector stores, checkpointers

For RAG and persistence you'll pair a chat model with these:

- **Vector stores**: `langchain-chroma`, `langchain-pinecone`, `langchain-qdrant`,
  `langchain-milvus`, `langchain-postgres` (pgvector), `langchain-mongodb` (Atlas),
  `langchain-redis`, `langchain-elasticsearch`, `langchain-astradb`, `langchain-neo4j`,
  `langchain-graph-retriever`.
- **Embeddings**: provider packages expose embedding classes (e.g. OpenAI, Cohere, HuggingFace,
  Google). Browse `https://docs.langchain.com/oss/python/integrations/embeddings.md`.
- **Checkpointers** (LangGraph persistence): `langgraph-checkpoint-postgres`,
  `langgraph-checkpoint-sqlite`, plus Redis/MongoDB. Listed at
  `https://docs.langchain.com/oss/python/integrations/checkpointers.md`.

---

## 5. Swapping providers / routing

Because every chat model implements the same interface, switching is usually a one-line model
string change. For dynamic selection at runtime use a router package (`langchain-litellm`) or
LangGraph conditional edges to pick a model per request. Keep the model string in config/env so
the rest of the app stays untouched.

---

## 6. Finding the exact current name

- Provider/model concepts and how to discover names:
  `https://docs.langchain.com/oss/python/concepts/providers-and-models.md`
- Full integrations index: `https://docs.langchain.com/oss/python/integrations/providers/overview.md`
  and `.../all_providers.md`
- API reference: `https://reference.langchain.com/python/`
- Page index for everything: `https://docs.langchain.com/llms.txt`

When in doubt, fetch the `.md` page rather than guessing a package name or model id.
