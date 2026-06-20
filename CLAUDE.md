# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este repo

App Streamlit interactiva en español para enseñar LangChain y LangGraph con 10
ejemplos cortos (`examples/01_*.py` a `examples/10_*.py`). Cada ejemplo se ve
desde dos lugares: la UI de Streamlit (`app.py`) y un modo CLI standalone
(bloque `if __name__ == "__main__":` de cada archivo).

## Comandos

```bash
pip install -r requirements.txt
streamlit run app.py                    # abre en http://localhost:8501

# Probar un ejemplo individual por consola (sin Streamlit):
export OPENAI_API_KEY="..."              # o GOOGLE_API_KEY / ANTHROPIC_API_KEY
python examples/01_chat_basico.py

# LCLG_PROVIDER elige el proveedor en modo CLI: "OpenAI" (default), "Gemini" o "Claude"
LCLG_PROVIDER=Claude ANTHROPIC_API_KEY="..." python examples/09_langgraph_react_agent.py
```

No hay suite de tests ni linter configurado en este repo.

## Arquitectura

**Regla de seguridad central, no negociable:** la API key del usuario nunca
toca disco. `app.py` y `llm_factory.py` jamás leen `os.environ`, `.env` ni
`st.secrets`; la key vive solo en `st.session_state` (RAM, dura la sesión de
navegador) y se pasa explícitamente como argumento `api_key` en cada llamada.
La lectura de variables de entorno (`OPENAI_API_KEY`, etc.) existe
*exclusivamente* dentro del bloque `if __name__ == "__main__":` de cada
`examples/*.py`, para el modo de prueba por consola — nunca desde la app. Si
se toca `app.py` o `llm_factory.py`, no introducir ninguna lectura de env vars
fuera de ese patrón.

**`llm_factory.py`** es el único punto que sabe instanciar un chat model:
`get_llm(provider, api_key, model=None, temperature=0)` devuelve
`ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatAnthropic` según
`provider in ("OpenAI", "Gemini", "Claude")`. También centraliza
`classify_llm_error(exc)`, que traduce excepciones de los 3 SDKs (sin
importar sus jerarquías propias) a uno de 4 mensajes en español por
heurística de `status_code`/palabras clave: key faltante, key inválida,
rate limit, error de conexión. Todo el manejo de errores de LLM en el repo
pasa por esta función — no agregar manejo de excepciones ad-hoc en otro lado.

**Contrato de `examples/*.py`:** cada módulo expone una función pura que
recibe `(provider, api_key, ...contenido..., model=None, temperature=0)` y no
depende de estado global ni de Streamlit. Se llama `run_example` en casi
todos; el ejemplo 10 (único con *human-in-the-loop*) expone en cambio
`iniciar(...)` y `resumir(...)`, porque su ejecución se parte en dos llamadas
separadas por una pausa humana. Cada módulo también define
`SNIPPET_OBJECTS: list` con los objetos (funciones/clases) cuyo código fuente
se muestra en la UI vía `inspect.getsource` — si se agrega o renombra una
función relevante en un ejemplo, hay que mantener `SNIPPET_OBJECTS`
sincronizado o el snippet mostrado quedará incompleto.

**`app.py`** carga cada `examples/<modulo>.py` dinámicamente
(`importlib.import_module(f"examples.{nombre}")`) según la lista `CONCEPTOS`,
y renderiza una vista por concepto con: explicación, código fuente real
(`_render_snippet`, leído en vivo, nunca un resumen hardcodeado), botón
"Ejecutar" deshabilitado si falta la key (`_boton_ejecutar`), y resultado.
`_ejecutar(fn, *args, **kwargs)` es el único lugar que atrapa excepciones de
una llamada a un ejemplo: nunca deja escapar un stack trace crudo, separa
`LLMConfigError` (config inválida, detectada antes de llamar a la API) del
resto (que pasa por `classify_llm_error`).

**División LangChain vs. LangGraph en los ejemplos:** 1–6 son flujos lineales
(`prompt | llm | parser`, LCEL) — no necesitan grafo. 7–10 usan
`StateGraph` porque necesitan algo que LCEL no expresa: bifurcación según
estado (8), ciclo de tool-calling tipo ReAct con loop manual (9), o pausar y
persistir ejecución entre llamadas con un checkpointer (10). El ejemplo 7 es
deliberadamente el caso más simple (2 nodos, sin ciclos) para introducir la
mecánica de `StateGraph`/`TypedDict` antes de necesitar las capacidades más
avanzadas.

**Ejemplo 6 (RAG):** usa embeddings TF-IDF locales (`scikit-learn`) en vez de
los del proveedor — funciona igual con los 3 proveedores, no gasta cuota de
API, y Anthropic no ofrece API de embeddings propia.

**Ejemplo 10 (checkpoint + HITL):** el `checkpointer` (`InMemorySaver`) y el
`thread_id` se crean *fuera* de `iniciar`/`resumir` y se pasan como parámetro
a ambas llamadas — son el estado que persiste entre la pausa (`interrupt()`)
y la reanudación (`Command(resume=...)`). En la app viven en
`st.session_state` (`hitl_checkpointer`, `hitl_thread_id`), igual que el
`store` de memoria de conversación del ejemplo 4.

## Versiones de dependencias

`requirements.txt` fija versiones validadas con una instalación limpia antes
de escribir código (ver comentarios en el propio archivo). Notar en
particular: la integración de FAISS para LangChain viene de
`langchain-community` (el paquete standalone `langchain-faiss` en PyPI está
vacío), y `langchain` v1 reexporta tipos de mensaje desde `langchain.messages`
en vez de `langchain_core.messages` directamente en los ejemplos.
