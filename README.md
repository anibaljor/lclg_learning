# LangChain & LangGraph paso a paso

App interactiva de Streamlit para aprender **LangChain** y **LangGraph** con 10
ejemplos cortos y reales: desde un chat model básico hasta un agente ReAct con
loop manual y un demo de checkpointing + human-in-the-loop.

Cada concepto muestra, en la propia app: una explicación breve, el código real
que se ejecuta, un botón para correrlo en vivo, y el resultado de esa ejecución.

## Seguridad: cómo se maneja tu API key

**La app nunca incluye ni necesita una API key propia.** Vos pegás tu propia
key (de OpenAI, Google Gemini o Anthropic Claude) en un campo de tipo
contraseña en la barra lateral. Esa key:

- vive únicamente en `st.session_state` (memoria del proceso, durante tu sesión
  de navegador);
- **nunca** se escribe a disco, no se loguea, y no se manda a ningún lado salvo
  a la API del proveedor que elegiste, cuando apretás "Ejecutar";
- desaparece sola al cerrar o recargar la pestaña, o de inmediato si apretás el
  botón **"Borrar mi key"**.

`app.py` no lee `os.environ`, `.env` ni `st.secrets` en ningún punto: esas
fuentes están reservadas exclusivamente al modo de prueba por línea de comandos
de cada archivo en `examples/` (más abajo), y jamás se usan dentro de la app.

## Estructura del proyecto

```
app.py                  # App Streamlit (UI, navegación, manejo de la API key)
llm_factory.py          # Factory get_llm(provider, api_key, model, temperature)
requirements.txt
data/                   # Documentos de ejemplo para el demo de RAG
examples/
  01_chat_basico.py            # Chat model básico + invoke
  02_lcel_prompt.py            # Prompt templates + LCEL (prompt | llm | parser)
  03_structured_output.py      # Salida estructurada a un modelo Pydantic
  04_memory.py                 # Memoria de conversación (RunnableWithMessageHistory)
  05_tools_agent.py             # Tools + agente (create_agent)
  06_rag_minimo.py              # RAG mínimo con FAISS in-memory
  07_langgraph_estado.py        # StateGraph + TypedDict, 2 nodos
  08_langgraph_branching.py     # add_conditional_edges
  09_langgraph_react_agent.py   # Agente ReAct con loop manual (think→act→observe)
  10_langgraph_checkpoint_hitl.py  # Checkpointing + human-in-the-loop (bonus)
```

Cada `examples/*.py` expone una función pura (`run_example(...)`, o `iniciar`/
`resumir` en el caso 10) que recibe el proveedor y la API key como argumentos.
`app.py` la llama con la key de `st.session_state`. Cada archivo también corre
solo por consola, leyendo la key de una variable de entorno **solo en ese modo**:

```bash
export OPENAI_API_KEY="tu-key-personal"     # o GOOGLE_API_KEY / ANTHROPIC_API_KEY
python examples/01_chat_basico.py
```

(`LCLG_PROVIDER` elige el proveedor del modo CLI: `OpenAI` (default), `Gemini` o
`Claude`.)

## Instalación y uso local

Requiere Python 3.11+.

```bash
git clone <tu-fork-o-este-repo>
cd lclg_learning

python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Se abre en `http://localhost:8501`. La app arranca **sin ninguna key
configurada**: vas a ver un aviso pidiéndote que ingreses la tuya en la barra
lateral antes de poder ejecutar los ejemplos en vivo (las explicaciones y el
código se pueden leer sin key).

### ¿De dónde saco una API key?

- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/app/apikey
- **Anthropic Claude**: https://console.anthropic.com/settings/keys

Cualquiera de las 3 alcanza para usar toda la app; no hace falta tener las 3.

## Deploy en Streamlit Community Cloud

1. Subí este repo a GitHub (puede ser un fork). Tiene que ser un repo al que
   tu cuenta de GitHub tenga acceso (público, o privado autorizando la app).
2. Entrá a https://share.streamlit.io e iniciá sesión con tu cuenta de GitHub.
3. Apretá **"New app"**.
4. Elegí el repositorio, la rama (por ejemplo `main`), y como archivo principal
   `app.py`.
5. Apretá **"Deploy"**.

**No hace falta configurar ningún Secret.** La app no necesita ninguna API key
propia para arrancar: cada persona que la usa pega su propia key en la barra
lateral, en su propia sesión de navegador. La sección "Secrets" del deploy de
Streamlit Cloud puede quedar vacía.

`requirements.txt` fija versiones compatibles entre sí (verificadas con una
instalación limpia antes de escribir el código), para que el build en la nube
no falle por conflictos de dependencias.

## Notas sobre los ejemplos

- **RAG (ejemplo 6)**: usa embeddings TF-IDF 100% locales (`scikit-learn`), no
  los del proveedor. Así el demo funciona igual con los 3 proveedores, no gasta
  cuota de API, y usa poca RAM (importante en el tier gratuito de Streamlit
  Cloud). Anthropic ni siquiera ofrece una API de embeddings. En un proyecto
  real con más documentos, convendría `OpenAIEmbeddings` o
  `GoogleGenerativeAIEmbeddings` para mejor calidad semántica.
- **Errores**: la app nunca muestra un stack trace crudo. Los errores se
  clasifican en "falta la key", "key inválida o sin permisos", "límite de uso
  alcanzado" o "error de conexión", con un mensaje en español para cada caso
  (`llm_factory.classify_llm_error`).
- **LangChain vs. LangGraph**: los ejemplos 1 a 6 son flujos lineales (chains
  LCEL) — alcanza con LangChain. Los ejemplos 7 a 10 necesitan algo que LCEL no
  puede expresar solo: bifurcaciones según el estado (8), ciclos de
  tool-calling (9), o pausar y persistir la ejecución entre llamadas (10). El
  ejemplo 7 es intencionalmente el caso más simple, para mostrar la mecánica
  básica de un grafo antes de necesitar de verdad esas capacidades.
