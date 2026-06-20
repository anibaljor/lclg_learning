"""App Streamlit — LangChain & LangGraph paso a paso.

Principio de seguridad central de este archivo: la API key del usuario SOLO
vive en `st.session_state`, durante esta sesión de navegador. Este archivo NO
usa `os.environ`, `.env` ni `st.secrets` en ningún momento — esas fuentes
quedan reservadas exclusivamente al modo CLI de cada `examples/*.py` (ver sus
bloques `if __name__ == "__main__":`), nunca a esta app. La key nunca se
escribe a disco, no se loguea, y solo se manda al proveedor de LLM elegido
cuando el usuario aprieta "Ejecutar".
"""
from __future__ import annotations

import importlib
import inspect
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from llm_factory import DEFAULT_MODELS, PROVIDERS, LLMConfigError, classify_llm_error

ROOT_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="LangChain & LangGraph paso a paso", page_icon="🧩", layout="wide")

CONCEPTOS = [
    {"id": "01", "modulo": "01_chat_basico", "titulo": "1. Chat básico + invoke", "seccion": "LangChain"},
    {"id": "02", "modulo": "02_lcel_prompt", "titulo": "2. Prompt templates + LCEL", "seccion": "LangChain"},
    {"id": "03", "modulo": "03_structured_output", "titulo": "3. Salida estructurada (Pydantic)", "seccion": "LangChain"},
    {"id": "04", "modulo": "04_memory", "titulo": "4. Memoria de conversación", "seccion": "LangChain"},
    {"id": "05", "modulo": "05_tools_agent", "titulo": "5. Tools + agente", "seccion": "LangChain"},
    {"id": "06", "modulo": "06_rag_minimo", "titulo": "6. RAG mínimo (FAISS)", "seccion": "LangChain"},
    {"id": "07", "modulo": "07_langgraph_estado", "titulo": "7. Estado tipado (StateGraph)", "seccion": "LangGraph"},
    {"id": "08", "modulo": "08_langgraph_branching", "titulo": "8. Branching condicional", "seccion": "LangGraph"},
    {"id": "09", "modulo": "09_langgraph_react_agent", "titulo": "9. Agente ReAct manual", "seccion": "LangGraph"},
    {"id": "10", "modulo": "10_langgraph_checkpoint_hitl", "titulo": "10 (bonus). Checkpoint + Human-in-the-loop", "seccion": "LangGraph"},
]


# -----------------------------------------------------------------------------
# Helpers compartidos por todas las vistas
# -----------------------------------------------------------------------------

def _init_session_state() -> None:
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {p: "" for p in PROVIDERS}
    if "memoria_store" not in st.session_state:
        st.session_state.memoria_store = {}
    if "hitl_pendiente" not in st.session_state:
        st.session_state.hitl_pendiente = False
    if "hitl_checkpointer" not in st.session_state:
        st.session_state.hitl_checkpointer = None
    if "hitl_thread_id" not in st.session_state:
        st.session_state.hitl_thread_id = None
    if "hitl_borrador" not in st.session_state:
        st.session_state.hitl_borrador = None


def _render_snippet(modulo) -> None:
    """Muestra el código REAL que se va a ejecutar, leído en vivo del módulo con `inspect`.

    Así no hay riesgo de que el snippet mostrado en la UI se desactualice respecto
    del código que efectivamente corre: es literalmente el mismo texto fuente.
    """
    codigo = "\n\n".join(inspect.getsource(obj) for obj in modulo.SNIPPET_OBJECTS)
    with st.expander("📄 Ver el código que se ejecuta", expanded=False):
        st.code(codigo, language="python")


def _ejecutar(fn, *args, **kwargs):
    """Corre `fn` atrapando errores y devuelve (resultado, None) o (None, mensaje_clasificado).

    Nunca deja escapar un stack trace crudo: `LLMConfigError` (key/proveedor mal
    configurados, detectado ANTES de llamar a la API) se muestra tal cual, y
    cualquier otra excepción pasa por `classify_llm_error` para traducirla.
    """
    try:
        return fn(*args, **kwargs), None
    except LLMConfigError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — frontera de la UI: hay que atrapar todo.
        return None, classify_llm_error(exc)


def _boton_ejecutar(api_key: str, provider: str, key_suffix: str, label: str = "▶️ Ejecutar") -> bool:
    clicked = st.button(label, disabled=not api_key, key=f"run_{key_suffix}")
    if not api_key:
        st.caption(f"⚠️ Ingresá tu API key de {provider} en la barra lateral para habilitar este botón.")
    return clicked


def _cargar_modulo(nombre_modulo: str):
    return importlib.import_module(f"examples.{nombre_modulo}")


# -----------------------------------------------------------------------------
# Vistas: una por concepto. Todas reciben (modulo, provider, api_key, model, temperature).
# -----------------------------------------------------------------------------

def _vista_home(provider: str, api_key: str) -> None:
    st.title("🧩 LangChain & LangGraph, paso a paso")
    st.markdown(
        """
Recorrido práctico por los conceptos centrales de **LangChain** y **LangGraph**, en
10 ejemplos cortos y reales. Cada sección de la barra lateral muestra:

- una explicación breve del concepto;
- el código que se ejecuta (tal cual vive en `examples/`, sin resúmenes);
- un botón para correrlo en vivo contra el proveedor de LLM que elijas;
- el resultado real de esa ejecución.

### ¿Cuándo LangChain y cuándo LangGraph?

- **LangChain** conviene cuando el flujo es esencialmente **lineal**: una secuencia
  fija de pasos (`prompt | llm | parser`) que siempre corre en el mismo orden, sin
  ciclos ni bifurcaciones complejas. Ejemplos 1 a 6.
- **LangGraph** conviene cuando hace falta **estado compartido entre pasos, ciclos
  (volver a un nodo ya visitado), bifurcaciones según condiciones dinámicas, o
  persistencia** entre llamadas (memoria de largo plazo, *human-in-the-loop*).
  Ejemplos 7 a 10. De hecho, `create_agent` (ejemplo 5) por dentro COMPILA un grafo
  de LangGraph — LangGraph es la capa de más bajo nivel sobre la que se construye
  buena parte de LangChain.

### Tu API key

Elegí un proveedor y pegá tu propia API key en la barra lateral: **OpenAI**,
**Google Gemini** o **Anthropic Claude**. Los 10 ejemplos corren igual con
cualquiera de los 3 (las diferencias internas — por ejemplo tool-calling vs. modo
JSON nativo para salida estructurada — quedan ocultas detrás de la misma interfaz
de LangChain).
        """
    )
    if not api_key:
        st.warning(
            f"Todavía no ingresaste una API key de {provider}. Hacelo en la barra lateral "
            "para ejecutar los ejemplos en vivo (podés leer explicaciones y código sin key)."
        )
    else:
        st.success(f"Key de {provider} cargada en esta sesión. Elegí un concepto en la barra lateral para empezar.")


def _vista_01(mod, provider, api_key, model, temperature) -> None:
    st.header("1. Chat model básico + `.invoke()`")
    st.markdown(
        """
Punto de partida de todo lo demás en esta app: instanciar un chat model con
`get_llm(...)` y llamarlo con `.invoke()`, pasándole una lista de **mensajes
con rol** en vez de un string suelto.

- `SystemMessage`: instrucciones generales para el modelo, invisibles para
  quien escribe el mensaje humano (acá, "respondé en máximo 3 oraciones").
- `HumanMessage`: el turno de la persona usuaria.
- La respuesta es un `AIMessage`; su texto vive en `.content`.

Este patrón mensaje-con-rol es universal en LangChain: una conversación de 10
turnos es simplemente una lista más larga de estos mismos 3 tipos de mensaje
(`SystemMessage`, `HumanMessage`, `AIMessage`), no un mecanismo distinto. Los
ejemplos 2 y 4 van a automatizar partes de esto (plantillas, historial), pero
por debajo siguen siendo listas de mensajes como esta.
        """
    )
    _render_snippet(mod)

    mensaje = st.text_input("Tu mensaje", value="¿Qué es LangChain en una oración?", key="c1_msg")
    if _boton_ejecutar(api_key, provider, "c1"):
        with st.spinner("Llamando al modelo..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, mensaje, model, temperature)
        if error:
            st.error(error)
        else:
            st.write(resultado)


def _vista_02(mod, provider, api_key, model, temperature) -> None:
    st.header("2. Prompt templates + LCEL")
    st.markdown(
        """
En el ejemplo 1 armábamos la lista de mensajes a mano en cada llamada. Acá
damos un paso más: una **plantilla** (`ChatPromptTemplate`) con un hueco
(`{tema}`) que se rellena distinto en cada invocación, y la conectamos al
modelo con **LCEL** (LangChain Expression Language) — el operador `|`:

```
prompt | llm | output_parser
```

Cada pieza es un *Runnable* (tiene `.invoke`, `.stream`, `.batch`) y la cadena
entera también lo es, así que se pueden anidar chains dentro de chains. Acá:
`prompt` rellena la plantilla → `llm` la invoca (mismo paso que en el ejemplo
1, ahora dentro de la chain) → `StrOutputParser()` extrae el `.content` del
`AIMessage`, el mismo dato que devolvíamos a mano antes.

Esta es la base de las **chains lineales**: datos fluyendo de izquierda a
derecha, en un orden fijo, sin estado compartido entre pasos ni ciclos. Los
ejemplos 3, 4 y 6 son variaciones de esta misma forma `prompt | llm | algo`.
        """
    )
    _render_snippet(mod)

    tema = st.text_input("Tema a explicar", value="los embeddings", key="c2_tema")
    if _boton_ejecutar(api_key, provider, "c2"):
        with st.spinner("Generando explicación..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, tema, model, temperature)
        if error:
            st.error(error)
        else:
            st.write(resultado)


def _vista_03(mod, provider, api_key, model, temperature) -> None:
    st.header("3. Salida estructurada (`with_structured_output`)")
    st.markdown(
        """
Hasta ahora (ejemplos 1 y 2) el modelo siempre devolvía **texto libre** en
`.content`. Muchas veces lo que se necesita es lo opuesto: datos con una
forma exacta y predecible para que el código que sigue los pueda usar sin
parsear texto a mano (llenar un formulario, guardar en una base de datos,
etc.).

`model.with_structured_output(MiModeloPydantic)` envuelve el chat model para
que, en vez de un `AIMessage` con texto, devuelva directamente una
**instancia validada** de tu clase Pydantic. Por debajo, según el proveedor,
esto se implementa con tool-calling (el LLM "llama" a una tool cuyo
schema es tu clase) o con un modo JSON nativo — esa diferencia queda oculta
detrás de la misma llamada `.invoke(...)`, sin que tengas que pensar en cuál
usa cada proveedor.

El `description` de cada `Field` del schema (ver el código) no es
documentación decorativa: es el texto que el modelo lee para saber qué poner
en cada campo, igual que el docstring de una tool en el ejemplo 5.
        """
    )
    _render_snippet(mod)

    descripcion = st.text_area(
        "Descripción de una receta",
        value=(
            "Milanesa a la napolitana: se cubre una milanesa de carne frita con salsa de "
            "tomate, jamón y mucho queso mozzarella, y se gratina al horno unos 10 minutos. "
            "Es un plato de dificultad baja, ideal para principiantes."
        ),
        key="c3_desc",
    )
    if _boton_ejecutar(api_key, provider, "c3"):
        with st.spinner("Extrayendo datos estructurados..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, descripcion, model, temperature)
        if error:
            st.error(error)
        else:
            st.json(resultado.model_dump())


def _vista_04(mod, provider, api_key, model, temperature) -> None:
    st.header("4. Memoria de conversación (`RunnableWithMessageHistory`)")
    st.markdown(
        """
Las chains de los ejemplos 2 y 3 son **sin estado**: cada `.invoke()` parte de
cero, no sabe nada de llamadas anteriores. Para que el modelo "se acuerde" de
lo que ya se habló (como en cualquier chat real), hace falta volver a
inyectarle el historial en cada turno — eso es justamente lo que automatiza
`RunnableWithMessageHistory`.

Envuelve la misma chain LCEL de siempre (`prompt | llm | parser`) y, en cada
`.invoke()`: primero busca el historial de la sesión (`get_session_history`)
y lo inyecta en el `MessagesPlaceholder("history")` de la plantilla, *antes*
del mensaje nuevo; al terminar, agrega tanto el mensaje del usuario como la
respuesta a ese mismo historial. El historial se identifica por
`session_id` — acá una sola conversación por pestaña de navegador, guardada
en `st.session_state.memoria_store` mientras dure la sesión.

Esta memoria vive solo en RAM del proceso: si recargás la página o el
proceso se reinicia, se pierde. Para memoria que sobreviva entre sesiones o
procesos hace falta un backend persistente (DB, archivo) — fuera del alcance
de este demo, pero el mismo patrón de `get_session_history` aplicaría.
        """
    )
    _render_snippet(mod)

    session_id = "sesion_streamlit"
    historia = st.session_state.memoria_store.get(session_id)
    if historia and historia.messages:
        with st.expander(f"📜 Historial actual ({len(historia.messages)} mensajes)", expanded=True):
            for m in historia.messages:
                rol = "🧑 Vos" if m.type == "human" else "🤖 Modelo"
                st.markdown(f"**{rol}:** {m.content}")

    mensaje = st.text_input("Tu mensaje", value="Hola, me llamo Ada.", key="c4_msg")
    col1, col2 = st.columns([3, 1])
    with col1:
        enviar = _boton_ejecutar(api_key, provider, "c4", label="▶️ Enviar")
    with col2:
        if st.button("🔄 Reiniciar memoria", key="c4_reset"):
            st.session_state.memoria_store.pop(session_id, None)
            st.rerun()

    if enviar:
        with st.spinner("Pensando..."):
            _, error = _ejecutar(
                mod.run_example, provider, api_key, mensaje, st.session_state.memoria_store, session_id, model, temperature
            )
        if error:
            st.error(error)
        else:
            st.rerun()


def _vista_05(mod, provider, api_key, model, temperature) -> None:
    st.header("5. Tools + agente (`create_agent`)")
    st.markdown(
        """
Diferencia clave con los ejemplos 1 a 4: ahí el ORDEN de pasos lo decidíamos
nosotros al escribir la chain (`prompt | llm | parser`, siempre igual). Acá
le damos al modelo 3 tools (calculadora, hora actual, búsqueda en una lista
local) y dejamos que **el LLM decida**, en tiempo de ejecución, cuáles
necesita, con qué argumentos, y cuántas veces — antes de responder.

`tool(función)` convierte cada función Python en una Tool: el nombre y los
parámetros se extraen del código, y la **descripción que ve el modelo es el
docstring** — por eso esos docstrings (ver el código) están escritos
pensando en qué necesita saber el LLM para decidir bien, no solo para quien
lea el código.

La entrada/salida de un agente es siempre `{"messages": [...]}`, no un string
suelto: el resultado incluye los mensajes intermedios (las llamadas a tools)
además de la respuesta final, que es lo que mostramos abajo como "traza".

`create_agent` es la API vigente de LangChain v1 para esto (reemplaza al
viejo `AgentExecutor`/`initialize_agent`). Por dentro compila un grafo de
LangGraph para manejar el loop de decidir→ejecutar→decidir, pero como
usuaria/o de LangChain no hace falta pensar en nodos — eso es justo lo que
el ejemplo 9 va a desarmar y mostrar a mano.
        """
    )
    _render_snippet(mod)

    pregunta = st.text_input("Tu pregunta", value="¿Cuánto es 23 * 47 y qué hora es en UTC?", key="c5_q")
    if _boton_ejecutar(api_key, provider, "c5"):
        with st.spinner("El agente está pensando..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, pregunta, model, temperature)
        if error:
            st.error(error)
        else:
            st.write(resultado["respuesta"])
            if resultado["tool_calls"]:
                st.caption("🔧 Tools usadas:")
                st.json(resultado["tool_calls"])
            else:
                st.caption("El agente respondió sin necesitar ninguna tool.")


def _vista_06(mod, provider, api_key, model, temperature) -> None:
    st.header("6. RAG mínimo con FAISS (embeddings locales TF-IDF)")
    st.markdown(
        """
RAG (Retrieval-Augmented Generation) combina dos cosas que ya viste por
separado: una búsqueda por similitud (nueva acá) y la chain LCEL
`prompt | llm | parser` del ejemplo 2 (y 4) para generar la respuesta. La
diferencia con esos ejemplos es de dónde sale el contexto del prompt: ahí lo
escribíamos nosotros en el template; acá lo **recuperamos dinámicamente**
según la pregunta.

Tres pasos:
1. **Indexar**: cada documento de `data/*.txt` se convierte en un vector
   (embedding) y se guarda en FAISS, una estructura para buscar por
   similitud entre vectores.
2. **Recuperar**: la pregunta también se vectoriza y se buscan los `k`
   documentos más parecidos (slider abajo).
3. **Generar**: esos documentos se insertan como `{context}` en el prompt —
   misma chain de siempre, pero ahora el LLM responde con información que no
   tenía de entrenamiento, solo la que le pasamos en este paso.

Usamos embeddings TF-IDF 100% locales (`scikit-learn`, no los del
proveedor) para que el demo funcione igual con los 3 proveedores, sin
gastar cuota de API ni RAM extra en el tier gratuito de Streamlit Cloud —
Anthropic ni siquiera ofrece una API de embeddings propia. En un proyecto
real con más documentos convendría `OpenAIEmbeddings` o
`GoogleGenerativeAIEmbeddings` para mejor calidad semántica.
        """
    )
    _render_snippet(mod)

    archivos = sorted(p.name for p in (ROOT_DIR / "data").glob("*.txt"))
    st.caption(f"📚 Documentos indexados: {', '.join(archivos)}")

    pregunta = st.text_input(
        "Tu pregunta", value="¿Cuándo conviene usar LangGraph en vez de LangChain?", key="c6_q"
    )
    k = st.slider("Documentos a recuperar (k)", 1, 3, 2, key="c6_k")
    if _boton_ejecutar(api_key, provider, "c6"):
        with st.spinner("Indexando y buscando..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, pregunta, model, temperature, k)
        if error:
            st.error(error)
        else:
            st.write(resultado["respuesta"])
            st.caption(f"📎 Fuentes recuperadas: {', '.join(resultado['fuentes'])}")


def _vista_07(mod, provider, api_key, model, temperature) -> None:
    st.header("7. Estado tipado con StateGraph (2 nodos)")
    st.markdown(
        """
Primer ejemplo de **LangGraph** en vez de LangChain. La unidad básica cambia:
ya no son Runnables encadenados con `|` (ejemplos 2, 4, 6), sino un **grafo**
explícito de nodos y aristas (edges), con un **estado tipado** (`TypedDict`)
que viaja de nodo en nodo.

Cada nodo es una función `estado -> dict parcial`: recibe el estado
completo, pero devuelve solo los campos que cambia (acá, `generar_idea`
escribe `idea`; `mejorar_idea` lee esa `idea` y escribe `idea_mejorada`).
LangGraph mergea ese dict parcial sobre el estado antes de pasarlo al
siguiente nodo — así un nodo nunca necesita reescribir lo que ya estaba.

**Importante:** este flujo de 2 nodos en línea recta (`START → generar_idea →
mejorar_idea → END`) TAMBIÉN se podría haber escrito como una chain LCEL
(`prompt1 | llm | prompt2 | llm`, parecido al ejemplo 2). Es intencional: el
objetivo acá es solo la mecánica básica de StateGraph (estado → nodo → estado
→ nodo → fin), antes de ver en el ejemplo 8 un caso donde LangGraph sí aporta
algo que una chain lineal de LCEL no puede expresar.
        """
    )
    _render_snippet(mod)

    tema = st.text_input("Tema", value="una app para aprender LangChain", key="c7_tema")
    if _boton_ejecutar(api_key, provider, "c7"):
        with st.spinner("Generando y mejorando la idea..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, tema, model, temperature)
        if error:
            st.error(error)
        else:
            st.markdown(f"**Idea inicial (nodo `generar_idea`):** {resultado['idea']}")
            st.markdown(f"**Idea mejorada (nodo `mejorar_idea`):** {resultado['idea_mejorada']}")


def _vista_08(mod, provider, api_key, model, temperature) -> None:
    st.header("8. Branching con `add_conditional_edges`")
    st.markdown(
        """
El ejemplo 7 conectaba sus 2 nodos con `add_edge`: el siguiente nodo era
siempre el mismo, fijo de antemano. Acá aparece la primera diferencia real
con una chain lineal: el siguiente nodo depende del **CONTENIDO** del
estado, decidido en tiempo de ejecución.

El flujo: un nodo `clasificar` le pide al LLM una categoría
(`saludo`/`tecnica`/`otro`) y la guarda en el estado; una función de
**routing**, `elegir_rama`, lee esa categoría y devuelve el NOMBRE del
próximo nodo (no modifica el estado, solo decide a dónde seguir);
`add_conditional_edges("clasificar", elegir_rama)` es lo que conecta ese
nombre devuelto con el nodo real. Después, cada una de las 3 ramas
(`responder_saludo`, `responder_tecnica`, `responder_generico`) tiene su
propio edge fijo hacia `END`.

Con LCEL puro esto se podría simular con un `if` en Python por fuera de la
chain, pero entonces ya no sería "una sola chain" invocable de punta a
punta. LangGraph lo modela como parte explícita del grafo: visualizable,
testeable nodo por nodo, y con la misma interfaz `.invoke(...)` que el resto.
        """
    )
    _render_snippet(mod)

    mensaje = st.text_input(
        "Tu mensaje", value="¿Qué diferencia hay entre un TypedDict y un dataclass?", key="c8_msg"
    )
    if _boton_ejecutar(api_key, provider, "c8"):
        with st.spinner("Clasificando y respondiendo..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, mensaje, model, temperature)
        if error:
            st.error(error)
        else:
            st.caption(f"🔀 Categoría detectada: `{resultado['categoria']}`")
            st.write(resultado["respuesta"])


def _vista_09(mod, provider, api_key, model, temperature) -> None:
    st.header("9. Agente ReAct con loop manual (think → act → observe)")
    st.markdown(
        """
Mismo problema que el ejemplo 5 (calculadora + hora actual como tools), pero
ahora desarmamos la caja negra: en vez de `create_agent`, construimos el loop
de tool-calling a mano con `StateGraph`, para ver el **ciclo** explícito que
antes quedaba escondido.

- `llm_call` ("pensar"): el LLM ve todo el historial de mensajes y decide si
  responder directo o pedir una tool, vía `llm.bind_tools(tools)` — el mismo
  mecanismo de bajo nivel que usa `create_agent` por dentro.
- Si pidió una tool, `tool_node` ("actuar + observar") la ejecuta de verdad y
  devuelve el resultado como un `ToolMessage`.
- Ese `ToolMessage` vuelve a `llm_call` (edge `tool_node -> llm_call`), que ve
  la observación y decide si ya puede responder o necesita otra tool más.

Esa vuelta atrás es un **ciclo real** en el grafo — el nodo `llm_call` se
puede ejecutar 1, 2 o N veces según haga falta, algo que ni una chain LCEL
lineal (ejemplos 1-6) ni el branching de un solo salto del ejemplo 8 pueden
expresar. La función `should_continue` es la que cierra el ciclo: mira si el
último mensaje tiene `tool_calls` pendientes y, si los tiene, vuelve a
`tool_node`; si no, el LLM ya respondió en texto y el grafo termina.

Notá también el estado: a diferencia de los ejemplos 7 y 8 (que pisan campos
sueltos como `idea` o `categoria`), acá `messages` usa el reducer
`add_messages`, que ACUMULA mensajes nuevos a la lista existente en vez de
pisarla — necesario porque el historial crece de a un mensaje por vuelta del
ciclo.
        """
    )
    _render_snippet(mod)

    pregunta = st.text_input("Tu pregunta", value="¿Cuánto es 12 * 8 y qué hora es en UTC?", key="c9_q")
    if _boton_ejecutar(api_key, provider, "c9"):
        with st.spinner("Pensando, actuando, observando..."):
            resultado, error = _ejecutar(mod.run_example, provider, api_key, pregunta, model, temperature)
        if error:
            st.error(error)
        else:
            st.write(resultado["respuesta"])
            st.caption("🔁 Traza completa del ciclo think → act → observe:")
            st.json(resultado["pasos"])


def _vista_10(mod, provider, api_key, model, temperature) -> None:
    st.header("10 (bonus). Checkpointing + Human-in-the-Loop")
    st.markdown(
        """
Todos los ejemplos anteriores (incluido el ciclo del 9) corren de punta a
punta en una sola llamada a `.invoke(...)`. Este es el único de **2 pasos**,
porque hay un humano (vos) en el medio de la ejecución:

1. `iniciar(...)` corre el grafo hasta el nodo `pedir_aprobacion`, que llama
   `interrupt({"borrador": ...})`. Eso **PAUSA** la ejecución ahí mismo —no
   es un error ni un return normal— y le devuelve el control a `app.py` con
   el borrador para mostrar.
2. Vos (acá abajo) aprobás o corregís el borrador.
3. `resumir(...)` llama de nuevo al grafo con `Command(resume=decision)`. El
   nodo `pedir_aprobacion` se vuelve a ejecutar desde el principio, pero esta
   vez `interrupt(...)` no pausa: directamente "devuelve" tu decisión, como
   si fuera el valor de una función que ya respondió, y el grafo sigue hasta
   `END`.

Lo que hace posible esa pausa entre los pasos 1 y 3 es el **checkpointer**
(`InMemorySaver`, pasado a `.compile(checkpointer=...)`): en cada paso del
grafo guarda el estado completo asociado a un `thread_id`, igual que el
`checkpoint` de un juego. Por eso `checkpointer` y `thread_id` se crean
*afuera* de `iniciar`/`resumir` (en `st.session_state`, igual que el `store`
de memoria del ejemplo 4) y se pasan como parámetro a ambas llamadas: si se
creara un checkpointer nuevo en cada una, no habría estado guardado para
retomar.
        """
    )
    _render_snippet(mod)

    if not st.session_state.hitl_pendiente:
        tema = st.text_input(
            "Tema del mensaje a redactar", value="agradecerle a un cliente por su compra", key="c10_tema"
        )
        if _boton_ejecutar(api_key, provider, "c10_iniciar", label="✏️ Generar borrador"):
            checkpointer = InMemorySaver()
            thread_id = str(uuid.uuid4())
            with st.spinner("Generando borrador..."):
                resultado, error = _ejecutar(
                    mod.iniciar, provider, api_key, tema, thread_id, checkpointer, model, temperature
                )
            if error:
                st.error(error)
            else:
                st.session_state.hitl_checkpointer = checkpointer
                st.session_state.hitl_thread_id = thread_id
                st.session_state.hitl_borrador = resultado["borrador"]
                st.session_state.hitl_pendiente = True
                st.rerun()
    else:
        st.info("⏸️ El grafo está PAUSADO en `interrupt()`, esperando tu decisión.")
        st.markdown(f"**Borrador generado:**\n\n> {st.session_state.hitl_borrador}")
        decision = st.text_input("¿Aprobás? Escribí 'si' o un texto corregido", value="si", key="c10_decision")

        col1, col2 = st.columns(2)
        with col1:
            confirmar = st.button("✅ Confirmar y resumir grafo", disabled=not api_key, key="c10_resumir")
        with col2:
            cancelar = st.button("↩️ Descartar / empezar de nuevo", key="c10_cancelar")

        if not api_key:
            st.caption(f"⚠️ Ingresá tu API key de {provider} en la barra lateral para confirmar.")

        if cancelar:
            st.session_state.hitl_pendiente = False
            st.session_state.hitl_checkpointer = None
            st.session_state.hitl_thread_id = None
            st.session_state.hitl_borrador = None
            st.rerun()

        if confirmar:
            with st.spinner("Resumiendo el grafo desde el interrupt()..."):
                resultado, error = _ejecutar(
                    mod.resumir,
                    provider,
                    api_key,
                    decision,
                    st.session_state.hitl_thread_id,
                    st.session_state.hitl_checkpointer,
                    model,
                    temperature,
                )
            if error:
                st.error(error)
            else:
                st.success("Grafo resumido y terminado:")
                st.write(resultado["mensaje_final"])
                st.caption(f"¿Se usó el borrador tal cual? {resultado['aprobado']}")
                st.session_state.hitl_pendiente = False
                st.session_state.hitl_checkpointer = None
                st.session_state.hitl_thread_id = None
                st.session_state.hitl_borrador = None


_VISTAS = {
    "01": _vista_01,
    "02": _vista_02,
    "03": _vista_03,
    "04": _vista_04,
    "05": _vista_05,
    "06": _vista_06,
    "07": _vista_07,
    "08": _vista_08,
    "09": _vista_09,
    "10": _vista_10,
}


# -----------------------------------------------------------------------------
# Main: sidebar (key + navegación) + dispatch a la vista elegida
# -----------------------------------------------------------------------------

_init_session_state()

with st.sidebar:
    st.header("🔑 Tu API key")
    st.caption(
        "Tu API key se usa solo en esta sesión: vive en memoria mientras esta pestaña esté "
        "abierta y se borra al cerrarla, recargarla, o al apretar 'Borrar mi key'. Nunca se "
        "guarda en disco ni se envía a ningún lado salvo al proveedor elegido."
    )

    provider = st.selectbox("Proveedor", PROVIDERS, key="provider_select")
    api_key = st.text_input(
        f"API key de {provider}",
        value=st.session_state.api_keys.get(provider, ""),
        type="password",
        key=f"input_key_{provider}",
    )
    st.session_state.api_keys[provider] = api_key

    if api_key:
        st.success(f"Key de {provider} cargada en esta sesión.")
    else:
        st.warning(f"Falta la key de {provider}: los botones 'Ejecutar' van a estar deshabilitados.")

    if st.button("🗑️ Borrar mi key"):
        for p in PROVIDERS:
            st.session_state.api_keys[p] = ""
            st.session_state.pop(f"input_key_{p}", None)
        st.rerun()
    st.caption("Esto borra las keys de los 3 proveedores guardadas en esta sesión.")

    st.divider()
    model_override = st.text_input(
        "Modelo (opcional)",
        value="",
        placeholder=DEFAULT_MODELS[provider],
        help="Dejalo vacío para usar el modelo por defecto del proveedor.",
        key=f"model_override_{provider}",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="temperature_slider")

    st.divider()
    opciones_nav = ["🏠 Inicio"] + [f"{c['seccion']} · {c['titulo']}" for c in CONCEPTOS]
    eleccion = st.radio("Navegación", opciones_nav, key="nav_concepto")

    st.divider()
    st.caption("Hecho con LangChain + LangGraph + Streamlit. Código completo en `examples/`.")

model = model_override.strip() or None

if eleccion == opciones_nav[0]:
    _vista_home(provider, api_key)
else:
    concepto = CONCEPTOS[opciones_nav.index(eleccion) - 1]
    modulo = _cargar_modulo(concepto["modulo"])
    _VISTAS[concepto["id"]](modulo, provider, api_key, model, temperature)
