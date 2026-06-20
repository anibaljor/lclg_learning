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
        "Lo más simple posible: instanciar un chat model y llamarlo con `.invoke()`, "
        "pasándole una lista de mensajes con roles (`SystemMessage`, `HumanMessage`). "
        "Así se modela cualquier conversación en LangChain, de 1 turno o de muchos."
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
        "LCEL (LangChain Expression Language) es el operador `|` para encadenar Runnables: "
        "`prompt | llm | output_parser`. Cada pieza es intercambiable y la cadena entera "
        "también es un Runnable. Es la base de las chains lineales de LangChain: datos "
        "fluyendo de izquierda a derecha, sin estado compartido entre pasos ni ciclos."
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
        "`model.with_structured_output(MiModeloPydantic)` le pide al proveedor una respuesta "
        "que calza exactamente con un schema Pydantic (por debajo, vía tool-calling o modo JSON "
        "nativo según el proveedor). El resultado ya es una instancia validada de la clase, no "
        "texto suelto para parsear a mano."
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
        "`RunnableWithMessageHistory` envuelve una chain LCEL y le agrega historial "
        "automático: antes de invocar, inyecta los mensajes previos; después, guarda la "
        "respuesta. El historial se identifica por `session_id` — acá usamos una sola "
        "conversación por pestaña de navegador."
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
        "Le damos al agente 3 tools (calculadora, hora actual, búsqueda en una lista local) "
        "y decide solo cuáles usar, con qué argumentos y cuántas veces, antes de responder. "
        "`create_agent` es la API vigente de LangChain v1 para agentes; por dentro compila un "
        "grafo de LangGraph, pero como usuaria/o de LangChain no hace falta pensar en nodos."
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
        "Indexamos `data/*.txt` en FAISS y, para responder, recuperamos primero los `k` "
        "documentos más parecidos a la pregunta y los insertamos como contexto del prompt. "
        "Usamos embeddings TF-IDF 100% locales (no los del proveedor) para que el demo "
        "funcione igual con los 3 proveedores, sin gastar cuota de API ni RAM extra en el "
        "tier gratuito de Streamlit Cloud — en un proyecto real con más documentos convendría "
        "`OpenAIEmbeddings` o `GoogleGenerativeAIEmbeddings`."
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
        "El caso más simple de LangGraph: un estado tipado (`TypedDict`) que viaja por 2 "
        "nodos conectados en línea recta. Importante: este flujo lineal TAMBIÉN se podría "
        "haber escrito como una chain LCEL. Lo usamos para entender la mecánica básica antes "
        "de ver, en el ejemplo 8, un caso donde LangGraph sí aporta algo que LCEL no puede."
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
        "Acá el siguiente nodo a ejecutar depende del CONTENIDO del estado, decidido en "
        "tiempo de ejecución: un nodo clasifica el mensaje y una función de routing manda la "
        "ejecución a una de 3 ramas distintas según esa categoría. Esto es lo que LCEL puro no "
        "puede expresar como una sola chain: acá es parte explícita y visualizable del grafo."
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
        "Mismo problema que el ejemplo 5, pero ahora el loop de tool-calling se construye a "
        "mano con `StateGraph`, sin `create_agent`. Se ve el ciclo explícito: el nodo "
        "`llm_call` (pensar) decide si pedir una tool; si la pidió, `tool_node` (actuar + "
        "observar) la ejecuta y vuelve a `llm_call`, que ve el resultado y decide si responder "
        "o pedir otra tool más. Esa vuelta atrás (`tool_node -> llm_call`) es un ciclo real en "
        "el grafo — algo que una chain LCEL lineal no puede expresar."
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
        "Único ejemplo de 2 pasos: el grafo se PAUSA con `interrupt()` esperando que un "
        "humano (vos) apruebe o corrija un borrador, y se retoma con `Command(resume=...)` "
        "desde el punto exacto donde se pausó. El `checkpointer` (`InMemorySaver`) es lo que "
        "hace posible esa pausa: guarda el estado completo del grafo mientras esperamos tu "
        "respuesta, igual que un `checkpoint` de un juego."
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
