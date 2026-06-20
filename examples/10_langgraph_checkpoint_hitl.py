"""Concepto 10 (bonus) — Checkpointing + Human-in-the-Loop con `interrupt()`.

Este es el único ejemplo que NO corre de punta a punta en una sola llamada:
se ejecuta en 2 pasos porque hay un humano en el medio.
  1. `iniciar(...)`: corre el grafo hasta el nodo "pedir_aprobacion", que llama
     `interrupt(...)`. Eso CONGELA la ejecución ahí mismo, guarda el estado en
     el checkpointer (`InMemorySaver`) y devuelve el control a Python con el
     payload que le pasamos a `interrupt(...)` (acá, el borrador a revisar).
  2. `resumir(...)`: alguien (un humano) decide qué hacer con ese borrador, y
     se llama de nuevo al grafo con `Command(resume=decision_humana)`. Eso hace
     que el nodo se re-ejecute desde el principio, pero esta vez `interrupt(...)`
     no pausa: directamente "devuelve" `decision_humana`, y el nodo sigue.
Sin el checkpointer no habría dónde guardar el estado entre el paso 1 y el 2:
por eso `checkpointer` se crea afuera (una sola vez por conversación, en la UI
viviría en `st.session_state`, igual que el `store` del ejemplo 4 de memoria) y
se pasa como parámetro a ambas funciones en vez de crearse adentro de cada una.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from llm_factory import get_llm  # noqa: E402


class EstadoAprobacion(TypedDict):
    tema: str
    borrador: str
    mensaje_final: str
    aprobado: bool


def construir_grafo(provider: str, api_key: str, model: str | None, temperature: float, checkpointer: InMemorySaver):
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    def generar_borrador(estado: EstadoAprobacion) -> dict:
        """Nodo 1: el LLM escribe un primer borrador, todavía sin intervención humana."""
        respuesta = llm.invoke(f"Escribí un mensaje corto (1-2 oraciones) sobre: {estado['tema']}")
        return {"borrador": respuesta.content}

    def pedir_aprobacion(estado: EstadoAprobacion) -> dict:
        """Nodo 2: pausa el grafo y espera la decisión humana sobre `borrador`.

        Si `decision` viene "si"/"sí"/"ok" se usa el borrador tal cual; cualquier
        otro texto se interpreta como una corrección del humano y se usa en su lugar.
        """
        # `interrupt(payload)` congela la ejecución ACÁ MISMO la primera vez que
        # se llama (en `iniciar`): el grafo devuelve el control a Python sin
        # terminar, llevándose `payload` consigo. Cuando después se resume con
        # `Command(resume=decision_humana)` (en `resumir`), este nodo se vuelve a
        # ejecutar desde el principio, pero esta vez `interrupt(...)` no pausa:
        # directamente "devuelve" `decision_humana`, como si fuera el valor de
        # retorno de una función que ya respondió.
        decision = interrupt({"borrador": estado["borrador"]})
        aprobado = str(decision).strip().lower() in ("si", "sí", "yes", "ok", "true")
        mensaje_final = estado["borrador"] if aprobado else str(decision)
        return {"mensaje_final": mensaje_final, "aprobado": aprobado}

    grafo = StateGraph(EstadoAprobacion)
    grafo.add_node("generar_borrador", generar_borrador)
    grafo.add_node("pedir_aprobacion", pedir_aprobacion)
    grafo.add_edge(START, "generar_borrador")
    grafo.add_edge("generar_borrador", "pedir_aprobacion")
    grafo.add_edge("pedir_aprobacion", END)

    # Pasar `checkpointer` a `.compile()` es lo que habilita pausar/resumir: en
    # cada paso, LangGraph guarda ahí el estado completo asociado al
    # `thread_id` de la llamada. Sin esto, `interrupt()` no tendría dónde
    # persistir el estado entre la llamada que pausa y la que resume.
    return grafo.compile(checkpointer=checkpointer)


def iniciar(
    provider: str,
    api_key: str,
    tema: str,
    thread_id: str,
    checkpointer: InMemorySaver,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Arranca el grafo y lo corre hasta el primer (y único) `interrupt()`.

    `checkpointer` debe ser el MISMO objeto que después se use en `resumir()`
    con el mismo `thread_id`: si se creara uno nuevo en cada llamada, no habría
    estado guardado para retomar.
    """
    app = construir_grafo(provider, api_key, model, temperature, checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    estado = app.invoke(
        {"tema": tema, "borrador": "", "mensaje_final": "", "aprobado": False},
        config=config,
    )
    payload = estado["__interrupt__"][0].value
    return {"borrador": payload["borrador"]}


def resumir(
    provider: str,
    api_key: str,
    decision_humana: str,
    thread_id: str,
    checkpointer: InMemorySaver,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Retoma el grafo desde el `interrupt()` con la decisión humana y corre hasta el final."""
    app = construir_grafo(provider, api_key, model, temperature, checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    estado_final = app.invoke(Command(resume=decision_humana), config=config)
    return {"mensaje_final": estado_final["mensaje_final"], "aprobado": estado_final["aprobado"]}


SNIPPET_OBJECTS = [EstadoAprobacion, construir_grafo, iniciar, resumir]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    checkpointer_cli = InMemorySaver()
    paso_1 = iniciar(provider, api_key, "agradecerle a un cliente por su compra", "demo-cli", checkpointer_cli)
    print("Borrador generado, esperando aprobación:", paso_1["borrador"])
    paso_2 = resumir(provider, api_key, "si", "demo-cli", checkpointer_cli)
    print("Resultado final:", paso_2)
