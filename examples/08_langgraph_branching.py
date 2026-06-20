"""Concepto 8 — Branching con `add_conditional_edges`.

Acá está el primer caso donde LangGraph aporta algo que una chain LCEL lineal
no puede dar fácil: el siguiente nodo a ejecutar depende del CONTENIDO del
estado, decidido en tiempo de ejecución. Clasificamos el mensaje del usuario
en una categoría ("saludo" / "tecnica" / "otro") y una función de routing
(`elegir_rama`) lee esa categoría del estado y devuelve el NOMBRE del próximo
nodo. `add_conditional_edges` conecta ese nombre devuelto con el nodo real
correspondiente. Con LCEL puro esto se podría simular con un `if` en Python
afuera de la chain, pero ya no sería "una sola chain": LangGraph lo modela
como parte explícita del grafo, visualizable y testeable nodo por nodo.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from llm_factory import get_llm  # noqa: E402


class EstadoMensaje(TypedDict):
    """Estado: el mensaje original, la categoría detectada y la respuesta final."""

    mensaje: str
    categoria: str
    respuesta: str


def construir_grafo(provider: str, api_key: str, model: str | None, temperature: float):
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    def clasificar(estado: EstadoMensaje) -> dict:
        """Nodo de entrada: le pide al LLM una sola palabra de categoría."""
        respuesta = llm.invoke(
            "Clasificá el siguiente mensaje en exactamente una palabra: "
            "'saludo' (si es un saludo o cortesía), 'tecnica' (si es una pregunta "
            f"técnica sobre programación/IA) u 'otro' (cualquier otra cosa).\n\nMensaje: {estado['mensaje']}"
        )
        categoria = respuesta.content.strip().lower()
        if categoria not in ("saludo", "tecnica", "otro"):
            categoria = "otro"
        return {"categoria": categoria}

    def responder_saludo(estado: EstadoMensaje) -> dict:
        """Rama A: respuesta corta y cordial, sin gastar razonamiento de más."""
        respuesta = llm.invoke(f"Respondé cordial y brevemente a este saludo: {estado['mensaje']}")
        return {"respuesta": respuesta.content}

    def responder_tecnica(estado: EstadoMensaje) -> dict:
        """Rama B: se le pide al LLM una respuesta técnica más detallada."""
        respuesta = llm.invoke(
            f"Respondé esta pregunta técnica con precisión y un ejemplo breve: {estado['mensaje']}"
        )
        return {"respuesta": respuesta.content}

    def responder_generico(estado: EstadoMensaje) -> dict:
        """Rama C: catch-all para todo lo que no es saludo ni pregunta técnica."""
        respuesta = llm.invoke(f"Respondé brevemente: {estado['mensaje']}")
        return {"respuesta": respuesta.content}

    def elegir_rama(estado: EstadoMensaje) -> Literal["responder_saludo", "responder_tecnica", "responder_generico"]:
        """Función de routing: lee `categoria` del estado y nombra el próximo nodo.

        Esto es lo que `add_conditional_edges` necesita: no modifica el estado,
        solo decide a dónde seguir.
        """
        if estado["categoria"] == "saludo":
            return "responder_saludo"
        if estado["categoria"] == "tecnica":
            return "responder_tecnica"
        return "responder_generico"

    grafo = StateGraph(EstadoMensaje)
    grafo.add_node("clasificar", clasificar)
    grafo.add_node("responder_saludo", responder_saludo)
    grafo.add_node("responder_tecnica", responder_tecnica)
    grafo.add_node("responder_generico", responder_generico)

    grafo.add_edge(START, "clasificar")
    grafo.add_conditional_edges("clasificar", elegir_rama)
    grafo.add_edge("responder_saludo", END)
    grafo.add_edge("responder_tecnica", END)
    grafo.add_edge("responder_generico", END)

    return grafo.compile()


def run_example(
    provider: str,
    api_key: str,
    mensaje: str,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Compila el grafo y lo corre; devuelve la categoría elegida y la respuesta de esa rama."""
    app = construir_grafo(provider, api_key, model, temperature)
    estado_final = app.invoke({"mensaje": mensaje, "categoria": "", "respuesta": ""})
    return {"categoria": estado_final["categoria"], "respuesta": estado_final["respuesta"]}


SNIPPET_OBJECTS = [EstadoMensaje, construir_grafo, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "¿Qué diferencia hay entre un TypedDict y un dataclass?"))
