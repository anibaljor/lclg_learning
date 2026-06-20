"""Concepto 7 — Grafo mínimo con estado tipado (StateGraph + TypedDict).

LangGraph modela una app como un grafo: un estado tipado (acá un `TypedDict`)
que viaja de nodo en nodo, y cada nodo es una función que recibe ese estado y
devuelve los campos que quiere actualizar. Este ejemplo tiene solo 2 nodos
conectados en línea recta (`START -> generar_idea -> mejorar_idea -> END`),
a propósito: es el caso más simple posible, y todavía podría haberse escrito
como una chain LCEL (`prompt1 | llm | prompt2 | llm`). La diferencia real de
LangGraph (ciclos, ramas condicionales, estado compartido complejo) se ve en
el ejemplo 8 (branching) y el 9 (agente ReAct manual). Acá el objetivo es
solamente entender la mecánica básica: estado -> nodo -> estado -> nodo -> fin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from llm_factory import get_llm  # noqa: E402


class EstadoIdea(TypedDict):
    """Estado que viaja por el grafo: cada nodo lee y/o escribe estos campos."""

    tema: str
    idea: str
    idea_mejorada: str


def construir_grafo(provider: str, api_key: str, model: str | None, temperature: float):
    """Arma el grafo de 2 nodos. Separado de `run_example` para que la UI
    pueda mostrar esta función como "la definición del grafo" en sí misma.
    """
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    def generar_idea(estado: EstadoIdea) -> dict:
        """Nodo 1: a partir de `tema`, propone una idea breve."""
        respuesta = llm.invoke(
            f"Proponé una idea breve (1 oración) sobre '{estado['tema']}'. Solo la idea, sin introducción."
        )
        return {"idea": respuesta.content}

    def mejorar_idea(estado: EstadoIdea) -> dict:
        """Nodo 2: toma la idea del nodo anterior y la expande con más detalle."""
        respuesta = llm.invoke(
            f"Mejorá y expandí esta idea en 2-3 oraciones, agregando un detalle concreto: {estado['idea']}"
        )
        return {"idea_mejorada": respuesta.content}

    grafo = StateGraph(EstadoIdea)
    grafo.add_node("generar_idea", generar_idea)
    grafo.add_node("mejorar_idea", mejorar_idea)
    grafo.add_edge(START, "generar_idea")
    grafo.add_edge("generar_idea", "mejorar_idea")
    grafo.add_edge("mejorar_idea", END)

    return grafo.compile()


def run_example(
    provider: str,
    api_key: str,
    tema: str,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Compila el grafo y lo ejecuta de punta a punta con `tema` como entrada."""
    app = construir_grafo(provider, api_key, model, temperature)
    estado_final = app.invoke({"tema": tema, "idea": "", "idea_mejorada": ""})
    return {"idea": estado_final["idea"], "idea_mejorada": estado_final["idea_mejorada"]}


SNIPPET_OBJECTS = [EstadoIdea, construir_grafo, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "una app para aprender LangChain"))
