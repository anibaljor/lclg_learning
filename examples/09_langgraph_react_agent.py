"""Concepto 9 — Agente ReAct con loop manual (think -> act -> observe).

En el ejemplo 5 usamos `create_agent`, que por dentro compila un grafo de
LangGraph pero ESCONDE el loop de tool-calling. Acá lo construimos a mano con
`StateGraph` para ver exactamente qué pasa en cada vuelta:
  1. "pensar" (llm_call): el LLM recibe los mensajes y decide si responde
     directo o pide ejecutar una o más tools (vía `.bind_tools(...)`).
  2. si pidió tools, "actuar + observar" (tool_node): ejecutamos cada tool
     pedida y devolvemos el resultado como un `ToolMessage`.
  3. ese `ToolMessage` vuelve al paso 1, y el LLM ve el resultado y decide si
     ya puede responder o necesita otra tool más. Este CICLO (volver a
     "llm_call" desde "tool_node") es justamente lo que una chain LCEL lineal
     no puede expresar y LangGraph sí.
La función de routing `should_continue` es la que cierra el ciclo: mira si el
último mensaje tiene `tool_calls` pendientes y manda al nodo "tool_node", o
termina el grafo si el LLM ya respondió en texto plano.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from zoneinfo import ZoneInfo, available_timezones

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import ToolMessage  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from llm_factory import get_llm  # noqa: E402


def calculadora(expresion: str) -> str:
    """Resuelve una expresión aritmética simple, por ejemplo "3 * (4 + 2)"."""
    permitidos = set("0123456789+-*/(). ")
    if not set(expresion) <= permitidos:
        return "Expresión inválida: solo se permiten números y + - * / ( )."
    return str(eval(expresion, {"__builtins__": {}}, {}))


def hora_actual(zona: str = "UTC") -> str:
    """Devuelve la fecha y hora actual en la zona horaria pedida (ej: 'America/Argentina/Buenos_Aires')."""
    if zona not in available_timezones():
        zona = "UTC"
    return datetime.now(ZoneInfo(zona)).strftime(f"%Y-%m-%d %H:%M:%S ({zona})")


class EstadoAgente(TypedDict):
    """`add_messages` es el reducer: sabe ACUMULAR mensajes nuevos a la lista
    existente en vez de pisarla, que es lo que necesita un loop que crece de a un
    mensaje por vuelta (a diferencia de los ejemplos 7 y 8, que pisan campos sueltos).
    """

    messages: Annotated[list, add_messages]


def construir_grafo(provider: str, api_key: str, model: str | None, temperature: float):
    llm = get_llm(provider, api_key, model=model, temperature=temperature)
    tools = [tool(calculadora), tool(hora_actual)]
    llm_con_tools = llm.bind_tools(tools)
    tools_por_nombre = {t.name: t for t in tools}

    def llm_call(estado: EstadoAgente) -> dict:
        """Nodo "pensar": el LLM ve todo el historial y decide responder o pedir una tool."""
        respuesta = llm_con_tools.invoke(estado["messages"])
        return {"messages": [respuesta]}

    def tool_node(estado: EstadoAgente) -> dict:
        """Nodo "actuar + observar": ejecuta cada tool call pedida por el último mensaje del LLM."""
        ultimo_mensaje = estado["messages"][-1]
        observaciones = []
        for tool_call in ultimo_mensaje.tool_calls:
            funcion = tools_por_nombre[tool_call["name"]]
            resultado = funcion.invoke(tool_call["args"])
            observaciones.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))
        return {"messages": observaciones}

    def should_continue(estado: EstadoAgente) -> Literal["tool_node", "__end__"]:
        """Routing: si el LLM pidió tools, hay que ejecutarlas; si no, ya respondió y termina."""
        ultimo_mensaje = estado["messages"][-1]
        if getattr(ultimo_mensaje, "tool_calls", None):
            return "tool_node"
        return END

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("llm_call", llm_call)
    grafo.add_node("tool_node", tool_node)
    grafo.add_edge(START, "llm_call")
    grafo.add_conditional_edges("llm_call", should_continue)
    grafo.add_edge("tool_node", "llm_call")  # el ciclo: vuelve a pensar con la observación nueva

    return grafo.compile()


def run_example(
    provider: str,
    api_key: str,
    pregunta: str,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Corre el agente manual y devuelve la respuesta final + la traza completa de pasos."""
    app = construir_grafo(provider, api_key, model, temperature)
    estado_final = app.invoke({"messages": [{"role": "user", "content": pregunta}]})

    pasos = []
    for msg in estado_final["messages"]:
        tipo = type(msg).__name__
        if getattr(msg, "tool_calls", None):
            pasos.append({"paso": tipo, "tool_calls": [f"{tc['name']}({tc['args']})" for tc in msg.tool_calls]})
        else:
            pasos.append({"paso": tipo, "contenido": msg.content})

    return {"respuesta": estado_final["messages"][-1].content, "pasos": pasos}


SNIPPET_OBJECTS = [calculadora, hora_actual, EstadoAgente, construir_grafo, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "¿Cuánto es 12 * 8 y qué hora es en UTC?"))
