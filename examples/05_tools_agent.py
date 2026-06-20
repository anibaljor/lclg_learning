"""Concepto 5 — Tools + agente que las usa de verdad.

Definimos 3 tools simples con `@tool` y se las damos a `create_agent` (la API
vigente de LangChain v1 para agentes; reemplaza al viejo `initialize_agent` /
`AgentExecutor` manual). El agente decide solo qué tool llamar, con qué
argumentos, y cuántas veces, antes de responder. Por dentro usa LangGraph
(create_agent compila un grafo), pero como usuario de LangChain no necesitás
pensar en nodos ni edges: por eso este ejemplo vive en la sección "LangChain".
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402

from llm_factory import get_llm  # noqa: E402

_BASE_CONOCIMIENTO = [
    {"tema": "LangChain", "dato": "Framework para construir aplicaciones con LLMs usando chains y agentes."},
    {"tema": "LangGraph", "dato": "Librería para construir agentes con estado como grafos, con ciclos y checkpointing."},
    {"tema": "FAISS", "dato": "Librería de Meta para búsqueda eficiente de similitud entre vectores."},
    {"tema": "Streamlit", "dato": "Framework para crear apps web de datos en Python sin escribir HTML/JS."},
]


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


def buscar_en_lista_local(query: str) -> str:
    """Busca `query` (case-insensitive) en una pequeña base de datos local de temas técnicos."""
    query_lower = query.lower()
    resultados = [
        f"{item['tema']}: {item['dato']}"
        for item in _BASE_CONOCIMIENTO
        if query_lower in item["tema"].lower() or query_lower in item["dato"].lower()
    ]
    return "\n".join(resultados) if resultados else "No se encontraron resultados en la base local."


def run_example(
    provider: str,
    api_key: str,
    pregunta: str,
    model: str | None = None,
    temperature: float = 0,
) -> dict:
    """Crea un agente con 3 tools y lo invoca; devuelve la respuesta y la traza de tool calls."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)
    # `tool(...)` convierte cada función Python en una Tool de LangChain: el
    # nombre, los parámetros (por type hints) y la descripción (por el docstring)
    # se extraen automáticamente. Esa descripción es lo único que el LLM ve para
    # decidir cuándo y cómo llamar a cada una — por eso los docstrings de arriba
    # están escritos pensando en el modelo, no solo en quien lea el código.
    tools = [tool(calculadora), tool(hora_actual), tool(buscar_en_lista_local)]

    # A diferencia de las chains LCEL de los ejemplos 1-4 (orden fijo de pasos),
    # acá NO decidimos nosotros qué se ejecuta: `create_agent` le da el control al
    # LLM para que elija, en un loop interno, qué tool(s) llamar y cuándo parar.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="Sos un asistente que responde en español y usa tools cuando hacen falta.",
    )
    # La entrada/salida de un agente siempre es `{"messages": [...]}`, no un string
    # suelto como en `chain.invoke(...)`. El resultado incluye TODOS los mensajes
    # intermedios (incluidas las llamadas a tools), no solo la respuesta final.
    result = agent.invoke({"messages": [{"role": "user", "content": pregunta}]})

    # Recorremos esos mensajes intermedios solo para mostrar en la UI qué tools
    # se usaron y con qué argumentos — el agente ya terminó, esto es inspección.
    traza = [
        {"tool": tc["name"], "args": tc["args"]}
        for msg in result["messages"]
        for tc in getattr(msg, "tool_calls", []) or []
    ]
    return {"respuesta": result["messages"][-1].content, "tool_calls": traza}


SNIPPET_OBJECTS = [calculadora, hora_actual, buscar_en_lista_local, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "¿Cuánto es 23 * 47 y qué hora es en UTC?"))
