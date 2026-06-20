"""Concepto 1 — Chat model básico + invoke.

Lo mínimo posible: instanciar un chat model y llamarlo con `.invoke()`.
Mostramos además el patrón de mensajes con roles (SystemMessage/HumanMessage),
que es como se modela cualquier conversación en LangChain, sea de 1 turno o de
muchos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import HumanMessage, SystemMessage  # noqa: E402

from llm_factory import get_llm  # noqa: E402


def run_example(
    provider: str,
    api_key: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0,
) -> str:
    """Invoca un chat model con un mensaje de sistema + un mensaje humano."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    messages = [
        SystemMessage("Sos un asistente conciso. Respondé siempre en español, en máximo 3 oraciones."),
        HumanMessage(user_message),
    ]
    response = llm.invoke(messages)
    return response.content


SNIPPET_OBJECTS = [run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "¿Qué es LangChain en una oración?"))
