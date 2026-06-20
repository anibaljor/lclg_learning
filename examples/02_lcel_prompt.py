"""Concepto 2 — Prompt templates + LCEL.

LCEL (LangChain Expression Language) es el operador `|` para encadenar
Runnables: `prompt | llm | output_parser`. Cada pieza es intercambiable y la
cadena entera también es un Runnable (tiene `.invoke`, `.stream`, `.batch`).
Esto es la base de "LangChain como chains lineales": sin estado entre pasos,
sin ciclos, solo datos fluyendo de izquierda a derecha.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from llm_factory import get_llm  # noqa: E402


def run_example(
    provider: str,
    api_key: str,
    tema: str,
    model: str | None = None,
    temperature: float = 0,
) -> str:
    """Construye una chain LCEL prompt -> llm -> parser y la ejecuta."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Sos un divulgador que explica temas técnicos de forma simple."),
            ("human", "Explicá '{tema}' como si yo tuviera 5 años, en 2 oraciones, en español."),
        ]
    )

    # El operador `|` compone Runnables: la salida de cada paso es la entrada del siguiente.
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"tema": tema})


SNIPPET_OBJECTS = [run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "los embeddings"))
