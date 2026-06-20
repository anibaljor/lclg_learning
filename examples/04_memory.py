"""Concepto 4 — Memory / historial de conversación con RunnableWithMessageHistory.

`RunnableWithMessageHistory` envuelve cualquier chain LCEL y le agrega
historial automático: antes de invocar, inyecta los mensajes previos de la
sesión; después de invocar, guarda la respuesta. El historial se identifica
por `session_id`, por eso `app.py` reutiliza el mismo `session_id` (y el mismo
diccionario `store`) durante toda la sesión del usuario en el navegador.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.chat_history import InMemoryChatMessageHistory  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: E402

from llm_factory import get_llm  # noqa: E402


def run_example(
    provider: str,
    api_key: str,
    user_message: str,
    store: dict[str, InMemoryChatMessageHistory],
    session_id: str = "default",
    model: str | None = None,
    temperature: float = 0,
) -> str:
    """Invoca una chain con memoria; `store` persiste el historial entre llamadas."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Sos un asistente conciso. Usá lo que el usuario ya te contó en esta charla."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    def get_session_history(sid: str) -> InMemoryChatMessageHistory:
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_history.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )


SNIPPET_OBJECTS = [run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    historial_cli: dict[str, InMemoryChatMessageHistory] = {}
    print(run_example(provider, api_key, "Hola, me llamo Ada.", historial_cli))
    print(run_example(provider, api_key, "¿Cómo me llamo?", historial_cli))
