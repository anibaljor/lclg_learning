"""Factory de chat models por proveedor.

Principio central del proyecto: este módulo NUNCA lee una API key de disco,
de variables de entorno ni de `st.secrets`. La key siempre llega como
argumento (`api_key`), provista por quien llama:
  - en la app Streamlit, desde `st.session_state` (dura solo la sesión);
  - en el modo CLI de los ejemplos (`python examples/01_....py`), desde una
    variable de entorno leída únicamente en ese bloque `__main__`.

Sintaxis verificada contra la documentación vigente de LangChain v1
(docs.langchain.com/oss/python) y de langchain-openai / langchain-anthropic /
langchain-google-genai antes de escribir este archivo.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

PROVIDERS: tuple[str, ...] = ("OpenAI", "Gemini", "Claude")

# Modelos por defecto: rápidos/económicos pero con buen soporte de tool-calling
# y salida estructurada, que es lo que usan la mayoría de los ejemplos.
DEFAULT_MODELS: dict[str, str] = {
    "OpenAI": "gpt-5.4",
    "Gemini": "gemini-3.5-flash",
    "Claude": "claude-sonnet-4-6",
}

# Variables de entorno estándar de cada SDK. Se usan SOLO para el modo CLI
# de prueba de los archivos en examples/, nunca desde la app Streamlit.
ENV_VAR_BY_PROVIDER: dict[str, str] = {
    "OpenAI": "OPENAI_API_KEY",
    "Gemini": "GOOGLE_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
}


class LLMConfigError(Exception):
    """Error de configuración detectado ANTES de llamar a la API (provider inválido o key faltante)."""


def get_llm(
    provider: str,
    api_key: str,
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Devuelve el chat model de LangChain correspondiente al proveedor pedido.

    Las 3 librerías de proveedor aceptan el parámetro `api_key` (en
    langchain-google-genai es un alias de `google_api_key`), así que la
    construcción queda uniforme pese a ser 3 clases distintas.
    """
    if provider not in PROVIDERS:
        raise LLMConfigError(
            f"Proveedor desconocido: {provider!r}. Opciones válidas: {', '.join(PROVIDERS)}."
        )
    if not api_key or not api_key.strip():
        raise LLMConfigError(f"Falta la API key de {provider}.")

    model_name = model.strip() if model and model.strip() else DEFAULT_MODELS[provider]

    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=temperature)
    return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)


def classify_llm_error(exc: Exception) -> str:
    """Traduce excepciones de los 3 SDKs a un mensaje claro en español.

    OpenAI, Google y Anthropic usan jerarquías de excepciones propias, así que
    en vez de importarlas todas (y acoplarnos a sus versiones internas)
    clasificamos por atributos comunes (status_code/code) y palabras clave del
    mensaje. Es una heurística deliberadamente simple pero cubre los 3 casos
    que pide la consigna: key faltante, key inválida y error del proveedor.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = f"{status} {exc}".lower()

    auth_keywords = (
        "api key", "api_key", "unauthorized", "authentication",
        "permission_denied", "invalid_api_key", "401", "403",
    )
    if status in (401, 403) or any(k in text for k in auth_keywords):
        return "La API key parece inválida, vencida o sin permisos para este modelo. Revisala e intentá de nuevo."

    quota_keywords = ("rate limit", "quota", "resource_exhausted", "429")
    if status == 429 or any(k in text for k in quota_keywords):
        return "Se alcanzó el límite de uso del proveedor (rate limit / cuota). Esperá un momento y reintentá."

    if any(k in text for k in ("timeout", "connection", "network")):
        return "No se pudo conectar con el proveedor. Revisá tu conexión a internet e intentá de nuevo."

    return f"Error del proveedor: {exc}"
