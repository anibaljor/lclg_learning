"""Concepto 3 — Output parsers estructurados (salida a un modelo Pydantic).

`model.with_structured_output(MiModeloPydantic)` le pide al proveedor que
devuelva datos que calzan exactamente con el schema (vía tool-calling o modo
JSON nativo, según el proveedor). El resultado ya es una instancia validada de
la clase, no texto suelto para parsear a mano.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from llm_factory import get_llm  # noqa: E402


class RecetaInfo(BaseModel):
    """Información estructurada extraída de la descripción de una receta.

    El `description` de cada `Field` no es decoración: es el texto que el
    proveedor recibe para saber qué poner en cada campo (parecido al docstring
    de una tool en el ejemplo 5). Un schema con buenas descripciones es lo que
    hace que la extracción sea confiable.
    """

    nombre_plato: str = Field(description="Nombre del plato")
    ingredientes: list[str] = Field(description="Lista de ingredientes principales")
    tiempo_preparacion_min: int = Field(description="Tiempo estimado de preparación en minutos")
    dificultad: str = Field(description="Una de: facil, media, dificil")


def run_example(
    provider: str,
    api_key: str,
    descripcion_receta: str,
    model: str | None = None,
    temperature: float = 0,
) -> RecetaInfo:
    """Extrae datos estructurados de una receta en texto libre, validados con Pydantic."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)
    # `with_structured_output` envuelve el chat model para que, en vez de
    # devolver texto libre (como en los ejemplos 1 y 2), devuelva directamente
    # una instancia de `RecetaInfo` ya validada. Sin esto habría que pedirle al
    # LLM "respondé en JSON" y parsear/validar esa respuesta a mano.
    structured_llm = llm.with_structured_output(RecetaInfo)
    return structured_llm.invoke(descripcion_receta)


SNIPPET_OBJECTS = [RecetaInfo, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    ejemplo = (
        "Milanesa a la napolitana: se cubre una milanesa de carne frita con salsa de "
        "tomate, jamón y mucho queso mozzarella, y se gratina al horno unos 10 minutos. "
        "Es un plato de dificultad baja, ideal para principiantes."
    )
    print(run_example(provider, api_key, ejemplo))
