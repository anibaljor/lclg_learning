"""Concepto 6 — RAG mínimo con FAISS in-memory.

Indexamos los documentos de `data/` en un vector store FAISS y respondemos
preguntas usando el contenido recuperado como contexto del prompt (LCEL otra
vez: armamos el contexto a mano y lo mandamos por `prompt | llm | parser`).

Nota sobre embeddings: para que este demo funcione igual con los 3
proveedores y sin gastar cuota de API ni RAM extra (límite del tier gratuito
de Streamlit Cloud), usamos un embedding 100% local (TF-IDF, vía
scikit-learn) en vez de pedirle embeddings al proveedor elegido. Anthropic ni
siquiera expone una API de embeddings, así que esto evita ese caso especial.
En un proyecto real con más documentos usarías `OpenAIEmbeddings` o
`GoogleGenerativeAIEmbeddings` para mejor calidad semántica.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from llm_factory import get_llm  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class EmbeddingsLocalesTfidf(Embeddings):
    """Embeddings locales (TF-IDF): livianos, sin costo de API y válidos para los 3 proveedores."""

    def __init__(self, corpus: list[str]):
        self._vectorizer = TfidfVectorizer().fit(corpus)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vectorizer.transform([text]).toarray()[0].tolist()


def _cargar_documentos() -> list[Document]:
    archivos = sorted(DATA_DIR.glob("*.txt"))
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in archivos
    ]


def run_example(
    provider: str,
    api_key: str,
    pregunta: str,
    model: str | None = None,
    temperature: float = 0,
    k: int = 2,
) -> dict:
    """Indexa data/*.txt en FAISS y responde `pregunta` usando esos documentos como contexto."""
    llm = get_llm(provider, api_key, model=model, temperature=temperature)
    documentos = _cargar_documentos()

    # 1. Indexar: cada documento se convierte en un vector (embedding) y se guarda
    # en FAISS, una estructura optimizada para buscar por similitud entre vectores.
    embeddings = EmbeddingsLocalesTfidf([d.page_content for d in documentos])
    vector_store = FAISS.from_documents(documentos, embeddings)
    # `as_retriever` envuelve el vector store como un Runnable: `k` es cuántos
    # documentos devolver por búsqueda (ajustable desde el slider de la UI).
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Recuperar: la pregunta también se convierte a vector y se buscan los `k`
    # documentos más cercanos — esto es la "R" de RAG (Retrieval).
    docs_recuperados = retriever.invoke(pregunta)
    contexto = "\n\n".join(d.page_content for d in docs_recuperados)

    # 3. Generar: misma chain LCEL de los ejemplos 2/4 (prompt | llm | parser),
    # pero ahora el prompt recibe el `contexto` recuperado además de la pregunta
    # — esto es la "AG" de RAG (Augmented Generation): el LLM responde con
    # información que NO tenía de entrenamiento, solo la que le pasamos acá.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Respondé solo con información del contexto. Si no está ahí, decí que no lo sabés."),
            ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    respuesta = chain.invoke({"context": contexto, "question": pregunta})

    return {"respuesta": respuesta, "fuentes": [d.metadata["source"] for d in docs_recuperados]}


SNIPPET_OBJECTS = [EmbeddingsLocalesTfidf, run_example]


if __name__ == "__main__":
    import os

    from llm_factory import ENV_VAR_BY_PROVIDER

    provider = os.environ.get("LCLG_PROVIDER", "OpenAI")
    env_var = ENV_VAR_BY_PROVIDER[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Seteá la variable de entorno {env_var} para probar este ejemplo por CLI.")

    print(run_example(provider, api_key, "¿Cuándo conviene usar LangGraph en vez de LangChain?"))
