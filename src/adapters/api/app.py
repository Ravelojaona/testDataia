"""
adapters/api/app.py — FastAPI application factory + dependency container.

The container is a simple dict wired at startup — no DI framework needed
at this scale. All concrete implementations are instantiated here;
the rest of the codebase sees only domain ports.
"""

from __future__ import annotations

import os

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.application.index_usecase import BuildIndexUseCase
from src.application.query_usecase import QueryUseCase
from src.infrastructure.chunker import WikipediaChunker
from src.infrastructure.embedder import OpenAIEmbedder
from src.infrastructure.generator import OpenAIGenerator, OpenAITranslator
from src.infrastructure.hybrid_retriever import HybridRetriever
from src.infrastructure.loader import WikipediaLoader
from src.infrastructure.reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Dependency container (populated at startup)
# ---------------------------------------------------------------------------

container: dict = {}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Madagascar RAG API",
        description=(
            "Retrieval-Augmented Generation system over the Wikipedia page "
            "for Madagascar. Answers factual questions in English or French."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup():
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        client = openai.OpenAI(api_key=api_key)
        index_path = os.environ.get("INDEX_PATH", "index")

        loader = WikipediaLoader()
        chunker = WikipediaChunker()
        embedder = OpenAIEmbedder(client)
        retriever = HybridRetriever(index_path=index_path)
        generator = OpenAIGenerator(client)
        translator = OpenAITranslator(client)

        container["retriever"] = retriever
        container["build_index_uc"] = BuildIndexUseCase(
            loader=loader,
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            index_path=index_path,
        )
        container["query_uc"] = QueryUseCase(
            retriever=retriever,
            embedder=embedder,
            generator=generator,
            translator=translator,
            reranker=CrossEncoderReranker(),
        )

        # Auto-load existing index on startup (non-blocking if absent)
        import os as _os
        if _os.path.exists(f"{index_path}/faiss.index"):
            print("Auto-loading index on startup…")
            retriever.load(index_path)

    from src.adapters.api.routes import router
    app.include_router(router)

    return app


app = create_app()
