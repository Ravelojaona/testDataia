"""
adapters/api/routes.py — FastAPI route definitions.

Endpoints:
  GET  /health          health check + index status
  GET  /stats           index metadata
  POST /index/build     trigger index build (or reload)
  POST /query           ask a question
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.adapters.api.schemas import (
    BuildIndexRequest,
    BuildIndexResponse,
    HealthResponse,
    IndexStatsResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from src.application.index_usecase import BuildIndexUseCase
from src.application.query_usecase import QueryUseCase

router = APIRouter()


def _get_build_uc() -> BuildIndexUseCase:
    """Dependency injector — resolved by the app container at startup."""
    from src.adapters.api.app import container
    return container["build_index_uc"]


def _get_query_uc() -> QueryUseCase:
    from src.adapters.api.app import container
    return container["query_uc"]


def _get_retriever():
    from src.adapters.api.app import container
    return container["retriever"]


# ── Health ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(retriever=Depends(_get_retriever)):
    return HealthResponse(status="ok", index_ready=retriever.is_ready())


# ── Stats ─────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=IndexStatsResponse, tags=["System"])
def stats(retriever=Depends(_get_retriever)):
    if not retriever.is_ready():
        raise HTTPException(status_code=503, detail="Index not built yet.")
    s = retriever.stats()
    return IndexStatsResponse(
        total_chunks=s.total_chunks,
        text_chunks=s.text_chunks,
        table_chunks=s.table_chunks,
        sections=s.sections,
        is_ready=s.is_ready,
    )


# ── Build index ───────────────────────────────────────────────────────────

@router.post("/index/build", response_model=BuildIndexResponse, tags=["Index"])
def build_index(
    body: BuildIndexRequest,
    uc: BuildIndexUseCase = Depends(_get_build_uc),
):
    stats = uc.execute(force_rebuild=body.force_rebuild)
    return BuildIndexResponse(
        status="built" if body.force_rebuild else "loaded_or_built",
        stats=IndexStatsResponse(
            total_chunks=stats.total_chunks,
            text_chunks=stats.text_chunks,
            table_chunks=stats.table_chunks,
            sections=stats.sections,
            is_ready=stats.is_ready,
        ),
    )


# ── Query ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(
    body: QueryRequest,
    uc: QueryUseCase = Depends(_get_query_uc),
):
    if not _get_retriever().is_ready():
        raise HTTPException(
            status_code=503,
            detail="Index not ready. POST /index/build first.",
        )
    result = uc.execute(body.question, top_k=body.top_k)
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceResponse(
                chunk_id=s.chunk_id,
                section=s.section,
                type=s.type.value,
                score=s.score,
            )
            for s in result.sources
        ],
        model=result.model,
        detected_language=result.detected_language,
        search_query_used=result.search_query_used,
        usage=result.usage,
    )
