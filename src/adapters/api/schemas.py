"""
adapters/api/schemas.py — FastAPI request/response Pydantic schemas.

Kept separate from domain models so API contract changes don't pollute
the domain layer, and vice versa.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=512,
                          example="What is the capital of Madagascar?")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class BuildIndexRequest(BaseModel):
    force_rebuild: bool = Field(False, description="Force re-fetch + re-embed even if index exists")


# ── Responses ─────────────────────────────────────────────────────────────

class SourceResponse(BaseModel):
    chunk_id: int
    section: str
    type: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceResponse]
    model: str
    detected_language: str
    search_query_used: str
    usage: Dict[str, Any] = Field(default_factory=dict)


class IndexStatsResponse(BaseModel):
    total_chunks: int
    text_chunks: int
    table_chunks: int
    sections: List[str]
    is_ready: bool


class BuildIndexResponse(BaseModel):
    status: str
    stats: IndexStatsResponse


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    version: str = "1.0.0"
