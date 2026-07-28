"""
domain/models.py — Core domain entities.

Pure Pydantic models with ZERO external infrastructure dependencies.
These are the language of the entire application — every layer speaks
in terms of these types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class Chunk(BaseModel):
    """A single indexable unit of content extracted from the Wikipedia page."""

    id: int
    type: ChunkType
    section: str
    content: str
    token_count: int

    model_config = {"frozen": True}  # immutable after creation


class SearchResult(BaseModel):
    """A chunk returned by the retriever, enriched with its retrieval score."""

    chunk: Chunk
    score: float
    rank: int


class Source(BaseModel):
    """Compact reference to a source chunk used in a QueryResult."""

    chunk_id: int
    section: str
    type: ChunkType
    score: Optional[float] = None


class QueryResult(BaseModel):
    """The complete output of the RAG pipeline for a single question."""

    query: str
    answer: str
    sources: List[Source]
    model: str
    detected_language: str = "en"
    search_query_used: str = ""
    usage: Dict[str, Any] = Field(default_factory=dict)


class IndexStats(BaseModel):
    """Metadata about the current vector index."""

    total_chunks: int
    text_chunks: int
    table_chunks: int
    sections: List[str]
    index_path: str
    is_ready: bool
