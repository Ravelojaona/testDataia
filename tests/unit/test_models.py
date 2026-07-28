"""
tests/unit/test_models.py — Domain model validation tests.
"""

import pytest
from pydantic import ValidationError

from src.domain.models import Chunk, ChunkType, QueryResult, SearchResult, Source


class TestChunk:
    def test_valid_text_chunk(self):
        c = Chunk(id=0, type=ChunkType.TEXT, section="Geography",
                  content="Madagascar is an island.", token_count=5)
        assert c.id == 0
        assert c.type == ChunkType.TEXT
        assert c.section == "Geography"

    def test_valid_table_chunk(self):
        c = Chunk(id=1, type=ChunkType.TABLE, section="Demographics",
                  content="| Region | Pop |", token_count=6)
        assert c.type == ChunkType.TABLE

    def test_chunk_is_immutable(self):
        from pydantic import ValidationError
        c = Chunk(id=0, type=ChunkType.TEXT, section="X", content="y", token_count=1)
        with pytest.raises((TypeError, ValidationError)):
            c.id = 99  # frozen model — Pydantic v2 raises ValidationError

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Chunk(type=ChunkType.TEXT, section="X", content="y", token_count=1)  # id missing

    def test_chunk_type_enum_values(self):
        assert ChunkType.TEXT == "text"
        assert ChunkType.TABLE == "table"


class TestQueryResult:
    def test_default_language(self):
        r = QueryResult(query="q", answer="a", sources=[], model="gpt-4o")
        assert r.detected_language == "en"
        assert r.usage == {}

    def test_sources_populated(self):
        source = Source(chunk_id=0, section="Geography", type=ChunkType.TEXT)
        r = QueryResult(query="q", answer="a", sources=[source], model="gpt-4o")
        assert len(r.sources) == 1
        assert r.sources[0].section == "Geography"

    def test_source_optional_score(self):
        s = Source(chunk_id=5, section="Economy", type=ChunkType.TEXT)
        assert s.score is None

        s2 = Source(chunk_id=5, section="Economy", type=ChunkType.TEXT, score=0.92)
        assert s2.score == pytest.approx(0.92)


class TestSearchResult:
    def test_search_result_rank(self, sample_chunks):
        from src.domain.models import SearchResult
        sr = SearchResult(chunk=sample_chunks[0], score=0.85, rank=1)
        assert sr.rank == 1
        assert sr.score == pytest.approx(0.85)
