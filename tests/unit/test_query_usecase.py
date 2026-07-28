"""
tests/unit/test_query_usecase.py — QueryUseCase unit tests with mocked ports.

All external dependencies (embedder, retriever, generator, translator)
are replaced by unittest.mock stubs so tests run with zero API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.query_usecase import QueryUseCase, _detect_language
from src.domain.models import Chunk, ChunkType, QueryResult, SearchResult, Source


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_english_detected(self):
        assert _detect_language("What is the capital of Madagascar?") == "en"

    def test_french_detected(self):
        assert _detect_language("Quelle est la capitale de Madagascar ?") == "fr"

    def test_short_text_defaults_to_english(self):
        assert _detect_language("hello") == "en"

    def test_mixed_with_fr_words_detects_french(self):
        assert _detect_language("Comment est le gouvernement de Madagascar ?") == "fr"


# ---------------------------------------------------------------------------
# QueryUseCase
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chunk():
    return Chunk(id=0, type=ChunkType.TEXT, section="Geography",
                 content="Madagascar is the fourth largest island.", token_count=10)


@pytest.fixture
def mock_search_result(mock_chunk):
    return SearchResult(chunk=mock_chunk, score=0.88, rank=1)


@pytest.fixture
def mock_query_result(mock_chunk):
    return QueryResult(
        query="What is Madagascar?",
        answer="Madagascar is the fourth largest island.",
        sources=[Source(chunk_id=0, section="Geography", type=ChunkType.TEXT)],
        model="gpt-4o",
    )


@pytest.fixture
def use_case(mock_search_result, mock_query_result):
    retriever = MagicMock()
    retriever.is_ready.return_value = True
    retriever.retrieve.return_value = [mock_search_result]

    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(8, dtype=np.float32)

    generator = MagicMock()
    generator.generate.return_value = mock_query_result

    translator = MagicMock()
    translator.translate.return_value = "What is the capital of Madagascar?"

    return QueryUseCase(
        retriever=retriever,
        embedder=embedder,
        generator=generator,
        translator=translator,
        top_k=5,
    )


class TestQueryUseCase:
    def test_english_query_no_translation(self, use_case):
        result = use_case.execute("What is the capital?")
        use_case._translator.translate.assert_not_called()
        assert result.detected_language == "en"

    def test_french_query_triggers_translation(self, use_case):
        result = use_case.execute("Quelle est la capitale de Madagascar ?")
        use_case._translator.translate.assert_called_once()
        assert result.detected_language == "fr"

    def test_retriever_is_called(self, use_case):
        use_case.execute("What is Madagascar?")
        use_case._retriever.retrieve.assert_called_once()

    def test_generator_is_called_with_chunks(self, use_case, mock_chunk):
        use_case.execute("What is Madagascar?")
        call_args = use_case._generator.generate.call_args
        chunks_arg = call_args[0][1]
        assert len(chunks_arg) == 1
        assert chunks_arg[0].id == mock_chunk.id

    def test_raises_if_index_not_ready(self):
        retriever = MagicMock()
        retriever.is_ready.return_value = False
        uc = QueryUseCase(
            retriever=retriever,
            embedder=MagicMock(),
            generator=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="Index not ready"):
            uc.execute("Some question")

    def test_top_k_override(self, use_case):
        use_case.execute("Question?", top_k=3)
        call_kwargs = use_case._retriever.retrieve.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_source_scores_enriched(self, use_case, mock_search_result):
        result = use_case.execute("What is Madagascar?")
        assert result.sources[0].score == pytest.approx(mock_search_result.score)


# ---------------------------------------------------------------------------
# Reranker wiring
# ---------------------------------------------------------------------------

class TestRerankerWiring:
    def test_no_reranker_retrieves_exactly_top_k(self, use_case):
        use_case.execute("Question?", top_k=3)
        assert use_case._retriever.retrieve.call_args[1]["top_k"] == 3

    def test_reranker_pulls_wider_candidate_pool(self, mock_search_result, mock_query_result):
        retriever = MagicMock()
        retriever.is_ready.return_value = True
        retriever.retrieve.return_value = [mock_search_result]

        embedder = MagicMock()
        embedder.embed_query.return_value = np.zeros(8, dtype=np.float32)

        generator = MagicMock()
        generator.generate.return_value = mock_query_result

        reranker = MagicMock()
        reranker.rerank.return_value = [mock_search_result]

        uc = QueryUseCase(
            retriever=retriever,
            embedder=embedder,
            generator=generator,
            reranker=reranker,
            top_k=3,
            candidate_pool_size=20,
        )
        uc.execute("Question?")

        assert retriever.retrieve.call_args[1]["top_k"] == 20
        reranker.rerank.assert_called_once()
        assert reranker.rerank.call_args[0][2] == 3

    def test_reranker_output_feeds_generator(self, mock_chunk, mock_query_result):
        other_chunk = Chunk(id=1, type=ChunkType.TEXT, section="Demographics",
                             content="Population figures.", token_count=5)
        result_a = SearchResult(chunk=mock_chunk, score=0.1, rank=1)
        result_b = SearchResult(chunk=other_chunk, score=0.9, rank=1)

        retriever = MagicMock()
        retriever.is_ready.return_value = True
        retriever.retrieve.return_value = [result_a, result_b]

        embedder = MagicMock()
        embedder.embed_query.return_value = np.zeros(8, dtype=np.float32)

        generator = MagicMock()
        generator.generate.return_value = mock_query_result

        reranker = MagicMock()
        # Reranker flips the order — only result_b should reach the generator.
        reranker.rerank.return_value = [result_b]

        uc = QueryUseCase(
            retriever=retriever, embedder=embedder, generator=generator,
            reranker=reranker, top_k=1,
        )
        uc.execute("Question?")

        chunks_arg = generator.generate.call_args[0][1]
        assert len(chunks_arg) == 1
        assert chunks_arg[0].id == other_chunk.id
