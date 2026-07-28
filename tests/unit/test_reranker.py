"""
tests/unit/test_reranker.py — CrossEncoderReranker unit tests.

The underlying transformers tokenizer/model are mocked out entirely: these
tests verify the reranking/sorting logic, not the ML model itself (which
would require downloading weights and is covered by manual/integration
testing instead).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.domain.models import Chunk, ChunkType, SearchResult


def _make_result(chunk_id: int, content: str) -> SearchResult:
    chunk = Chunk(id=chunk_id, type=ChunkType.TEXT, section="Test",
                  content=content, token_count=5)
    return SearchResult(chunk=chunk, score=0.5, rank=chunk_id + 1)


def _set_logits(reranker, values):
    reranker._model.return_value.logits = torch.tensor([[v] for v in values])


@pytest.fixture
def reranker():
    """CrossEncoderReranker with mocked tokenizer/model (no download)."""
    fake_tokenizer = MagicMock(return_value={})
    fake_model = MagicMock()
    fake_model.eval.return_value = None

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
    ), patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=fake_model,
    ):
        from src.infrastructure.reranker import CrossEncoderReranker

        instance = CrossEncoderReranker()

    return instance


class TestCrossEncoderReranker:
    def test_empty_results_returns_empty(self, reranker):
        assert reranker.rerank("query", [], top_k=5) == []

    def test_reorders_by_score(self, reranker):
        low = _make_result(0, "irrelevant passage")
        high = _make_result(1, "the passage that actually answers the query")
        _set_logits(reranker, [0.1, 0.9])

        out = reranker.rerank("query", [low, high], top_k=2)

        assert [r.chunk.id for r in out] == [1, 0]
        assert out[0].rank == 1
        assert out[1].rank == 2

    def test_truncates_to_top_k(self, reranker):
        results = [_make_result(i, f"passage {i}") for i in range(5)]
        _set_logits(reranker, [float(i) for i in range(5)])

        out = reranker.rerank("query", results, top_k=2)

        assert len(out) == 2
        assert [r.chunk.id for r in out] == [4, 3]

    def test_score_reflects_model_output(self, reranker):
        result = _make_result(0, "passage")
        _set_logits(reranker, [0.42])

        out = reranker.rerank("query", [result], top_k=1)

        assert out[0].score == pytest.approx(0.42, abs=1e-4)
