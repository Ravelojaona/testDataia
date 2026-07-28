"""
tests/unit/test_retriever.py — HybridRetriever unit tests.

Uses small in-memory fixtures — no disk I/O, no real FAISS file.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.models import ChunkType
from src.infrastructure.hybrid_retriever import HybridRetriever


@pytest.fixture
def built_retriever(sample_chunks, sample_embeddings):
    """A HybridRetriever already built with the sample fixtures."""
    r = HybridRetriever()
    r.build(sample_chunks, sample_embeddings)
    return r


class TestBuild:
    def test_is_ready_after_build(self, built_retriever):
        assert built_retriever.is_ready() is True

    def test_not_ready_before_build(self):
        r = HybridRetriever()
        assert r.is_ready() is False

    def test_stats_after_build(self, built_retriever):
        stats = built_retriever.stats()
        assert stats.total_chunks == 4
        assert stats.text_chunks == 3
        assert stats.table_chunks == 1
        assert stats.is_ready is True
        assert "Geography" in stats.sections


class TestRetrieve:
    def test_returns_top_k_results(self, built_retriever, sample_embeddings):
        q_vec = sample_embeddings[0]
        results = built_retriever.retrieve("Madagascar geography", q_vec, top_k=2)
        assert len(results) == 2

    def test_results_have_scores(self, built_retriever, sample_embeddings):
        results = built_retriever.retrieve("economy", sample_embeddings[3], top_k=3)
        assert all(r.score > 0 for r in results)

    def test_ranks_are_sequential(self, built_retriever, sample_embeddings):
        results = built_retriever.retrieve("question", sample_embeddings[0], top_k=4)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_top_k_capped_at_corpus_size(self, built_retriever, sample_embeddings):
        results = built_retriever.retrieve("anything", sample_embeddings[0], top_k=100)
        assert len(results) <= 4

    def test_retrieve_returns_chunks(self, built_retriever, sample_embeddings):
        results = built_retriever.retrieve("population", sample_embeddings[2], top_k=2)
        assert all(hasattr(r, "chunk") for r in results)

    def test_retrieve_before_build_raises(self, sample_embeddings):
        r = HybridRetriever()
        with pytest.raises(RuntimeError, match="not initialised"):
            r.retrieve("question", sample_embeddings[0])


class TestRRFFusion:
    def test_alpha_0_uses_only_bm25(self, sample_chunks, sample_embeddings):
        """With alpha=0, dense scores contribute 0 — BM25 drives ranking."""
        r = HybridRetriever(alpha=0.0)
        r.build(sample_chunks, sample_embeddings)
        results = r.retrieve("agriculture economy", sample_embeddings[0], top_k=2)
        # Economy chunk (id=3) contains "agriculture" — BM25 should rank it first
        assert any(res.chunk.section == "Economy" for res in results)

    def test_alpha_1_uses_only_dense(self, sample_chunks, sample_embeddings):
        """With alpha=1, BM25 contributes 0 — dense drives ranking."""
        r = HybridRetriever(alpha=1.0)
        r.build(sample_chunks, sample_embeddings)
        q_vec = sample_embeddings[0]  # closest to chunk 0 (Geography)
        results = r.retrieve("random query", q_vec, top_k=1)
        assert results[0].chunk.id == 0


class TestPersistence:
    def test_save_and_load(self, built_retriever, sample_embeddings, tmp_path):
        save_dir = str(tmp_path / "test_index")
        built_retriever.save(save_dir)

        r2 = HybridRetriever()
        r2.load(save_dir)

        assert r2.is_ready()
        assert r2.stats().total_chunks == 4

        # Results should be identical
        q_vec = sample_embeddings[0]
        r1_results = built_retriever.retrieve("question", q_vec, top_k=2)
        r2_results = r2.retrieve("question", q_vec, top_k=2)
        assert [r.chunk.id for r in r1_results] == [r.chunk.id for r in r2_results]
