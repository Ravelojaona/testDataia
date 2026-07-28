"""
infrastructure/hybrid_retriever.py — HybridRetriever implementing IRetriever.

Combines dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank
Fusion (RRF).

Design decisions:
  FAISS IndexFlatIP  — exact cosine search (L2-normed vectors). The corpus
    is small (< 1 000 chunks), so approximate methods add complexity with
    no measurable speed benefit.
  BM25 Okapi  — complements dense retrieval for exact keyword matches:
    dates, proper nouns, precise figures.
  RRF  — rank-based fusion avoids the scale mismatch between cosine scores
    ∈ [0, 1] and BM25 scores (unbounded). Standard k=60 constant.

Persistence:
  FAISS index   → binary (faiss.write_index)
  BM25 model    → pickle
  Chunk list    → JSON (human-readable, inspectable)
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.domain.models import Chunk, ChunkType, IndexStats, SearchResult
from src.domain.ports import IRetriever

RRF_K = 60


class HybridRetriever(IRetriever):
    def __init__(self, index_path: str = "index", alpha: float = 0.5):
        """
        Parameters
        ----------
        index_path : directory where FAISS/BM25 files are saved/loaded
        alpha      : RRF weight for dense retrieval (0=BM25 only, 1=dense only)
        """
        self._path = index_path
        self._alpha = alpha
        self._faiss_index: faiss.Index = None
        self._bm25: BM25Okapi = None
        self._chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # IRetriever — build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        self._chunks = chunks

        # FAISS — cosine similarity via L2-normalised inner product
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        normed = embeddings.copy()
        faiss.normalize_L2(normed)
        self._faiss_index.add(normed)

        # BM25
        tokenised = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)

        n_text = sum(1 for c in chunks if c.type == ChunkType.TEXT)
        n_table = sum(1 for c in chunks if c.type == ChunkType.TABLE)
        print(f"Index built: {len(chunks)} chunks  ({n_text} text, {n_table} tables)  dim={dim}")

    # ------------------------------------------------------------------
    # IRetriever — retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not self.is_ready():
            raise RuntimeError("Retriever not initialised — call build() or load() first.")

        n = len(self._chunks)
        candidate_k = min(top_k * 4, n)

        # Dense
        q_vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self._faiss_index.search(q_vec, candidate_k)
        dense_ranked = list(zip(indices[0].tolist(), scores[0].tolist()))

        # BM25
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:candidate_k].tolist()
        bm25_ranked = [(idx, float(bm25_scores[idx])) for idx in bm25_top]

        # RRF fusion
        rrf: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + self._alpha / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + (1 - self._alpha) / (RRF_K + rank + 1)

        top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]

        return [
            SearchResult(chunk=self._chunks[idx], score=round(rrf[idx], 6), rank=rank + 1)
            for rank, idx in enumerate(top_ids)
        ]

    # ------------------------------------------------------------------
    # IRetriever — persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._faiss_index, f"{path}/faiss.index")
        with open(f"{path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self._chunks], f, ensure_ascii=False, indent=2)
        with open(f"{path}/bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        print(f"Index saved → {path}/")

    def load(self, path: str) -> None:
        self._faiss_index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/chunks.json", "r", encoding="utf-8") as f:
            self._chunks = [Chunk(**c) for c in json.load(f)]
        with open(f"{path}/bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        self._path = path
        print(f"Index loaded from {path}/  ({len(self._chunks)} chunks)")

    # ------------------------------------------------------------------
    # IRetriever — status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._faiss_index is not None and bool(self._chunks)

    def stats(self) -> IndexStats:
        return IndexStats(
            total_chunks=len(self._chunks),
            text_chunks=sum(1 for c in self._chunks if c.type == ChunkType.TEXT),
            table_chunks=sum(1 for c in self._chunks if c.type == ChunkType.TABLE),
            sections=sorted({c.section for c in self._chunks}),
            index_path=self._path,
            is_ready=self.is_ready(),
        )
