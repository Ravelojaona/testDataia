"""
application/index_usecase.py — BuildIndexUseCase.

Orchestrates: load → chunk → embed → index → persist.
Depends only on domain ports (interfaces), not concrete implementations.
Supports incremental rebuild via cached intermediate files.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from src.domain.models import Chunk, IndexStats
from src.domain.ports import IChunker, IDocumentLoader, IEmbedder, IRetriever

_RAW_HTML_CACHE = "data/raw.html"
_CHUNKS_CACHE = "data/chunks.json"


class BuildIndexUseCase:
    """
    Builds or reloads the retrieval index.

    If the index already exists on disk and force_rebuild=False, it is
    loaded immediately (avoiding expensive re-embedding API calls).
    """

    def __init__(
        self,
        loader: IDocumentLoader,
        chunker: IChunker,
        embedder: IEmbedder,
        retriever: IRetriever,
        index_path: str = "index",
    ):
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._retriever = retriever
        self._index_path = index_path

    def execute(self, force_rebuild: bool = False) -> IndexStats:
        """
        Execute the indexing pipeline.

        Returns the IndexStats of the resulting index.
        """
        index_exists = os.path.exists(os.path.join(self._index_path, "faiss.index"))

        if index_exists and not force_rebuild:
            print("Loading existing index from disk…")
            self._retriever.load(self._index_path)
            return self._retriever.stats()

        os.makedirs("data", exist_ok=True)
        os.makedirs(self._index_path, exist_ok=True)

        # 1 — Load
        chunks = self._load_or_build_chunks(force_rebuild)

        # 2 — Embed
        print(f"Generating embeddings for {len(chunks)} chunks…")
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        # 3 — Index
        print("Building FAISS + BM25 index…")
        self._retriever.build(chunks, embeddings)
        self._retriever.save(self._index_path)

        stats = self._retriever.stats()
        print(f"\nPipeline ready — {stats.total_chunks} chunks indexed.")
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_build_chunks(self, force: bool) -> List[Chunk]:
        if not force and os.path.exists(_CHUNKS_CACHE):
            print("Loading cached chunks…")
            return self._load_chunks_from_cache()

        if not force and os.path.exists(_RAW_HTML_CACHE):
            print("Loading cached HTML…")
            with open(_RAW_HTML_CACHE, "r", encoding="utf-8") as f:
                html = f.read()
        else:
            print("Fetching Wikipedia page (Madagascar)…")
            html = self._loader.load()
            with open(_RAW_HTML_CACHE, "w", encoding="utf-8") as f:
                f.write(html)

        print("Chunking content…")
        chunks = self._chunker.chunk(html)
        self._save_chunks_to_cache(chunks)
        return chunks

    def _save_chunks_to_cache(self, chunks: List[Chunk]) -> None:
        with open(_CHUNKS_CACHE, "w", encoding="utf-8") as f:
            json.dump([c.dict() for c in chunks], f, ensure_ascii=False, indent=2)
        n_t = sum(1 for c in chunks if c.type.value == "table")
        print(f"Saved {len(chunks)} chunks to cache ({n_t} tables)")

    def _load_chunks_from_cache(self) -> List[Chunk]:
        with open(_CHUNKS_CACHE, "r", encoding="utf-8") as f:
            return [Chunk(**d) for d in json.load(f)]
