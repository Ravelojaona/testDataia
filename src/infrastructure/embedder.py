"""
infrastructure/embedder.py — OpenAIEmbedder implementing IEmbedder.

Model: text-embedding-3-small
  - 1 536-dimension vectors
  - Best cost/quality ratio for factual retrieval
  - Same tokeniser family as GPT-4o → consistent token handling

Batching at 100 texts/call stays well below the 2 048-item API limit
while providing readable progress logging.
"""

from __future__ import annotations

from typing import List

import numpy as np
import openai

from src.domain.ports import IEmbedder

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


class OpenAIEmbedder(IEmbedder):
    def __init__(self, client: openai.OpenAI, model: str = MODEL):
        self._client = client
        self._model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches; returns (n, dim) float32 array."""
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            all_embeddings.extend(e.embedding for e in response.data)
            done = min(i + BATCH_SIZE, len(texts))
            print(f"  Embedded {done}/{len(texts)} chunks…")

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single string; returns (dim,) float32 array."""
        response = self._client.embeddings.create(model=self._model, input=[query])
        return np.array(response.data[0].embedding, dtype=np.float32)
