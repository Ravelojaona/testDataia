"""
infrastructure/reranker.py — CrossEncoderReranker implementing IReranker.

Why a re-ranker at all:
  Dense + BM25 fusion (RRF) ranks each chunk independently of the others,
  purely on similarity to the query embedding / term overlap. On chunks
  that mix many numbers (large tables, dense factual paragraphs), this can
  bury the one passage that actually answers the question outside the
  retriever's top-k, especially when the query is phrased loosely (e.g.
  "the most recent census" vs "census 2018").

  A cross-encoder reranker scores (query, passage) pairs jointly instead
  of independently, which is much more precise at this kind of fine-grained
  relevance judgment — at the cost of being too slow to run over the whole
  corpus, hence why it only re-scores a wider candidate pool pulled from
  the cheap hybrid retriever.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — small, CPU-friendly, trained
for exactly this passage-reranking task.

Implementation note: loaded directly via `transformers` (AutoTokenizer +
AutoModelForSequenceClassification) rather than the `sentence-transformers`
convenience wrapper. The wrapper pulls in scikit-learn as a transitive
dependency purely for an unrelated similarity utility, and its compiled
extension was blocked by this machine's application-control policy
(unrelated to the RAG logic itself). Going through `transformers` directly
avoids that dependency entirely while using the exact same model weights.
"""

from __future__ import annotations

from typing import List

from src.domain.models import SearchResult
from src.domain.ports import IReranker

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(IReranker):
    def __init__(self, model_name: str = MODEL_NAME):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def rerank(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        if not results:
            return results

        pairs = [(query, sr.chunk.content) for sr in results]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        with self._torch.no_grad():
            logits = self._model(**inputs).logits.view(-1)

        scores = logits.tolist()
        reranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(chunk=sr.chunk, score=round(float(score), 6), rank=rank + 1)
            for rank, (sr, score) in enumerate(reranked[:top_k])
        ]
