"""
application/query_usecase.py — QueryUseCase.

Orchestrates: detect language → translate → embed → retrieve → generate.
Depends only on domain ports — zero infrastructure imports.

Cross-lingual strategy:
  The Wikipedia source is English. French queries are detected via
  function-word overlap and translated to English before embedding/BM25,
  since both indices are built on English text. The original query is
  passed unchanged to the generator so the model answers in the user's
  language naturally.
"""

from __future__ import annotations

from typing import Optional

from src.domain.models import QueryResult
from src.domain.ports import IEmbedder, IGenerator, IReranker, IRetriever, ITranslator

# French function words for lightweight language detection
_FR_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "est",
    "que", "qui", "quoi", "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "combien", "quand", "où", "sont",
    "était", "avait", "avez", "avoir", "est-ce",
}


def _detect_language(text: str) -> str:
    """Return 'fr' if text appears French, 'en' otherwise."""
    words = set(text.lower().split())
    return "fr" if len(words & _FR_WORDS) >= 2 else "en"


class QueryUseCase:
    """Answer a natural-language question using the RAG pipeline."""

    def __init__(
        self,
        retriever: IRetriever,
        embedder: IEmbedder,
        generator: IGenerator,
        translator: Optional[ITranslator] = None,
        reranker: Optional[IReranker] = None,
        top_k: int = 5,
        candidate_pool_size: int = 20,
    ):
        self._retriever = retriever
        self._embedder = embedder
        self._generator = generator
        self._translator = translator
        self._reranker = reranker
        self._top_k = top_k
        # When a reranker is present, the hybrid retriever first pulls a
        # wider candidate pool (cheap), which the cross-encoder then
        # re-scores down to top_k (precise but too slow to run corpus-wide).
        self._candidate_pool_size = candidate_pool_size

    def execute(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        """
        Execute a query end-to-end.

        Parameters
        ----------
        question : user question (English or French)
        top_k    : override the default number of retrieved chunks

        Returns
        -------
        QueryResult with answer, sources, language metadata
        """
        if not self._retriever.is_ready():
            raise RuntimeError(
                "Index not ready. Run BuildIndexUseCase.execute() first."
            )

        k = top_k or self._top_k

        # Cross-lingual
        lang = _detect_language(question)
        if lang == "fr" and self._translator is not None:
            search_query = self._translator.translate(question, target_language="en")
        else:
            search_query = question

        # Embed the (potentially translated) query
        q_embedding = self._embedder.embed_query(search_query)

        # Hybrid retrieval — pull a wider pool when a reranker will refine it
        retrieve_k = max(k, self._candidate_pool_size) if self._reranker else k
        search_results = self._retriever.retrieve(
            query=search_query,
            query_embedding=q_embedding,
            top_k=retrieve_k,
        )

        if self._reranker is not None:
            search_results = self._reranker.rerank(search_query, search_results, k)

        chunks = [sr.chunk for sr in search_results]

        # Generate
        result = self._generator.generate(question, chunks)

        # Enrich with retrieval scores and language metadata
        for source, sr in zip(result.sources, search_results):
            source.score = sr.score

        result.detected_language = lang
        result.search_query_used = search_query

        return result
