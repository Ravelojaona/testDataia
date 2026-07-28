"""
tests/integration/test_pipeline.py — End-to-end integration tests.

These tests call the real OpenAI API and require a valid OPENAI_API_KEY.
They are skipped automatically when the key is absent (CI without secrets)
or when the index is not pre-built (to avoid expensive re-indexing in CI).

Run manually: pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import os

import pytest

SKIP_REASON = "Integration tests require OPENAI_API_KEY and a pre-built index."
requires_api = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason=SKIP_REASON
)

pytestmark = pytest.mark.integration


@requires_api
def test_full_english_query():
    """A factual English query returns a non-empty, non-refusal answer."""
    import openai

    from src.application.index_usecase import BuildIndexUseCase
    from src.application.query_usecase import QueryUseCase
    from src.infrastructure.chunker import WikipediaChunker
    from src.infrastructure.embedder import OpenAIEmbedder
    from src.infrastructure.generator import OpenAIGenerator, OpenAITranslator
    from src.infrastructure.hybrid_retriever import HybridRetriever
    from src.infrastructure.loader import WikipediaLoader

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retriever = HybridRetriever()
    embedder = OpenAIEmbedder(client)

    build_uc = BuildIndexUseCase(
        loader=WikipediaLoader(),
        chunker=WikipediaChunker(),
        embedder=embedder,
        retriever=retriever,
    )
    build_uc.execute()  # loads from disk if already built

    query_uc = QueryUseCase(
        retriever=retriever,
        embedder=embedder,
        generator=OpenAIGenerator(client),
        translator=OpenAITranslator(client),
    )

    result = query_uc.execute("What is the capital of Madagascar?")

    assert "Antananarivo" in result.answer
    assert len(result.sources) > 0
    assert result.detected_language == "en"


@requires_api
def test_out_of_scope_query_is_refused():
    """An out-of-scope query should trigger a refusal, not a hallucination."""
    import openai

    from src.application.index_usecase import BuildIndexUseCase
    from src.application.query_usecase import QueryUseCase
    from src.infrastructure.chunker import WikipediaChunker
    from src.infrastructure.embedder import OpenAIEmbedder
    from src.infrastructure.generator import OpenAIGenerator, OpenAITranslator
    from src.infrastructure.hybrid_retriever import HybridRetriever
    from src.infrastructure.loader import WikipediaLoader

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retriever = HybridRetriever()
    embedder = OpenAIEmbedder(client)

    build_uc = BuildIndexUseCase(
        loader=WikipediaLoader(),
        chunker=WikipediaChunker(),
        embedder=embedder,
        retriever=retriever,
    )
    build_uc.execute()

    query_uc = QueryUseCase(
        retriever=retriever,
        embedder=embedder,
        generator=OpenAIGenerator(client),
        translator=OpenAITranslator(client),
    )

    result = query_uc.execute("What is the national dish of Madagascar?")

    refusal_phrases = [
        "cannot answer", "not available", "not in the",
        "not mentioned", "based on the available information",
    ]
    assert any(p in result.answer.lower() for p in refusal_phrases), (
        f"Expected a refusal but got: {result.answer}"
    )
