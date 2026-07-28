"""
infrastructure/generator.py — OpenAIGenerator implementing IGenerator.

Prompt engineering choices:
  - System prompt hard-bans external knowledge and specifies the exact
    refusal phrase so evaluate.py can detect it reliably.
  - Each chunk is labelled (section + type + id) so the model can cite
    precisely.
  - temperature=0 maximises factual consistency.
  - Temporal ambiguity is explicitly addressed: the model must distinguish
    figures from different periods.
"""

from __future__ import annotations

from typing import List

import openai

from src.domain.models import Chunk, ChunkType, QueryResult, Source
from src.domain.ports import IGenerator

MODEL = "gpt-4o"

_SYSTEM_PROMPT = """\
You are a precise, grounded question-answering assistant.
Your ONLY source of knowledge is the context passages provided below, \
extracted from the English Wikipedia article about Madagascar.

RULES — follow them without exception:
1. Answer ONLY based on the provided context. Never use external knowledge.
2. Cite your source(s) at the end of every answer:
   → Source: [Section name] | Chunk #[id]
   If multiple chunks were used, cite all of them.
3. When the answer is NOT present in the context, respond with exactly:
   "I cannot answer this question based on the available information \
from the Madagascar Wikipedia page."
   Do NOT guess, infer, or use outside knowledge.
4. When multiple time periods are mentioned for the same fact (population
   figures, presidents, poverty rates…), always specify which period
   each figure refers to. Never mix values from different periods.
5. Be concise and factual. Do not speculate beyond what the text states.
"""


def _format_context(chunks: List[Chunk]) -> str:
    parts = [
        f"[Chunk #{c.id} | Section: {c.section} | Type: {c.type.value}]\n{c.content}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(parts)


class OpenAIGenerator(IGenerator):
    def __init__(self, client: openai.OpenAI, model: str = MODEL):
        self._client = client
        self._model = model

    def generate(self, query: str, chunks: List[Chunk]) -> QueryResult:
        if not chunks:
            return QueryResult(
                query=query,
                answer=(
                    "I cannot answer this question based on the available "
                    "information from the Madagascar Wikipedia page."
                ),
                sources=[],
                model=self._model,
            )

        user_message = (
            "Context passages from the Wikipedia article on Madagascar:\n\n"
            f"{_format_context(chunks)}\n\n"
            "---\n\n"
            f"Question: {query}\n\n"
            "Answer (strictly based on the context above):"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()

        return QueryResult(
            query=query,
            answer=answer,
            sources=[
                Source(
                    chunk_id=c.id,
                    section=c.section,
                    type=c.type,
                    score=None,
                )
                for c in chunks
            ],
            model=self._model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )


class OpenAITranslator:
    """Lightweight translator using GPT-4o-mini (cheap & fast)."""

    def __init__(self, client: openai.OpenAI):
        self._client = client

    def translate(self, text: str, target_language: str = "en") -> str:
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_language}. Return only the translation.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
