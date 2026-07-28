"""
domain/ports.py — Abstract interfaces (ports) for the hexagonal architecture.

The application layer depends ONLY on these abstractions.
The infrastructure layer provides concrete implementations.
This makes every external dependency swappable (different LLM, different
vector store, different embedder) without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .models import Chunk, IndexStats, QueryResult, SearchResult


class IDocumentLoader(ABC):
    """Port: loads raw HTML content from a source (Wikipedia API, local file…)."""

    @abstractmethod
    def load(self) -> str:
        """Return the raw HTML of the source document."""


class IChunker(ABC):
    """Port: splits raw HTML into indexable Chunk objects."""

    @abstractmethod
    def chunk(self, html: str) -> List[Chunk]:
        """Parse HTML and return a list of chunks ready for embedding."""


class IEmbedder(ABC):
    """Port: converts text to dense vector representations."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts; returns (n, dim) float32 array."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string; returns (dim,) float32 array."""


class IRetriever(ABC):
    """Port: builds a retrieval index and answers similarity queries."""

    @abstractmethod
    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Index all chunks with their pre-computed embeddings."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Return the top_k most relevant chunks for the query."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True if the index has been built or loaded."""

    @abstractmethod
    def stats(self) -> IndexStats:
        """Return metadata about the current index."""


class IGenerator(ABC):
    """Port: generates a natural-language answer from a query and context chunks."""

    @abstractmethod
    def generate(self, query: str, chunks: List[Chunk]) -> QueryResult:
        """Generate a grounded answer and return a complete QueryResult."""


class ITranslator(ABC):
    """Port: translates text between languages."""

    @abstractmethod
    def translate(self, text: str, target_language: str = "en") -> str:
        """Translate text to the target language."""
