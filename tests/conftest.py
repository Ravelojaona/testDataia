"""
tests/conftest.py — Shared pytest fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.models import Chunk, ChunkType


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(id=0, type=ChunkType.TEXT, section="Geography",
              content="[Section: Geography]\nMadagascar is an island nation in the Indian Ocean.",
              token_count=18),
        Chunk(id=1, type=ChunkType.TEXT, section="History",
              content="[Section: History]\nMadagascar gained independence from France in 1960.",
              token_count=16),
        Chunk(id=2, type=ChunkType.TABLE, section="Demographics",
              content="[TABLE | Section: Demographics]\n| Region | Population |\n|---|---|\n| Analamanga | 3 971 227 |",
              token_count=24),
        Chunk(id=3, type=ChunkType.TEXT, section="Economy",
              content="[Section: Economy]\nAgriculture accounts for a large portion of Madagascar's GDP.",
              token_count=17),
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """4 random unit vectors in 8-d space (tiny for speed)."""
    rng = np.random.default_rng(42)
    vecs = rng.random((4, 8)).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
