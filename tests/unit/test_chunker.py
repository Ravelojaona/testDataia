"""
tests/unit/test_chunker.py — WikipediaChunker unit tests.

Uses a minimal HTML fixture instead of real Wikipedia to keep tests
fast and deterministic (no network calls).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.domain.models import ChunkType
from src.infrastructure.chunker import WikipediaChunker, _clean, _split_text, _table_to_markdown

SAMPLE_HTML = """
<html><body>
  <h2>Geography</h2>
  <p>Madagascar is the world's fourth largest island. It covers 587,041 km².</p>
  <p>The island lies in the Indian Ocean off the east coast of Africa.</p>

  <h2>Demographics</h2>
  <table>
    <tr><th>Region</th><th>Capital</th><th>Population</th></tr>
    <tr><td>Analamanga</td><td>Antananarivo</td><td>3,971,227</td></tr>
    <tr><td>Vakinankaratra</td><td>Antsirabe</td><td>2,066,944</td></tr>
  </table>

  <h2>References</h2>
  <p>This section should be skipped.</p>
</body></html>
"""


@pytest.fixture
def chunker():
    return WikipediaChunker()


class TestCleanText:
    def test_removes_citation_markers(self):
        assert _clean("Madagascar[1] has a population[23].") == "Madagascar has a population."

    def test_normalises_whitespace(self):
        assert _clean("Hello  \n  world") == "Hello world"

    def test_removes_note_markers(self):
        assert _clean("See [note 1] and [note 2].") == "See and ."


class TestTableToMarkdown:
    def test_basic_table(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        table = BeautifulSoup(html, "html.parser").find("table")
        md = _table_to_markdown(table)
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md
        assert "---" in md

    def test_empty_table_returns_empty(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><td>only one row</td></tr></table>"
        table = BeautifulSoup(html, "html.parser").find("table")
        assert _table_to_markdown(table) == ""

    def test_pipe_in_cell_is_escaped(self):
        from bs4 import BeautifulSoup
        html = "<table><tr><th>X</th></tr><tr><td>a|b</td></tr></table>"
        table = BeautifulSoup(html, "html.parser").find("table")
        md = _table_to_markdown(table)
        assert "a\\|b" in md


class TestSplitText:
    def test_short_text_not_split(self):
        text = "Short text."
        parts = _split_text(text)
        assert len(parts) == 1
        assert parts[0] == text

    def test_long_text_is_split(self):
        # Generate a text definitely > 500 words (word-based fallback)
        long_text = " ".join([f"word{i}" for i in range(600)])
        parts = _split_text(long_text)
        assert len(parts) > 1

    def test_multiple_parts_produced(self):
        long_text = " ".join([f"token{i}" for i in range(700)])
        parts = _split_text(long_text)
        assert len(parts) >= 2


class TestWikipediaChunker:
    def test_extracts_text_chunks(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        text_chunks = [c for c in chunks if c.type == ChunkType.TEXT]
        assert len(text_chunks) >= 2

    def test_extracts_table_chunk(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        table_chunks = [c for c in chunks if c.type == ChunkType.TABLE]
        assert len(table_chunks) == 1
        assert "Analamanga" in table_chunks[0].content

    def test_references_section_skipped(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        contents = " ".join(c.content for c in chunks)
        assert "This section should be skipped" not in contents

    def test_chunk_ids_are_sequential(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        ids = [c.id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_section_prefix_in_text_chunk(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        text_chunks = [c for c in chunks if c.type == ChunkType.TEXT]
        assert any("Section: Geography" in c.content for c in text_chunks)

    def test_table_prefix_contains_section(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        table_chunks = [c for c in chunks if c.type == ChunkType.TABLE]
        assert "TABLE" in table_chunks[0].content
        assert "Demographics" in table_chunks[0].content

    def test_token_count_is_positive(self, chunker):
        chunks = chunker.chunk(SAMPLE_HTML)
        assert all(c.token_count > 0 for c in chunks)

    def test_citation_markers_removed(self, chunker):
        html = "<html><body><h2>Geo</h2><p>Area is 587,041 km²[1].</p></body></html>"
        chunks = chunker.chunk(html)
        assert all("[1]" not in c.content for c in chunks)
