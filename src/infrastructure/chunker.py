"""
infrastructure/chunker.py — WikipediaChunker implementing IChunker.

Chunking strategy:
  TEXT → sliding window (500 tokens, 50-token overlap).
         Each chunk is prefixed with its section heading so it is
         self-contained when retrieved out of context.
  TABLE → one atomic chunk per table (tables lose meaning if split
          mid-row; every table on this page fits under 1 000 tokens).

Token counting uses tiktoken (cl100k_base) — same tokeniser family as
text-embedding-3-small and GPT-4o, ensuring consistent size estimates.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from bs4 import BeautifulSoup

from src.domain.models import Chunk, ChunkType
from src.domain.ports import IChunker

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    _ENCODING = None
    _USE_TIKTOKEN = False

_SKIP_SECTIONS = {
    "references", "notes", "external links", "see also",
    "further reading", "bibliography", "footnotes",
}

MAX_TOKENS = 500
OVERLAP_TOKENS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    if _USE_TIKTOKEN and _ENCODING is not None:
        return len(_ENCODING.encode(text))
    # Fallback: approximate 1 token ≈ 4 chars (good enough for chunking)
    return max(1, len(text) // 4)


def _clean(text: str) -> str:
    """Strip citation markers and normalise whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note \d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _table_to_markdown(table) -> str:
    """Convert a BeautifulSoup <table> element to a Markdown table string."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [
            _clean(td.get_text(separator=" ", strip=True)).replace("|", "\\|")
            for td in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return ""

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * max_cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{header}\n{sep}\n{body}"


def _split_text(text: str) -> List[str]:
    """Sliding window split with overlap. Uses tiktoken if available, else word-based."""
    if _USE_TIKTOKEN and _ENCODING is not None:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= MAX_TOKENS:
            return [text]
        parts: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_TOKENS, len(tokens))
            parts.append(_ENCODING.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += MAX_TOKENS - OVERLAP_TOKENS
        return parts

    # Fallback: word-based split (approx 4 chars/token → MAX_TOKENS words ≈ 500 tokens)
    words = text.split()
    word_limit = MAX_TOKENS  # treat 1 word ≈ 1 token for simplicity
    if len(words) <= word_limit:
        return [text]
    parts = []
    start = 0
    while start < len(words):
        end = min(start + word_limit, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += word_limit - OVERLAP_TOKENS
    return parts


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class WikipediaChunker(IChunker):
    """Parses Wikipedia HTML and returns a flat list of Chunk objects."""

    def chunk(self, html: str) -> List[Chunk]:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noisy elements
        for tag in soup.find_all(["sup", "style", "script", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(class_=["reference", "mw-editsection", "navbox", "thumb"]):
            tag.decompose()

        raw_blocks: List[Tuple[str, str, str]] = []  # (type, section, content)
        current_section = "Introduction"

        for elem in soup.find_all(["h2", "h3", "h4", "p", "table", "ul", "ol"]):
            if elem.name in ("h2", "h3", "h4"):
                heading = re.sub(r"\[edit\]$", "", _clean(elem.get_text())).strip()
                if heading and heading.lower() not in _SKIP_SECTIONS:
                    current_section = heading
                continue

            if current_section.lower() in _SKIP_SECTIONS:
                continue

            if elem.name in ("p", "ul", "ol"):
                text = _clean(elem.get_text(separator=" ", strip=True))
                if len(text) > 60:
                    raw_blocks.append(("text", current_section, text))

            elif elem.name == "table":
                md = _table_to_markdown(elem)
                if md and len(md) > 50:
                    raw_blocks.append(("table", current_section, md))

        return self._build_chunks(raw_blocks)

    def _build_chunks(self, raw_blocks: List[Tuple[str, str, str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        cid = 0

        for kind, section, content in raw_blocks:
            if kind == "table":
                full = f"[TABLE | Section: {section}]\n{content}"
                chunks.append(Chunk(
                    id=cid,
                    type=ChunkType.TABLE,
                    section=section,
                    content=full,
                    token_count=_count_tokens(full),
                ))
                cid += 1
            else:
                parts = _split_text(content)
                for i, part in enumerate(parts):
                    label = section if i == 0 else f"{section} (cont.)"
                    full = f"[Section: {label}]\n{part}"
                    chunks.append(Chunk(
                        id=cid,
                        type=ChunkType.TEXT,
                        section=section,
                        content=full,
                        token_count=_count_tokens(full),
                    ))
                    cid += 1

        return chunks
