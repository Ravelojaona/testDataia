"""
infrastructure/loader.py — WikipediaLoader implementing IDocumentLoader.

Uses the Wikipedia Action API (action=parse) for reliable structured HTML
rather than raw web scraping, which is fragile against layout changes.
"""

from __future__ import annotations

import requests

from src.domain.ports import IDocumentLoader

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia rejects requests with no descriptive User-Agent (403):
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "rag-madagascar/1.0 (technical-test; contact: test_ia_072026@tikasa.net)"


class WikipediaLoader(IDocumentLoader):
    """Fetches a Wikipedia article's HTML via the MediaWiki Action API."""

    def __init__(self, page_title: str = "Madagascar", timeout: int = 30):
        self.page_title = page_title
        self.timeout = timeout

    def load(self) -> str:
        params = {
            "action": "parse",
            "page": self.page_title,
            "prop": "text",
            "format": "json",
            "disableeditsection": "1",
        }
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(
            WIKIPEDIA_API, params=params, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(f"Wikipedia API error: {data['error']['info']}")

        return data["parse"]["text"]["*"]
