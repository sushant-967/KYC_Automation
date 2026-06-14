"""
TavilyClient — async wrapper around the Tavily Search API.

Used by the screening agent to fetch live adverse media and PEP news.
Entirely optional: if TAVILY_API_KEY is not set, from_env() returns None
and all callers fall back to DB-only behaviour.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx


class TavilyClient:
    _BASE = "https://api.tavily.com/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def search_news(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Return a list of {title, url, content, score} from Tavily news search."""
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                r = await client.post(self._BASE, json={
                    "api_key":       self._key,
                    "query":         query,
                    "search_depth":  "basic",
                    "topic":         "news",
                    "include_answer": False,
                    "max_results":   max_results,
                })
                r.raise_for_status()
                return r.json().get("results", [])
        except Exception:
            return []   # network / quota failure → graceful degradation

    @classmethod
    def from_env(cls) -> Optional["TavilyClient"]:
        key = os.environ.get("TAVILY_API_KEY", "").strip()
        return cls(key) if key else None
