from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup

DDG_HTML = "https://html.duckduckgo.com/html/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Referer": "https://duckduckgo.com/",
}


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if "uddg=" in url:
        qs = parse_qs(urlparse(url).query)
        target = qs.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


async def search_duckduckgo(
    query: str,
    max_results: int = 5,
    timeout: float = 12.0,
) -> list[SearchResult]:
    if not query or not query.strip():
        return []

    max_results = max(1, min(int(max_results), 10))

    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=client_timeout,
    ) as session:
        async with session.post(
            DDG_HTML,
            data={"q": query, "kl": "wt-wt"},
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            html = await response.text()

    soup = BeautifulSoup(html, "lxml")

    results: list[SearchResult] = []

    for block in soup.select("div.result, div.web-result"):
        title_el = block.select_one("a.result__a")
        if not title_el:
            continue

        title = title_el.get_text(" ", strip=True)
        url = _clean_url(title_el.get("href", ""))

        snippet_el = block.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        if not url or not title:
            continue

        results.append(SearchResult(title=title, url=url, snippet=snippet))

        if len(results) >= max_results:
            break

    return results


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "Sin resultados."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   URL: {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        lines.append("")
    return "\n".join(lines).strip()
