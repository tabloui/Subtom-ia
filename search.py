from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup

DDG_HTML = "https://html.duckduckgo.com/html/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"

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
    domain: str = ""


@dataclass
class _CacheEntry:
    results: list[SearchResult]
    expires_at: float


class SearchCache:
    """Caché en memoria con TTL para no repetir búsquedas idénticas."""

    def __init__(self, ttl: float = 300.0, max_size: int = 256) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._store: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[list[SearchResult]]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            self._store.pop(key, None)
            return None
        return entry.results

    def set(self, key: str, results: list[SearchResult]) -> None:
        if len(self._store) >= self.max_size:
            oldest = min(self._store, key=lambda k: self._store[k].expires_at)
            self._store.pop(oldest, None)
        self._store[key] = _CacheEntry(results, time.monotonic() + self.ttl)


_cache = SearchCache()


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


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


class SearchError(Exception):
    """Error controlado de búsqueda tras agotar reintentos."""


async def _fetch(
    session: aiohttp.ClientSession,
    endpoint: str,
    query: str,
    region: str,
    safesearch: str,
) -> str:
    data = {"q": query, "kl": region}
    if safesearch in ("on", "moderate", "off"):
        data["kp"] = {"on": "1", "moderate": "-1", "off": "-2"}[safesearch]
    async with session.post(endpoint, data=data, allow_redirects=True) as resp:
        resp.raise_for_status()
        return await resp.text()


def _parse_html(html: str, max_results: int) -> list[SearchResult]:
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
        results.append(
            SearchResult(title=title, url=url, snippet=snippet, domain=_domain_of(url))
        )
        if len(results) >= max_results:
            break

    return results


def _parse_lite(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    results: list[SearchResult] = []

    for a in soup.select("a.result-link"):
        title = a.get_text(" ", strip=True)
        url = _clean_url(a.get("href", ""))
        if not url or not title:
            continue
        snippet = ""
        row = a.find_parent("tr")
        if row is not None:
            nxt = row.find_next_sibling("tr")
            if nxt is not None:
                snippet = nxt.get_text(" ", strip=True)
        results.append(
            SearchResult(title=title, url=url, snippet=snippet, domain=_domain_of(url))
        )
        if len(results) >= max_results:
            break

    return results


async def search_duckduckgo(
    query: str,
    max_results: int = 5,
    timeout: float = 12.0,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    retries: int = 2,
    use_cache: bool = True,
    session: Optional[aiohttp.ClientSession] = None,
) -> list[SearchResult]:
    """Busca en DuckDuckGo con reintentos, fallback a lite y caché opcional."""
    if not query or not query.strip():
        return []

    max_results = max(1, min(int(max_results), 10))
    cache_key = f"{query.strip().lower()}|{max_results}|{region}|{safesearch}"

    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=timeout),
        )

    last_error: Optional[Exception] = None
    results: list[SearchResult] = []

    try:
        for attempt in range(retries + 1):
            try:
                html = await _fetch(session, DDG_HTML, query, region, safesearch)
                results = _parse_html(html, max_results)
                if results:
                    break
                html = await _fetch(session, DDG_LITE, query, region, safesearch)
                results = _parse_lite(html, max_results)
                if results:
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
    finally:
        if owns_session and session is not None:
            await session.close()

    if not results and last_error is not None:
        raise SearchError(f"Búsqueda fallida tras {retries + 1} intentos: {last_error}")

    if use_cache and results:
        _cache.set(cache_key, results)

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
