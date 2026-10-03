from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import aiohttp
from bs4 import BeautifulSoup


DDG_HTML = "https://html.duckduckgo.com/html/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"
WIKI_API = "https://es.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://duckduckgo.com/",
    "DNT": "1",
    "Connection": "keep-alive",
}

DOMINIOS_BLOQUEADOS = {
    "adf.ly",
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "doubleclick.net",
    "googlesyndication.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
}

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
    "ref",
    "ref_src",
}

MAX_QUERY_LENGTH = 500
MAX_TITLE_LENGTH = 300
MAX_SNIPPET_LENGTH = 1500


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str = ""
    source: str = "ddg"


@dataclass
class _CacheEntry:
    results: list[SearchResult]
    expires_at: float


class SearchError(Exception):
    pass


class SearchCache:
    def __init__(
        self,
        ttl: float = 300.0,
        max_size: int = 256,
    ) -> None:
        self.ttl = max(1.0, float(ttl))
        self.max_size = max(1, int(max_size))
        self._store: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        key: str,
    ) -> Optional[list[SearchResult]]:
        async with self._lock:
            entry = self._store.get(key)

            if entry is None:
                return None

            if time.monotonic() >= entry.expires_at:
                self._store.pop(key, None)
                return None

            return list(entry.results)

    async def set(
        self,
        key: str,
        results: list[SearchResult],
    ) -> None:
        async with self._lock:
            if len(self._store) >= self.max_size:
                oldest_key = min(
                    self._store,
                    key=lambda item: (
                        self._store[item].expires_at
                    ),
                )
                self._store.pop(oldest_key, None)

            self._store[key] = _CacheEntry(
                results=list(results),
                expires_at=(
                    time.monotonic()
                    + self.ttl
                ),
            )

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def size(self) -> int:
        async with self._lock:
            now = time.monotonic()

            expired = [
                key
                for key, entry in self._store.items()
                if now >= entry.expires_at
            ]

            for key in expired:
                self._store.pop(key, None)

            return len(self._store)


_cache = SearchCache()


def _clean_text(
    value: str,
    max_length: int | None = None,
) -> str:
    if not value:
        return ""

    value = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    if max_length is not None:
        value = value[:max_length].strip()

    return value


def _clean_url(url: str) -> str:
    if not url:
        return ""

    url = str(url).strip()

    if url.startswith("//"):
        url = "https:" + url

    try:
        parsed = urlparse(url)

        if "uddg" in parse_qs(parsed.query):
            target = parse_qs(
                parsed.query
            ).get(
                "uddg",
                [""],
            )[0]

            if target:
                url = unquote(target)
                parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return ""

        query = parse_qs(
            parsed.query,
            keep_blank_values=True,
        )

        cleaned_query: list[tuple[str, str]] = []

        for key, values in query.items():
            if key.lower() in TRACKING_PARAMS:
                continue

            for value in values:
                cleaned_query.append(
                    (key, value)
                )

        new_query = urlencode(
            cleaned_query,
            doseq=True,
        )

        parsed = parsed._replace(
            query=new_query,
            fragment="",
        )

        return parsed.geturl()

    except Exception:
        return ""


def _domain_of(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()

        if "@" in domain:
            domain = domain.rsplit(
                "@",
                1,
            )[-1]

        if ":" in domain:
            domain = domain.split(
                ":",
                1,
            )[0]

        return domain.removeprefix("www.")

    except Exception:
        return ""


def _is_blocked(
    domain: str,
) -> bool:
    if not domain:
        return False

    domain = domain.lower().strip()

    for blocked in DOMINIOS_BLOQUEADOS:
        if (
            domain == blocked
            or domain.endswith(
                "." + blocked
            )
        ):
            return True

    return False


def _detect_block(
    html: str,
    status: int,
) -> str | None:
    if status == 429:
        return "rate_limit"

    if status == 403:
        return "forbidden"

    if status == 401:
        return "unauthorized"

    if status >= 500:
        return "server_error"

    lowered = (
        html[:15000].lower()
        if html
        else ""
    )

    if (
        "cf-browser-verification"
        in lowered
    ):
        return "cloudflare"

    if (
        "challenge-platform"
        in lowered
    ):
        return "cloudflare"

    if (
        "cloudflare" in lowered
        and "challenge" in lowered
    ):
        return "cloudflare"

    if "captcha" in lowered:
        return "captcha"

    if "recaptcha" in lowered:
        return "captcha"

    if "access denied" in lowered:
        return "access_denied"

    if "temporarily blocked" in lowered:
        return "temporarily_blocked"

    return None


def _deduplicate(
    results: list[SearchResult],
    max_results: int,
) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()

    for result in results:
        url = _clean_url(result.url)

        if not url:
            continue

        normalized = url.lower().rstrip("/")

        if normalized in seen:
            continue

        seen.add(normalized)

        title = _clean_text(
            result.title,
            MAX_TITLE_LENGTH,
        )

        snippet = _clean_text(
            result.snippet,
            MAX_SNIPPET_LENGTH,
        )

        domain = _domain_of(url)

        if (
            not title
            or not domain
            or _is_blocked(domain)
        ):
            continue

        output.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                source=result.source or "ddg",
            )
        )

        if len(output) >= max_results:
            break

    return output


async def _fetch(
    session: aiohttp.ClientSession,
    endpoint: str,
    query: str,
    region: str,
    safesearch: str,
    method: str = "POST",
) -> tuple[str, int]:
    data = {
        "q": query,
        "kl": region,
    }

    if safesearch in {
        "on",
        "moderate",
        "off",
    }:
        data["kp"] = {
            "on": "1",
            "moderate": "-1",
            "off": "-2",
        }[safesearch]

    if method.upper() == "GET":
        async with session.get(
            endpoint,
            params=data,
            allow_redirects=True,
        ) as response:
            text = await response.text(
                errors="replace"
            )
            return text, response.status

    async with session.post(
        endpoint,
        data=data,
        allow_redirects=True,
    ) as response:
        text = await response.text(
            errors="replace"
        )
        return text, response.status


def _parse_html(
    html: str,
    max_results: int,
) -> list[SearchResult]:
    if not html:
        return []

    try:
        soup = BeautifulSoup(
            html,
            "lxml",
        )
    except Exception:
        soup = BeautifulSoup(
            html,
            "html.parser",
        )

    blocks = soup.select(
        "div.result, "
        "div.web-result, "
        "article.result"
    )

    results: list[SearchResult] = []

    for block in blocks:
        title_el = block.select_one(
            "a.result__a"
        )

        if title_el is None:
            title_el = block.select_one(
                "a.result-link"
            )

        if title_el is None:
            continue

        title = _clean_text(
            title_el.get_text(
                " ",
                strip=True,
            ),
            MAX_TITLE_LENGTH,
        )

        url = _clean_url(
            title_el.get(
                "href",
                "",
            )
        )

        snippet_el = block.select_one(
            ".result__snippet"
        )

        if snippet_el is None:
            snippet_el = block.select_one(
                ".snippet"
            )

        snippet = ""

        if snippet_el is not None:
            snippet = _clean_text(
                snippet_el.get_text(
                    " ",
                    strip=True,
                ),
                MAX_SNIPPET_LENGTH,
            )

        domain = _domain_of(url)

        if (
            not title
            or not url
            or not domain
            or _is_blocked(domain)
        ):
            continue

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                source="ddg",
            )
        )

        if len(results) >= max_results:
            break

    return results


def _parse_lite(
    html: str,
    max_results: int,
) -> list[SearchResult]:
    if not html:
        return []

    try:
        soup = BeautifulSoup(
            html,
            "lxml",
        )
    except Exception:
        soup = BeautifulSoup(
            html,
            "html.parser",
        )

    results: list[SearchResult] = []

    links = soup.select(
        "a.result-link"
    )

    if not links:
        links = soup.select(
            "a.result-link, "
            "a.result__a"
        )

    for link in links:
        title = _clean_text(
            link.get_text(
                " ",
                strip=True,
            ),
            MAX_TITLE_LENGTH,
        )

        url = _clean_url(
            link.get(
                "href",
                "",
            )
        )

        domain = _domain_of(url)

        if (
            not title
            or not url
            or not domain
            or _is_blocked(domain)
        ):
            continue

        snippet = ""

        row = link.find_parent("tr")

        if row is not None:
            next_row = row.find_next_sibling(
                "tr"
            )

            if next_row is not None:
                snippet = _clean_text(
                    next_row.get_text(
                        " ",
                        strip=True,
                    ),
                    MAX_SNIPPET_LENGTH,
                )

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                source="ddg",
            )
        )

        if len(results) >= max_results:
            break

    return results


def _wiki_snippet(
    value: str,
) -> str:
    if not value:
        return ""

    try:
        soup = BeautifulSoup(
            value,
            "html.parser",
        )

        value = soup.get_text(
            " ",
            strip=True,
        )

    except Exception:
        value = re.sub(
            r"<[^>]+>",
            "",
            value,
        )

    return _clean_text(
        value,
        MAX_SNIPPET_LENGTH,
    )


async def _search_wikipedia(
    session: aiohttp.ClientSession,
    query: str,
    max_results: int,
    timeout: float,
) -> list[SearchResult]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
        "srprop": "snippet",
    }

    request_timeout = aiohttp.ClientTimeout(
        total=timeout,
        connect=min(
            timeout,
            10.0,
        ),
        sock_read=timeout,
    )

    try:
        async with session.get(
            WIKI_API,
            params=params,
            timeout=request_timeout,
        ) as response:
            if response.status != 200:
                return []

            data = await response.json(
                content_type=None
            )

    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ValueError,
    ):
        return []

    if not isinstance(
        data,
        dict,
    ):
        return []

    query_data = data.get(
        "query",
        {},
    )

    if not isinstance(
        query_data,
        dict,
    ):
        return []

    items = query_data.get(
        "search",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        return []

    results: list[SearchResult] = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        title = _clean_text(
            str(
                item.get(
                    "title",
                    "",
                )
            ),
            MAX_TITLE_LENGTH,
        )

        snippet = _wiki_snippet(
            str(
                item.get(
                    "snippet",
                    "",
                )
            )
        )

        if not title:
            continue

        page_url = (
            "https://es.wikipedia.org/wiki/"
            + title.replace(" ", "_")
        )

        page_url = _clean_url(page_url)

        domain = _domain_of(page_url)

        if not page_url or not domain:
            continue

        results.append(
            SearchResult(
                title=title,
                url=page_url,
                snippet=snippet,
                domain=domain,
                source="wikipedia",
            )
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
    if not query:
        return []

    query = _clean_text(
        query,
        MAX_QUERY_LENGTH,
    )

    if not query:
        return []

    try:
        max_results = int(max_results)
    except Exception:
        max_results = 5

    max_results = max(
        1,
        min(max_results, 10),
    )

    if safesearch not in {
        "on",
        "moderate",
        "off",
    }:
        safesearch = "moderate"

    if not region:
        region = "wt-wt"

    cache_key = (
        f"{query.lower()}|"
        f"{max_results}|"
        f"{region}|"
        f"{safesearch}"
    )

    if use_cache:
        cached = await _cache.get(
            cache_key
        )

        if cached is not None:
            return cached

    owns_session = session is None

    if owns_session:
        session = aiohttp.ClientSession(
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(
                total=timeout,
                connect=min(
                    timeout,
                    10.0,
                ),
                sock_read=timeout,
            ),
        )

    last_error: Optional[Exception] = None
    results: list[SearchResult] = []

    try:
        for attempt in range(
            retries + 1
        ):
            try:
                html, status = await _fetch(
                    session,
                    DDG_HTML,
                    query,
                    region,
                    safesearch,
                    method="POST",
                )

                block = _detect_block(
                    html,
                    status,
                )

                if not block:
                    parsed = _parse_html(
                        html,
                        max_results,
                    )

                    if parsed:
                        results = parsed
                        break

                html, status = await _fetch(
                    session,
                    DDG_LITE,
                    query,
                    region,
                    safesearch,
                    method="POST",
                )

                block = _detect_block(
                    html,
                    status,
                )

                if not block:
                    parsed = _parse_lite(
                        html,
                        max_results,
                    )

                    if parsed:
                        results = parsed
                        break

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as exc:
                last_error = exc

                if attempt < retries:
                    await asyncio.sleep(
                        0.5 * (2 ** attempt)
                    )

        if not results:
            try:
                wiki_results = (
                    await _search_wikipedia(
                        session,
                        query,
                        max_results,
                        timeout,
                    )
                )

                if wiki_results:
                    results = wiki_results

            except Exception:
                pass

        results = _deduplicate(
            results,
            max_results,
        )

    finally:
        if (
            owns_session
            and session is not None
        ):
            await session.close()

    if (
        not results
        and last_error is not None
    ):
        raise SearchError(
            "Búsqueda fallida tras "
            f"{retries + 1} intentos: "
            f"{last_error}"
        )

    if use_cache and results:
        await _cache.set(
            cache_key,
            results,
        )

    return results


def format_results(
    results: list[SearchResult],
) -> str:
    if not results:
        return "Sin resultados."

    lines: list[str] = []

    for index, result in enumerate(
        results,
        1,
    ):
        title = _clean_text(
            result.title,
            MAX_TITLE_LENGTH,
        )

        url = result.url

        snippet = _clean_text(
            result.snippet,
            MAX_SNIPPET_LENGTH,
        )

        lines.append(
            f"{index}. {title}"
        )

        lines.append(
            f"   URL: {url}"
        )

        if snippet:
            lines.append(
                f"   {snippet}"
            )

        if (
            result.source
            and result.source != "ddg"
        ):
            lines.append(
                f"   Fuente: {result.source}"
            )

        lines.append("")

    return "\n".join(lines).strip()
