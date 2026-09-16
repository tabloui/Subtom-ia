from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp

from config import config
from connector import connector, detect_provider


class TaskType(str, Enum):
    IMAGE_READ = "image_read"
    IMAGE_GEN  = "image_gen"
    CODE       = "code"
    SEARCH     = "search"
    FETCH      = "fetch"
    FILE       = "file"
    PDF        = "pdf"
    DATA       = "data"
    CHAT       = "chat"


class PromptLang(str, Enum):
    ES = "es"
    EN = "en"
    UNKNOWN = "?"


_RE_IMAGE_PATH = re.compile(r"->\s*ruta local:\s*(\S+)", re.IGNORECASE)
_RE_PDF_PATH = re.compile(r"\.pdf\b", re.IGNORECASE)
_RE_DATA_PATH = re.compile(r"\.(csv|json|yaml|yml|tsv)\b", re.IGNORECASE)
_RE_CODE_PATH = re.compile(
    r"\.(py|js|ts|tsx|jsx|java|rs|go|rb|php|c|cpp|h|hpp|cs|kt|swift)\b",
    re.IGNORECASE,
)

_RE_IMAGE_GEN = re.compile(
    r"\b(genera|generar|generame|genérame|dibuja|dibujame|"
    r"dibújame|pinta|pintame|píntame|ilustra|ilustrame|"
    r"ilústrame|crea|creame|créame|hazme|haz|imagina|"
    r"imaginame|diseña|diseñame|diséñame|renderiza|"
    r"renderizame|muestrame|muéstrame|quiero ver|dame|"
    r"necesito|me gustaría ver)\b"
    r".{0,80}\b(imagen|imágenes|imagenes|foto|fotos|dibujo|dibujos|"
    r"ilustraci[oó]n|ilustraciones|logo|logotipo|"
    r"wallpaper|fondo de pantalla|p[oó]ster|poster|"
    r"avatar|banner|portada|miniatura|thumbnail|"
    r"escena|paisaje|personaje|retrato|caricatura|"
    r"emoji|sticker|arte|dise[ñn]o gr[aá]fico)\b",
    re.IGNORECASE | re.DOTALL,
)

_RE_IMAGE_GEN_SHORT = re.compile(
    r"^\s*(genera|dibuja|pinta|crea|hazme|haz|ilustra|"
    r"dis[eé]?[ñn]a|renderiza|muestrame|mu[eé]strame)\b",
    re.IGNORECASE,
)

_RE_SEARCH_STRONG = re.compile(
    r"\b(busca en (google|internet|la web|online|el navegador)|"
    r"b[uú]scame en internet|b[uú]scalo en google|"
    r"googlea|googlealo|googleame|"
    r"investiga|invest[íi]game|averigua|aver[íi]guame|"
    r"consulta en internet|saca de internet|"
    r"[uú]ltima hora|[uú]ltimas noticias|noticias de hoy|"
    r"noticias recientes|qu[eé] ha pasado (hoy|ayer)|"
    r"qu[eé] pas[oó] (hoy|ayer|esta semana)|"
    r"actualidad|qu[eé] est[aá] pasando|"
    r"precio (actual|de hoy|del d[ií]a|de ahora)|"
    r"cu[aá]nto (cuesta|vale|est[aá])|cotizaci[oó]n|"
    r"a cu[aá]nto est[aá]|valor (actual|del d[ií]a)|"
    r"clima en|tiempo en|temperatura en|"
    r"va a llover|qu[eé] tiempo (hace|har[aá])|"
    r"qui[eé]n gan[oó]|qui[eé]n va ganando|resultado (de|del)|"
    r"c[oó]mo va (el|la)|c[oó]mo qued[oó]|"
    r"cu[aá]ndo (fue|es|se|sale|estrena)|"
    r"en qu[eé] fecha|qu[eé] d[ií]a (es|fue|ser[aá])|"
    r"qui[eé]n es|qui[eé]n fue|qui[eé]nes son|"
    r"qu[eé] es (el|la|un|una)|qu[eé] significa (el|la|un|una)|"
    r"d[oó]nde (est[aá]|queda|se encuentra)|"
    r"cu[aá]l es (el|la|un|una))\b",
    re.IGNORECASE,
)

_RE_SEARCH_WEAK = re.compile(
    r"\b(busca|buscar|b[uú]scame|b[uú]scalo|b[uú]scala|"
    r"investiga|averigua|consulta|encuentra|"
    r"me gustar[íi]a saber|quiero saber|quisiera saber|"
    r"sabes (si|qui[eé]n|cu[aá]ndo|d[oó]nde|qu[eé]))\b",
    re.IGNORECASE,
)

_RE_CODE = re.compile(
    r"\b(c[oó]digo|code|c[oó]digo fuente|programa|programar|"
    r"script|funci[oó]n|function|m[eé]todo|method|clase|class|"
    r"variable|constante|array|lista|diccionario|objeto|"
    r"bucle|loop|condicional|if|else|while|for|"
    r"python|javascript|typescript|html|css|sql|java|rust|go|"
    r"c\+\+|c#|php|ruby|kotlin|swift|dart|scala|bash|shell|"
    r"bug|error|traceback|exception|excepci[oó]n|"
    r"no funciona|no compila|no corre|falla|se rompe|"
    r"debug|debuggear|depurar|"
    r"compil[ao]|compilar|refactor|refactoriza|refactorizar|"
    r"implementa|implementar|optimiza|optimizar|"
    r"arregla|arreglar|corrige|corregir|soluciona|"
    r"mejora el c[oó]digo|limpia el c[oó]digo|"
    r"escribe (un|una|el|la)|crea (un|una) (funci[oó]n|clase|script)|"
    r"api|endpoint|servidor|server|servicio|microservicio|"
    r"deploy|despliega|deployar|"
    r"commit|push|pull|merge|branch|rama|"
    r"repositorio|repo|github|gitlab|bitbucket|"
    r"vercel|railway|heroku|aws|azure|docker|kubernetes|"
    r"base de datos|database|query|consulta sql|migraci[oó]n|"
    r"algoritmo|algoritmia|complejidad|big o|"
    r"funci[oó]n recursiva|recursividad|"
    r"estructura de datos|"
    r"regex|regexp|expresi[oó]n regular)\b",
    re.IGNORECASE,
)

_RE_FILE_HINT = re.compile(
    r"\b(archivo|archivos|fichero|ficheros|carpeta|carpetas|"
    r"directorio|directorios|"
    r"en (mis|los|el|la) (archivos?|carpetas?|workspace|disco)|"
    r"en mi espacio de trabajo|"
    r"\.(py|js|ts|tsx|jsx|json|txt|md|yaml|yml|csv|pdf|zip|log|"
    r"html|css|sql|sh|bat|ini|conf|toml|xml)\b|"
    r"mi workspace|el workspace)\b",
    re.IGNORECASE,
)

_RE_FILE_ACTION = re.compile(
    r"\b(leer|lee|leelo|escribir|escribe|guarda|guardar|"
    r"guardalo|guardala|salva|salvar|"
    r"borra|borrar|b[oó]rralo|elimina|eliminar|elim[íi]nalo|"
    r"crea (una|la) carpeta|crea (un|el) archivo|nueva carpeta|"
    r"nuevo archivo|"
    r"zip|zíp|comprime|comprimir|empaqueta|empaquetar|"
    r"descomprime|descomprimir|deszipa|extrae el zip|"
    r"copiar|copia|c[oó]pialo|mueve|mover|mu[eé]velo|"
    r"renombra|renombrar|ren[oó]mbralo|"
    r"grep|busca en (mis|los) archivos|"
    r"busca dentro de los archivos|"
    r"reemplaza en|reemplazar en|sustituye en|"
    r"lista (los|las|el|la) (archivos?|carpetas?|contenido)|"
    r"muestra (los|las) (archivos?|carpetas?|contenido)|"
    r"qu[eé] hay en (la|el|el directorio|la carpeta))\b",
    re.IGNORECASE,
)

_RE_URL = re.compile(r"https?://[^\s,;)\]\}<>\"']+")

_RE_ES = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|al|"
    r"que|qu[eé]|c[oó]mo|cu[aá]l|d[oó]nde|cu[aá]ndo|"
    r"por|para|con|sin|sobre|entre|"
    r"es|son|est[aá]|est[aá]n|hay|tiene|tienen|"
    r"hola|gracias|quiero|puedes|dime|dame|"
    r"pero|porque|aunque|mientras|"
    r"yo|t[uú]|[eé]l|ella|nosotros|ustedes)\b",
    re.IGNORECASE,
)

_RE_EN = re.compile(
    r"\b(the|a|an|of|to|in|on|at|for|with|without|about|"
    r"is|are|was|were|be|been|have|has|had|do|does|did|"
    r"hello|thanks|thank|please|could|would|should|"
    r"what|when|where|why|how|which|who|"
    r"i|you|he|she|we|they|it|"
    r"but|because|although|while)\b",
    re.IGNORECASE,
)


def detect_lang(prompt: str) -> PromptLang:
    if not prompt:
        return PromptLang.UNKNOWN
    es = len(_RE_ES.findall(prompt))
    en = len(_RE_EN.findall(prompt))
    if es > en:
        return PromptLang.ES
    if en > es:
        return PromptLang.EN
    return PromptLang.UNKNOWN


def classify(prompt: str) -> TaskType:
    if not prompt:
        return TaskType.CHAT
    if _RE_IMAGE_PATH.search(prompt):
        return TaskType.IMAGE_READ
    if _RE_IMAGE_GEN.search(prompt):
        return TaskType.IMAGE_GEN
    if _RE_IMAGE_GEN_SHORT.search(prompt):
        return TaskType.IMAGE_GEN
    if _RE_URL.search(prompt):
        return TaskType.FETCH
    if _RE_PDF_PATH.search(prompt):
        return TaskType.PDF
    if _RE_DATA_PATH.search(prompt):
        return TaskType.DATA
    if _RE_CODE_PATH.search(prompt) and not _RE_CODE.search(prompt):
        return TaskType.FILE
    if _RE_CODE.search(prompt):
        return TaskType.CODE
    if _RE_SEARCH_STRONG.search(prompt):
        return TaskType.SEARCH
    if _RE_FILE_HINT.search(prompt) or _RE_FILE_ACTION.search(prompt):
        return TaskType.FILE
    if _RE_SEARCH_WEAK.search(prompt):
        return TaskType.SEARCH
    return TaskType.CHAT


class SlidingWindow:
    def __init__(self, size: int = 50) -> None:
        self.size = size
        self._samples: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._samples.append(value)

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]

    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


@dataclass
class SensorStats:
    calls: int = 0
    errors: int = 0
    total_latency: float = 0.0
    window: SlidingWindow = field(default_factory=SlidingWindow)
    last_error: str = ""
    last_error_at: float = 0.0
    last_ok_at: float = 0.0
    consecutive_errors: int = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.calls if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        return 1 - (self.errors / self.calls) if self.calls else 1.0

    @property
    def p50(self) -> float:
        return self.window.percentile(0.50)

    @property
    def p95(self) -> float:
        return self.window.percentile(0.95)

    def observe(self, latency: float, error: bool = False) -> None:
        self.calls += 1
        self.total_latency += latency
        self.window.add(latency)
        if error:
            self.errors += 1
            self.consecutive_errors += 1
            self.last_error_at = time.time()
        else:
            self.consecutive_errors = 0
            self.last_ok_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "ok_rate": round(self.success_rate, 3),
            "avg_ms": round(self.avg_latency * 1000, 1),
            "p50_ms": round(self.p50 * 1000, 1),
            "p95_ms": round(self.p95 * 1000, 1),
            "streak": self.consecutive_errors,
        }


class Metrics:
    def __init__(self) -> None:
        self.per_model: dict[str, SensorStats] = {}
        self.per_key: dict[int, SensorStats] = {}
        self.per_task: dict[str, SensorStats] = {}
        self.per_lang: dict[str, SensorStats] = {}
        self.events: deque = deque(maxlen=200)
        self.t_start = time.time()

    def _get(self, bucket: dict, key: Any) -> SensorStats:
        s = bucket.get(key)
        if s is None:
            s = SensorStats()
            bucket[key] = s
        return s

    def record(
        self,
        *,
        model: str | None = None,
        key_index: int | None = None,
        task: TaskType | None = None,
        lang: PromptLang | None = None,
        latency: float,
        error: bool = False,
        error_msg: str = "",
    ) -> None:
        if model is not None:
            s = self._get(self.per_model, model)
            s.observe(latency, error)
            if error and error_msg:
                s.last_error = error_msg[:200]
        if key_index is not None:
            self._get(self.per_key, key_index).observe(latency, error)
        if task is not None:
            self._get(self.per_task, task.value).observe(latency, error)
        if lang is not None:
            self._get(self.per_lang, lang.value).observe(latency, error)

        self.events.append({
            "t": round(time.time(), 3),
            "model": model,
            "task": task.value if task else None,
            "lang": lang.value if lang else None,
            "latency_ms": round(latency * 1000, 1),
            "error": error,
        })

    def snapshot(self) -> dict[str, Any]:
        def dump(b: dict) -> dict:
            return {k: v.to_dict() for k, v in b.items()}

        return {
            "uptime_s": round(time.time() - self.t_start, 1),
            "provider": connector.provider,
            "models": dump(self.per_model),
            "keys": dump(self.per_key),
            "tasks": dump(self.per_task),
            "langs": dump(self.per_lang),
            "events": list(self.events)[-20:],
        }


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._open_until: dict[str, float] = {}

    def is_open(self, ident: str) -> bool:
        return time.monotonic() < self._open_until.get(ident, 0)

    def trip(self, ident: str, seconds: float | None = None) -> None:
        t = seconds if seconds is not None else self.cooldown
        self._open_until[ident] = time.monotonic() + t

    def reset(self, ident: str) -> None:
        self._open_until.pop(ident, None)

    def state(self) -> dict[str, float]:
        now = time.monotonic()
        return {
            k: round(v - now, 1)
            for k, v in self._open_until.items()
            if v > now
        }


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class ResultCache:
    def __init__(self, max_size: int = 300, ttl: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.per_prefix: dict[str, dict[str, int]] = {}

    def _stats_for(self, prefix: str) -> dict[str, int]:
        s = self.per_prefix.get(prefix)
        if s is None:
            s = {"hits": 0, "misses": 0}
            self.per_prefix[prefix] = s
        return s

    @staticmethod
    def _key(prefix: str, args: tuple) -> str:
        raw = prefix + "\x00" + "\x00".join(str(a) for a in args)
        return hashlib.blake2b(
            raw.encode("utf-8", "ignore"), digest_size=12
        ).hexdigest()

    def get(self, prefix: str, *args: Any) -> Any | None:
        k = self._key(prefix, args)
        entry = self._data.get(k)
        stats = self._stats_for(prefix)
        if entry is None:
            self.misses += 1
            stats["misses"] += 1
            return None
        if entry.expires_at < time.time():
            self._data.pop(k, None)
            self.misses += 1
            stats["misses"] += 1
            return None
        self._data.move_to_end(k)
        self.hits += 1
        stats["hits"] += 1
        return entry.value

    def set(self, prefix: str, value: Any, *args: Any) -> None:
        k = self._key(prefix, args)
        self._data[k] = CacheEntry(
            value=value,
            expires_at=time.time() + self.ttl,
        )
        self._data.move_to_end(k)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def invalidate(self, prefix: str | None = None) -> int:
        if prefix is None:
            n = len(self._data)
            self._data.clear()
            return n
        to_del = [k for k in self._data if k.startswith(prefix)]
        for k in to_del:
            self._data.pop(k, None)
        return len(to_del)

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._data),
            "per_prefix": self.per_prefix,
        }


class ModelRouter:
    def __init__(self, metrics: Metrics, breaker: CircuitBreaker) -> None:
        self.metrics = metrics
        self.breaker = breaker

    def order(self, models: list[str]) -> list[str]:
        alive: list[tuple[float, str]] = []
        dead: list[str] = []

        for m in models:
            if self.breaker.is_open(f"model:{m}"):
                dead.append(m)
                continue
            s = self.metrics.per_model.get(m)
            if s is None or s.calls < 2:
                score = 0.30
            else:
                ok_rate = s.success_rate
                lat_score = 1.0 / (1.0 + s.p50 * 1000)
                score = (1.0 - ok_rate) * 0.7 + (1.0 - lat_score) * 0.3
            alive.append((score, m))

        alive.sort(key=lambda x: x[0])
        return [m for _, m in alive] + dead

    def pick(self, candidates: list[str], top: int = 3) -> list[str]:
        return self.order(candidates)[:top]


class Motor:
    def __init__(self) -> None:
        self.metrics = Metrics()
        self.cache = ResultCache(max_size=300, ttl=300.0)
        self.breaker = CircuitBreaker(threshold=3, cooldown=60.0)
        self.router = ModelRouter(self.metrics, self.breaker)

    def classify(self, prompt: str) -> TaskType:
        t0 = time.perf_counter()
        task = classify(prompt)
        dt = (time.perf_counter() - t0) * 1000
        if dt > 1.0:
            print(f"[MOTOR] classify lento: {dt:.2f}ms")
        return task

    def lang(self, prompt: str) -> PromptLang:
        return detect_lang(prompt)

    def pick_models(
        self,
        candidates: list[str],
        task: TaskType,
        top: int = 3,
    ) -> list[str]:
        if task in (TaskType.CODE, TaskType.PDF):
            top = max(top, 4)
        elif task == TaskType.CHAT:
            top = min(top, 2)
        return self.router.pick(candidates, top=top)

    def mark_model_error(self, model: str) -> None:
        s = self.metrics.per_model.get(model)
        if s and s.consecutive_errors >= self.breaker.threshold:
            self.breaker.trip(f"model:{model}", seconds=90)
            print(
                f"[MOTOR] circuit breaker → {model} "
                f"({s.consecutive_errors} fallos seguidos)"
            )

    def mark_model_ok(self, model: str) -> None:
        self.breaker.reset(f"model:{model}")

    def cache_get_search(self, query: str) -> Any | None:
        return self.cache.get("search", query.lower().strip())

    def cache_set_search(self, query: str, value: Any) -> None:
        self.cache.set("search", value, query.lower().strip())

    def cache_get_fetch(self, url: str) -> Any | None:
        return self.cache.get("fetch", url.strip())

    def cache_set_fetch(self, url: str, value: Any) -> None:
        self.cache.set("fetch", value, url.strip())

    def record(
        self,
        *,
        model: str | None = None,
        key_index: int | None = None,
        task: TaskType | None = None,
        lang: PromptLang | None = None,
        latency: float,
        error: bool = False,
        error_msg: str = "",
    ) -> None:
        self.metrics.record(
            model=model,
            key_index=key_index,
            task=task,
            lang=lang,
            latency=latency,
            error=error,
            error_msg=error_msg,
        )
        if model is not None:
            if error:
                self.mark_model_error(model)
            else:
                self.mark_model_ok(model)

    def snapshot(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats(),
            "breaker": self.breaker.state(),
            **self.metrics.snapshot(),
        }

    # ============================================================
    # PREWARM (solo ping a servicios que SÍ tienen /models)
    # ============================================================

    async def prewarm(self) -> None:
        timeout = aiohttp.ClientTimeout(total=10)
        results: dict[str, Any] = {}

        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def ping(url: str, headers: dict | None = None) -> None:
                try:
                    async with session.get(url, headers=headers or {}) as r:
                        await r.read()
                        results[url] = r.status
                except Exception as exc:
                    results[url] = f"err:{type(exc).__name__}"

            await asyncio.gather(
                ping("https://html.duckduckgo.com/html/"),
                ping(config.flux_base_url.rstrip("/")),
                return_exceptions=True,
            )

        print(f"[MOTOR] prewarm → {results}")

    # ============================================================
    # HEALTHCHECK (deshabilitado: SambaNova no tiene /models)
    # ============================================================

    async def healthcheck_loop(self, interval: float = 300.0) -> None:
        return

    def start_healthcheck(self, interval: float = 300.0) -> None:
        return

    def stop_healthcheck(self) -> None:
        return


motor = Motor()
