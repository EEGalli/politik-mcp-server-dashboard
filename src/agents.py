"""
pydantic-ai agenter for politik-mcp-server.

Anvander lokala Ollama-modeller for analys och jamforelse.
Inga API-nycklar kravs.
"""

import json
import os
import re
from collections import deque
from functools import lru_cache
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import fitz  # PyMuPDF
import httpx
from pydantic_ai import Agent

from src.models import ComparisonResult, ExtractedPromises

def _get_ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _get_primary_agent_model() -> str:
    default_model = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    return os.getenv("OLLAMA_AGENT_MODEL", default_model)


def _get_fallback_agent_model(primary_model: str) -> str:
    return os.getenv("OLLAMA_FALLBACK_MODEL", primary_model)


def _ensure_ollama_base_url(host: str) -> None:
    # pydantic-ai anvander OLLAMA_BASE_URL for OpenAI-kompatibla anrop.
    os.environ["OLLAMA_BASE_URL"] = f"{host}/v1"


# -- Textextraktion ------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Enkel HTML-till-text-parser. Skippar script/style-taggar."""

    def __init__(self):
        super().__init__()
        self._result: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._result.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._result.append(data)

    def get_text(self) -> str:
        return "".join(self._result).strip()


class _HTMLLinkExtractor(HTMLParser):
    """Enkel länkextraktor för att crawla interna undersidor."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(str(value))
                return


def _html_to_text(html: str) -> str:
    """Konvertera HTML till ren text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extrahera text fran en PDF med PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


_SKIP_LINK_PREFIXES = (
    "javascript:",
    "mailto:",
    "tel:",
    "#",
)

_SKIP_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".txt",
    ".zip",
    ".rar",
    ".7z",
    ".mp3",
    ".wav",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

_SKIP_PATH_SUBSTRINGS = (
    "/wp-json/",
    "/wp-admin/",
    "/feed",
    "/sitemap",
    "/tag/",
    "/author/",
    "/search",
    "/kommentar",
    "/comment",
    "/konto",
    "/account",
    "/login",
)

_MAX_ENQUEUED_URLS = 800
_MAX_LINKS_PER_PAGE = 180


def _canonicalize_url(raw_url: str, keep_query: bool = True) -> str:
    cleaned = str(raw_url or "").strip()
    if not cleaned:
        return ""

    cleaned, _ = urldefrag(cleaned)
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query = parsed.query if keep_query else ""
    return urlunparse((scheme, netloc, path, "", query, ""))


def _scope_prefix_from_seed(path: str) -> str:
    parts = [part for part in str(path or "").split("/") if part]
    if not parts:
        return "/"
    return f"/{parts[0]}/"


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    parser = _HTMLLinkExtractor()
    parser.feed(html)
    absolute_links: list[str] = []
    for href in parser.links:
        link = str(href or "").strip()
        if not link:
            continue
        lowered = link.lower()
        if lowered.startswith(_SKIP_LINK_PREFIXES):
            continue
        absolute_links.append(urljoin(base_url, link))
    return absolute_links


def _is_allowed_crawl_target(url: str, seed_domain: str, scope_prefix: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() != seed_domain.lower():
        return False
    if parsed.query:
        return False

    path = parsed.path or "/"
    if scope_prefix != "/":
        scope_no_trailing = scope_prefix[:-1] if scope_prefix.endswith("/") else scope_prefix
        if not (path == scope_no_trailing or path.startswith(scope_prefix)):
            return False

    lowered_path = path.lower()
    for marker in _SKIP_PATH_SUBSTRINGS:
        if marker in lowered_path:
            return False
    for ext in _SKIP_FILE_EXTENSIONS:
        if lowered_path.endswith(ext):
            return False

    return True


async def _fetch_page_payload(client: httpx.AsyncClient, url: str) -> dict:
    response = await client.get(url)
    response.raise_for_status()

    final_url = str(response.url)
    content_type = str(response.headers.get("content-type", "")).lower()
    is_pdf = "pdf" in content_type or final_url.lower().endswith(".pdf")

    if is_pdf:
        text = _pdf_to_text(response.content)
        return {
            "url": final_url,
            "content_type": content_type,
            "text": text,
            "links": [],
        }

    html = response.text
    return {
        "url": final_url,
        "content_type": content_type,
        "text": _html_to_text(html),
        "links": _extract_links_from_html(html, final_url),
    }


async def fetch_page_text(url: str) -> str:
    """Hamta text fran en URL (HTML eller PDF)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        payload = await _fetch_page_payload(client, url)
    return str(payload.get("text", ""))


async def crawl_site_texts(
    start_url: str,
    max_pages: int = 25,
    max_depth: int = 2,
) -> dict:
    """
    Crawla en site från start-URL och hämta text från interna undersidor.

    Returnerar:
      {
        "pages": [{"url": "...", "text": "...", "depth": 0}, ...],
        "errors": [{"url": "...", "error": "..."}]
      }
    """
    max_pages = max(1, min(int(max_pages or 1), 80))
    max_depth = max(0, min(int(max_depth or 0), 5))

    seed_url = _canonicalize_url(start_url, keep_query=False)
    if not seed_url:
        return {"pages": [], "errors": [{"url": start_url, "error": "Ogiltig start-URL"}]}

    seed_parsed = urlparse(seed_url)
    seed_domain = seed_parsed.netloc
    scope_prefix = _scope_prefix_from_seed(seed_parsed.path)
    if scope_prefix == "/":
        # Säkerhetsläge: om seed ligger på domänroten, följ inte vidare länkar.
        max_depth = 0

    queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
    enqueued: set[str] = {seed_url}
    visited: set[str] = set()
    pages: list[dict] = []
    errors: list[dict] = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        while queue and len(pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                payload = await _fetch_page_payload(client, current_url)
            except Exception as exc:
                errors.append({"url": current_url, "error": str(exc)})
                continue

            final_url = _canonicalize_url(str(payload.get("url", current_url)), keep_query=False)
            page_text = str(payload.get("text", "") or "").strip()
            if page_text:
                pages.append(
                    {
                        "url": final_url or current_url,
                        "text": page_text,
                        "depth": depth,
                    }
                )

            if depth >= max_depth:
                continue

            links = payload.get("links", [])
            if not isinstance(links, list):
                continue

            links_seen_on_page = 0
            for raw_link in links:
                if links_seen_on_page >= _MAX_LINKS_PER_PAGE:
                    break
                if len(enqueued) >= _MAX_ENQUEUED_URLS:
                    break

                candidate = _canonicalize_url(str(raw_link), keep_query=False)
                if not candidate:
                    continue
                if candidate in visited or candidate in enqueued:
                    continue
                if not _is_allowed_crawl_target(candidate, seed_domain=seed_domain, scope_prefix=scope_prefix):
                    continue
                queue.append((candidate, depth + 1))
                enqueued.add(candidate)
                links_seen_on_page += 1

    return {"pages": pages, "errors": errors}


# -- Gemensamma prompts -------------------------------------------


_COMPARISON_INSTRUCTIONS = """Du ar en politisk analyst som jamfor politikers sociala medier-innehall
med deras valloeften. Du ar saklig, balanserad och partipolitiskt neutral.

Nar du jamfor:
- 'supports': Innehallet stodjer eller bekraftar valloftet
- 'contradicts': Innehallet motsager eller motarbetar valloftet
- 'unrelated': Ingen tydlig koppling mellan innehall och lofte

Match-poang:
- 1.0 = Direkt och tydligt stod/motsagelse
- 0.7-0.9 = Stark koppling
- 0.4-0.6 = Viss koppling
- 0.1-0.3 = Svag koppling
- 0.0 = Ingen koppling alls

Svara alltid pa svenska. Citera relevanta delar av transkriptionen."""

_SCRAPE_INSTRUCTIONS = """Du ar en expert pa att identifiera konkreta politiska valloften i text.

Regler:
- Extrahera BARA konkreta loften (specifika ataganden, inte vaga visioner)
- Omformulera till enkel, neutral policiesvenska utan partislogans och vag retorik.
- Hitta INTE pa nya sakuppgifter. Behall endast det som faktiskt star i kallan.
- Om texten beskriver malbild men inte metod (t.ex. "fler ska kunna..." utan hur), ta inte med den.
- Om metod finns i bisats (t.ex. "genom ...", "via ...", "med ..."), lyft fram metoden som huvudatgard.
- Om texten inte innehaller nagra konkreta loften, returnera en tom lista
- Behall siffror, procentsatser, datum, malnivaer och andra konkreta detaljer om de finns.
- Skippa slogans, visioner och allmanna mal utan tydlig atgard.
- Svara ALLTID pa svenska.
- Kategorisera varje lofte med en av dessa kategorier:
  Sjukvard, Skola & Utbildning, Rattsvasende, Ekonomi, Skatt,
  Arbetsmarknad, Socialpolitik, Forsvar, Migration, Klimat & Miljo,
  Bostader, Infrastruktur, Kultur, Jamstalldhet, EU & Utrikes,
  Digitalisering, Landsbygd, Ovrigt

Exempel pa konkreta loften:
- "Vi ska oka antalet poliser till 50 000"  (konkret, matbart)
- "Vi vill forbjuda vinster i valfarden"    (konkret atagande)

Exempel pa saker som INTE ar loften:
- "Vi tror pa ett starkt Sverige"           (vag vision)
- "Trygghet ar viktigt"                     (allmant uttalande)

Svara alltid pa svenska."""

_CATEGORY_MAP_TO_SWEDISH = {
    "healthcare": "Sjukvard",
    "health care": "Sjukvard",
    "education": "Skola & Utbildning",
    "school": "Skola & Utbildning",
    "law and order": "Rattsvasende",
    "justice": "Rattsvasende",
    "economy": "Ekonomi",
    "tax": "Skatt",
    "taxes": "Skatt",
    "labor market": "Arbetsmarknad",
    "employment": "Arbetsmarknad",
    "social policy": "Socialpolitik",
    "defense": "Forsvar",
    "defence": "Forsvar",
    "migration": "Migration",
    "climate": "Klimat & Miljo",
    "environment": "Klimat & Miljo",
    "housing": "Bostader",
    "infrastructure": "Infrastruktur",
    "culture": "Kultur",
    "gender equality": "Jamstalldhet",
    "eu and foreign policy": "EU & Utrikes",
    "foreign policy": "EU & Utrikes",
    "digitalization": "Digitalisering",
    "digitalisation": "Digitalisering",
    "rural policy": "Landsbygd",
    "other": "Ovrigt",
}


# -- Agent-byggare ------------------------------------------------


def _build_comparison_agent(model_name: str) -> Agent:
    return Agent(
        f"ollama:{model_name}",
        output_type=ComparisonResult,
        instructions=_COMPARISON_INSTRUCTIONS,
    )


def _build_scrape_agent(model_name: str) -> Agent:
    return Agent(
        f"ollama:{model_name}",
        output_type=ExtractedPromises,
        instructions=_SCRAPE_INSTRUCTIONS,
    )


@lru_cache(maxsize=16)
def _comparison_agent_for_model(model_name: str) -> Agent:
    return _build_comparison_agent(model_name)


@lru_cache(maxsize=16)
def _scrape_agent_for_model(model_name: str) -> Agent:
    return _build_scrape_agent(model_name)


# -- Jamforelse-agent --------------------------------------------


async def compare_analysis_with_promise(
    transcription: str,
    summary: str,
    promise_text: str,
    promise_category: str,
    politician_name: str,
    party: str,
) -> ComparisonResult:
    """Kor AI-agenten for att jamfora en videoanalys med ett vallofte."""

    host = _get_ollama_host()
    _ensure_ollama_base_url(host)
    primary_model = _get_primary_agent_model()
    fallback_model = _get_fallback_agent_model(primary_model)

    comparison_agent = _comparison_agent_for_model(primary_model)
    comparison_fallback_agent = None
    if fallback_model and fallback_model != primary_model:
        comparison_fallback_agent = _comparison_agent_for_model(fallback_model)

    prompt = f"""Jamfor foljande videoinnehall med valloftet.

## Politiker
{politician_name} ({party})

## Videoinnehall (fran sociala medier)
Sammanfattning: {summary}

Transkription:
{transcription}

## Vallofte
Kategori: {promise_category}
Lofte: {promise_text}

Bedom hur val videoinnehallet stammer overens med valloftet."""

    try:
        result = await comparison_agent.run(prompt)
        return result.output
    except Exception as exc:
        if comparison_fallback_agent is None:
            raise
        try:
            result = await comparison_fallback_agent.run(prompt)
            return result.output
        except Exception:
            raise exc


# -- Scrape-agent -------------------------------------------------


def _extract_first_json_object(raw: str) -> dict | None:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return None


def _normalize_category_label(category: str) -> str:
    cleaned = str(category or "").strip()
    if not cleaned:
        return "Ovrigt"

    key = re.sub(r"\s+", " ", cleaned.lower())
    return _CATEGORY_MAP_TO_SWEDISH.get(key, cleaned)


_MIN_PROMISE_CHARS = 20
_MIN_PROMISE_WORDS = 4
_MAX_PROMISE_CHARS = 320


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _sentence_case(text: str) -> str:
    cleaned = str(text or "").strip(" .;:,")
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _clean_extracted_promise_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^[\-\*\u2022\(\)\[\]\d\.\s]+", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'` ")
    return _sentence_case(cleaned)


def _is_concrete_promise_text(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    if len(normalized) < _MIN_PROMISE_CHARS:
        return False
    if len(normalized) > _MAX_PROMISE_CHARS:
        return False
    if len(normalized.split()) < _MIN_PROMISE_WORDS:
        return False
    if normalized.endswith("?"):
        return False
    if re.fullmatch(r"[\W\d_]+", normalized):
        return False
    return True


def _dedupe_promises(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        key = _normalize_for_match(str(row.get("text", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _compress_text_for_extraction(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    # Sprakneutral komprimering: inledning + slut ger bred kontext
    # utan ordlistor som riskerar feltolkning.
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return f"{text[:head]}\n...\n{text[-tail:]}"


async def _ollama_chat(prompt: str, model_name: str) -> str:
    host = _get_ollama_host()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{host}/api/chat", json=payload)
        response.raise_for_status()

    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()


def _normalize_extracted_promises(payload: dict) -> ExtractedPromises:
    raw_promises = payload.get("promises")
    if not isinstance(raw_promises, list):
        raw_promises = []

    normalized: list[dict] = []
    for item in raw_promises:
        if not isinstance(item, dict):
            continue
        text = _clean_extracted_promise_text(str(item.get("text", "")).strip())
        category = _normalize_category_label(str(item.get("category", "")).strip() or "Ovrigt")
        if not text:
            continue
        if not _is_concrete_promise_text(text):
            continue
        normalized.append({"text": text, "category": category})

    normalized = _dedupe_promises(normalized)
    return ExtractedPromises.model_validate({"promises": normalized})


async def _extract_promises_via_plain_ollama(
    text: str,
    party_name: str,
    source_url: str,
    model_name: str,
) -> ExtractedPromises:
    prompt = f"""Extrahera konkreta valloften ur texten nedan.

Parti: {party_name}
Kalla: {source_url}

Svara ENDAST med giltig JSON i exakt detta format:
{{"promises":[{{"text":"...","category":"..."}}]}}

Regler:
- Ta bara konkreta ataganden, inte vaga visioner.
- Skriv loftet i neutral och enkel policy-svenska.
- Om metod saknas (bara malbild) ska loftet inte tas med.
- Om metod uttrycks i bisats (genom/via/med), skriv metoden som huvudatgard.
- Hall dig till uppgifter som finns i kalltexten.
- Skriv allt pa svenska.
- Varje lofte ska vara en kort mening som beskriver konkret atgard.
- Behall siffror, procentsatser, belopp, datum och malnivaer om de finns.
- Om inga konkreta loften finns: {{"promises":[]}}

Text:
---
{text}
---"""

    raw = await _ollama_chat(prompt, model_name)
    payload = _extract_first_json_object(raw)
    if payload is None:
        raise ValueError("Kunde inte hitta giltig JSON i modellsvaret")

    return _normalize_extracted_promises(payload)


async def extract_promises_from_text(text: str, party_name: str, source_url: str) -> ExtractedPromises:
    """Kor AI-agenten for att extrahera valloften fran text."""

    host = _get_ollama_host()
    _ensure_ollama_base_url(host)
    primary_model = _get_primary_agent_model()
    fallback_model = _get_fallback_agent_model(primary_model)

    scrape_agent = _scrape_agent_for_model(primary_model)
    scrape_fallback_agent = None
    if fallback_model and fallback_model != primary_model:
        scrape_fallback_agent = _scrape_agent_for_model(fallback_model)

    max_chars = 12000
    text = _compress_text_for_extraction(text, max_chars=max_chars)

    prompt = f"""Extrahera alla konkreta valloften fran foljande text fran {party_name}s hemsida ({source_url}).

---
{text}
---

Returnera bara konkreta, specifika loften. Skippa vaga visioner och allmanna uttalanden.
Skriv varje lofte i neutral och enkel policy-svenska utan slogans.
Om metod saknas (bara malbild) ska loftet inte tas med.
Om metod uttrycks i bisats (genom/via/med), skriv metoden som huvudatgard.
Varje lofte ska vara en kort mening som beskriver konkret atgard.
Behall siffror, procentsatser, belopp, datum och malnivaer nar de finns."""

    pydantic_error: Exception | None = None
    try:
        result = await scrape_agent.run(prompt)
        return result.output
    except Exception as exc:
        pydantic_error = exc

    # Prova fallback-agent oavsett exakt feltyp (inte bara tool-stod).
    if scrape_fallback_agent is not None:
        try:
            result = await scrape_fallback_agent.run(prompt)
            return result.output
        except Exception:
            pass

    # Sista fallback: plain JSON-mode via Ollama API.
    last_error: Exception = pydantic_error if pydantic_error is not None else RuntimeError("Unknown extraction error")
    candidate_models: list[str] = []
    for model_name in (fallback_model, primary_model):
        if model_name and model_name not in candidate_models:
            candidate_models.append(model_name)

    for model_name in candidate_models:
        try:
            plain_result = await _extract_promises_via_plain_ollama(text, party_name, source_url, model_name)
            return plain_result
        except Exception as plain_exc:
            last_error = plain_exc

    raise last_error
