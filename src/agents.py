"""
pydantic-ai agenter for politik-mcp-server.

Anvander lokala Ollama-modeller for analys och jamforelse.
Inga API-nycklar kravs.
"""

import json
import os
import re
from functools import lru_cache
from html.parser import HTMLParser

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


async def fetch_page_text(url: str) -> str:
    """Hamta text fran en URL (HTML eller PDF)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _pdf_to_text(response.content)
    return _html_to_text(response.text)


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
- Anvand texten sa nara originalet som mojligt -- hitta INTE pa egna formuleringar
- Om texten inte innehaller nagra konkreta loften, returnera en tom lista
- Behall siffror, procentsatser, datum, malnivaer och andra konkreta detaljer om de finns.
- Skippa slogans, visioner och allmanna mal utan tydlig atgard.
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


_GENERIC_PROMISE_PATTERNS = [
    "vi vill ha",
    "vi tror pa",
    "vi tror att",
    "det ar viktigt",
    "trygghet ar viktigt",
    "ett starkt sverige",
    "for ett battre",
    "vi prioriterar",
    "vi arbetar for",
]

_COMMITMENT_MARKERS = [
    "ska ",
    "kommer att",
    "vi vill ",
    "genom att",
    "genomfora",
    "infors",
    "infora",
    "avskaffa",
    "forbjuda",
    "sanka",
    "hoja",
    "starka",
    "utoka",
    "bygga ut",
    "garantera",
]

_POLICY_DETAIL_MARKERS = [
    "lag",
    "lagstiftning",
    "reform",
    "budget",
    "anslag",
    "myndighet",
    "kommun",
    "region",
    "skola",
    "sjukvard",
    "skatt",
    "polis",
    "domstol",
    "arbetsloshet",
    "ersattning",
    "bidrag",
    "migration",
    "klimat",
    "miljo",
]


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _promise_specificity_score(text: str) -> int:
    normalized = _normalize_for_match(text)
    score = 0

    if any(marker in normalized for marker in _COMMITMENT_MARKERS):
        score += 1
    if any(marker in normalized for marker in _POLICY_DETAIL_MARKERS):
        score += 1
    if re.search(r"\d", normalized):
        score += 1
    if len(normalized.split()) >= 9:
        score += 1

    return score


def _is_concrete_promise_text(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    if len(normalized) < 35:
        return False
    if any(pattern in normalized for pattern in _GENERIC_PROMISE_PATTERNS):
        # Tillat generic fras endast om den ocksa innehaller tydliga detaljer.
        return _promise_specificity_score(normalized) >= 3
    return _promise_specificity_score(normalized) >= 2


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

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    # Prioritera rader med lofte-signaler/siffror.
    selected: list[str] = []
    total = 0
    for line in lines:
        lower = line.lower()
        is_signal = any(marker.strip() in lower for marker in _COMMITMENT_MARKERS) or bool(re.search(r"\d", lower))
        if not is_signal:
            continue
        if total + len(line) + 1 > max_chars:
            break
        selected.append(line)
        total += len(line) + 1

    if selected and total >= max_chars // 2:
        return "\n".join(selected)[:max_chars]

    # Fallback: ta inledning + slut.
    half = max_chars // 2
    return f"{text[:half]}\n...\n{text[-half:]}"


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
        text = str(item.get("text", "")).strip()
        category = str(item.get("category", "")).strip() or "Ovrigt"
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
- Hall texten nara originalet.
- Varje lofte ska vara 1-3 meningar och ange vad + vem + hur (nar det framgar i kallan).
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
Varje lofte ska vara 1-3 meningar och tydliggora vad + vem + hur.
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
            return await _extract_promises_via_plain_ollama(text, party_name, source_url, model_name)
        except Exception as plain_exc:
            last_error = plain_exc

    raise last_error
