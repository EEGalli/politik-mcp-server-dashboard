"""
Politik MCP Server — Analysera politikers sociala medier vs vallöften.

Exponerar tools för Claude Desktop:
  - add_party, list_parties, get_party_details
  - add_politician, list_politicians, get_politician_details
  - add_promise, get_promises
  - save_analysis, search_analyses
  - scrape_party_promises (AI-driven scraping av vallöften)
  - compare_content_vs_promise (AI-driven via pydantic-ai + Ollama)
  - get_consistency_report
"""

import json
import re
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.db.models import init_db
from src.tools.politicians import (
    add_party as _add_party,
    list_parties as _list_parties,
    get_party as _get_party,
    add_politician as _add_politician,
    list_politicians as _list_politicians,
    get_politician as _get_politician,
    add_promise as _add_promise,
    get_promises as _get_promises,
    delete_promise as _delete_promise,
)
from src.tools.comparison import (
    save_comparison as _save_comparison,
    get_consistency_report as _get_consistency_report,
)
from src.agents import (
    compare_analysis_with_promise,
    crawl_site_texts,
    extract_promises_from_text,
    fetch_page_text,
)

mcp = FastMCP(
    "Politik Analyzer",
    instructions="Analysera politikers sociala medier och jämför med deras vallöften",
)

# Databasanslutning (lazy init)
_db = None


def get_db():
    global _db
    if _db is None:
        _db = init_db()
    return _db


# ── Parti-tools ────────────────────────────────────────────────


@mcp.tool()
def add_party(name: str, abbreviation: str, block: str = "", website_url: str = "") -> str:
    """Lägg till ett parti i databasen.

    Args:
        name: Partiets fullständiga namn, t.ex. "Moderaterna"
        abbreviation: Partiförkortning, t.ex. "M"
        block: Politiskt block: vänster, höger, mitt (valfritt)
        website_url: Partiets webbplats (valfritt)
    """
    result = _add_party(get_db(), name, abbreviation, block, website_url)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def list_parties() -> str:
    """Lista alla partier i databasen."""
    results = _list_parties(get_db())
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
def get_party_details(party_key: str) -> str:
    """Hämta detaljerad info om ett parti, inklusive dess vallöften.

    Args:
        party_key: Partiets databas-nyckel
    """
    db = get_db()
    party = _get_party(db, party_key)
    if not party:
        return json.dumps({"error": f"Parti med nyckel '{party_key}' hittades inte"})

    promises = _get_promises(db, party_key)
    party["promises"] = promises
    return json.dumps(party, ensure_ascii=False, default=str)


# ── Politiker-tools ──────────────────────────────────────────────


@mcp.tool()
def add_politician(name: str, party_key: str, tiktok: str = "", instagram: str = "", twitter: str = "") -> str:
    """Lägg till en ny politiker i databasen.

    Args:
        name: Politikerns fullständiga namn, t.ex. "Ulf Kristersson"
        party_key: Partiets databas-nyckel (hämta med list_parties)
        tiktok: TikTok-användarnamn (valfritt)
        instagram: Instagram-användarnamn (valfritt)
        twitter: Twitter/X-användarnamn (valfritt)
    """
    social_media = {}
    if tiktok:
        social_media["tiktok"] = tiktok
    if instagram:
        social_media["instagram"] = instagram
    if twitter:
        social_media["twitter"] = twitter

    result = _add_politician(get_db(), name, party_key, social_media)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def list_politicians(party_key: str = "") -> str:
    """Lista alla politiker i databasen, valfritt filtrerade per parti.

    Args:
        party_key: Filtrera per parti-nyckel (lämna tomt för alla)
    """
    results = _list_politicians(get_db(), party_key or None)
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
def get_politician_details(politician_key: str) -> str:
    """Hämta detaljerad info om en politiker, inklusive partiets vallöften.

    Args:
        politician_key: Politikerns databas-nyckel
    """
    db = get_db()
    politician = _get_politician(db, politician_key)
    if not politician:
        return json.dumps({"error": f"Politiker med nyckel '{politician_key}' hittades inte"})

    # Hämta partiinfo
    party = _get_party(db, politician.get("party_key", ""))
    if party:
        politician["party"] = party

    # Hämta partiets vallöften
    promises = _get_promises(db, politician.get("party_key", ""))
    politician["promises"] = promises

    return json.dumps(politician, ensure_ascii=False, default=str)


# ── Vallöften-tools ──────────────────────────────────────────────


@mcp.tool()
def add_promise(
    party_key: str,
    text: str,
    category: str,
    source_url: str = "",
    source_name: str = "",
    date: str = "",
) -> str:
    """Lägg till ett vallöfte kopplat till ett parti.

    Args:
        party_key: Partiets databas-nyckel
        text: Vallöftets text, t.ex. "Vi ska sänka skatten med 10%"
        category: Kategori, t.ex. "Skatt", "Sjukvård", "Skola"
        source_url: URL till källan (valfritt)
        source_name: Namn på källan, t.ex. "Valmanifest 2022" (valfritt)
        date: Datum för löftet, YYYY-MM-DD (valfritt)
    """
    result = _add_promise(get_db(), party_key, text, category, source_url, source_name, date)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_promises(party_key: str, category: str = "") -> str:
    """Hämta vallöften för ett parti, valfritt filtrerade per kategori.

    Args:
        party_key: Partiets databas-nyckel
        category: Filtrera per kategori (lämna tomt för alla)
    """
    results = _get_promises(get_db(), party_key, category or None)
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
def delete_promise(promise_key: str) -> str:
    """Radera ett vallöfte via dess nyckel.

    Args:
        promise_key: Dokumentnyckel i promises-collectionen
    """
    result = _delete_promise(get_db(), promise_key)
    return json.dumps(result, ensure_ascii=False, default=str)


# ── Scraping-tools ───────────────────────────────────────────────


def _normalize_promise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _coerce_int(value: Any, default: int = 0, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, parsed)


def _extract_source_urls(promise: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    primary_url = str(promise.get("source_url", "") or "").strip()
    if primary_url:
        urls.append(primary_url)

    extra_urls = promise.get("source_urls")
    if isinstance(extra_urls, list):
        for url in extra_urls:
            cleaned = str(url or "").strip()
            if cleaned and cleaned not in urls:
                urls.append(cleaned)
    return urls


@mcp.tool()
async def scrape_party_promises(
    party_key: str,
    urls: list[str],
    crawl: bool = True,
    max_pages_per_url: int = 25,
    max_depth: int = 2,
) -> str:
    """Hämta och extrahera vallöften automatiskt från webbsidor eller PDF:er.
    Använder AI (Ollama/Gemma3) för att identifiera konkreta vallöften i texten.
    Om crawl=True (default) följer den interna länkar automatiskt.

    Args:
        party_key: Partiets databas-nyckel
        urls: Lista med URL:er till partiets policy-sidor (HTML eller PDF)
        crawl: Om True, crawla interna undersidor från varje start-URL
        max_pages_per_url: Max antal sidor att hämta per start-URL vid crawl
        max_depth: Hur många länknivåer från start-URL som ska följas
    """
    db = get_db()

    # Verifiera att partiet finns
    party = _get_party(db, party_key)
    if not party:
        return json.dumps({"error": f"Parti med nyckel '{party_key}' hittades inte"})

    party_name = party["name"]
    all_saved = []
    errors = []
    pages_processed = 0
    duplicates_skipped = 0
    effective_max_pages_per_url = max(1, min(int(max_pages_per_url or 1), 80))
    effective_max_depth = max(0, min(int(max_depth or 0), 4))
    promises_collection = db.collection("promises")

    # Förhindra dubbletter både mot tidigare databasinnehåll och inom samma körning.
    existing_promises = _get_promises(db, party_key)
    seen_promises: dict[str, dict[str, Any]] = {}
    for row in existing_promises:
        if not isinstance(row, dict):
            continue
        normalized_text = _normalize_promise_text(str(row.get("text", "")))
        promise_key = str(row.get("_key", "")).strip()
        if not normalized_text or not promise_key:
            continue

        mention_count = _coerce_int(row.get("mention_count"), default=1, minimum=1)
        source_urls = _extract_source_urls(row)
        source_count = max(
            _coerce_int(row.get("source_count"), default=0, minimum=0),
            len(source_urls),
        )

        if normalized_text in seen_promises:
            existing = seen_promises[normalized_text]
            existing["mention_count"] = max(existing["mention_count"], mention_count)
            existing["source_urls"].update(source_urls)
            existing["source_count"] = max(existing["source_count"], source_count, len(existing["source_urls"]))
            continue

        seen_promises[normalized_text] = {
            "key": promise_key,
            "mention_count": mention_count,
            "source_urls": set(source_urls),
            "source_count": source_count,
        }

    for url in urls:
        try:
            pages: list[dict] = []

            if crawl:
                crawl_result = await crawl_site_texts(
                    url,
                    max_pages=effective_max_pages_per_url,
                    max_depth=effective_max_depth,
                )
                crawl_pages = crawl_result.get("pages", [])
                crawl_errors = crawl_result.get("errors", [])
                if isinstance(crawl_pages, list):
                    pages = crawl_pages
                if isinstance(crawl_errors, list):
                    for err in crawl_errors:
                        if isinstance(err, dict):
                            errors.append(
                                {
                                    "url": str(err.get("url", url)),
                                    "error": str(err.get("error", "Crawl-fel")),
                                    "stage": "crawl",
                                }
                            )
                if not pages:
                    errors.append({"url": url, "error": "Crawlern hittade ingen läsbar text", "stage": "crawl"})
                    continue
            else:
                text = await fetch_page_text(url)
                pages = [{"url": url, "text": text, "depth": 0}]

            for page in pages:
                page_url = str(page.get("url", url))
                page_text = str(page.get("text", "") or "")
                if not page_text.strip():
                    continue

                pages_processed += 1
                extracted = await extract_promises_from_text(page_text, party_name, page_url)

                # Spara varje löfte i databasen
                for promise in extracted.promises:
                    normalized_text = _normalize_promise_text(promise.text)
                    if not normalized_text:
                        continue

                    if normalized_text in seen_promises:
                        existing = seen_promises[normalized_text]
                        existing["mention_count"] += 1
                        if page_url:
                            existing["source_urls"].add(page_url)
                        existing["source_count"] = max(existing["source_count"], len(existing["source_urls"]))

                        update_payload = {
                            "_key": existing["key"],
                            "mention_count": existing["mention_count"],
                            "source_urls": sorted(existing["source_urls"]),
                            "source_count": existing["source_count"],
                            "last_seen_at": datetime.now(timezone.utc).isoformat(),
                        }
                        promises_collection.update(update_payload, keep_none=False)
                        duplicates_skipped += 1
                        continue

                    source_urls = [page_url] if page_url else []
                    saved = _add_promise(
                        db,
                        party_key,
                        promise.text,
                        promise.category,
                        source_url=page_url,
                        source_name=f"{party_name}s hemsida",
                        mention_count=1,
                        source_urls=source_urls,
                        source_count=len(source_urls),
                        last_seen_at=datetime.now(timezone.utc).isoformat(),
                    )
                    seen_promises[normalized_text] = {
                        "key": str(saved.get("_key", "")).strip(),
                        "mention_count": 1,
                        "source_urls": set(source_urls),
                        "source_count": len(source_urls),
                    }
                    all_saved.append(saved)

        except Exception as e:
            errors.append({"url": url, "error": str(e), "stage": "extract"})

    result = {
        "party": party_name,
        "crawl": crawl,
        "max_pages_per_url": effective_max_pages_per_url,
        "max_depth": effective_max_depth,
        "urls_processed": len(urls),
        "pages_processed": pages_processed,
        "promises_extracted": len(all_saved),
        "duplicates_skipped": duplicates_skipped,
        "promises": all_saved,
        "errors": errors,
    }
    return json.dumps(result, ensure_ascii=False, default=str)


# ── Analys-tools ─────────────────────────────────────────────────


@mcp.tool()
def save_analysis(
    politician_key: str,
    video_file: str,
    transcription: str,
    summary: str,
    category: str,
    political_color: str = "",
    platform: str = "",
) -> str:
    """Spara en videoanalys i databasen.

    Args:
        politician_key: Politikerns databas-nyckel
        video_file: Filnamn på videon
        transcription: Full transkription av videon
        summary: Sammanfattning av innehållet
        category: Innehållskategori
        political_color: Politisk färg (vänster/höger/centrum/neutral)
        platform: Plattform (tiktok, instagram, etc.)
    """
    db = get_db()
    collection = db.collection("analyses")

    doc = {
        "politician_key": politician_key,
        "video_file": video_file,
        "transcription": transcription,
        "summary": summary,
        "category": category,
        "political_color": political_color,
        "platform": platform,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    result = collection.insert(doc)
    doc["_key"] = result["_key"]
    return json.dumps(doc, ensure_ascii=False, default=str)


@mcp.tool()
def search_analyses(
    politician_key: str = "",
    category: str = "",
    platform: str = "",
    search_text: str = "",
) -> str:
    """Sök bland videoanalyser med valfria filter.

    Args:
        politician_key: Filtrera per politiker (valfritt)
        category: Filtrera per kategori (valfritt)
        platform: Filtrera per plattform (valfritt)
        search_text: Fritextsökning i transkription/sammanfattning (valfritt)
    """
    db = get_db()
    filters = []
    bind_vars = {}

    if politician_key:
        filters.append("a.politician_key == @politician_key")
        bind_vars["politician_key"] = politician_key
    if category:
        filters.append("a.category == @category")
        bind_vars["category"] = category
    if platform:
        filters.append("a.platform == @platform")
        bind_vars["platform"] = platform
    if search_text:
        filters.append("(CONTAINS(LOWER(a.transcription), LOWER(@search_text)) OR CONTAINS(LOWER(a.summary), LOWER(@search_text)))")
        bind_vars["search_text"] = search_text

    where_clause = " FILTER " + " AND ".join(filters) if filters else ""
    query = f"FOR a IN analyses{where_clause} SORT a.analyzed_at DESC LIMIT 50 RETURN a"

    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return json.dumps(list(cursor), ensure_ascii=False, default=str)


# ── Jämförelse-tools ─────────────────────────────────────────────


@mcp.tool()
async def compare_content_vs_promise(
    analysis_key: str,
    promise_key: str,
) -> str:
    """Jämför en videoanalys med ett vallöfte med hjälp av AI.
    Använder en lokal Ollama-modell (Gemma3) för att bedöma
    hur väl innehållet stämmer överens med löftet.

    Args:
        analysis_key: Nyckel för videoanalysen
        promise_key: Nyckel för vallöftet
    """
    db = get_db()

    # Hämta analysen
    analyses_col = db.collection("analyses")
    if not analyses_col.has(analysis_key):
        return json.dumps({"error": f"Analys '{analysis_key}' hittades inte"})
    analysis = analyses_col.get(analysis_key)

    # Hämta löftet
    promises_col = db.collection("promises")
    if not promises_col.has(promise_key):
        return json.dumps({"error": f"Löfte '{promise_key}' hittades inte"})
    promise = promises_col.get(promise_key)

    # Hämta politikern och partiet
    politician = _get_politician(db, analysis.get("politician_key", ""))
    politician_name = politician["name"] if politician else "Okänd"
    party_doc = _get_party(db, politician.get("party_key", "")) if politician else None
    party = party_doc["name"] if party_doc else "Okänt"

    # Kör AI-agenten för jämförelse
    comparison_result = await compare_analysis_with_promise(
        transcription=analysis.get("transcription", ""),
        summary=analysis.get("summary", ""),
        promise_text=promise.get("text", ""),
        promise_category=promise.get("category", ""),
        politician_name=politician_name,
        party=party,
    )

    # Spara resultatet i databasen
    saved = _save_comparison(
        db,
        analysis_key,
        promise_key,
        comparison_result.match_score,
        comparison_result.match_type,
        comparison_result.explanation,
    )

    # Returnera både AI-resultatet och den sparade jämförelsen
    return json.dumps({
        "comparison": comparison_result.model_dump(),
        "saved": saved,
    }, ensure_ascii=False, default=str)


@mcp.tool()
def get_consistency_report(politician_key: str) -> str:
    """Generera en konsistensrapport för en politiker.
    Visar hur väl deras sociala medier-innehåll stämmer överens med partiets vallöften.

    Args:
        politician_key: Politikerns databas-nyckel
    """
    result = _get_consistency_report(get_db(), politician_key)
    return json.dumps(result, ensure_ascii=False, default=str)


# ── MCP Resources (kontext) ──────────────────────────────────────


@mcp.resource("politik://categories")
def get_categories() -> str:
    """Lista tillgängliga kategorier för vallöften."""
    categories = [
        "Skatt", "Sjukvård", "Skola & Utbildning", "Försvar",
        "Migration", "Klimat & Miljö", "Rättsväsende", "Arbetsmarknad",
        "Bostäder", "Infrastruktur", "Kultur", "Jämställdhet",
        "EU & Utrikes", "Ekonomi", "Socialpolitik", "Landsbygd",
        "Digitalisering", "Övrigt",
    ]
    return json.dumps(categories, ensure_ascii=False)


@mcp.resource("politik://parties")
def get_parties() -> str:
    """Lista alla partier från databasen."""
    results = _list_parties(get_db())
    return json.dumps(results, ensure_ascii=False, default=str)


# ── Prompts ──────────────────────────────────────────────────────


@mcp.prompt()
def analyze_politician(politician_name: str) -> str:
    """Analysera en politikers sociala medier vs partiets vallöften.

    Args:
        politician_name: Politikerns namn
    """
    return f"""Analysera politikern {politician_name}:

1. Sök efter politikern i databasen med list_politicians
2. Hämta deras parti och partiets vallöften med get_politician_details
3. Sök deras videoanalyser med search_analyses
4. Jämför varje videoanalys med relevanta vallöften med compare_content_vs_promise
5. Generera en konsistensrapport med get_consistency_report

Var saklig och balanserad i din analys."""


@mcp.prompt()
def scrape_promises(party_name: str) -> str:
    """Instruktioner för att skrapa vallöften från ett partis hemsida.

    Args:
        party_name: Partiets namn
    """
    return f"""Hämta vallöften för {party_name}:

1. Sök efter partiets officiella hemsida och valmanifest
2. Identifiera konkreta vallöften (specifika åtaganden, inte vaga visioner)
3. Kategorisera varje löfte (Skatt, Sjukvård, Skola, etc.)
4. Lägg till partiet med add_party om det inte redan finns
5. Spara varje löfte med add_promise (använd partiets party_key)

Fokusera på:
- Konkreta, mätbara löften
- Ange källan (URL och namn)
- Ange datum om möjligt"""


# ── Startpunkt ───────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
