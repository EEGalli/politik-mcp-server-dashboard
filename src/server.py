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
from datetime import datetime, timezone

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
from src.agents import compare_analysis_with_promise, extract_promises_from_text, fetch_page_text

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


@mcp.tool()
async def scrape_party_promises(party_key: str, urls: list[str]) -> str:
    """Hämta och extrahera vallöften automatiskt från webbsidor eller PDF:er.
    Använder AI (Ollama/Gemma3) för att identifiera konkreta vallöften i texten.

    Args:
        party_key: Partiets databas-nyckel
        urls: Lista med URL:er till partiets policy-sidor (HTML eller PDF)
    """
    db = get_db()

    # Verifiera att partiet finns
    party = _get_party(db, party_key)
    if not party:
        return json.dumps({"error": f"Parti med nyckel '{party_key}' hittades inte"})

    party_name = party["name"]
    all_saved = []
    errors = []

    for url in urls:
        try:
            # Hämta och extrahera text
            text = await fetch_page_text(url)
            if not text.strip():
                errors.append({"url": url, "error": "Ingen text kunde extraheras"})
                continue

            # Kör AI-agenten för att hitta vallöften
            extracted = await extract_promises_from_text(text, party_name, url)

            # Spara varje löfte i databasen
            for promise in extracted.promises:
                saved = _add_promise(
                    db,
                    party_key,
                    promise.text,
                    promise.category,
                    source_url=url,
                    source_name=f"{party_name}s hemsida",
                )
                all_saved.append(saved)

        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    result = {
        "party": party_name,
        "urls_processed": len(urls),
        "promises_extracted": len(all_saved),
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
