"""
Politik MCP Server — Analysera politikers sociala medier vs vallöften.

Exponerar tools för Claude Desktop:
  - add_politician
  - list_politicians
  - add_promise
  - get_promises
  - save_analysis
  - compare_content_vs_promise
  - get_consistency_report
"""

import json
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from src.db.models import init_db
from src.tools.politicians import (
    add_politician as _add_politician,
    list_politicians as _list_politicians,
    get_politician as _get_politician,
    add_promise as _add_promise,
    get_promises as _get_promises,
)
from src.tools.comparison import (
    save_comparison as _save_comparison,
    get_consistency_report as _get_consistency_report,
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


# ── Politiker-tools ──────────────────────────────────────────────


@mcp.tool()
def add_politician(name: str, party: str, tiktok: str = "", instagram: str = "", twitter: str = "") -> str:
    """Lägg till en ny politiker i databasen.

    Args:
        name: Politikerns fullständiga namn, t.ex. "Ulf Kristersson"
        party: Partinamn, t.ex. "Moderaterna"
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

    result = _add_politician(get_db(), name, party, social_media)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def list_politicians(party: str = "") -> str:
    """Lista alla politiker i databasen, valfritt filtrerade per parti.

    Args:
        party: Filtrera per parti (lämna tomt för alla)
    """
    results = _list_politicians(get_db(), party or None)
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
def get_politician_details(politician_key: str) -> str:
    """Hämta detaljerad info om en politiker, inklusive deras vallöften.

    Args:
        politician_key: Politikerns databas-nyckel
    """
    db = get_db()
    politician = _get_politician(db, politician_key)
    if not politician:
        return json.dumps({"error": f"Politiker med nyckel '{politician_key}' hittades inte"})

    promises = _get_promises(db, politician_key)
    politician["promises"] = promises
    return json.dumps(politician, ensure_ascii=False, default=str)


# ── Vallöften-tools ──────────────────────────────────────────────


@mcp.tool()
def add_promise(
    politician_key: str,
    text: str,
    category: str,
    source_url: str = "",
    source_name: str = "",
    date: str = "",
) -> str:
    """Lägg till ett vallöfte kopplat till en politiker.

    Args:
        politician_key: Politikerns databas-nyckel
        text: Vallöftets text, t.ex. "Vi ska sänka skatten med 10%"
        category: Kategori, t.ex. "Skatt", "Sjukvård", "Skola"
        source_url: URL till källan (valfritt)
        source_name: Namn på källan, t.ex. "Valmanifest 2022" (valfritt)
        date: Datum för löftet, YYYY-MM-DD (valfritt)
    """
    result = _add_promise(get_db(), politician_key, text, category, source_url, source_name, date)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_promises(politician_key: str, category: str = "") -> str:
    """Hämta vallöften för en politiker, valfritt filtrerade per kategori.

    Args:
        politician_key: Politikerns databas-nyckel
        category: Filtrera per kategori (lämna tomt för alla)
    """
    results = _get_promises(get_db(), politician_key, category or None)
    return json.dumps(results, ensure_ascii=False, default=str)


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
def compare_content_vs_promise(
    analysis_key: str,
    promise_key: str,
    match_score: float,
    match_type: str,
    explanation: str,
) -> str:
    """Spara en jämförelse mellan en videoanalys och ett vallöfte.

    Args:
        analysis_key: Nyckel för videoanalysen
        promise_key: Nyckel för vallöftet
        match_score: Matchpoäng 0.0-1.0 (hur väl stämmer de överens)
        match_type: "supports" (stöder), "contradicts" (motsäger), eller "unrelated" (orelaterat)
        explanation: Förklaring av jämförelsen på svenska
    """
    result = _save_comparison(get_db(), analysis_key, promise_key, match_score, match_type, explanation)
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_consistency_report(politician_key: str) -> str:
    """Generera en konsistensrapport för en politiker.
    Visar hur väl deras sociala medier-innehåll stämmer överens med vallöften.

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
    """Lista svenska riksdagspartier."""
    parties = [
        {"name": "Socialdemokraterna", "abbr": "S", "block": "vänster"},
        {"name": "Moderaterna", "abbr": "M", "block": "höger"},
        {"name": "Sverigedemokraterna", "abbr": "SD", "block": "höger"},
        {"name": "Centerpartiet", "abbr": "C", "block": "mitt"},
        {"name": "Vänsterpartiet", "abbr": "V", "block": "vänster"},
        {"name": "Kristdemokraterna", "abbr": "KD", "block": "höger"},
        {"name": "Liberalerna", "abbr": "L", "block": "höger"},
        {"name": "Miljöpartiet", "abbr": "MP", "block": "vänster"},
    ]
    return json.dumps(parties, ensure_ascii=False)


# ── Prompts ──────────────────────────────────────────────────────


@mcp.prompt()
def analyze_politician(politician_name: str) -> str:
    """Analysera en politikers sociala medier vs vallöften.

    Args:
        politician_name: Politikerns namn
    """
    return f"""Analysera politikern {politician_name}:

1. Sök efter politikern i databasen med list_politicians
2. Hämta deras vallöften med get_promises
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
4. Lägg till politikern med add_politician om de inte redan finns
5. Spara varje löfte med add_promise

Fokusera på:
- Konkreta, mätbara löften
- Ange källan (URL och namn)
- Ange datum om möjligt"""


# ── Startpunkt ───────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
