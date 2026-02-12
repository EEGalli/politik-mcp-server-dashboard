"""
Automatisk bevakning av politikers sociala medier via RSSHub + feedparser.

Hämtar RSS-flöden från en lokal RSSHub-instans, identifierar nytt innehåll,
analyserar text med Ollama och jämför mot partiets vallöften.

Fas 1: Bara TikTok. Utöka _PLATFORM_ROUTE_MAP för fler plattformar.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from src.agents import (
    _extract_first_json_object,
    _get_ollama_host,
    _get_primary_agent_model,
    _ollama_chat,
    compare_analysis_with_promise,
)
from src.db.models import init_db
from src.tools.comparison import save_comparison
from src.tools.politicians import (
    get_party,
    get_politician,
    get_promises,
    list_politicians,
)
from src.video_bridge import tokenize

logger = logging.getLogger(__name__)

RSSHUB_URL = os.getenv("RSSHUB_URL", "http://localhost:1200").rstrip("/")

# Fas 1: bara TikTok. Lägg till fler plattformar här i framtida faser.
_PLATFORM_ROUTE_MAP = {
    "tiktok": "/tiktok/user/{handle}",
    # "twitter": "/twitter/user/{handle}",       # Fas 2
    # "instagram": "/instagram/user/{handle}",   # Fas 3
}


# ── Feed-URL-konstruktion ─────────────────────────────────────────


def _build_feed_url(platform: str, handle: str) -> str | None:
    """Konstruera RSSHub-feed-URL för en plattform + handle.

    Returnerar None om plattformen inte stöds eller handle saknas.
    """
    route_template = _PLATFORM_ROUTE_MAP.get(platform.lower())
    if not route_template:
        return None

    clean_handle = handle.strip().lstrip("@")
    if not clean_handle:
        return None

    route = route_template.format(handle=clean_handle)
    return f"{RSSHUB_URL}{route}"


# ── Hälsokontroll ──────────────────────────────────────────────────


async def check_rsshub_health() -> dict[str, Any]:
    """Kontrollera att RSSHub är nåbar och svarar."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{RSSHUB_URL}")
            return {
                "status": "ok" if response.status_code == 200 else "error",
                "url": RSSHUB_URL,
                "status_code": response.status_code,
            }
    except Exception as exc:
        return {
            "status": "unreachable",
            "url": RSSHUB_URL,
            "error": str(exc),
        }


# ── Databashjälpare ────────────────────────────────────────────────


def _ensure_monitored_content(db) -> None:
    """Skapa monitored_content-collection om den inte redan finns."""
    if not db.has_collection("monitored_content"):
        db.create_collection("monitored_content")


def _is_already_processed(
    db, politician_key: str, platform: str, item_id: str
) -> bool:
    """Kontrollera om ett feed-item redan har bearbetats."""
    query = """
    FOR mc IN monitored_content
        FILTER mc.politician_key == @politician_key
           AND mc.platform == @platform
           AND mc.item_id == @item_id
        LIMIT 1
        RETURN 1
    """
    cursor = db.aql.execute(
        query,
        bind_vars={
            "politician_key": politician_key,
            "platform": platform,
            "item_id": item_id,
        },
    )
    return len(list(cursor)) > 0


# ── RSS-parsning ───────────────────────────────────────────────────


async def _fetch_feed(feed_url: str) -> dict[str, Any]:
    """Hämta ett RSS-flöde från RSSHub och parsa med feedparser."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(feed_url)
            response.raise_for_status()
    except httpx.ConnectError:
        return {
            "entries": [],
            "feed_title": "",
            "error": f"Kan inte ansluta till RSSHub på {RSSHUB_URL}",
        }
    except httpx.HTTPStatusError as exc:
        return {
            "entries": [],
            "feed_title": "",
            "error": f"HTTP {exc.response.status_code} från {feed_url}",
        }
    except Exception as exc:
        return {"entries": [], "feed_title": "", "error": str(exc)}

    parsed = feedparser.parse(response.text)
    return {
        "entries": parsed.entries,
        "feed_title": getattr(parsed.feed, "title", ""),
        "error": "",
    }


def _item_id_from_entry(entry) -> str:
    """Extrahera ett stabilt unikt ID från ett feedparser-entry."""
    guid = getattr(entry, "id", "") or ""
    if guid.strip():
        return guid.strip()
    link = getattr(entry, "link", "") or ""
    if link.strip():
        return link.strip()
    title = getattr(entry, "title", "") or ""
    return f"title:{title.strip()[:200]}"


def _extract_text_from_entry(entry) -> str:
    """Extrahera bästa tillgängliga text från ett feedparser-entry."""
    parts = []

    title = getattr(entry, "title", "") or ""
    if title.strip():
        parts.append(title.strip())

    summary = getattr(entry, "summary", "") or ""
    if summary.strip():
        parts.append(summary.strip())

    content_list = getattr(entry, "content", []) or []
    for content_item in content_list:
        if isinstance(content_item, dict):
            value = content_item.get("value", "")
            if value and value.strip() and value.strip() != summary.strip():
                parts.append(value.strip())

    return "\n\n".join(parts)


# ── AI-kategorisering ──────────────────────────────────────────────


async def _categorize_and_summarize(
    text: str, politician_name: str, platform: str
) -> dict[str, str]:
    """Använd Ollama för att kategorisera och sammanfatta ett inlägg."""
    model = _get_primary_agent_model()

    prompt = f"""Analysera följande sociala medier-inlägg från politikern {politician_name} på {platform}.

Text:
---
{text[:4000]}
---

Svara ENDAST med giltig JSON:
{{"summary":"kort sammanfattning på svenska (max 2 meningar)","category":"en av: Sjukvård, Skola & Utbildning, Rättsväsende, Ekonomi, Skatt, Arbetsmarknad, Socialpolitik, Försvar, Migration, Klimat & Miljö, Bostäder, Infrastruktur, Kultur, Jämställdhet, EU & Utrikes, Digitalisering, Landsbygd, Övrigt","political_color":"vänster/höger/centrum/neutral"}}"""

    try:
        raw = await _ollama_chat(prompt, model)
        parsed = _extract_first_json_object(raw)
        if parsed and isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary", "")).strip() or text[:200],
                "category": str(parsed.get("category", "")).strip() or "Övrigt",
                "political_color": str(parsed.get("political_color", "")).strip()
                or "neutral",
            }
    except Exception as exc:
        logger.warning("Ollama-kategorisering misslyckades: %s", exc)

    return {
        "summary": text[:200],
        "category": "Övrigt",
        "political_color": "neutral",
    }


# ── Relevans-scoring ───────────────────────────────────────────────


def _score_feed_promise_relevance(
    text: str, category: str, promise: dict[str, Any]
) -> int:
    """Poängsätt ett löftes relevans mot ett feed-items text.

    Återanvänder tokenize() från video_bridge.py.
    """
    content_tokens = tokenize(text)

    promise_text = " ".join(
        [
            str(promise.get("text", "") or ""),
            str(promise.get("category", "") or ""),
        ]
    )
    promise_tokens = tokenize(promise_text)

    overlap = len(content_tokens & promise_tokens)

    cat_a = category.strip().lower()
    cat_b = str(promise.get("category", "") or "").strip().lower()
    category_bonus = 1 if cat_a and cat_b and (cat_a in cat_b or cat_b in cat_a) else 0

    return overlap + category_bonus


# ── Bearbetning av enskilt feed-item ───────────────────────────────


async def _process_feed_item(
    db,
    politician: dict[str, Any],
    platform: str,
    entry,
    promises: list[dict[str, Any]],
    max_comparisons_per_item: int = 3,
) -> dict[str, Any]:
    """Bearbeta ett RSS-entry: analysera, spara, jämför."""
    politician_key = str(politician["_key"])
    politician_name = str(politician.get("name", "Okänd"))
    party_key = str(politician.get("party_key", ""))

    item_id = _item_id_from_entry(entry)
    item_url = str(getattr(entry, "link", "") or "").strip()
    item_title = str(getattr(entry, "title", "") or "").strip()

    # Deduplicering
    if _is_already_processed(db, politician_key, platform, item_id):
        return {"status": "skipped", "reason": "already_processed", "item_id": item_id}

    # Extrahera text
    text = _extract_text_from_entry(entry)
    if not text.strip() or len(text.strip()) < 10:
        db.collection("monitored_content").insert(
            {
                "politician_key": politician_key,
                "platform": platform,
                "item_id": item_id,
                "item_url": item_url,
                "item_title": item_title,
                "analysis_key": "",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": "skipped",
                "comparisons_done": 0,
            }
        )
        return {"status": "skipped", "reason": "empty_text", "item_id": item_id}

    # Kategorisera och sammanfatta med AI
    analysis_info = await _categorize_and_summarize(text, politician_name, platform)

    # Spara analys (samma mönster som save_analysis i server.py)
    analyses_col = db.collection("analyses")
    analysis_doc = {
        "politician_key": politician_key,
        "video_file": f"feed:{platform}:{item_id[:80]}",
        "transcription": text,
        "summary": analysis_info["summary"],
        "category": analysis_info["category"],
        "political_color": analysis_info["political_color"],
        "platform": platform,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "source_url": item_url,
        "source_type": "feed",
    }
    result = analyses_col.insert(analysis_doc)
    analysis_key = result["_key"]

    # Hitta relevanta löften och jämför
    comparisons_done = 0
    comparison_errors = []

    if promises:
        scored = [
            (p, _score_feed_promise_relevance(text, analysis_info["category"], p))
            for p in promises
            if str(p.get("_key", "")).strip()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_promises = [
            p for p, score in scored[:max_comparisons_per_item] if score > 0
        ]

        party_doc = get_party(db, party_key) if party_key else None
        party_name = str(party_doc["name"]) if party_doc else "Okänt"

        for promise in top_promises:
            promise_key = str(promise["_key"])
            try:
                comparison_result = await compare_analysis_with_promise(
                    transcription=text,
                    summary=analysis_info["summary"],
                    promise_text=str(promise.get("text", "")),
                    promise_category=str(promise.get("category", "")),
                    politician_name=politician_name,
                    party=party_name,
                )
                save_comparison(
                    db,
                    analysis_key,
                    promise_key,
                    comparison_result.match_score,
                    comparison_result.match_type,
                    comparison_result.explanation,
                )
                comparisons_done += 1
            except Exception as exc:
                comparison_errors.append(
                    {"promise_key": promise_key, "error": str(exc)}
                )

    # Spara som bearbetad
    db.collection("monitored_content").insert(
        {
            "politician_key": politician_key,
            "platform": platform,
            "item_id": item_id,
            "item_url": item_url,
            "item_title": item_title,
            "analysis_key": analysis_key,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "status": "analyzed",
            "comparisons_done": comparisons_done,
        }
    )

    return {
        "status": "analyzed",
        "item_id": item_id,
        "analysis_key": analysis_key,
        "comparisons_done": comparisons_done,
        "comparison_errors": comparison_errors,
    }


# ── Huvudfunktioner (anropas av MCP-tools och cron) ───────────────


async def check_feeds(
    db=None,
    politician_key: str = "",
    max_comparisons_per_item: int = 3,
) -> dict[str, Any]:
    """Kontrollera RSS-flöden för en eller alla politiker.

    Args:
        db: Databasanslutning (None → init_db())
        politician_key: Kontrollera bara denna politiker (tomt = alla)
        max_comparisons_per_item: Max antal löften att jämföra per nytt item
    """
    if db is None:
        db = init_db()

    _ensure_monitored_content(db)

    # Kontrollera att RSSHub är tillgänglig
    health = await check_rsshub_health()
    if health["status"] == "unreachable":
        return {
            "error": (
                f"RSSHub är inte nåbar på {RSSHUB_URL}. "
                "Starta med: docker compose -f docker-compose.rsshub.yml up -d"
            ),
            "rsshub_status": health,
            "results": [],
        }

    # Hämta politiker att kontrollera
    if politician_key:
        politician = get_politician(db, politician_key)
        if not politician:
            return {
                "error": f"Politiker '{politician_key}' hittades inte",
                "results": [],
            }
        politicians = [politician]
    else:
        politicians = list_politicians(db)

    results = []

    for pol in politicians:
        pol_key = str(pol.get("_key", "")).strip()
        pol_name = str(pol.get("name", "")).strip()
        party_key = str(pol.get("party_key", "")).strip()
        social_media = pol.get("social_media", {})

        if not isinstance(social_media, dict) or not social_media:
            results.append(
                {
                    "politician_key": pol_key,
                    "politician_name": pol_name,
                    "status": "skipped",
                    "reason": "no_social_media_handles",
                    "platforms": {},
                }
            )
            continue

        # Hämta partiets löften en gång per politiker
        promises = get_promises(db, party_key) if party_key else []

        platform_results = {}

        for platform, handle in social_media.items():
            if not handle or not str(handle).strip():
                continue

            feed_url = _build_feed_url(platform, str(handle))
            if not feed_url:
                platform_results[platform] = {
                    "status": "skipped",
                    "reason": "unsupported_platform",
                }
                continue

            feed_data = await _fetch_feed(feed_url)

            if feed_data["error"]:
                platform_results[platform] = {
                    "status": "error",
                    "error": feed_data["error"],
                    "feed_url": feed_url,
                }
                continue

            entries = feed_data["entries"]
            items_new = 0
            items_skipped = 0
            items_errors = []

            for entry in entries:
                try:
                    item_result = await _process_feed_item(
                        db,
                        pol,
                        platform,
                        entry,
                        promises,
                        max_comparisons_per_item=max_comparisons_per_item,
                    )

                    if item_result["status"] == "analyzed":
                        items_new += 1
                    elif item_result["status"] == "skipped":
                        items_skipped += 1
                except Exception as exc:
                    items_errors.append(str(exc))

            platform_results[platform] = {
                "status": "ok",
                "feed_url": feed_url,
                "total_entries": len(entries),
                "items_new": items_new,
                "items_skipped": items_skipped,
                "errors": items_errors,
            }

        results.append(
            {
                "politician_key": pol_key,
                "politician_name": pol_name,
                "status": "checked",
                "platforms": platform_results,
            }
        )

    return {
        "rsshub_status": health,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "politicians_checked": len(results),
        "results": results,
    }


def get_monitor_status(
    db=None, politician_key: str = ""
) -> dict[str, Any]:
    """Visa bevakningsstatus: senaste kontroll, antal hämtade inlägg per plattform."""
    if db is None:
        db = init_db()

    _ensure_monitored_content(db)

    filters = []
    bind_vars: dict[str, Any] = {}

    if politician_key:
        filters.append("mc.politician_key == @politician_key")
        bind_vars["politician_key"] = politician_key

    where_clause = " FILTER " + " AND ".join(filters) if filters else ""

    # Sammanfattning per politiker + plattform
    query = f"""
    FOR mc IN monitored_content
        {where_clause}
        COLLECT pol_key = mc.politician_key, platform = mc.platform
        AGGREGATE
            total = LENGTH(1),
            analyzed = SUM(mc.status == "analyzed" ? 1 : 0),
            skipped = SUM(mc.status == "skipped" ? 1 : 0),
            failed = SUM(mc.status == "failed" ? 1 : 0),
            last_fetched = MAX(mc.fetched_at)
        SORT last_fetched DESC
        RETURN {{
            politician_key: pol_key,
            platform: platform,
            total_items: total,
            analyzed: analyzed,
            skipped: skipped,
            failed: failed,
            last_fetched: last_fetched
        }}
    """

    cursor = db.aql.execute(query, bind_vars=bind_vars)
    summary_rows = list(cursor)

    # Senaste hämtade items
    recent_query = f"""
    FOR mc IN monitored_content
        {where_clause}
        SORT mc.fetched_at DESC
        LIMIT 20
        LET pol = DOCUMENT(CONCAT("politicians/", mc.politician_key))
        RETURN {{
            politician_name: pol.name,
            platform: mc.platform,
            item_title: mc.item_title,
            item_url: mc.item_url,
            status: mc.status,
            fetched_at: mc.fetched_at,
            analysis_key: mc.analysis_key
        }}
    """
    recent_cursor = db.aql.execute(recent_query, bind_vars=bind_vars)
    recent_items = list(recent_cursor)

    return {
        "summary": summary_rows,
        "recent_items": recent_items,
    }
