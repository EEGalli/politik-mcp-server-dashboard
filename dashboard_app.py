import asyncio
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src import server as mcp_server
from src.video_bridge import analyze_video_file, get_video_analyzer_root, tokenize

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CHAT_MODEL = "qwen3:8b"
CHAT_SYSTEM_PROMPT = (
    "Du är en hjälpsam politisk assistent. "
    "Svara sakligt, kortfattat och neutralt på svenska."
)

st.set_page_config(page_title="Politik MCP Dashboard", layout="wide")
st.title("Politik MCP Dashboard")
st.caption("Interagera med MCP-serverns tools utan Claude Desktop")


def _parse_json_result(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            result["error"] = exc

    import threading

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def _call_tool(fn: Callable, args: dict[str, Any], is_async: bool = False) -> Any:
    if is_async:
        raw = _run_async(fn(**args))
    else:
        raw = fn(**args)
    return _parse_json_result(raw)


def _extract_first_json_object(raw: str) -> dict | None:
    cleaned = str(raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
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


DB_COLLECTIONS = ["parties", "politicians", "promises", "analyses", "comparisons"]
COLLECTION_SORT_FIELDS = {
    "parties": "created_at",
    "politicians": "created_at",
    "promises": "created_at",
    "analyses": "analyzed_at",
    "comparisons": "compared_at",
}
_APP_ROOT = Path(__file__).resolve().parent
_SCRAPE_JOB_DIR = _APP_ROOT / ".runtime" / "scrape_jobs"
_ACTIVE_SCRAPE_JOB_FILE = _SCRAPE_JOB_DIR / "active_scrape_job.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _is_stale_scrape_job(job: dict[str, Any], max_age_minutes: int = 25) -> bool:
    status = str(job.get("status", "")).strip().lower()
    if status not in {"queued", "running"}:
        return False
    started_at = _parse_iso_datetime(job.get("started_at"))
    if started_at is None:
        started_at = _parse_iso_datetime(job.get("created_at"))
    if started_at is None:
        return False
    age_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()

    timeout_seconds = _coerce_int(job.get("timeout_seconds", 0))
    if timeout_seconds > 0:
        # Ge lite marginal ovanpå timeout innan vi klassar jobbet som fastnat.
        stale_after_seconds = min(7200, max(timeout_seconds + 180, 600))
        return age_seconds > stale_after_seconds

    return age_seconds > (max_age_minutes * 60)


def _job_age_seconds(job: dict[str, Any]) -> int:
    started_at = _parse_iso_datetime(job.get("started_at"))
    if started_at is None:
        started_at = _parse_iso_datetime(job.get("created_at"))
    if started_at is None:
        return 0
    return max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))


def _estimate_scrape_timeout_seconds(
    urls: list[str],
    max_pages_per_url: int,
    max_depth: int,
) -> int:
    # Enkel tumregel: fler sidor/djup ger längre timeout, men alltid inom rimliga gränser.
    safe_urls = max(1, len(urls or []))
    safe_pages = max(1, int(max_pages_per_url or 1))
    safe_depth = max(0, int(max_depth or 0))
    estimated = safe_urls * safe_pages * (20 + (safe_depth * 6))
    return max(300, min(5400, estimated))


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_active_scrape_job() -> dict[str, Any] | None:
    if not _ACTIVE_SCRAPE_JOB_FILE.exists():
        return None
    try:
        raw = _ACTIVE_SCRAPE_JOB_FILE.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _set_active_scrape_job(job: dict[str, Any]) -> None:
    _write_json_file(_ACTIVE_SCRAPE_JOB_FILE, job)


def _clear_active_scrape_job() -> None:
    try:
        _ACTIVE_SCRAPE_JOB_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _spawn_scrape_worker(
    job_id: str,
    party_key: str,
    urls: list[str],
    crawl: bool,
    max_pages_per_url: int,
    max_depth: int,
) -> None:
    timeout_seconds = _estimate_scrape_timeout_seconds(
        urls=urls,
        max_pages_per_url=max_pages_per_url,
        max_depth=max_depth,
    )

    def _worker() -> None:
        current = _read_active_scrape_job() or {}
        current.update(
            {
                "job_id": job_id,
                "status": "running",
                "started_at": _utc_now_iso(),
                "timeout_seconds": timeout_seconds,
            }
        )
        _set_active_scrape_job(current)

        try:
            raw = asyncio.run(
                asyncio.wait_for(
                    mcp_server.scrape_party_promises(
                        party_key=party_key,
                        urls=urls,
                        crawl=crawl,
                        max_pages_per_url=max_pages_per_url,
                        max_depth=max_depth,
                    ),
                    timeout=timeout_seconds,
                )
            )
            result = _parse_json_result(raw)
            current = _read_active_scrape_job() or {}
            if str(current.get("job_id", "")) != job_id:
                return
            current.update(
                {
                    "status": "completed",
                    "finished_at": _utc_now_iso(),
                    "result": result,
                    "error": "",
                }
            )
            _set_active_scrape_job(current)
        except Exception as exc:  # pragma: no cover - extern modell/nät
            current = _read_active_scrape_job() or {}
            if str(current.get("job_id", "")) != job_id:
                return
            error_text = str(exc)
            if isinstance(exc, TimeoutError):
                error_text = (
                    f"Scraping timeout efter {timeout_seconds} sekunder. "
                    "Kör med färre sidor/lägre djup eller dela upp URL:erna."
                )
            current.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now_iso(),
                    "error": error_text,
                }
            )
            _set_active_scrape_job(current)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _db_counts() -> dict[str, int]:
    db = mcp_server.get_db()
    result: dict[str, int] = {}
    for name in DB_COLLECTIONS:
        result[name] = int(db.collection(name).count())
    return result


def _fetch_collection_docs(collection_name: str, limit: int) -> list[dict]:
    if collection_name not in DB_COLLECTIONS:
        return []
    db = mcp_server.get_db()
    sort_field = COLLECTION_SORT_FIELDS.get(collection_name, "created_at")
    query = f"""
    FOR d IN @@collection
        SORT d.{sort_field} DESC
        LIMIT @limit
        RETURN d
    """
    cursor = db.aql.execute(
        query,
        bind_vars={"@collection": collection_name, "limit": int(limit)},
    )
    return list(cursor)


def _render_db_inspector() -> None:
    with st.expander("Databas-inspektör (under huven)", expanded=False):
        st.caption("Visar rådata från ArangoDB-collections som MCP-servern använder.")
        try:
            counts = _db_counts()
        except Exception as exc:
            st.error(f"Kunde inte läsa databasstatistik: {exc}")
            return

        count_cols = st.columns(len(DB_COLLECTIONS))
        for idx, name in enumerate(DB_COLLECTIONS):
            count_cols[idx].metric(name, counts.get(name, 0))

        selected_collection = st.selectbox(
            "Collection",
            options=DB_COLLECTIONS,
            index=2,
            key="db_inspector_collection",
        )
        limit = st.slider(
            "Antal dokument att visa",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            key="db_inspector_limit",
        )

        if st.button("Ladda dokument", key="db_inspector_load"):
            try:
                docs = _fetch_collection_docs(selected_collection, limit)
            except Exception as exc:
                st.error(f"Kunde inte läsa collection '{selected_collection}': {exc}")
                return

            st.success(f"Hämtade {len(docs)} dokument från {selected_collection}.")
            st.json(docs)


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _count_match_type(rows: list[dict[str, Any]], match_type: str) -> int:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("match_type", "")).strip().lower() == match_type:
            return _coerce_int(row.get("count", 0))
    return 0


@st.cache_data(show_spinner=False, ttl=90)
def _fetch_party_summary(party_key: str) -> dict[str, Any]:
    db = mcp_server.get_db()
    query = """
    LET party = DOCUMENT(CONCAT("parties/", @party_key))

    LET politicians_for_party = (
        FOR p IN politicians
            FILTER p.party_key == @party_key
            SORT p.name ASC
            RETURN p
    )

    LET promises_count = LENGTH(
        FOR pr IN promises
            FILTER pr.party_key == @party_key
            RETURN 1
    )

    LET promises_by_category = (
        FOR pr IN promises
            FILTER pr.party_key == @party_key
            LET category = TRIM(pr.category || "")
            COLLECT category_name = (category != "" ? category : "Ovrigt") WITH COUNT INTO count
            SORT count DESC, category_name ASC
            RETURN {category: category_name, count: count}
    )

    LET politician_keys = (
        FOR p IN politicians
            FILTER p.party_key == @party_key
            RETURN p._key
    )

    LET analysis_keys = (
        FOR a IN analyses
            FILTER a.politician_key IN politician_keys
            RETURN a._key
    )

    LET analyses_count = LENGTH(analysis_keys)

    LET analyses_by_platform = (
        FOR a IN analyses
            FILTER a._key IN analysis_keys
            LET platform = TRIM(a.platform || "")
            COLLECT platform_name = (platform != "" ? platform : "okänd") WITH COUNT INTO count
            SORT count DESC, platform_name ASC
            RETURN {platform: platform_name, count: count}
    )

    LET analyses_by_category = (
        FOR a IN analyses
            FILTER a._key IN analysis_keys
            LET category = TRIM(a.category || "")
            COLLECT category_name = (category != "" ? category : "Ovrigt") WITH COUNT INTO count
            SORT count DESC, category_name ASC
            RETURN {category: category_name, count: count}
    )

    LET comparisons_count = LENGTH(
        FOR c IN comparisons
            FILTER PARSE_IDENTIFIER(c._from).key IN analysis_keys
            RETURN 1
    )

    LET match_types = (
        FOR c IN comparisons
            FILTER PARSE_IDENTIFIER(c._from).key IN analysis_keys
            LET match_type = TRIM(c.match_type || "")
            COLLECT match_type_name = (match_type != "" ? match_type : "unknown") WITH COUNT INTO count
            SORT count DESC, match_type_name ASC
            RETURN {match_type: match_type_name, count: count}
    )

    LET politician_stats = (
        FOR p IN politicians
            FILTER p.party_key == @party_key
            LET politician_analysis_keys = (
                FOR a IN analyses
                    FILTER a.politician_key == p._key
                    RETURN a._key
            )
            LET politician_comparison_count = LENGTH(
                FOR c IN comparisons
                    FILTER PARSE_IDENTIFIER(c._from).key IN politician_analysis_keys
                    RETURN 1
            )
            SORT p.name ASC
            RETURN {
                politician_key: p._key,
                name: p.name,
                analyses_count: LENGTH(politician_analysis_keys),
                comparisons_count: politician_comparison_count
            }
    )

    LET top_promises = (
        FOR pr IN promises
            FILTER pr.party_key == @party_key
            LET normalized_category = TRIM(pr.category || "") != "" ? TRIM(pr.category || "") : "Ovrigt"
            LET category_weight = FIRST(
                FOR row IN promises_by_category
                    FILTER row.category == normalized_category
                    RETURN row.count
            ) || 0
            LET mention_raw = pr.mention_count
            LET mention_count = (IS_NUMBER(mention_raw) && mention_raw > 0) ? mention_raw : 1
            LET source_raw = pr.source_count
            LET inferred_source_count = LENGTH(IS_ARRAY(pr.source_urls) ? pr.source_urls : [])
                + (TRIM(pr.source_url || "") != "" ? 1 : 0)
            LET source_count = (IS_NUMBER(source_raw) && source_raw >= 0) ? source_raw : inferred_source_count
            LET emphasis_score = (mention_count * 100) + (category_weight * 10) + source_count
            SORT emphasis_score DESC, mention_count DESC, category_weight DESC, source_count DESC, pr.created_at DESC
            LIMIT 10
            RETURN {
                promise_key: pr._key,
                text: pr.text,
                category: normalized_category,
                source_url: pr.source_url,
                source_count: source_count,
                created_at: pr.created_at,
                mention_count: mention_count,
                category_weight: category_weight,
                emphasis_score: emphasis_score
            }
    )

    RETURN {
        party: party,
        politician_count: LENGTH(politicians_for_party),
        promises_count: promises_count,
        promises_by_category: promises_by_category,
        analyses_count: analyses_count,
        analyses_by_platform: analyses_by_platform,
        analyses_by_category: analyses_by_category,
        comparisons_count: comparisons_count,
        match_types: match_types,
        politician_stats: politician_stats,
        top_promises: top_promises
    }
    """
    cursor = db.aql.execute(query, bind_vars={"party_key": party_key})
    rows = list(cursor)
    if not rows:
        return {}
    return rows[0] if isinstance(rows[0], dict) else {}


@st.cache_data(show_spinner=False, ttl=90)
def _fetch_party_promises_for_category(party_key: str, category: str) -> list[dict[str, Any]]:
    if not str(party_key or "").strip():
        return []

    normalized_category = str(category or "").strip()
    db = mcp_server.get_db()
    query = """
    FOR pr IN promises
        FILTER pr.party_key == @party_key
        LET normalized = TRIM(pr.category || "")
        FILTER @category == "Ovrigt"
            ? (normalized == "" OR normalized == "Ovrigt")
            : normalized == @category
        SORT TO_NUMBER(pr.mention_count || 1) DESC, pr.created_at DESC
        RETURN {
            promise_key: pr._key,
            text: pr.text,
            category: (normalized != "" ? normalized : "Ovrigt"),
            source_url: pr.source_url,
            mention_count: TO_NUMBER(pr.mention_count || 1),
            source_count: TO_NUMBER(pr.source_count || 0),
            created_at: pr.created_at
        }
    """
    cursor = db.aql.execute(
        query,
        bind_vars={"party_key": party_key, "category": normalized_category or "Ovrigt"},
    )
    return [row for row in cursor if isinstance(row, dict)]


def _extract_selected_row_indices(selection_obj: Any) -> list[int]:
    rows: list[Any] = []
    if isinstance(selection_obj, dict):
        rows = selection_obj.get("selection", {}).get("rows", []) or []
    else:
        selection = getattr(selection_obj, "selection", None)
        if selection is not None:
            rows = getattr(selection, "rows", []) or []

    selected: list[int] = []
    for item in rows:
        try:
            selected.append(int(item))
        except Exception:
            continue
    return selected


def _render_party_tabs() -> None:
    with st.expander("Partiöversikt (flikar)", expanded=True):
        refresh_col, _ = st.columns([1, 5])
        with refresh_col:
            if st.button("Uppdatera partiöversikt", key="refresh_party_tabs"):
                _fetch_party_summary.clear()
                _fetch_party_promises_for_category.clear()

        try:
            parties_raw = _call_tool(mcp_server.list_parties, {})
        except Exception as exc:
            st.error(f"Kunde inte läsa partier: {exc}")
            return

        parties = [p for p in parties_raw if isinstance(p, dict) and str(p.get("_key", "")).strip()] if isinstance(parties_raw, list) else []
        if not parties:
            st.info("Inga partier ännu. Lägg till ett parti först.")
            return

        parties.sort(key=lambda p: str(p.get("name", "")).lower())
        labels: list[str] = []
        for party in parties:
            name = str(party.get("name", "Okänt parti")).strip() or "Okänt parti"
            abbreviation = str(party.get("abbreviation", "")).strip()
            labels.append(f"{name} ({abbreviation})" if abbreviation else name)

        tabs = st.tabs(labels)
        for tab, party in zip(tabs, parties):
            with tab:
                party_key = str(party.get("_key", "")).strip()
                summary: dict[str, Any] = {}
                if party_key:
                    try:
                        summary = _fetch_party_summary(party_key)
                    except Exception as exc:
                        st.error(f"Kunde inte läsa partiöversikt: {exc}")
                        continue

                party_doc = summary.get("party") if isinstance(summary, dict) else None
                if not isinstance(party_doc, dict):
                    party_doc = party

                block = str(party_doc.get("block", "")).strip()
                website = str(party_doc.get("website_url", "")).strip()
                if block or website:
                    details: list[str] = []
                    if block:
                        details.append(f"Block: `{block}`")
                    if website:
                        details.append(f"Webb: {website}")
                    st.caption(" • ".join(details))

                politician_count = _coerce_int(summary.get("politician_count", 0))
                promises_count = _coerce_int(summary.get("promises_count", 0))
                analyses_count = _coerce_int(summary.get("analyses_count", 0))
                comparisons_count = _coerce_int(summary.get("comparisons_count", 0))
                match_rows = summary.get("match_types", [])
                if not isinstance(match_rows, list):
                    match_rows = []

                supports_count = _count_match_type(match_rows, "supports")
                contradicts_count = _count_match_type(match_rows, "contradicts")
                unrelated_count = _count_match_type(match_rows, "unrelated")
                supports_share = (supports_count / comparisons_count * 100.0) if comparisons_count > 0 else 0.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Politiker", politician_count)
                m2.metric("Vallöften", promises_count)
                m3.metric("Analyser", analyses_count)
                m4.metric("Jämförelser", comparisons_count)

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Supports", supports_count)
                m6.metric("Contradicts", contradicts_count)
                m7.metric("Unrelated", unrelated_count)
                m8.metric("Stödandel", f"{supports_share:.1f}%")

                split_left, split_right = st.columns(2)

                with split_left:
                    st.markdown("**Löften per kategori**")
                    promises_by_category = summary.get("promises_by_category", [])
                    categories_col, promises_col = st.columns(2)
                    selected_category = ""

                    with categories_col:
                        if isinstance(promises_by_category, list) and promises_by_category:
                            table_key = f"party_promises_category_table_{party_key or 'unknown'}"
                            selection_state = st.dataframe(
                                promises_by_category,
                                hide_index=True,
                                use_container_width=True,
                                key=table_key,
                                on_select="rerun",
                                selection_mode="single-row",
                            )
                            selected_rows = _extract_selected_row_indices(selection_state)
                            if selected_rows:
                                selected_index = selected_rows[0]
                                if 0 <= selected_index < len(promises_by_category):
                                    selected_row = promises_by_category[selected_index]
                                    if isinstance(selected_row, dict):
                                        selected_category = str(selected_row.get("category", "") or "").strip() or "Ovrigt"
                        else:
                            st.caption("Inga löften registrerade ännu.")

                    with promises_col:
                        st.markdown("**Löften i vald kategori**")
                        if not selected_category:
                            st.caption("Klicka på en rad i tabellen för att visa löften.")
                        else:
                            try:
                                category_promises = _fetch_party_promises_for_category(party_key, selected_category)
                            except Exception as exc:
                                st.error(f"Kunde inte läsa löften för kategori: {exc}")
                                category_promises = []

                            if category_promises:
                                st.caption(f"{selected_category} ({len(category_promises)})")
                                with st.container(height=290, border=True):
                                    for idx, promise in enumerate(category_promises, start=1):
                                        text = str(promise.get("text", "") or "").strip()
                                        source_url = str(promise.get("source_url", "") or "").strip()
                                        mention_count = _coerce_int(promise.get("mention_count", 1))
                                        st.markdown(f"**{idx}.** {_promise_short_summary(text)}")
                                        st.caption(f"Omnämnanden: {mention_count}")
                                        if source_url:
                                            st.caption(source_url)
                                        if idx < len(category_promises):
                                            st.divider()
                            else:
                                st.caption(f"Inga löften hittades i kategorin `{selected_category}`.")

                    st.markdown("**Analyser per kategori**")
                    analyses_by_category = summary.get("analyses_by_category", [])
                    if isinstance(analyses_by_category, list) and analyses_by_category:
                        st.dataframe(analyses_by_category, hide_index=True, use_container_width=True)
                    else:
                        st.caption("Inga analyser registrerade ännu.")

                with split_right:
                    st.markdown("**Analyser per plattform**")
                    analyses_by_platform = summary.get("analyses_by_platform", [])
                    if isinstance(analyses_by_platform, list) and analyses_by_platform:
                        st.dataframe(analyses_by_platform, hide_index=True, use_container_width=True)
                    else:
                        st.caption("Inga plattformsdata ännu.")

                    st.markdown("**Jämförelser per match-typ**")
                    if match_rows:
                        st.dataframe(match_rows, hide_index=True, use_container_width=True)
                    else:
                        st.caption("Inga jämförelser registrerade ännu.")

                st.markdown("**10 viktigaste löften (partiets egen betoning)**")
                st.caption("Rangordnas efter hur ofta löftet återkommer i källmaterialet och hur centralt området verkar vara för partiet.")
                top_promises = summary.get("top_promises", [])
                if isinstance(top_promises, list) and top_promises:
                    with st.container(height=340, border=True):
                        for idx, promise in enumerate(top_promises, start=1):
                            if not isinstance(promise, dict):
                                continue
                            text = str(promise.get("text", "") or "").strip()
                            category = str(promise.get("category", "") or "").strip() or "Ovrigt"
                            mention_count = _coerce_int(promise.get("mention_count", 1))
                            source_count = _coerce_int(promise.get("source_count", 0))
                            category_weight = _coerce_int(promise.get("category_weight", 0))
                            source_url = str(promise.get("source_url", "") or "").strip()

                            st.markdown(f"**{idx}. {category}**")
                            st.write(_promise_short_summary(text))
                            meta = f"Omnämnanden: {mention_count} • Källor: {source_count} • Kategorityngd: {category_weight}"
                            st.caption(meta)
                            if source_url:
                                st.caption(source_url)
                            if idx < len(top_promises):
                                st.divider()
                else:
                    st.caption("Inga löften att ranka ännu.")

                st.markdown("**Politiker i partiet**")
                politician_stats = summary.get("politician_stats", [])
                if isinstance(politician_stats, list) and politician_stats:
                    st.dataframe(politician_stats, hide_index=True, use_container_width=True)
                else:
                    st.caption("Inga politiker registrerade för partiet ännu.")


def _render_scrape_result(result: Any) -> None:
    if not isinstance(result, dict):
        st.json(result)
        return

    promises = result.get("promises", [])
    errors = result.get("errors", [])
    pages_processed = _coerce_int(result.get("pages_processed", 0))
    urls_processed = _coerce_int(result.get("urls_processed", 0))
    extracted_count = _coerce_int(result.get("promises_extracted", 0))
    duplicates_skipped = _coerce_int(result.get("duplicates_skipped", 0))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("URL:er", urls_processed)
    m2.metric("Sidor", pages_processed)
    m3.metric("Nya löften", extracted_count)
    m4.metric("Dubbletter", duplicates_skipped)
    m5.metric("Fel", len(errors) if isinstance(errors, list) else 0)

    if isinstance(promises, list) and promises:
        st.markdown("**Löften (i ordningen de hittades)**")
        st.caption("Visas som korta utdrag ur originaltexten för att vara enkla men sanningsenliga.")
        with st.container(height=420, border=True):
            total = len(promises)
            for idx, promise in enumerate(promises, start=1):
                if not isinstance(promise, dict):
                    continue
                promise_text = str(promise.get("text", "")).strip()
                short_summary = _promise_short_summary(promise_text)
                category = str(promise.get("category", "")).strip() or "Ovrigt"
                source_url = str(promise.get("source_url", "")).strip()
                header = f"**{idx}/{total} · {category}**"
                st.markdown(header)
                st.write(short_summary or "(tomt löfte)")
                if promise_text and short_summary and promise_text != short_summary:
                    with st.expander("Visa originaltext", expanded=False):
                        st.write(promise_text)
                if source_url:
                    st.caption(source_url)
                if idx < total:
                    st.divider()
    else:
        st.info("Inga nya löften extraherades i den senaste körningen.")

    if isinstance(errors, list) and errors:
        with st.expander(f"Fel ({len(errors)})", expanded=False):
            st.json(errors)

    with st.expander("Rått resultat (JSON)", expanded=False):
        st.json(result)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _promise_short_summary(text: str, max_chars: int = 220) -> str:
    """
    Skapa en kort och sanningsenlig sammanfattning genom att ta ett utdrag
    ur originaltexten (ingen fri omformulering).
    """
    cleaned = _compact_whitespace(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]

    for sentence in sentences:
        if len(sentence) > max_chars:
            continue
        if re.search(r"\d", sentence):
            return sentence

    for sentence in sentences:
        if 40 <= len(sentence) <= max_chars:
            return sentence

    clipped = cleaned[:max_chars].rstrip(" ,;:-")
    return f"{clipped}..."


def _fetch_party_promises_live(party_key: str) -> list[dict[str, Any]]:
    if not str(party_key or "").strip():
        return []
    try:
        payload = _call_tool(
            mcp_server.get_promises,
            {"party_key": str(party_key).strip(), "category": ""},
        )
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    rows = [row for row in payload if isinstance(row, dict)]
    rows.sort(
        key=lambda row: str(row.get("created_at", "") or ""),
        reverse=True,
    )
    return rows


def _render_live_promises_panel(
    party_key: str,
    baseline_count: int = 0,
    max_items: int = 30,
) -> None:
    promises = _fetch_party_promises_live(party_key)
    total = len(promises)
    new_since_start = max(0, total - max(0, int(baseline_count or 0)))

    st.markdown("**Löften hittills i databasen**")
    k1, k2 = st.columns(2)
    k1.metric("Totalt", total)
    k2.metric("Nya i denna körning", new_since_start)

    if not promises:
        st.caption("Inga löften sparade ännu.")
        return

    with st.container(height=320, border=True):
        for idx, promise in enumerate(promises[:max_items], start=1):
            text = str(promise.get("text", "") or "").strip()
            category = str(promise.get("category", "") or "").strip() or "Ovrigt"
            source = str(promise.get("source_url", "") or "").strip()
            created = str(promise.get("created_at", "") or "").strip()
            st.markdown(f"**{idx}. {category}**")
            st.write(_promise_short_summary(text))
            meta_bits: list[str] = []
            if source:
                meta_bits.append(source)
            if created:
                meta_bits.append(created)
            if meta_bits:
                st.caption(" • ".join(meta_bits))
            if idx < min(len(promises), max_items):
                st.divider()


TOOL_MAP: dict[str, tuple[Callable, bool]] = {
    "add_party": (mcp_server.add_party, False),
    "list_parties": (mcp_server.list_parties, False),
    "get_party_details": (mcp_server.get_party_details, False),
    "add_politician": (mcp_server.add_politician, False),
    "list_politicians": (mcp_server.list_politicians, False),
    "get_politician_details": (mcp_server.get_politician_details, False),
    "add_promise": (mcp_server.add_promise, False),
    "get_promises": (mcp_server.get_promises, False),
    "delete_promise": (mcp_server.delete_promise, False),
    "scrape_party_promises": (mcp_server.scrape_party_promises, True),
    "save_analysis": (mcp_server.save_analysis, False),
    "search_analyses": (mcp_server.search_analyses, False),
    "compare_content_vs_promise": (mcp_server.compare_content_vs_promise, True),
    "get_consistency_report": (mcp_server.get_consistency_report, False),
}

READ_TOOL_SET = {
    "list_parties",
    "get_party_details",
    "list_politicians",
    "get_politician_details",
    "get_promises",
    "search_analyses",
    "get_consistency_report",
}
WRITE_TOOL_SET = {
    "add_party",
    "add_politician",
    "add_promise",
    "delete_promise",
    "save_analysis",
    "scrape_party_promises",
}


def _tool_schema_text(allowed_tools: set[str]) -> str:
    schema = {
        "add_party": {"name": "str", "abbreviation": "str", "block": "str", "website_url": "str"},
        "list_parties": {},
        "get_party_details": {"party_key": "str"},
        "add_politician": {"name": "str", "party_key": "str", "tiktok": "str", "instagram": "str", "twitter": "str"},
        "list_politicians": {"party_key": "str"},
        "get_politician_details": {"politician_key": "str"},
        "add_promise": {"party_key": "str", "text": "str", "category": "str", "source_url": "str", "source_name": "str", "date": "str"},
        "get_promises": {"party_key": "str", "category": "str"},
        "delete_promise": {"promise_key": "str"},
        "scrape_party_promises": {
            "party_key": "str",
            "urls": ["str"],
            "crawl": "bool",
            "max_pages_per_url": "int",
            "max_depth": "int",
        },
        "save_analysis": {"politician_key": "str", "video_file": "str", "transcription": "str", "summary": "str", "category": "str", "political_color": "str", "platform": "str"},
        "search_analyses": {"politician_key": "str", "category": "str", "platform": "str", "search_text": "str"},
        "compare_content_vs_promise": {"analysis_key": "str", "promise_key": "str"},
        "get_consistency_report": {"politician_key": "str"},
    }
    filtered = {name: schema[name] for name in sorted(allowed_tools) if name in schema}
    return json.dumps(filtered, ensure_ascii=False, indent=2)


def _truncate_tool_result(result: Any, max_chars: int = 3500) -> str:
    text = json.dumps(result, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _run_qwen_tool_agent(user_prompt: str, allow_writes: bool) -> dict[str, Any]:
    allowed_tools = set(READ_TOOL_SET)
    if allow_writes:
        allowed_tools.update(WRITE_TOOL_SET)

    system_prompt = f"""Du är en verktygsagent som får anropa MCP-tools i en databas.
Du ska antingen anropa exakt ett tool-steg eller ge ett slutligt svar.
Svara ALLTID med EN JSON-objekt-sträng:

Stegformat:
{{"action":"call_tool","tool":"tool_namn","args":{{...}},"reason":"kort motivering"}}

Slutformat:
{{"action":"final","answer":"slutsvar på svenska"}}

Regler:
- Använd ENDAST tools i listan nedan.
- Hitta aldrig på nycklar. Om nycklar saknas, hämta dem först med list/get-tools.
- Vid databasändringar, gör minsta möjliga ändring.
- Om användaren ber om rensning av löften, läs först löften och radera sedan med delete_promise.

Tillåtna tools + args-schema:
{_tool_schema_text(allowed_tools)}
"""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    performed_actions: list[dict[str, Any]] = []

    for _ in range(8):
        raw = _chat_with_ollama(messages)
        decision = _extract_first_json_object(raw)
        if not decision:
            return {
                "answer": raw,
                "actions": performed_actions,
            }

        action = str(decision.get("action", "")).strip().lower()
        if action == "final":
            return {
                "answer": str(decision.get("answer", "")).strip() or "Klart.",
                "actions": performed_actions,
            }

        if action != "call_tool":
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Ogiltigt action-värde. Svara med JSON enligt formatet.",
                }
            )
            continue

        tool_name = str(decision.get("tool", "")).strip()
        args = decision.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if tool_name not in allowed_tools or tool_name not in TOOL_MAP:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool '{tool_name}' är inte tillåtet. Välj ett giltigt tool.",
                }
            )
            continue

        fn, is_async = TOOL_MAP[tool_name]
        try:
            result = _call_tool(fn, args, is_async=is_async)
        except Exception as exc:
            result = {"error": str(exc)}

        performed_actions.append({"tool": tool_name, "args": args, "result": result})
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool result ({tool_name}): {_truncate_tool_result(result)}\n"
                    "Fortsätt. Om uppgiften är klar, svara med action=final."
                ),
            }
        )

    return {
        "answer": "Jag nådde max antal agentsteg. Försök med en mer avgränsad begäran.",
        "actions": performed_actions,
    }


_VIDEO_UPLOAD_DIR = Path("videos/uploads")
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(name or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"


def _save_uploaded_video_file(uploaded_file) -> Path:
    _VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = _safe_filename(getattr(uploaded_file, "name", "upload.mp4"))
    suffix = Path(original_name).suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        suffix = ".mp4"
    stem = Path(original_name).stem
    target_path = _VIDEO_UPLOAD_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    with open(target_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target_path


def _score_promise_relevance(clip_row: dict[str, Any], promise: dict[str, Any]) -> int:
    clip_text = " ".join(
        [
            str(clip_row.get("innehall", "") or ""),
            str(clip_row.get("transkribering", "") or "")[:2400],
            str(clip_row.get("kategori", "") or ""),
        ]
    )
    promise_text = " ".join(
        [
            str(promise.get("text", "") or ""),
            str(promise.get("category", "") or ""),
        ]
    )
    clip_tokens = tokenize(clip_text)
    promise_tokens = tokenize(promise_text)
    overlap = len(clip_tokens & promise_tokens)
    # Liten bonus om kategorifält råkar överlappa.
    cat_a = str(clip_row.get("kategori", "") or "").strip().lower()
    cat_b = str(promise.get("category", "") or "").strip().lower()
    category_bonus = 1 if cat_a and cat_b and (cat_a in cat_b or cat_b in cat_a) else 0
    return overlap + category_bonus


def _select_candidate_promises(
    clip_row: dict[str, Any],
    promises: list[dict[str, Any]],
    max_promises_per_clip: int,
) -> list[dict[str, Any]]:
    if not promises:
        return []
    if max_promises_per_clip <= 0 or max_promises_per_clip >= len(promises):
        return promises

    scored = [
        (promise, _score_promise_relevance(clip_row, promise))
        for promise in promises
        if str(promise.get("_key", "")).strip()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored:
        return []

    # Om alla scorer är 0, ta första N förutsägbart.
    if scored[0][1] <= 0:
        return [item[0] for item in scored[:max_promises_per_clip]]

    return [item[0] for item in scored[:max_promises_per_clip]]


def _run_video_policy_match(
    politician_key: str,
    uploaded_file,
    platform: str,
    custom_categories: str,
    max_promises_per_clip: int,
) -> dict[str, Any]:
    saved_path = _save_uploaded_video_file(uploaded_file)

    clip_rows = _run_async(analyze_video_file(saved_path, categories=(custom_categories or None)))
    if not clip_rows:
        return {
            "saved_video_path": str(saved_path),
            "clip_count": 0,
            "analyses_saved": [],
            "comparisons_saved": [],
            "comparison_errors": [],
            "consistency_report": {},
        }

    politician_details = _call_tool(
        mcp_server.get_politician_details,
        {"politician_key": politician_key},
    )
    if isinstance(politician_details, dict) and politician_details.get("error"):
        raise RuntimeError(str(politician_details.get("error")))

    promises = politician_details.get("promises", []) if isinstance(politician_details, dict) else []
    promises = [p for p in promises if isinstance(p, dict) and str(p.get("_key", "")).strip()]

    analyses_saved: list[dict[str, Any]] = []
    comparisons_saved: list[dict[str, Any]] = []
    comparison_errors: list[dict[str, Any]] = []

    for idx, clip_row in enumerate(clip_rows, start=1):
        analysis_doc = _call_tool(
            mcp_server.save_analysis,
            {
                "politician_key": politician_key,
                "video_file": f"{saved_path.name}#clip{idx}",
                "transcription": str(clip_row.get("transkribering", "") or ""),
                "summary": str(clip_row.get("innehall", "") or ""),
                "category": str(clip_row.get("kategori", "") or "Annat"),
                "political_color": str(clip_row.get("politiskt_farg", "") or "neutral"),
                "platform": platform,
            },
        )
        analyses_saved.append(analysis_doc if isinstance(analysis_doc, dict) else {"raw": analysis_doc})

        analysis_key = str((analysis_doc or {}).get("_key", "")).strip() if isinstance(analysis_doc, dict) else ""
        if not analysis_key or not promises:
            continue

        selected_promises = _select_candidate_promises(clip_row, promises, max_promises_per_clip)
        for promise in selected_promises:
            promise_key = str(promise.get("_key", "")).strip()
            if not promise_key:
                continue
            try:
                comparison_doc = _call_tool(
                    mcp_server.compare_content_vs_promise,
                    {"analysis_key": analysis_key, "promise_key": promise_key},
                    is_async=True,
                )
                comparisons_saved.append(
                    comparison_doc if isinstance(comparison_doc, dict) else {"raw": comparison_doc}
                )
            except Exception as exc:  # pragma: no cover - extern modell/db
                comparison_errors.append(
                    {
                        "analysis_key": analysis_key,
                        "promise_key": promise_key,
                        "error": str(exc),
                    }
                )

    consistency_report = _call_tool(
        mcp_server.get_consistency_report,
        {"politician_key": politician_key},
    )

    return {
        "saved_video_path": str(saved_path),
        "clip_count": len(clip_rows),
        "analyses_saved": analyses_saved,
        "comparisons_saved": comparisons_saved,
        "comparison_errors": comparison_errors,
        "consistency_report": consistency_report,
    }


def _init_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT}
        ]
    st.session_state.setdefault("chat_widget_open", False)


def _chat_with_ollama(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        response.raise_for_status()

    data = response.json()
    text = str(data.get("message", {}).get("content", "")).strip()
    return text or "Jag fick inget svar från modellen."


def _messages_for_request() -> list[dict[str, str]]:
    messages = st.session_state.get("chat_messages", [])
    if len(messages) <= 18:
        return messages

    system = [messages[0]]
    tail = [m for m in messages[1:] if m.get("role") in {"user", "assistant"}][-16:]
    return system + tail


def _render_local_chat() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(#chat-open-panel-anchor) [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            opacity: 1 !important;
            backdrop-filter: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not bool(st.session_state.get("chat_widget_open", False)):
        if st.button("💬 Öppna chatt", key="chat_widget_open_button", use_container_width=True):
            st.session_state["chat_widget_open"] = True
            st.rerun()
        return

    with st.container(border=True):
        st.markdown('<div id="chat-open-panel-anchor"></div>', unsafe_allow_html=True)
        top_left, top_mid, top_right = st.columns([3, 1, 1])
        with top_left:
            st.caption(f"Qwen assistent • `{CHAT_MODEL}`")
        with top_mid:
            reset_clicked = st.button("Nytt", use_container_width=True, key="chat_reset_button")
        with top_right:
            minimize_clicked = st.button("Minimera", use_container_width=True, key="chat_minimize_button")

        if minimize_clicked:
            st.session_state["chat_widget_open"] = False
            st.rerun()
        if reset_clicked:
            st.session_state["chat_messages"] = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT}
            ]
            st.rerun()

        with st.expander("Inställningar", expanded=False):
            agent_mode_col, write_mode_col = st.columns(2)
            with agent_mode_col:
                agent_mode = st.toggle("Agentläge", value=True, key="chat_agent_mode")
            with write_mode_col:
                allow_writes = st.toggle("Tillåt skrivning", value=True, key="chat_agent_allow_writes")
            st.caption(f"Host: `{OLLAMA_HOST}`")

        with st.container(height=360, border=True):
            for message in st.session_state["chat_messages"]:
                role = str(message.get("role", "")).strip()
                content = str(message.get("content", "")).strip()
                if role == "system" or not content:
                    continue
                with st.chat_message("assistant" if role == "assistant" else "user"):
                    st.markdown(content)
                    if role == "assistant":
                        actions = message.get("actions", [])
                        if isinstance(actions, list) and actions:
                            with st.expander(f"Agentsteg ({len(actions)})", expanded=False):
                                st.json(actions)

        with st.form("local_chat_form", clear_on_submit=True):
            prompt = st.text_area(
                "Meddelande",
                value="",
                height=90,
                label_visibility="collapsed",
                placeholder="Skriv till Qwen...",
            )
            send = st.form_submit_button("Skicka", use_container_width=True)

    if not send:
        return

    prompt = str(prompt or "").strip()
    if not prompt:
        return

    agent_mode = bool(st.session_state.get("chat_agent_mode", True))
    allow_writes = bool(st.session_state.get("chat_agent_allow_writes", True))

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    actions: list[Any] = []
    with st.spinner("Qwen svarar..."):
        try:
            if agent_mode:
                agent_result = _run_qwen_tool_agent(prompt, allow_writes=allow_writes)
                reply = str(agent_result.get("answer", "")).strip() or "Klart."
                raw_actions = agent_result.get("actions", [])
                if isinstance(raw_actions, list):
                    actions = raw_actions
            else:
                reply = _chat_with_ollama(_messages_for_request())
        except Exception as exc:
            reply = f"Kunde inte nå Ollama på {OLLAMA_HOST}. Fel: {exc}"

    assistant_message: dict[str, Any] = {"role": "assistant", "content": reply}
    if actions:
        assistant_message["actions"] = actions
    st.session_state["chat_messages"].append(assistant_message)
    st.rerun()

with st.expander("Snabbflöde: Parti + scraping", expanded=True):
    try:
        parties = _call_tool(mcp_server.list_parties, {})
    except Exception:
        parties = []

    party_options = {
        f"{p.get('name', '')} ({p.get('abbreviation', '')}) [{p.get('_key', '')}]": p.get("_key", "")
        for p in (parties if isinstance(parties, list) else [])
        if p.get("_key")
    }
    party_placeholder = "Välj parti..."
    party_select_options = (
        [party_placeholder] + list(party_options.keys())
        if party_options
        else ["(inga partier ännu)"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Lägg till parti")
        with st.form("add_party_form"):
            party_name = st.text_input("Partinamn", value="")
            abbreviation = st.text_input("Förkortning", value="")
            block = st.text_input("Block", value="")
            website_url = st.text_input("Webbplats", value="")
            add_party_submit = st.form_submit_button("Lägg till parti")

        if add_party_submit:
            try:
                result = _call_tool(
                    mcp_server.add_party,
                    {
                        "name": party_name,
                        "abbreviation": abbreviation,
                        "block": block,
                        "website_url": website_url,
                    },
                )
                st.success("Parti sparat")
                st.json(result)
            except Exception as exc:
                st.error(f"Kunde inte lägga till parti: {exc}")

        st.subheader("2) Lägg till politiker")
        if not party_options:
            st.caption("Lägg till minst ett parti först.")
        with st.form("add_politician_form", clear_on_submit=True):
            politician_name = st.text_input("Namn", value="")
            politician_party_label = st.selectbox(
                "Parti",
                options=party_select_options,
                index=0,
                key="quick_add_politician_party",
            )
            tiktok_handle = st.text_input("TikTok (valfritt)", value="")
            instagram_handle = st.text_input("Instagram (valfritt)", value="")
            twitter_handle = st.text_input("X/Twitter (valfritt)", value="")
            add_politician_submit = st.form_submit_button(
                "Lägg till politiker",
                disabled=not bool(party_options),
            )

        if add_politician_submit:
            selected_party_for_politician = party_options.get(politician_party_label, "")
            if not str(politician_name or "").strip():
                st.warning("Fyll i politikerns namn.")
            elif not selected_party_for_politician:
                st.warning("Välj parti för politikern.")
            else:
                try:
                    result = _call_tool(
                        mcp_server.add_politician,
                        {
                            "name": str(politician_name).strip(),
                            "party_key": str(selected_party_for_politician).strip(),
                            "tiktok": str(tiktok_handle).strip(),
                            "instagram": str(instagram_handle).strip(),
                            "twitter": str(twitter_handle).strip(),
                        },
                    )
                    st.success("Politiker sparad")
                    st.json(result)
                except Exception as exc:
                    st.error(f"Kunde inte lägga till politiker: {exc}")

    with col2:
        st.subheader("3) Skrapa vallöften")
        selected_label = st.selectbox(
            "Välj parti",
            options=party_select_options,
            index=0,
            key="quick_scrape_selected_party",
        )
        selected_party_key = party_options.get(selected_label, "")

        urls_text = st.text_area(
            "URL:er (en per rad)",
            value="",
            height=120,
        )
        crawl_enabled = st.checkbox(
            "Crawla interna undersidor automatiskt",
            value=True,
            key="quick_scrape_crawl_enabled",
        )
        scrape_col1, scrape_col2 = st.columns(2)
        with scrape_col1:
            max_pages_per_url = st.slider(
                "Max sidor per URL",
                min_value=1,
                max_value=80,
                value=25,
                step=1,
                key="quick_scrape_max_pages_per_url",
            )
        with scrape_col2:
            max_depth = st.slider(
                "Max länkdjup",
                min_value=0,
                max_value=4,
                value=2,
                step=1,
                key="quick_scrape_max_depth",
            )

        active_scrape_job = _read_active_scrape_job()
        if isinstance(active_scrape_job, dict):
            job_status = str(active_scrape_job.get("status", "")).strip().lower()
            if _is_stale_scrape_job(active_scrape_job):
                started_raw = str(active_scrape_job.get("started_at", "") or active_scrape_job.get("created_at", "")).strip()
                if started_raw:
                    st.warning(f"Ett tidigare scrapingjobb verkar ha fastnat (start: {started_raw}). Jobblåset återställs.")
                else:
                    st.warning("Ett tidigare scrapingjobb verkar ha fastnat. Jobblåset återställs.")
                _clear_active_scrape_job()
                active_scrape_job = None
            elif job_status in {"completed", "failed"}:
                if job_status == "completed":
                    result_payload = active_scrape_job.get("result")
                    if isinstance(result_payload, dict):
                        st.session_state["last_scrape_result"] = result_payload
                        st.success("Bakgrundsjobb klart: vallöften är uppdaterade.")
                else:
                    st.error(f"Bakgrundsjobb misslyckades: {active_scrape_job.get('error', 'okänt fel')}")
                _clear_active_scrape_job()
                active_scrape_job = None

        active_is_running = isinstance(active_scrape_job, dict) and str(active_scrape_job.get("status", "")).strip().lower() in {"queued", "running"}
        if active_is_running:
            running_since = str(active_scrape_job.get("started_at", "") or active_scrape_job.get("created_at", "")).strip()
            running_age_seconds = _job_age_seconds(active_scrape_job)
            timeout_seconds = _coerce_int(active_scrape_job.get("timeout_seconds", 0))
            st.info("Scraping körs i bakgrunden. Du kan refresha sidan utan att avbryta jobbet.")
            if running_since:
                st.caption(f"Startad: `{running_since}`")
            if running_age_seconds > 0:
                elapsed_minutes = running_age_seconds // 60
                if timeout_seconds > 0:
                    timeout_minutes = max(1, timeout_seconds // 60)
                    st.caption(f"Körtid: `{elapsed_minutes} min` • Timeout: `{timeout_minutes} min`")
                    if running_age_seconds > timeout_seconds:
                        st.warning("Jobbet har passerat timeout och kan vara fastnat. Återställ och kör igen med lägre crawl-gränser.")
                else:
                    st.caption(f"Körtid: `{elapsed_minutes} min`")
            if st.button("Återställ låst scrapingjobb", key="clear_active_scrape_job"):
                _clear_active_scrape_job()
                st.rerun()
            active_party_key = str(active_scrape_job.get("party_key", "") or "").strip()
            baseline_count = _coerce_int(active_scrape_job.get("baseline_promises_count", 0))
            if active_party_key:
                _render_live_promises_panel(active_party_key, baseline_count=baseline_count)

        if st.button("Skrapa och spara löften", disabled=(not bool(selected_party_key) or active_is_running)):
            urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
            if not urls:
                st.warning("Lägg till minst en URL")
            else:
                try:
                    job_id = uuid4().hex
                    existing_promises = _fetch_party_promises_live(selected_party_key)
                    job = {
                        "job_id": job_id,
                        "status": "queued",
                        "created_at": _utc_now_iso(),
                        "party_key": selected_party_key,
                        "baseline_promises_count": len(existing_promises),
                        "urls": urls,
                        "crawl": bool(crawl_enabled),
                        "max_pages_per_url": int(max_pages_per_url),
                        "max_depth": int(max_depth),
                    }
                    _set_active_scrape_job(job)
                    _spawn_scrape_worker(
                        job_id=job_id,
                        party_key=selected_party_key,
                        urls=urls,
                        crawl=bool(crawl_enabled),
                        max_pages_per_url=int(max_pages_per_url),
                        max_depth=int(max_depth),
                    )
                    st.info("Scraping startad i bakgrunden.")
                except Exception as exc:
                    st.error(f"Scraping misslyckades: {exc}")

        if "last_scrape_result" in st.session_state:
            _render_scrape_result(st.session_state["last_scrape_result"])


# ── Bevakning (sociala medier via RSSHub) ────────────────────────

with st.expander("Bevakning (sociala medier via RSSHub)", expanded=False):
    st.caption(
        "Automatisk bevakning av politikers TikTok via RSSHub. "
        "Kräver att RSSHub körs lokalt via Docker."
    )

    bev_col1, bev_col2 = st.columns(2)

    with bev_col1:
        st.subheader("Bevakningsstatus")
        bev_politician_filter = st.text_input(
            "Filtrera per politiker-nyckel (valfritt)",
            value="",
            key="monitor_politician_filter",
        )
        if st.button("Visa status", key="monitor_show_status"):
            try:
                status = _call_tool(
                    mcp_server.get_feed_monitor_status,
                    {"politician_key": bev_politician_filter.strip()},
                )
                if isinstance(status, dict):
                    summary = status.get("summary", [])
                    if isinstance(summary, list) and summary:
                        st.dataframe(summary, hide_index=True, use_container_width=True)
                    else:
                        st.info("Ingen bevakningsdata hittad ännu.")

                    recent = status.get("recent_items", [])
                    if isinstance(recent, list) and recent:
                        st.markdown("**Senaste hämtade inlägg**")
                        for item in recent:
                            if not isinstance(item, dict):
                                continue
                            name = str(item.get("politician_name", "") or "")
                            platform = str(item.get("platform", "") or "")
                            title = str(item.get("item_title", "") or "")
                            url = str(item.get("item_url", "") or "")
                            item_status = str(item.get("status", "") or "")
                            fetched = str(item.get("fetched_at", "") or "")
                            st.markdown(f"**{name}** ({platform}) — {item_status}")
                            if title:
                                st.caption(title[:200])
                            if url:
                                st.caption(url)
                            if fetched:
                                st.caption(f"Hämtad: {fetched}")
                            st.divider()
                else:
                    st.json(status)
            except Exception as exc:
                st.error(f"Kunde inte hämta bevakningsstatus: {exc}")

    with bev_col2:
        st.subheader("Kör bevakning nu")
        st.caption("Kontrollera RSSHub-flöden och analysera nya inlägg")

        run_politician_key = st.text_input(
            "Specifik politiker-nyckel (valfritt, lämna tomt för alla)",
            value="",
            key="monitor_run_politician",
        )

        if st.button("Kör bevakning nu", key="monitor_run_now", type="primary"):
            with st.spinner("Kontrollerar flöden och analyserar nya inlägg..."):
                try:
                    result = _call_tool(
                        mcp_server.check_feeds_now,
                        {"politician_key": run_politician_key.strip()},
                        is_async=True,
                    )
                    if isinstance(result, dict):
                        if result.get("error"):
                            st.error(str(result["error"]))
                        else:
                            checked = result.get("politicians_checked", 0)
                            checked_at = result.get("checked_at", "")
                            st.success(
                                f"Bevakning klar! {checked} politiker kontrollerade."
                            )
                            if checked_at:
                                st.caption(f"Tid: {checked_at}")

                        for pol_result in result.get("results", []):
                            if not isinstance(pol_result, dict):
                                continue
                            pol_name = str(
                                pol_result.get("politician_name", "")
                            ).strip()
                            pol_status = str(
                                pol_result.get("status", "")
                            ).strip()
                            reason = str(pol_result.get("reason", "")).strip()

                            label = f"**{pol_name}**: {pol_status}"
                            if reason:
                                label += f" ({reason})"
                            st.markdown(label)

                            platforms = pol_result.get("platforms", {})
                            if isinstance(platforms, dict):
                                for pf_name, pf_data in platforms.items():
                                    if not isinstance(pf_data, dict):
                                        continue
                                    pf_status = str(pf_data.get("status", ""))
                                    new_items = pf_data.get("items_new", 0)
                                    total = pf_data.get("total_entries", 0)
                                    pf_error = str(
                                        pf_data.get("error", "")
                                    ).strip()
                                    line = (
                                        f"  {pf_name}: {pf_status} "
                                        f"({new_items} nya av {total} i flödet)"
                                    )
                                    if pf_error:
                                        line += f" — {pf_error}"
                                    st.caption(line)
                    else:
                        st.json(result)
                except Exception as exc:
                    st.error(f"Bevakning misslyckades: {exc}")


with st.expander("Video -> Match mot partiets politik", expanded=True):
    try:
        analyzer_root = get_video_analyzer_root()
        st.caption(f"Video-analyzer hittad: `{analyzer_root}`")
        analyzer_ready = True
    except Exception as exc:
        st.error(f"VIDEO_ANALYZER_PATH problem: {exc}")
        analyzer_ready = False

    try:
        politicians_data = _call_tool(mcp_server.list_politicians, {"party_key": ""})
    except Exception as exc:
        st.error(f"Kunde inte läsa politiker: {exc}")
        politicians_data = []

    politician_options: dict[str, str] = {}
    for row in politicians_data if isinstance(politicians_data, list) else []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("_key", "")).strip()
        if not key:
            continue
        name = str(row.get("name", "Okänd")).strip()
        party_key = str(row.get("party_key", "")).strip()
        label = f"{name} [{key}] · parti={party_key}"
        politician_options[label] = key

    selected_politician_label = st.selectbox(
        "Politiker",
        options=list(politician_options.keys()) if politician_options else ["(inga politiker ännu)"],
        key="video_match_politician",
    )
    selected_politician_key = politician_options.get(selected_politician_label, "")

    uploaded_video = st.file_uploader(
        "Ladda upp videoklipp",
        type=["mp4", "mov", "avi", "mkv", "webm", "flv"],
        key="video_match_upload",
    )

    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        platform_value = st.selectbox(
            "Plattform",
            options=["tiktok", "instagram", "youtube", "facebook", "x", "okänd"],
            index=0,
            key="video_match_platform",
        )
    with vcol2:
        max_promises_per_clip = st.slider(
            "Max löften per klipp",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            key="video_match_max_promises",
        )
    with vcol3:
        custom_categories = st.text_input(
            "Egna kategorier (valfritt)",
            value="",
            help="Kommaseparerat, skickas till video-analysen",
            key="video_match_categories",
        )

    run_video_match = st.button(
        "Analysera video och matcha mot partiets politik",
        disabled=not (analyzer_ready and selected_politician_key and uploaded_video is not None),
        key="video_match_run",
    )

    if run_video_match:
        with st.spinner("Kör videoanalys, sparar resultat och jämför mot vallöften..."):
            try:
                summary = _run_video_policy_match(
                    politician_key=selected_politician_key,
                    uploaded_file=uploaded_video,
                    platform=platform_value,
                    custom_categories=custom_categories,
                    max_promises_per_clip=max_promises_per_clip,
                )
                st.success("Videoklipp analyserat och matchat mot partiets politik.")
                k1, k2, k3 = st.columns(3)
                k1.metric("Klipprader", int(summary.get("clip_count", 0)))
                k2.metric("Sparade analyser", len(summary.get("analyses_saved", [])))
                k3.metric("Jämförelser", len(summary.get("comparisons_saved", [])))

                comparison_errors = summary.get("comparison_errors", [])
                if comparison_errors:
                    st.warning(f"{len(comparison_errors)} jämförelser misslyckades.")
                    with st.expander("Jämförelsefel", expanded=False):
                        st.json(comparison_errors)

                with st.expander("Konsistensrapport", expanded=True):
                    st.json(summary.get("consistency_report", {}))
                with st.expander("Sparade analyser", expanded=False):
                    st.json(summary.get("analyses_saved", []))
                with st.expander("Sparade jämförelser", expanded=False):
                    st.json(summary.get("comparisons_saved", []))
            except Exception as exc:
                st.error(f"Video-matchning misslyckades: {exc}")


_render_party_tabs()

with st.expander("Snabbvisning: partier, löften, politiker", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Lista partier"):
            try:
                st.json(_call_tool(mcp_server.list_parties, {}))
            except Exception as exc:
                st.error(str(exc))

    with c2:
        party_key_for_promises = st.text_input("party_key för get_promises", value="")
        if st.button("Visa löften"):
            if not party_key_for_promises.strip():
                st.warning("Ange party_key")
            else:
                try:
                    st.json(_call_tool(mcp_server.get_promises, {"party_key": party_key_for_promises.strip(), "category": ""}))
                except Exception as exc:
                    st.error(str(exc))

    with c3:
        if st.button("Lista politiker"):
            try:
                st.json(_call_tool(mcp_server.list_politicians, {"party_key": ""}))
            except Exception as exc:
                st.error(str(exc))


_render_db_inspector()

st.divider()
st.subheader(f"Tool Playground (alla {len(TOOL_MAP)} tools)")

selected_tool = st.selectbox("Tool", options=list(TOOL_MAP.keys()))
default_payloads = {
    "add_party": {"name": "", "abbreviation": "", "block": "", "website_url": ""},
    "list_parties": {},
    "get_party_details": {"party_key": ""},
    "add_politician": {"name": "", "party_key": "", "tiktok": "", "instagram": "", "twitter": ""},
    "list_politicians": {"party_key": ""},
    "get_politician_details": {"politician_key": ""},
    "add_promise": {"party_key": "", "text": "", "category": "", "source_url": "", "source_name": "", "date": ""},
    "get_promises": {"party_key": "", "category": ""},
    "delete_promise": {"promise_key": ""},
    "scrape_party_promises": {
        "party_key": "",
        "urls": [],
        "crawl": True,
        "max_pages_per_url": 25,
        "max_depth": 2,
    },
    "save_analysis": {"politician_key": "", "video_file": "", "transcription": "", "summary": "", "category": "", "political_color": "", "platform": ""},
    "search_analyses": {"politician_key": "", "category": "", "platform": "", "search_text": ""},
    "compare_content_vs_promise": {"analysis_key": "", "promise_key": ""},
    "get_consistency_report": {"politician_key": ""},
}

payload_raw = st.text_area(
    "JSON payload",
    value=json.dumps(default_payloads[selected_tool], ensure_ascii=False, indent=2),
    height=180,
)

if st.button("Kör tool"):
    try:
        payload = json.loads(payload_raw) if payload_raw.strip() else {}
        fn, is_async = TOOL_MAP[selected_tool]
        result = _call_tool(fn, payload, is_async=is_async)
        st.success("Klart")
        st.json(result)
    except Exception as exc:
        st.error(f"Tool-körning misslyckades: {exc}")

st.caption("Tips: Starta med add_party -> scrape_party_promises -> get_promises -> add_politician -> save_analysis -> compare_content_vs_promise -> get_consistency_report")

_init_chat_state()
_render_local_chat()
