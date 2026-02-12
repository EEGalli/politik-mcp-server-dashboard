import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
        "scrape_party_promises": {"party_key": "str", "urls": ["str"]},
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
    st.subheader("Lokal chatt (Qwen via Ollama)")
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.caption(f"Host: `{OLLAMA_HOST}`  •  Modell: `{CHAT_MODEL}`")
    with top_right:
        if st.button("Nytt samtal", use_container_width=True):
            st.session_state["chat_messages"] = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT}
            ]
            st.rerun()

    agent_mode_col, write_mode_col = st.columns(2)
    with agent_mode_col:
        agent_mode = st.toggle("Agentläge (Qwen kan anropa MCP-tools)", value=True, key="chat_agent_mode")
    with write_mode_col:
        allow_writes = st.toggle("Tillåt databasändringar", value=True, key="chat_agent_allow_writes")

    for message in st.session_state["chat_messages"]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role == "system" or not content:
            continue
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content)

    prompt = st.chat_input("Skriv till Qwen...")
    if not prompt:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Qwen svarar..."):
            try:
                if agent_mode:
                    agent_result = _run_qwen_tool_agent(prompt, allow_writes=allow_writes)
                    reply = str(agent_result.get("answer", "")).strip() or "Klart."
                    actions = agent_result.get("actions", [])
                    if actions:
                        with st.expander(f"Agentsteg ({len(actions)})", expanded=False):
                            st.json(actions)
                else:
                    reply = _chat_with_ollama(_messages_for_request())
            except Exception as exc:
                reply = f"Kunde inte nå Ollama på {OLLAMA_HOST}. Fel: {exc}"
        st.markdown(reply)

    st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


_init_chat_state()
_render_local_chat()
st.divider()

with st.expander("Snabbflöde: Parti + scraping", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Lägg till parti")
        with st.form("add_party_form"):
            party_name = st.text_input("Partinamn", value="Socialdemokraterna")
            abbreviation = st.text_input("Förkortning", value="S")
            block = st.text_input("Block", value="vänster")
            website_url = st.text_input("Webbplats", value="https://www.socialdemokraterna.se")
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

    with col2:
        st.subheader("2) Skrapa vallöften")
        try:
            parties = _call_tool(mcp_server.list_parties, {})
        except Exception:
            parties = []

        party_options = {
            f"{p.get('name', '')} ({p.get('abbreviation', '')}) [{p.get('_key', '')}]": p.get("_key", "")
            for p in (parties if isinstance(parties, list) else [])
            if p.get("_key")
        }

        selected_label = st.selectbox(
            "Välj parti",
            options=list(party_options.keys()) if party_options else ["(inga partier ännu)"],
        )
        selected_party_key = party_options.get(selected_label, "")

        urls_text = st.text_area(
            "URL:er (en per rad)",
            value="https://www.socialdemokraterna.se/var-politik",
            height=120,
        )

        if st.button("Skrapa och spara löften", disabled=not bool(selected_party_key)):
            urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
            if not urls:
                st.warning("Lägg till minst en URL")
            else:
                try:
                    result = _call_tool(
                        mcp_server.scrape_party_promises,
                        {"party_key": selected_party_key, "urls": urls},
                        is_async=True,
                    )
                    st.success("Scraping klar")
                    st.json(result)
                except Exception as exc:
                    st.error(f"Scraping misslyckades: {exc}")


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
    "add_party": {"name": "Socialdemokraterna", "abbreviation": "S", "block": "vänster", "website_url": "https://www.socialdemokraterna.se"},
    "list_parties": {},
    "get_party_details": {"party_key": ""},
    "add_politician": {"name": "", "party_key": "", "tiktok": "", "instagram": "", "twitter": ""},
    "list_politicians": {"party_key": ""},
    "get_politician_details": {"politician_key": ""},
    "add_promise": {"party_key": "", "text": "", "category": "", "source_url": "", "source_name": "", "date": ""},
    "get_promises": {"party_key": "", "category": ""},
    "delete_promise": {"promise_key": ""},
    "scrape_party_promises": {"party_key": "", "urls": ["https://www.socialdemokraterna.se/var-politik"]},
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
