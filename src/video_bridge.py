"""
Bridge mot befintlig video-analyzer i separat projekt.

Läser VIDEO_ANALYZER_PATH från miljön och anropar analyze_content_iter
för att få strukturerad analys av ett uppladdat videoklipp.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any


def get_video_analyzer_root() -> Path:
    """Returnera absolut sökväg till video-analyzer-projektet."""
    raw = os.getenv("VIDEO_ANALYZER_PATH", "").strip()
    if not raw:
        raise RuntimeError("VIDEO_ANALYZER_PATH saknas i miljön")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        # Relativa sökvägar tolkas från detta repos rot.
        path = (Path(__file__).resolve().parents[1] / path).resolve()

    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"VIDEO_ANALYZER_PATH hittades inte: {path}")
    return path


def _prepare_analyzer_import(root: Path) -> None:
    """Säkerställ att import av extern app.services.* går mot rätt projekt."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Om "app" är laddad från fel ställe, rensa cache så rätt paket importeras.
    cached_app = sys.modules.get("app")
    cached_file = str(getattr(cached_app, "__file__", "")) if cached_app else ""
    if cached_app is not None and root_str not in cached_file and not hasattr(cached_app, "__path__"):
        for mod_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
            del sys.modules[mod_name]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    """Normalisera analysrad till dashboardens gemensamma format."""
    return {
        "avsandare": _safe_text(row.get("avsandare")) or "Okänd",
        "transkribering": _safe_text(row.get("transkribering")),
        "innehall": _safe_text(row.get("innehall")),
        "kategori": _safe_text(row.get("kategori")) or "Annat",
        "politiskt_farg": _safe_text(row.get("politiskt_farg")) or "neutral",
    }


async def analyze_video_file(video_path: Path, categories: str | None = None) -> list[dict[str, str]]:
    """
    Kör extern video-analyzer på en videofil och returnerar normaliserade klipprader.
    """
    analyzer_root = get_video_analyzer_root()
    _prepare_analyzer_import(analyzer_root)

    from app.services.content_analyzer import analyze_content_iter  # type: ignore[import-not-found]

    rows: list[dict[str, str]] = []
    async for result in analyze_content_iter(str(video_path), kategorier=(categories or None)):
        if not isinstance(result, dict):
            continue
        rows.append(_normalize_row(result))

    return rows


def tokenize(text: str) -> set[str]:
    """Enkel tokenisering för grov relevans-ranking mot löften."""
    normalized = re.sub(r"[^a-zA-Z0-9åäöÅÄÖ ]+", " ", str(text or "").lower())
    words = [w for w in normalized.split() if len(w) >= 3]
    return set(words)

