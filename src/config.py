"""Centralised runtime configuration (techstack.md §6, §7.1).

Paths and tunable processing/retrieval parameters live here so they are not
embedded throughout the application (techstack.md §6: "Paths should be
configurable rather than embedded throughout application code"). Every value
can be overridden with an environment variable, which keeps the module usable
on an approved offline workstation without editing source.

Configuration is validated once at startup via :func:`validate` and the data
locations are created via :func:`ensure_directories` (techstack.md §7.1).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


# --- Data locations (techstack.md §6) -------------------------------------- #
DOCUMENTS_DIR = _env_path("ACBS_DOCUMENTS_DIR", PROJECT_ROOT / "documents")
DATA_DIR = _env_path("ACBS_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = _env_path("ACBS_DB_PATH", DATA_DIR / "knowledge.db")

EXPORTS_DIR = _env_path("ACBS_EXPORTS_DIR", PROJECT_ROOT / "exports")
NOTES_DIR = EXPORTS_DIR / "notes"
DOCUMENTATION_DIR = EXPORTS_DIR / "documentation"
REPORTS_DIR = EXPORTS_DIR / "reports"

# --- Extraction quality thresholds (system-design.md §6.2) ----------------- #
MIN_TEXT_CHARS = _env_int("ACBS_MIN_TEXT_CHARS", 10)
LOW_TEXT_THRESHOLD = _env_float("ACBS_LOW_TEXT_THRESHOLD", 0.5)

# --- Chunking (system-design.md §6.3) -------------------------------------- #
CHUNK_SIZE = _env_int("ACBS_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("ACBS_CHUNK_OVERLAP", 150)
MIN_CHUNK = _env_int("ACBS_MIN_CHUNK", 100)

# --- Retrieval weights (system-design.md §6.4) ----------------------------- #
BM25_WEIGHT = _env_float("ACBS_BM25_WEIGHT", 0.65)
TFIDF_WEIGHT = _env_float("ACBS_TFIDF_WEIGHT", 0.35)

# --- UI (architecture.md §11) ---------------------------------------------- #
UI_HOST = os.environ.get("ACBS_UI_HOST", "127.0.0.1")


def ensure_directories() -> None:
    """Create the configured data/export directories if they do not exist."""
    for directory in (DOCUMENTS_DIR, DATA_DIR, NOTES_DIR, DOCUMENTATION_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def validate() -> None:
    """Validate configuration at application startup (techstack.md §7.1)."""
    if MIN_TEXT_CHARS < 1:
        raise ValueError("MIN_TEXT_CHARS must be at least 1")
    if not 0.0 <= LOW_TEXT_THRESHOLD <= 1.0:
        raise ValueError("LOW_TEXT_THRESHOLD must be between 0.0 and 1.0")
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
    if MIN_CHUNK < 0:
        raise ValueError("MIN_CHUNK must be >= 0")
    if BM25_WEIGHT < 0 or TFIDF_WEIGHT < 0:
        raise ValueError("Retrieval weights must not be negative")
    if BM25_WEIGHT + TFIDF_WEIGHT <= 0:
        raise ValueError("At least one retrieval weight must be positive")
