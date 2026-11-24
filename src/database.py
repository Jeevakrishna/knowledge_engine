"""SQLite persistence layer for the ACBS Knowledge Engine.

Implements the logical data architecture defined in architecture.md section 5:
DOCUMENTS -> PAGES -> CHUNKS, plus INGESTION_RUNS for run statistics.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from . import config

PROJECT_ROOT = config.PROJECT_ROOT
DEFAULT_DB_PATH = config.DB_PATH

# Document ingestion status values.
STATUS_INDEXED = "INDEXED"
STATUS_LOW_TEXT = "LOW_TEXT"
STATUS_OCR_REQUIRED = "OCR_REQUIRED"
STATUS_ERROR = "ERROR"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT    NOT NULL,
    filepath      TEXT    NOT NULL UNIQUE,
    category      TEXT,
    sha256        TEXT    NOT NULL,
    total_pages   INTEGER NOT NULL DEFAULT 0,
    indexed_pages INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL,
    ingested_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    page_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER NOT NULL,
    page_number    INTEGER NOT NULL,
    extracted_text TEXT,
    character_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL,
    page_number  INTEGER NOT NULL,
    chunk_number INTEGER NOT NULL,
    chunk_text   TEXT    NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL,
    completed_at  TEXT,
    files_seen    INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    files_skipped INTEGER NOT NULL DEFAULT 0,
    files_failed  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pages_document  ON pages (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_documents_path  ON documents (filepath);
"""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (and if necessary create the directory for) the SQLite database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection and guarantee it is closed afterwards.

    ``sqlite3.Connection`` used directly as a context manager only commits or
    rolls back — it does not close the handle, which leaks file locks on
    Windows. Always use this helper instead.
    """
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create the schema if it does not already exist."""
    with connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Provide a transaction per unit of work (one document, one run)."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------------- #
# Document queries
# --------------------------------------------------------------------------- #
def get_document_by_path(conn: sqlite3.Connection, filepath: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE filepath = ?", (filepath,)
    ).fetchone()


def get_document(conn: sqlite3.Connection, document_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents ORDER BY category, filename"
    ).fetchall()


def list_categories(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT category FROM documents "
        "WHERE category IS NOT NULL AND category <> '' ORDER BY category"
    ).fetchall()
    return [r["category"] for r in rows]


def upsert_document(
    conn: sqlite3.Connection,
    *,
    filename: str,
    filepath: str,
    category: str,
    sha256: str,
    total_pages: int,
    indexed_pages: int,
    status: str,
) -> int:
    """Insert or replace a document row and clear its previous pages/chunks.

    Re-indexing is transactional: old derived rows are removed before the new
    ones are written, so duplicates cannot accumulate (system-design.md §10).
    Returns the document_id.
    """
    existing = get_document_by_path(conn, filepath)
    if existing is not None:
        document_id = existing["document_id"]
        conn.execute(
            """
            UPDATE documents
               SET filename = ?, category = ?, sha256 = ?, total_pages = ?,
                   indexed_pages = ?, status = ?, ingested_at = ?
             WHERE document_id = ?
            """,
            (
                filename,
                category,
                sha256,
                total_pages,
                indexed_pages,
                status,
                utc_now(),
                document_id,
            ),
        )
        conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        return document_id

    cursor = conn.execute(
        """
        INSERT INTO documents
            (filename, filepath, category, sha256, total_pages, indexed_pages,
             status, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            filepath,
            category,
            sha256,
            total_pages,
            indexed_pages,
            status,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def insert_pages(conn: sqlite3.Connection, document_id: int, pages: Sequence[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO pages (document_id, page_number, extracted_text, character_count)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                document_id,
                p["page_number"],
                p["text"],
                len(p["text"] or ""),
            )
            for p in pages
        ],
    )


def insert_chunks(conn: sqlite3.Connection, document_id: int, chunks: Sequence[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO chunks (document_id, page_number, chunk_number, chunk_text)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                document_id,
                c["page_number"],
                c["chunk_number"],
                c["chunk_text"],
            )
            for c in chunks
        ],
    )


def delete_document(conn: sqlite3.Connection, document_id: int) -> None:
    """Force delete of a document plus its pages and chunks."""
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def reconcile_deleted_files(conn: sqlite3.Connection, existing_paths: set[str]) -> int:
    """Remove documents whose source files no longer exist.

    Explicit reconciliation operation (system-design.md §10) — never run
    implicitly during a normal ingestion pass. Returns the number removed.
    """
    removed = 0
    for row in list_documents(conn):
        if row["filepath"] not in existing_paths:
            delete_document(conn, row["document_id"])
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# Chunk / page retrieval
# --------------------------------------------------------------------------- #
def fetch_all_chunks(conn: sqlite3.Connection, category: str | None = None) -> list[sqlite3.Row]:
    """Return searchable chunks joined with their document metadata."""
    sql = """
        SELECT c.chunk_id, c.document_id, c.page_number, c.chunk_number,
               c.chunk_text, d.filename, d.category, d.status
          FROM chunks c
          JOIN documents d ON d.document_id = c.document_id
    """
    params: list = []
    if category:
        sql += " WHERE d.category = ?"
        params.append(category)
    sql += " ORDER BY c.chunk_id"
    return conn.execute(sql, params).fetchall()


def fetch_document_chunks(conn: sqlite3.Connection, document_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number, chunk_number",
        (document_id,),
    ).fetchall()


def fetch_document_pages(conn: sqlite3.Connection, document_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
        (document_id,),
    ).fetchall()


# --------------------------------------------------------------------------- #
# Ingestion runs
# --------------------------------------------------------------------------- #
def start_ingestion_run(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "INSERT INTO ingestion_runs (started_at) VALUES (?)", (utc_now(),)
    )
    return int(cursor.lastrowid)


def finish_ingestion_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    files_seen: int,
    files_indexed: int,
    files_skipped: int,
    files_failed: int,
) -> None:
    conn.execute(
        """
        UPDATE ingestion_runs
           SET completed_at = ?, files_seen = ?, files_indexed = ?,
               files_skipped = ?, files_failed = ?
         WHERE run_id = ?
        """,
        (utc_now(), files_seen, files_indexed, files_skipped, files_failed, run_id),
    )


def get_last_ingestion_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM ingestion_runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()


# --------------------------------------------------------------------------- #
# Dashboard statistics
# --------------------------------------------------------------------------- #
def get_stats(conn: sqlite3.Connection) -> dict:
    def scalar(sql: str, params: Iterable = ()) -> int:
        row = conn.execute(sql, tuple(params)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    total_pages = scalar("SELECT COALESCE(SUM(total_pages), 0) FROM documents")
    indexed_pages = scalar("SELECT COALESCE(SUM(indexed_pages), 0) FROM documents")
    return {
        "documents": scalar("SELECT COUNT(*) FROM documents"),
        "pages": scalar("SELECT COUNT(*) FROM pages"),
        "total_pages": total_pages,
        "indexed_pages": indexed_pages,
        "chunks": scalar("SELECT COUNT(*) FROM chunks"),
        "low_text": scalar(
            "SELECT COUNT(*) FROM documents WHERE status IN ('LOW_TEXT', 'OCR_REQUIRED')"
        ),
        "failed": scalar("SELECT COUNT(*) FROM documents WHERE status = 'ERROR'"),
        "ingestion_runs": scalar("SELECT COUNT(*) FROM ingestion_runs"),
    }
