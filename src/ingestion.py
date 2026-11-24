"""PDF ingestion orchestration (architecture.md §3).

Discovers PDFs recursively, fingerprints them with SHA-256 for change
detection, extracts and chunks text, and persists the result in one
transaction per document. A failure for a single file never aborts the batch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from . import database as db
from .chunker import ChunkConfig, chunk_pages
from .extractor import extract_pdf

# Files matching these prefixes/suffixes are transient artefacts and skipped.
SKIP_PREFIXES = ("~$", ".")
SKIP_SUFFIXES = (".tmp",)

# How many directory levels below the documents root form the category label.
CATEGORY_DEPTH = 1

DEFAULT_DOCUMENTS_DIR = config.DOCUMENTS_DIR


@dataclass
class FileOutcome:
    filepath: str
    filename: str
    category: str
    status: str  # INDEXED, SKIPPED, ERROR, LOW_TEXT, OCR_REQUIRED
    total_pages: int = 0
    indexed_pages: int = 0
    chunks: int = 0
    error: str | None = None


@dataclass
class IngestionSummary:
    run_id: int | None = None
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    outcomes: list[FileOutcome] = field(default_factory=list)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 content fingerprint of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    """Coordinates discovery, extraction, chunking and persistence."""

    def __init__(
        self,
        db_path: Path | str = db.DEFAULT_DB_PATH,
        documents_dir: Path | str = DEFAULT_DOCUMENTS_DIR,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.documents_dir = Path(documents_dir)
        self.chunk_config = chunk_config or ChunkConfig()
        db.init_db(self.db_path)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def scan(self) -> list[Path]:
        """Recursively find candidate PDF files in the documents directory."""
        if not self.documents_dir.exists():
            return []
        found: list[Path] = []
        for path in sorted(self.documents_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() != ".pdf":
                continue
            if path.name.startswith(SKIP_PREFIXES) or path.name.lower().endswith(SKIP_SUFFIXES):
                continue
            found.append(path)
        return found

    def category_for(self, path: Path) -> str:
        """Derive the category label from the folder structure."""
        try:
            relative = path.resolve().relative_to(self.documents_dir.resolve())
        except ValueError:
            return "Uncategorised"
        parts = relative.parts[:-1]  # drop the filename
        if not parts:
            return "Uncategorised"
        return "/".join(parts[:CATEGORY_DEPTH])

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, force: bool = False) -> IngestionSummary:
        """Run a full ingestion pass and return a summary.

        ``force=True`` re-extracts every file, ignoring the SHA-256 shortcut.
        """
        summary = IngestionSummary()
        files = self.scan()
        summary.files_seen = len(files)

        with db.connection(self.db_path) as conn:
            summary.run_id = db.start_ingestion_run(conn)
            conn.commit()

            for path in files:
                outcome = self._ingest_one(conn, path, force=force)
                summary.outcomes.append(outcome)
                if outcome.status == "SKIPPED":
                    summary.files_skipped += 1
                elif outcome.status == "ERROR":
                    summary.files_failed += 1
                else:
                    summary.files_indexed += 1

            db.finish_ingestion_run(
                conn,
                summary.run_id,
                files_seen=summary.files_seen,
                files_indexed=summary.files_indexed,
                files_skipped=summary.files_skipped,
                files_failed=summary.files_failed,
            )
            conn.commit()

        return summary

    def _ingest_one(self, conn, path: Path, force: bool) -> FileOutcome:
        filepath = str(path.resolve())
        filename = path.name
        category = self.category_for(path)

        try:
            digest = sha256_of(path)
        except OSError as exc:
            return self._record_error(conn, filepath, filename, category, f"hash failed: {exc}")

        existing = db.get_document_by_path(conn, filepath)
        if (
            not force
            and existing is not None
            and existing["sha256"] == digest
            and existing["status"] != db.STATUS_ERROR
        ):
            return FileOutcome(
                filepath=filepath,
                filename=filename,
                category=existing["category"] or category,
                status="SKIPPED",
                total_pages=existing["total_pages"],
                indexed_pages=existing["indexed_pages"],
            )

        try:
            extraction = extract_pdf(path)
        except Exception as exc:  # noqa: BLE001 - report any parser failure
            return self._record_error(conn, filepath, filename, category, str(exc), digest)

        chunks = chunk_pages(extraction.pages, self.chunk_config)
        status = extraction.quality_status

        try:
            with db.transaction(conn):
                document_id = db.upsert_document(
                    conn,
                    filename=filename,
                    filepath=filepath,
                    category=category,
                    sha256=digest,
                    total_pages=extraction.total_pages,
                    indexed_pages=extraction.text_pages,
                    status=status,
                )
                db.insert_pages(conn, document_id, extraction.pages)
                if chunks:
                    db.insert_chunks(conn, document_id, chunks)
        except Exception as exc:  # noqa: BLE001 - persistence failure
            return self._record_error(conn, filepath, filename, category, str(exc), digest)

        return FileOutcome(
            filepath=filepath,
            filename=filename,
            category=category,
            status=status,
            total_pages=extraction.total_pages,
            indexed_pages=extraction.text_pages,
            chunks=len(chunks),
        )

    def _record_error(
        self,
        conn,
        filepath: str,
        filename: str,
        category: str,
        message: str,
        digest: str = "",
    ) -> FileOutcome:
        """Persist an ERROR status for a file without aborting the batch."""
        try:
            with db.transaction(conn):
                db.upsert_document(
                    conn,
                    filename=filename,
                    filepath=filepath,
                    category=category,
                    sha256=digest,
                    total_pages=0,
                    indexed_pages=0,
                    status=db.STATUS_ERROR,
                )
        except Exception:  # noqa: BLE001 - last-resort, keep the batch alive
            pass
        return FileOutcome(
            filepath=filepath,
            filename=filename,
            category=category,
            status=db.STATUS_ERROR,
            error=message,
        )

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #
    def reconcile(self) -> int:
        """Explicitly remove records whose source PDFs no longer exist."""
        existing_paths = {str(p.resolve()) for p in self.scan()}
        with db.connection(self.db_path) as conn:
            with db.transaction(conn):
                removed = db.reconcile_deleted_files(conn, existing_paths)
        return removed
