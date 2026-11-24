"""Page-aware overlapping chunking (system-design.md §6.3).

Each chunk is kept within its source page so that every search result can be
cited back to an exact PDF page (architecture.md §3 ingestion rules).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config

# Initial tuning values — starting points to be evaluated against real PDFs.
DEFAULT_CHUNK_SIZE = config.CHUNK_SIZE
DEFAULT_OVERLAP = config.CHUNK_OVERLAP
DEFAULT_MIN_CHUNK = config.MIN_CHUNK


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    min_chunk: int = DEFAULT_MIN_CHUNK

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        if self.min_chunk < 0:
            raise ValueError("min_chunk must be >= 0")


def split_page_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split a single page's text into overlapping chunks.

    A trailing chunk shorter than ``min_chunk`` is merged into the previous
    chunk rather than emitted as a fragment.
    """
    config = config or ChunkConfig()
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= config.chunk_size:
        return [text] if len(text) >= config.min_chunk else []

    step = config.chunk_size - config.overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + config.chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    # Fold a too-short tail chunk into the previous chunk.
    if len(chunks) >= 2 and len(chunks[-1]) < config.min_chunk:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}".strip()
    return chunks


def chunk_pages(pages: list[dict], config: ChunkConfig | None = None) -> list[dict]:
    """Chunk an extracted document's pages.

    ``pages`` is the list produced by ``extractor.extract_pdf`` (each item has
    ``page_number`` and ``text``). Returns chunk dicts with ``page_number``,
    ``chunk_number`` (1-based within the page) and ``chunk_text``.
    """
    config = config or ChunkConfig()
    chunks: list[dict] = []
    for page in pages:
        for number, piece in enumerate(split_page_text(page["text"], config), start=1):
            chunks.append(
                {
                    "page_number": page["page_number"],
                    "chunk_number": number,
                    "chunk_text": piece,
                }
            )
    return chunks
