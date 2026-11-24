"""Page-level PDF text extraction using PyMuPDF.

Text is extracted page by page and the original 1-based page number is
preserved for every page so citations stay reliable (system-design.md §6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # PyMuPDF (import name `fitz` is deprecated)

from . import config

# A page is considered text-bearing when it exceeds this many characters.
MIN_TEXT_CHARS = config.MIN_TEXT_CHARS

# Below this fraction of text-bearing pages a PDF is flagged as needing OCR.
LOW_TEXT_THRESHOLD = config.LOW_TEXT_THRESHOLD


@dataclass
class ExtractionResult:
    """Result of extracting a single PDF, including quality metrics."""

    total_pages: int = 0
    pages: list[dict] = field(default_factory=list)
    empty_pages: int = 0
    text_pages: int = 0
    total_characters: int = 0

    @property
    def text_coverage(self) -> float:
        """Fraction of pages that contain extractable text."""
        if self.total_pages == 0:
            return 0.0
        return self.text_pages / self.total_pages

    @property
    def quality_status(self) -> str:
        """Map extraction coverage onto a document status value."""
        if self.text_pages == 0:
            return "OCR_REQUIRED"
        if self.text_coverage < LOW_TEXT_THRESHOLD:
            return "LOW_TEXT"
        return "INDEXED"


def extract_pdf(path: Path | str) -> ExtractionResult:
    """Extract text page by page from ``path``.

    Raises the underlying PyMuPDF exception if the file cannot be opened or
    parsed; callers are expected to catch it and record an ERROR status.
    """
    path = Path(path)
    result = ExtractionResult()

    with pymupdf.open(str(path)) as doc:
        result.total_pages = doc.page_count
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = _tidy(text)
            if len(text) < MIN_TEXT_CHARS:
                result.empty_pages += 1
            else:
                result.text_pages += 1
            result.total_characters += len(text)
            result.pages.append({"page_number": index, "text": text})

    return result


def _tidy(text: str) -> str:
    """Normalise extracted text without destroying page semantics."""
    # Normalise line endings and strip trailing whitespace on each line, then
    # collapse three-or-more blank lines down to a single blank line.
    lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()
