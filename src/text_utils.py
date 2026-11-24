"""Shared text normalisation and tokenisation helpers.

Deterministic, dependency-free processing used by both the search engine and
the learning utilities. No AI models, embeddings or external services.
"""

from __future__ import annotations

import re

# Minimal English stop-word list. Technical terms and acronyms are deliberately
# preserved; this list only removes low-signal function words.
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having he her here him his how i if in into is it its me my of
    on or our out over she should so such than that the their them then there
    these they this those to too up was we were what when where which while who
    why will with would you your
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.]*")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def tokenise(text: str, remove_stopwords: bool = True) -> list[str]:
    """Split text into lower-cased tokens suitable for lexical retrieval.

    Keeps alphanumerics plus ``-_.`` so identifiers such as ``imft``,
    ``sha-256`` and ``3.12`` survive tokenisation.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    # Drop pure punctuation artefacts and single characters with no digits.
    tokens = [t for t in tokens if t.strip("-_.") != ""]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Naive but deterministic sentence splitter using the standard library."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
