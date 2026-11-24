"""Hybrid lexical retrieval: BM25 + TF-IDF (system-design.md §6.4, §8).

Both retrievers are lexical and deterministic. The initial weighting is
0.65 * BM25 + 0.35 * TF-IDF cosine, which is configurable and intended to be
tuned against a labelled query set (architecture.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from . import database as db
from .text_utils import tokenise

VALID_MODES = ("hybrid", "bm25", "tfidf")


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    filename: str
    category: str
    page_number: int
    chunk_number: int
    chunk_text: str
    score: float
    bm25_score: float
    tfidf_score: float

    @property
    def citation(self) -> str:
        return f"{self.filename} — page {self.page_number}"


class SearchEngine:
    """In-memory BM25 + TF-IDF index over persisted chunks."""

    def __init__(
        self,
        db_path: Path | str = db.DEFAULT_DB_PATH,
        *,
        bm25_weight: float = config.BM25_WEIGHT,
        tfidf_weight: float = config.TFIDF_WEIGHT,
    ) -> None:
        self.db_path = Path(db_path)
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    # ------------------------------------------------------------------ #
    # Index lifecycle
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return bool(self._chunks) and self._bm25 is not None and self._matrix is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def build(self) -> None:
        """Load chunks from SQL and construct the BM25 and TF-IDF indexes."""
        with db.connection(self.db_path) as conn:
            rows = db.fetch_all_chunks(conn)

        self._chunks = [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "filename": r["filename"],
                "category": r["category"] or "Uncategorised",
                "page_number": r["page_number"],
                "chunk_number": r["chunk_number"],
                "chunk_text": r["chunk_text"],
                "tokens": tokenise(r["chunk_text"]),
            }
            for r in rows
        ]

        if not self._chunks:
            self._bm25 = None
            self._vectorizer = None
            self._matrix = None
            return

        corpus_tokens = [c["tokens"] for c in self._chunks]
        # BM25 needs at least one non-empty document to compute IDF.
        safe_corpus = [toks if toks else ["_empty_"] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(safe_corpus)

        joined = [" ".join(c["tokens"]) if c["tokens"] else "_empty_" for c in self._chunks]
        self._vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
        self._matrix = self._vectorizer.fit_transform(joined)

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.build()

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")

        self.ensure_loaded()
        if not self.is_loaded:
            return []

        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        candidates = list(range(len(self._chunks)))
        if category and category != "All":
            candidates = [i for i in candidates if self._chunks[i]["category"] == category]
        if not candidates:
            return []

        bm25_all = np.asarray(self._bm25.get_scores(query_tokens), dtype=float)
        query_vector = self._vectorizer.transform([" ".join(query_tokens)])
        tfidf_all = cosine_similarity(query_vector, self._matrix).ravel()

        cand_bm25 = bm25_all[candidates]
        cand_tfidf = tfidf_all[candidates]

        norm_bm25 = _minmax(cand_bm25)
        norm_tfidf = _minmax(cand_tfidf)

        if mode == "bm25":
            combined = norm_bm25
        elif mode == "tfidf":
            combined = norm_tfidf
        else:
            combined = self.bm25_weight * norm_bm25 + self.tfidf_weight * norm_tfidf

        top_k = max(1, int(top_k))
        order = np.argsort(-combined)[:top_k]

        results: list[SearchResult] = []
        for rank in order:
            score = float(combined[rank])
            if score <= 0:
                continue
            idx = candidates[rank]
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    filename=chunk["filename"],
                    category=chunk["category"],
                    page_number=chunk["page_number"],
                    chunk_number=chunk["chunk_number"],
                    chunk_text=chunk["chunk_text"],
                    score=score,
                    bm25_score=float(cand_bm25[rank]),
                    tfidf_score=float(cand_tfidf[rank]),
                )
            )
        return results


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalise a score vector to [0, 1].

    A constant vector normally normalises to zeros, but when a single positive
    score exists (for example an index with one passage) it is kept as 1.0 so
    the passage is not silently discarded.
    """
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.where(values > 0, 1.0, 0.0)
    return (values - low) / (high - low)
