"""Extractive learning utilities (system-design.md §6.5, product.md §8).

Everything here is extractive: summaries select existing source sentences,
term lists report observed frequencies, and revision prompts are derived from
source statements. No generative reasoning is performed, and every item keeps
its source file and page reference.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .text_utils import split_sentences, tokenise

MIN_SENTENCE_TOKENS = 4
MIN_PROMPT_TOKENS = 6

# Content governance evidence labels (product.md §12, template §1.1). Every
# exported content item must be classifiable as one of these so source text,
# extractive output, reviewer interpretation and open questions stay distinct.
EVIDENCE_SOURCE = "[SOURCE]"
EVIDENCE_EXTRACTIVE = "[EXTRACTIVE]"
EVIDENCE_INTERPRETATION = "[REVIEWER INTERPRETATION]"
EVIDENCE_OPEN_QUESTION = "[OPEN QUESTION]"
EVIDENCE_CONFLICT = "[CONFLICT]"

# Standard wording for incomplete or unresolved evidence (product.md §12,
# template §7). Never invent content to fill an unsupported section.
MISSING_EVIDENCE = "Not found in the indexed source documents. Confirmation required."
CONFLICT_EVIDENCE = (
    "The available source documents do not establish a single confirmed answer. "
    "Refer to the cited document versions and obtain confirmation from the responsible "
    "technical owner."
)
INTERPRETATION_NOTICE = (
    "Reviewer interpretation — this explanation is not stated verbatim in the source "
    "document and requires technical-owner validation."
)

# Canonical note structure suggested by product.md §9.
NOTE_SECTIONS = [
    "Topic name",
    "Plain-language description (learner-authored)",
    "Purpose and business context",
    "Components and responsibilities",
    "Process / workflow",
    "Inputs and outputs",
    "Upstream and downstream dependencies",
    "Protocols, formats and interfaces",
    "Error handling and operational considerations",
    "Security controls",
    "Acronyms and terminology",
    "Open questions",
    "Source references",
]


@dataclass
class StudyItem:
    """A source sentence or prompt with its citation."""

    text: str
    filename: str
    page_number: int

    @property
    def citation(self) -> str:
        return f"{self.filename} — page {self.page_number}"


def _iter_sentences(sources: list[dict]):
    """Yield (filename, page_number, sentence, tokens) for usable sentences."""
    for source in sources:
        filename = source.get("filename", "unknown")
        page = source.get("page_number", 0)
        for sentence in split_sentences(source.get("text", "")):
            tokens = tokenise(sentence)
            if len(tokens) < MIN_SENTENCE_TOKENS:
                continue
            yield filename, page, sentence, tokens


def extractive_summary(sources: list[dict], num_sentences: int = 7) -> list[StudyItem]:
    """Select the highest-signal source sentences, preserving their order.

    ``sources`` items look like ``{"filename", "page_number", "text"}``.
    """
    sentences: list[tuple] = []
    frequencies: Counter = Counter()
    for filename, page, sentence, tokens in _iter_sentences(sources):
        frequencies.update(tokens)
        sentences.append((filename, page, sentence, tokens))
    if not sentences:
        return []

    peak = max(frequencies.values()) or 1
    scored: list[tuple[float, str, int, str]] = []
    for filename, page, sentence, tokens in sentences:
        score = sum(frequencies[t] for t in tokens) / (len(tokens) * peak)
        scored.append((score, filename, page, sentence))

    scored.sort(key=lambda item: -item[0])
    selected = scored[: max(1, num_sentences)]
    selected.sort(key=lambda item: (item[1], item[2]))
    return [
        StudyItem(text=sentence, filename=filename, page_number=page)
        for _, filename, page, sentence in selected
    ]


def frequent_terms(texts: list[str], top_n: int = 20, min_length: int = 3) -> list[tuple[str, int]]:
    """Return the most frequent terms observed in the supplied text."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenise(text))
    filtered = ((term, n) for term, n in counts.items() if len(term) >= min_length)
    return sorted(filtered, key=lambda item: (-item[1], item[0]))[:top_n]


def revision_prompts(sources: list[dict], max_prompts: int = 10) -> list[dict]:
    """Turn source statements into cloze-style revision prompts."""
    frequencies: Counter = Counter()
    prepared: list[tuple] = []
    for filename, page, sentence, tokens in _iter_sentences(sources):
        if len(tokens) < MIN_PROMPT_TOKENS:
            continue
        frequencies.update(tokens)
        prepared.append((filename, page, sentence, tokens))

    prompts: list[dict] = []
    for filename, page, sentence, tokens in prepared:
        if len(prompts) >= max_prompts:
            break
        key_term = max(tokens, key=lambda t: (frequencies[t], len(t)))
        pattern = re.compile(re.escape(key_term), re.IGNORECASE)
        cloze, replacements = pattern.subn("____", sentence, count=1)
        if replacements == 0:
            continue
        prompts.append(
            {
                "question": f"Fill in the blank: {cloze}",
                "answer": key_term,
                "statement": sentence,
                "filename": filename,
                "page_number": page,
            }
        )
    return prompts


def comparison_markdown(results: list[dict], max_chars: int = 300) -> str:
    """Render side-by-side evidence from multiple documents as a Markdown table."""
    lines = ["| Document | Category | Page | Passage |", "|---|---|---|---|"]
    for item in results:
        passage = " ".join(str(item.get("chunk_text", "")).split())
        if len(passage) > max_chars:
            passage = passage[:max_chars].rstrip() + "…"
        passage = passage.replace("|", "\\|")
        lines.append(
            f"| {item.get('filename', '')} | {item.get('category', '')} "
            f"| {item.get('page_number', '')} | {passage} |"
        )
    return "\n".join(lines)


def notes_to_markdown(
    title: str,
    sections: list[tuple[str, str]],
    sources: list[str] | None = None,
) -> str:
    """Assemble a source-linked Markdown note from ordered (heading, body) pairs."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append((body or "").strip() or "_Not yet completed._")
        lines.append("")
    if sources:
        lines.append("## Source references")
        lines.append("")
        for citation in sources:
            lines.append(f"- {citation}")
        lines.append("")
    return "\n".join(lines)


def study_note_template(title: str) -> str:
    """Return an empty Markdown scaffold using the canonical note structure."""
    sections = [(heading, "") for heading in NOTE_SECTIONS]
    return notes_to_markdown(title, sections)
