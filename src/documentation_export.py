"""Post-ingestion knowledge outputs (product.md §9, system-design.md §6.5).

Two documentation levels are produced from indexed evidence:

- **Level 1 — Quick Reference Notes:** a concise, learner-friendly note.
- **Level 2 — Formal Technical Documentation Draft:** a structured,
  corporate-style draft for review.

Both are *extractive*: every factual line is a source sentence that was
selected and organised, retaining its PDF filename and page. Sections the
indexed sources cannot support are marked with the standard missing-evidence
wording (product.md §12, pdf-knowledge-extraction-template.md §7) rather than
being invented. The formal draft is a draft until a technical owner validates
it and must never be presented as approved corporate documentation
(product.md §9.2, system-design.md §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import learning
from .learning import (
    EVIDENCE_EXTRACTIVE,
    EVIDENCE_OPEN_QUESTION,
    EVIDENCE_SOURCE,
    INTERPRETATION_NOTICE,
    MISSING_EVIDENCE,
    StudyItem,
)
from .text_utils import split_sentences, tokenise

DRAFT_STATUS = "Draft — technical-owner review required"

# Evidence cues used to organise already-extracted source sentences into
# sections. Selecting a sentence by cue is organisation, not generation.
_PURPOSE_CUES = (
    "purpose",
    "designed to",
    "used to",
    "enables",
    "provides",
    "intended to",
    "responsible for",
    "allows",
)
_COMPONENT_CUES = (
    "component",
    "module",
    "service",
    "server",
    "manager",
    "engine",
    "node",
    "queue",
    "database",
    "interface",
    "gateway",
)
_WORKFLOW_CUES = (
    "step",
    "first",
    "then",
    "next",
    "finally",
    "trigger",
    "starts",
    "receives",
    "sends",
    "processes",
    "flow",
    "workflow",
)
_INTERFACE_CUES = (
    "interface",
    "protocol",
    "api",
    "format",
    "message",
    "port",
    "http",
    "tcp",
    "ftp",
    "sftp",
    "xml",
    "json",
    "csv",
    "file naming",
)
_CONFIG_CUES = (
    "configuration",
    "config",
    "parameter",
    "setting",
    "environment",
    "variable",
    "directory",
    "version",
    "install",
)
_OPERATIONAL_CUES = (
    "error",
    "fail",
    "failure",
    "retry",
    "restart",
    "monitor",
    "log",
    "warning",
    "exception",
    "troubleshoot",
    "recovery",
    "alert",
)
_SECURITY_CUES = (
    "security",
    "authentication",
    "authorisation",
    "authorization",
    "access control",
    "encryption",
    "audit",
    "permission",
    "credential",
    "retention",
)
_ASSUMPTION_CUES = (
    "assumption",
    "assumes",
    "constraint",
    "limitation",
    "depends on",
    "prerequisite",
    "requires",
)
_DEPENDENCY_CUES = (
    "depends on",
    "dependency",
    "upstream",
    "downstream",
    "integrates",
    "integration",
    "calls",
    "consumes",
    "connects",
)


@dataclass
class KnowledgeOutput:
    """A structured, extractive knowledge output ready for review/export."""

    title: str
    level: str
    sections: list[tuple[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    review_status: str = DRAFT_STATUS

    def to_markdown(self) -> str:
        """Render the output as reviewable Markdown with governance notices."""
        body = learning.notes_to_markdown(self.title, self.sections, sources=self.sources)
        governance = "\n".join(
            [
                f"> **{self.level}**",
                f"> Output status: {self.review_status}",
                f"> Evidence labels: {EVIDENCE_SOURCE} source · {EVIDENCE_EXTRACTIVE} extractive · "
                f"{EVIDENCE_OPEN_QUESTION} open question",
                f"> Missing evidence wording: _{MISSING_EVIDENCE}_",
                "",
            ]
        )
        return f"{body}\n\n{governance}"


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #
def _iter_sources(sources: list[dict]):
    for source in sources:
        filename = source.get("filename", "unknown")
        page = source.get("page_number", 0)
        for sentence in split_sentences(source.get("text", "")):
            if len(tokenise(sentence)) < learning.MIN_SENTENCE_TOKENS:
                continue
            yield filename, page, sentence


def _match_sentences(sources: list[dict], cues: tuple[str, ...], limit: int) -> list[StudyItem]:
    """Select, in source order, sentences that mention any of the cue terms."""
    wanted = tuple(cue.lower() for cue in cues)
    found: list[StudyItem] = []
    for filename, page, sentence in _iter_sources(sources):
        low = sentence.lower()
        if any(cue in low for cue in wanted):
            found.append(StudyItem(text=sentence, filename=filename, page_number=page))
            if len(found) >= limit:
                break
    return found


def _extract_block(items: list[StudyItem], fallback: list[StudyItem] | None = None) -> str:
    """Render selected source sentences as an evidence-labelled list."""
    items = items or fallback or []
    if not items:
        return f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}"
    return "\n".join(
        f"- {item.text}  \n  {EVIDENCE_EXTRACTIVE} `{item.filename}`, p. {item.page_number}"
        for item in items
    )


def _terms_table(sources: list[dict], top_n: int) -> str:
    terms = learning.frequent_terms(
        [s.get("text", "") for s in sources if s.get("text")], top_n=top_n
    )
    if not terms:
        return f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}"
    lines = [
        "| Term | Occurrences | Treatment |",
        "|---|---:|---|",
    ]
    for term, count in terms:
        lines.append(
            f"| {term} | {count} | {EVIDENCE_EXTRACTIVE} frequency signal only — not a definition |"
        )
    return "\n".join(lines)


def _scope_block(sources: list[dict], documents: list[dict] | None = None) -> str:
    """List the source documents that make up the scope of the output."""
    names: list[str] = []
    for document in documents or []:
        name = document.get("filename")
        if name and name not in names:
            names.append(name)
    for source in sources:
        name = source.get("filename")
        if name and name not in names:
            names.append(name)
    if not names:
        return f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}"
    return "\n".join(f"- `{name}` {EVIDENCE_SOURCE}" for name in names)


def _document_control_table(documents: list[dict]) -> str:
    if not documents:
        return f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}"
    lines = [
        "| Filename | Category | Status | Indexed / total pages | SHA-256 (truncated) |",
        "|---|---|---|---|---|",
    ]
    for document in documents:
        digest = str(document.get("sha256", ""))[:12]
        lines.append(
            f"| {document.get('filename', '')} | {document.get('category', '')} "
            f"| {document.get('status', '')} "
            f"| {document.get('indexed_pages', 0)} / {document.get('total_pages', 0)} "
            f"| {digest}… |"
        )
    return "\n".join(lines)


def _revision_history_table() -> str:
    today = date.today().isoformat()
    return "\n".join(
        [
            "| Version | Date | Author | Summary of change | Review / approval |",
            "|---|---|---|---|---|",
            f"| 0.1 | {today} | ACBS Knowledge Engine (extractive) "
            "| Initial evidence-based draft | Pending |",
        ]
    )


def _open_questions_table() -> str:
    return "\n".join(
        [
            "| ID | Question / decision | Why it matters | Responsible owner | Status |",
            "|---|---|---|---|---|",
            "| OQ-001 | All extractive statements require technical-owner validation "
            "| This draft is not approved corporate documentation "
            "| Document owner / technical owner | Open |",
            "| OQ-002 | Undocumented sections must be confirmed against the source PDFs "
            "| Missing detail must not be inferred "
            "| Document owner / technical owner | Open |",
        ]
    )


def _reference_register(sources: list[dict]) -> str:
    pages: dict[str, set[int]] = {}
    order: list[str] = []
    for source in sources:
        name = source.get("filename", "unknown")
        if name not in pages:
            pages[name] = set()
            order.append(name)
        pages[name].add(int(source.get("page_number", 0)))
    if not order:
        return f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}"
    lines = ["| Ref ID | Filename | Pages referenced |", "|---|---|---|"]
    for index, name in enumerate(order, start=1):
        referenced = sorted(p for p in pages[name] if p)
        if referenced:
            span = (
                f"p. {referenced[0]}"
                if referenced[0] == referenced[-1]
                else f"pp. {referenced[0]}–{referenced[-1]} ({len(referenced)} pages)"
            )
        else:
            span = "—"
        lines.append(f"| REF-{index:03d} | {name} | {span} |")
    return "\n".join(lines)


def _governance_block() -> str:
    return "\n".join(
        [
            f"{EVIDENCE_SOURCE} Source text extracted verbatim from a PDF.",
            f"{EVIDENCE_EXTRACTIVE} Source sentences selected or reorganised without adding a claim.",
            f"{EVIDENCE_OPEN_QUESTION} Information missing, ambiguous or requiring confirmation.",
            "",
            f"_{INTERPRETATION_NOTICE}_",
        ]
    )


# --------------------------------------------------------------------------- #
# Level 1 — Quick Reference Notes
# --------------------------------------------------------------------------- #
def quick_reference_notes(
    sources: list[dict],
    *,
    title: str = "Quick Reference Notes",
    num_sentences: int = 8,
    num_terms: int = 12,
) -> KnowledgeOutput:
    """Build a concise, extractive Level 1 note from indexed source evidence."""
    summary = learning.extractive_summary(sources, num_sentences=num_sentences)
    sections = [
        ("1. Topic and source document", _scope_block(sources)),
        (
            "2. Purpose — what the source states",
            _extract_block(_match_sentences(sources, _PURPOSE_CUES, num_sentences), fallback=summary[:4]),
        ),
        ("3. Key concepts and terminology", _terms_table(sources, num_terms)),
        (
            "4. Main components and responsibilities",
            _extract_block(_match_sentences(sources, _COMPONENT_CUES, num_sentences)),
        ),
        (
            "5. Simplified workflow",
            _extract_block(_match_sentences(sources, _WORKFLOW_CUES, num_sentences)),
        ),
        (
            "6. Important interfaces, formats or configuration",
            _extract_block(_match_sentences(sources, _INTERFACE_CUES, num_sentences)),
        ),
        (
            "7. Operational notes and known errors",
            _extract_block(_match_sentences(sources, _OPERATIONAL_CUES, num_sentences)),
        ),
        ("8. Key takeaways", _extract_block(summary[:3])),
        ("9. Source PDF and page references", _reference_register(sources)),
    ]
    return KnowledgeOutput(
        title=title,
        level="Level 1 — Quick Reference Notes",
        sections=sections,
        sources=[item.citation for item in summary],
    )


# --------------------------------------------------------------------------- #
# Level 2 — Formal Technical Documentation Draft
# --------------------------------------------------------------------------- #
def formal_documentation_draft(
    sources: list[dict],
    documents: list[dict] | None = None,
    *,
    title: str = "Formal Technical Documentation Draft",
    num_sentences: int = 10,
    num_terms: int = 20,
) -> KnowledgeOutput:
    """Build a structured, extractive Level 2 documentation draft."""
    documents = documents or []
    summary = learning.extractive_summary(sources, num_sentences=max(num_sentences * 2, 12))

    def matched(cues: tuple[str, ...]) -> str:
        return _extract_block(_match_sentences(sources, cues, num_sentences))

    sections = [
        (
            "1. Document control and revision history",
            "\n\n".join(
                [
                    "### Source documents",
                    "",
                    _document_control_table(documents),
                    "",
                    "### Revision history",
                    "",
                    _revision_history_table(),
                ]
            ),
        ),
        (
            "2. Executive overview and business context",
            _extract_block(_match_sentences(sources, _PURPOSE_CUES, num_sentences), fallback=summary[:5]),
        ),
        (
            "3. Purpose, scope and exclusions",
            "\n\n".join(
                [
                    "### In scope (evidence-derived sources)",
                    "",
                    _scope_block(sources, documents),
                    "",
                    "### Exclusions",
                    "",
                    f"{EVIDENCE_OPEN_QUESTION} {MISSING_EVIDENCE}",
                ]
            ),
        ),
        (
            "4. System context and architecture",
            _extract_block(summary[:num_sentences]),
        ),
        (
            "5. Components and responsibilities",
            matched(_COMPONENT_CUES),
        ),
        (
            "6. End-to-end workflows and sequence details",
            matched(_WORKFLOW_CUES),
        ),
        (
            "7. Interfaces, APIs, protocols and data formats",
            matched(_INTERFACE_CUES),
        ),
        (
            "8. Dependencies and integration relationships",
            matched(_DEPENDENCY_CUES),
        ),
        (
            "9. Configuration and environment requirements",
            matched(_CONFIG_CUES),
        ),
        (
            "10. Operations, monitoring and support procedures",
            matched(_OPERATIONAL_CUES),
        ),
        (
            "11. Error handling, retries and recovery",
            matched(_OPERATIONAL_CUES),
        ),
        (
            "12. Security and access controls",
            matched(_SECURITY_CUES),
        ),
        (
            "13. Assumptions, constraints and known limitations",
            matched(_ASSUMPTION_CUES),
        ),
        ("14. Risks and open questions", _open_questions_table()),
        ("15. Acronyms and glossary", _terms_table(sources, num_terms)),
        ("16. Source references by PDF and page", _reference_register(sources)),
        ("Content governance", _governance_block()),
    ]
    return KnowledgeOutput(
        title=title,
        level="Level 2 — Formal Technical Documentation Draft",
        sections=sections,
        sources=[item.citation for item in summary],
    )
