"""ACBS Knowledge Engine — Streamlit user interface.

Screens: Dashboard, Index documents, Search, Knowledge notes, Revision
(product.md §11). Runs bound to localhost only; no external calls are made.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src import config
from src import database as db
from src import documentation_export
from src import learning
from src.ingestion import IngestionService
from src.search_engine import SearchEngine

DB_PATH = config.DB_PATH
DOCUMENTS_DIR = config.DOCUMENTS_DIR
NOTES_DIR = config.NOTES_DIR
DOCUMENTATION_DIR = config.DOCUMENTATION_DIR
REPORTS_DIR = config.REPORTS_DIR

config.validate()
config.ensure_directories()

STATUS_ICONS = {
    db.STATUS_INDEXED: "✅ INDEXED",
    db.STATUS_LOW_TEXT: "⚠️ LOW_TEXT",
    db.STATUS_OCR_REQUIRED: "🛑 OCR_REQUIRED",
    db.STATUS_ERROR: "❌ ERROR",
}

st.set_page_config(page_title="ACBS Knowledge Engine", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resources — the version key forces a rebuild after ingestion.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _build_engine(db_path: str, version: int) -> SearchEngine:
    engine = SearchEngine(db_path)
    engine.build()
    return engine


@st.cache_data(show_spinner=False)
def _cached_stats(db_path: str, version: int) -> dict:
    with db.connection(db_path) as conn:
        return db.get_stats(conn)


def index_version() -> int:
    return st.session_state.get("index_version", 0)


def bump_version() -> None:
    st.session_state["index_version"] = index_version() + 1


def get_engine() -> SearchEngine:
    return _build_engine(str(DB_PATH), index_version())


def collect_sources(document_id: int | None = None, category: str | None = None) -> list[dict]:
    """Gather {filename, page_number, text} sources for the learning utilities."""
    sources: list[dict] = []
    with db.connection(DB_PATH) as conn:
        if document_id is not None:
            document = db.get_document(conn, document_id)
            documents = [document] if document else []
        elif category and category != "All":
            documents = [d for d in db.list_documents(conn) if d["category"] == category]
        else:
            documents = db.list_documents(conn)

        for document in documents:
            for page in db.fetch_document_pages(conn, document["document_id"]):
                text = page["extracted_text"] or ""
                if text.strip():
                    sources.append(
                        {
                            "filename": document["filename"],
                            "page_number": page["page_number"],
                            "text": text,
                        }
                    )
    return sources


def collect_documents(
    document_id: int | None = None, category: str | None = None
) -> list[dict]:
    """Return document metadata records for the selected scope."""
    with db.connection(DB_PATH) as conn:
        if document_id is not None:
            document = db.get_document(conn, document_id)
            documents = [document] if document else []
        elif category and category != "All":
            documents = [d for d in db.list_documents(conn) if d["category"] == category]
        else:
            documents = db.list_documents(conn)
    return [dict(document) for document in documents]


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def dashboard_screen() -> None:
    st.header("Dashboard")
    stats = _cached_stats(str(DB_PATH), index_version())

    row1 = st.columns(4)
    row1[0].metric("PDFs discovered", stats["documents"])
    row1[1].metric("Pages processed", stats["pages"])
    row1[2].metric("Chunks created", stats["chunks"])
    row1[3].metric("Indexed pages / PDF pages", f"{stats['indexed_pages']} / {stats['total_pages']}")

    row2 = st.columns(3)
    row2[0].metric("Low-text PDFs", stats["low_text"])
    row2[1].metric("Failed files", stats["failed"])
    row2[2].metric("Ingestion runs", stats["ingestion_runs"])

    with db.connection(DB_PATH) as conn:
        last_run = db.get_last_ingestion_run(conn)
        documents = db.list_documents(conn)

    if last_run is not None:
        st.subheader("Last ingestion run")
        st.write(
            f"Run #{last_run['run_id']} started {last_run['started_at']} "
            f"(completed {last_run['completed_at'] or '—'}) — "
            f"seen {last_run['files_seen']}, indexed {last_run['files_indexed']}, "
            f"skipped {last_run['files_skipped']}, failed {last_run['files_failed']}."
        )
    else:
        st.info("No ingestion has been run yet. Go to **Index documents** to begin.")

    st.subheader("Indexed documents")
    if documents:
        frame = pd.DataFrame(
            [
                {
                    "Filename": d["filename"],
                    "Category": d["category"],
                    "Status": STATUS_ICONS.get(d["status"], d["status"]),
                    "Total pages": d["total_pages"],
                    "Indexed pages": d["indexed_pages"],
                    "Ingested at": d["ingested_at"],
                }
                for d in documents
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download inventory CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="acbs_document_inventory.csv",
            mime="text/csv",
        )
    else:
        st.caption("No documents indexed yet.")


def ingest_screen() -> None:
    st.header("Index documents")
    st.caption(f"Documents directory: {DOCUMENTS_DIR}")

    service = IngestionService(db_path=DB_PATH, documents_dir=DOCUMENTS_DIR)

    found = service.scan()
    st.write(f"**{len(found)}** PDF file(s) discovered under the documents directory.")
    if found:
        st.dataframe(
            pd.DataFrame(
                [
                    {"File": p.name, "Category": service.category_for(p)}
                    for p in found
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No PDFs found. Place PDF files into category folders under "
            f"`{DOCUMENTS_DIR}` (for example `documents/IMFT/`)."
        )

    col1, col2 = st.columns(2)
    start = col1.button("Start ingestion", type="primary", disabled=not found)
    rescan = col2.button("Force re-index all files", disabled=not found)

    if start or rescan:
        with st.spinner("Extracting and indexing PDFs…"):
            summary = service.ingest(force=rescan)
        bump_version()
        st.success(
            f"Run #{summary.run_id}: seen {summary.files_seen}, "
            f"indexed {summary.files_indexed}, skipped {summary.files_skipped}, "
            f"failed {summary.files_failed}."
        )
        _render_outcomes(summary)

    st.divider()
    st.subheader("Reconciliation")
    st.caption(
        "Remove database records for source PDFs that have been deleted. "
        "This is never run automatically during ingestion."
    )
    if st.button("Reconcile deleted files"):
        removed = service.reconcile()
        bump_version()
        st.success(f"Removed {removed} record(s) for missing source files.")


def _render_outcomes(summary) -> None:
    if not summary.outcomes:
        return
    records = []
    for outcome in summary.outcomes:
        records.append(
            {
                "Filename": outcome.filename,
                "Category": outcome.category,
                "Result": STATUS_ICONS.get(outcome.status, outcome.status),
                "Pages": outcome.total_pages,
                "Indexed pages": outcome.indexed_pages,
                "Chunks": outcome.chunks,
                "Detail": outcome.error or "",
            }
        )
    frame = pd.DataFrame(records)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"ingestion_report_{timestamp}.csv"
    report_path.write_text(frame.to_csv(index=False), encoding="utf-8")
    st.caption(f"Report saved to {report_path}")
    st.download_button(
        "Download ingestion report CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=report_path.name,
        mime="text/csv",
    )


def search_screen() -> None:
    st.header("Search")

    with db.connection(DB_PATH) as conn:
        categories = ["All"] + db.list_categories(conn)

    with st.form("search_form"):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        query = col1.text_input("Query", placeholder="e.g. managed file transfer")
        category = col2.selectbox("Category", categories)
        top_k = col3.number_input("Results", min_value=1, max_value=50, value=10)
        mode = col4.selectbox("Mode", ["hybrid", "bm25", "tfidf"])
        submitted = st.form_submit_button("Search", type="primary")

    if not submitted:
        st.caption(
            "Lexical search matches wording, not meaning. Use exact technical terms, "
            "acronyms and alternate phrasings for best coverage."
        )
        return

    if not query.strip():
        st.warning("Enter a query.")
        return

    engine = get_engine()
    if not engine.is_loaded:
        st.warning("The index is empty. Index some documents first.")
        return

    results = engine.search(query, category=category, top_k=int(top_k), mode=mode)
    if not results:
        st.info("No matching passages. Try different terms or a broader category filter.")
        return

    st.success(f"{len(results)} passage(s) ranked using the '{mode}' retrieval mode.")

    as_dicts = [
        {
            "filename": r.filename,
            "category": r.category,
            "page_number": r.page_number,
            "chunk_text": r.chunk_text,
        }
        for r in results
    ]

    for rank, result in enumerate(results, start=1):
        label = (
            f"{rank}. {result.filename} — page {result.page_number} "
            f"(score {result.score:.3f})"
        )
        with st.expander(label, expanded=rank <= 3):
            st.caption(
                f"Category: {result.category} · BM25 {result.bm25_score:.3f} · "
                f"TF-IDF {result.tfidf_score:.3f}"
            )
            st.write(result.chunk_text)

    with st.expander("Compare passages as a Markdown table"):
        table = learning.comparison_markdown(as_dicts)
        st.markdown(table)
        st.download_button(
            "Download comparison (Markdown)",
            table.encode("utf-8"),
            file_name="evidence_comparison.md",
            mime="text/markdown",
        )


def notes_screen() -> None:
    st.header("Knowledge notes")
    st.caption(
        "Post-ingestion knowledge outputs (product.md §9). Both levels are extractive "
        "and remain drafts until a technical owner reviews them."
    )

    with db.connection(DB_PATH) as conn:
        documents = db.list_documents(conn)
        categories = ["All"] + db.list_categories(conn)

    if not documents:
        st.info("No documents indexed yet.")
        return

    output_type = st.radio(
        "Output",
        [
            "Level 1 — Quick Reference Notes",
            "Level 2 — Formal Technical Documentation Draft",
        ],
        horizontal=False,
    )

    scope = st.radio("Scope", ["Whole category", "Single document"], horizontal=True)
    document_id = None
    category = None
    scope_label = ""

    if scope == "Single document":
        options = {f"{d['filename']} [{d['category']}]": d["document_id"] for d in documents}
        chosen = st.selectbox("Document", list(options))
        document_id = options[chosen]
        scope_label = chosen
    else:
        category = st.selectbox("Category", categories)
        scope_label = f"Category: {category}"

    col1, col2 = st.columns(2)
    num_sentences = col1.slider("Evidence sentences per section", 3, 20, 8)
    top_terms = col2.slider("Term candidates", 5, 40, 20)

    if not st.button("Generate draft", type="primary"):
        return

    sources = collect_sources(document_id=document_id, category=category)
    if not sources:
        st.warning("No extractable text found for this scope.")
        return

    base_title = f"{output_type.split('—', 1)[1].strip()} — {scope_label}"
    if output_type.startswith("Level 1"):
        output = documentation_export.quick_reference_notes(
            sources, title=base_title, num_sentences=num_sentences, num_terms=top_terms
        )
        save_dir = NOTES_DIR
        save_name = "quick_reference_notes"
    else:
        documents_meta = collect_documents(document_id=document_id, category=category)
        output = documentation_export.formal_documentation_draft(
            sources,
            documents_meta,
            title=base_title,
            num_sentences=num_sentences,
            num_terms=top_terms,
        )
        save_dir = DOCUMENTATION_DIR
        save_name = "technical_documentation_draft"

    st.warning(f"Output status: {output.review_status}")
    st.caption(
        "Evidence labels: [SOURCE] · [EXTRACTIVE] · [OPEN QUESTION]. "
        f'Unsupported sections read: "{learning.MISSING_EVIDENCE}"'
    )

    for heading, body in output.sections:
        st.subheader(heading)
        st.markdown(body)

    markdown = output.to_markdown()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.subheader("Export")
    st.download_button(
        "Download (Markdown)",
        markdown.encode("utf-8"),
        file_name=f"{save_name}_{timestamp}.md",
        mime="text/markdown",
    )
    if st.button(f"Save to {save_dir.relative_to(config.PROJECT_ROOT)}"):
        path = save_dir / f"{save_name}_{timestamp}.md"
        path.write_text(markdown, encoding="utf-8")
        st.success(f"Saved to {path}")


def revision_screen() -> None:
    st.header("Revision")

    with db.connection(DB_PATH) as conn:
        documents = db.list_documents(conn)
        categories = ["All"] + db.list_categories(conn)

    if not documents:
        st.info("No documents indexed yet.")
        return

    scope = st.radio("Source", ["Whole category", "Single document"], horizontal=True)
    document_id = None
    category = None

    if scope == "Single document":
        options = {f"{d['filename']} [{d['category']}]": d["document_id"] for d in documents}
        chosen = st.selectbox("Document", list(options))
        document_id = options[chosen]
    else:
        category = st.selectbox("Category", categories)

    max_prompts = st.slider("Number of prompts", 3, 30, 10)

    if not st.button("Generate revision prompts", type="primary"):
        return

    sources = collect_sources(document_id=document_id, category=category)
    prompts = learning.revision_prompts(sources, max_prompts=max_prompts)
    if not prompts:
        st.warning("Not enough source text to build prompts.")
        return

    for index, prompt in enumerate(prompts, start=1):
        st.markdown(f"**{index}. {prompt['question']}**")
        with st.expander("Reveal answer"):
            st.write(f"**Answer:** {prompt['answer']}")
            st.caption(
                f"Source statement: {prompt['statement']}  \n"
                f"Reference: {prompt['filename']} — page {prompt['page_number']}"
            )

    lines = ["# Revision prompts", ""]
    for index, prompt in enumerate(prompts, start=1):
        lines.append(f"{index}. {prompt['question']}")
        lines.append(f"   - Answer: {prompt['answer']}")
        lines.append(f"   - Source: {prompt['filename']} — page {prompt['page_number']}")
        lines.append("")
    markdown = "\n".join(lines)
    st.download_button(
        "Download prompts (Markdown)",
        markdown.encode("utf-8"),
        file_name="revision_prompts.md",
        mime="text/markdown",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    db.init_db(DB_PATH)

    st.sidebar.title("📚 ACBS Knowledge Engine")
    st.sidebar.caption("Local, non-generative PDF knowledge retrieval.")
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Index documents", "Search", "Knowledge notes", "Revision"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Evidence before interpretation. Every result cites its source PDF and page."
    )

    if page == "Dashboard":
        dashboard_screen()
    elif page == "Index documents":
        ingest_screen()
    elif page == "Search":
        search_screen()
    elif page == "Knowledge notes":
        notes_screen()
    else:
        revision_screen()


if __name__ == "__main__":
    main()
