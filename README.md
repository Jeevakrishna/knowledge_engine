# ACBS Knowledge Engine

A local, non-generative knowledge retrieval and study application for a controlled
collection of PDF documents. It extracts page-level text, stores document metadata,
pages and passages in SQLite, and retrieves passages with classical information-retrieval
techniques (BM25 and TF-IDF).

## What it does

- Discovers PDFs recursively from category folders.
- Extracts text page by page and preserves the original PDF page number.
- Detects changed files with SHA-256 and re-indexes them without creating duplicates.
- Searches all indexed documents and returns ranked passages with file/page citations.
- Filters results by document category.
- Produces extractive summaries, frequent-term lists and source-based revision prompts.
- Produces two post-ingestion knowledge outputs: **Level 1 Quick Reference Notes** and
  **Level 2 Formal Technical Documentation Draft**, both extractive and source-linked.
- Exports notes and documentation drafts to Markdown.

## Content governance

Every content item is classified so source text, extractive output and open questions stay
distinguishable (product.md §12). Exported drafts use the evidence labels `[SOURCE]`,
`[EXTRACTIVE]`, `[REVIEWER INTERPRETATION]`, `[OPEN QUESTION]` and `[CONFLICT]`. Sections the
indexed PDFs cannot support are marked `Not found in the indexed source documents.
Confirmation required.` — the application does not invent missing content. Formal
documentation drafts are drafts until a responsible technical owner reviews and approves them.

## What it deliberately does not do

No generative AI or language models, no embeddings, no vector database, no cloud APIs and
no external hosted services. See `system-design.md` and `techstack.md` for the full design.

## Project layout

```text
ACBS-Knowledge-Engine/
├── documents/            # Place source PDFs in category sub-folders
│   ├── ACBS/
│   ├── IMFT/
│   ├── Airflow/
│   ├── Solace/
│   └── MQ/
├── data/                 # knowledge.db (SQLite)
├── exports/              # notes/, documentation/ and reports/ output
├── src/
│   ├── config.py         # centralised paths and tunable parameters
│   ├── database.py       # SQLite schema and persistence
│   ├── extractor.py      # PyMuPDF page-level extraction
│   ├── chunker.py        # page-aware overlapping chunking
│   ├── ingestion.py      # discovery, hashing, orchestration
│   ├── search_engine.py  # BM25 + TF-IDF hybrid retrieval
│   ├── learning.py       # summaries, terms, prompts, governance labels
│   ├── documentation_export.py  # Level 1 / Level 2 knowledge outputs
│   └── text_utils.py     # normalisation and tokenisation
├── app.py                # Streamlit interface
└── requirements.txt      # pinned dependency set
```

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the environment is offline, install from an approved local wheel repository instead:

```powershell
python -m pip install --no-index --find-links .\wheels -r requirements.txt
```

## Running

```powershell
streamlit run app.py
```

The UI binds to localhost. Add PDFs to the relevant `documents/<category>/` folder, then use
**Index documents** to scan and ingest them. Use **Search** to retrieve ranked passages,
**Knowledge notes** to generate the Level 1 / Level 2 outputs, and **Revision** for study
prompts.

## Configuration

Paths and tunable parameters are centralised in `src/config.py` (techstack.md §6, §7.1) and
can be overridden with environment variables without editing source:

| Variable | Default | Purpose |
|---|---|---|
| `ACBS_DOCUMENTS_DIR` | `documents/` | Source PDF root |
| `ACBS_DATA_DIR` / `ACBS_DB_PATH` | `data/` / `data/knowledge.db` | SQLite location |
| `ACBS_EXPORTS_DIR` | `exports/` | Notes, documentation and report root |
| `ACBS_CHUNK_SIZE` / `ACBS_CHUNK_OVERLAP` / `ACBS_MIN_CHUNK` | `1000` / `150` / `100` | Chunking |
| `ACBS_BM25_WEIGHT` / `ACBS_TFIDF_WEIGHT` | `0.65` / `0.35` | Hybrid retrieval weights |
| `ACBS_MIN_TEXT_CHARS` / `ACBS_LOW_TEXT_THRESHOLD` | `10` / `0.5` | Extraction quality |

Configuration is validated at startup (techstack.md §7.1).

## Run statistics and statuses

Each document is recorded with a status of `INDEXED`, `LOW_TEXT`, `OCR_REQUIRED` or `ERROR`.
A PDF with no extractable text is flagged `OCR_REQUIRED`; one where fewer than half the pages
contain text is flagged `LOW_TEXT`. Both are surfaced on the dashboard rather than being
silently treated as fully indexed.

## Quality tuning

Chunk size, overlap, and the BM25/TF-IDF weighting (`0.65`/`0.35` by default) are starting
values defined in `src/config.py`. They are intended to be tuned against a labelled query set
(see `system-design.md` §11).

## Security

All processing stays within the local machine. Do not place confidential documents in
cloud-synchronised folders, and do not send document text to external services. Bind the UI
to `127.0.0.1` and restrict access to the document and database directories with OS permissions.
