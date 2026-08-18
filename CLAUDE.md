# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese-language **RAG (Retrieval-Augmented Generation) Q&A system** for computer science textbooks. It parses PDFs, chunks text, vectorizes content, and uses hybrid retrieval (BM25 + semantic embeddings) to answer student questions via an LLM.

## Pipeline Steps

The processing pipeline runs in sequence:

1. **PDF to Markdown** — two compatibility options, reading local PDFs from `data/raw/`:
   - `parsingPDF.py`: Uses Docling + EasyOCR (GPU). Forces OCR on all pages. Known issue: misses content on scanned pages.
   - `parsingPDF_mineru.py`: Uses MinerU 2.7.6 (pipeline backend). `parse_method="auto"` detects per page whether OCR is needed. Produces ~50% more content than Docling on mixed digital/scanned PDFs. Output files are named `*_mineru.md` to distinguish from Docling output. Always uses `formula_enable=True`; for long PDFs that would OOM, automatically splits into `CHUNK_PAGES`-page segments (default 100) using pymupdf, processes each segment separately, then merges the Markdown. **Do NOT set `formula_enable=False`** — MinerU will treat formula regions as images (`![](images/...)`) which are useless for RAG.
2. **Clean Markdown** — `clean_markdown.py`: Normalizes heading hierarchy (SmartMarkdownCleaner), outputs `*_cleaned.md`
3. **Chunk** — `chunk_textbooks.py`: Splits cleaned Markdown into JSON chunks (`*_chunks.json`) using SmartTextbookChunker (max 800 chars, min 100 chars, 50 char overlap). HTML tables (MinerU output) are kept as single chunks regardless of size to preserve table integrity.
4. **Vectorize** — `vectorize_chunks.py`: Embeds `data/chunks/` files with `BAAI/bge-large-zh-v1.5` and stores them in `artifacts/vector_db/`. Each book gets its own collection named `textbook_{book_name}`.
5. **Query/RAG** — `rag_engine.py`: Hybrid retrieval (embedding + BM25/jieba) → Cross-Encoder reranking (`BAAI/bge-reranker-base`) → prompt construction → LLM call via `llm_client.py`. Optional HyDE (`enable_hyde=True`) generates a hypothetical document via LLM before embedding the query.
6. **Evaluate** — `ragas_evaluation.py`: full RAGAS metrics (faithfulness, answer relevancy, context precision/recall) via LangChain + LLM. Also runs a **no-RAG baseline** (direct LLM, no retrieval) and prints a side-by-side comparison. Falls back to local HuggingFace embeddings if API embeddings are unavailable.

## Running Scripts

Preferred commands are run from the repository root:

```bash
# Step 1: Parse the default PDF under data/raw/
python project/parsingPDF.py          # Docling version
python project/parsingPDF_mineru.py   # MinerU version (recommended for scanned PDFs)

# Step 2: Clean markdown
rag-qa ingest clean INPUT.md --output data/cleaned/BOOK_cleaned.md

# Step 3: Chunk a textbook
rag-qa ingest chunk data/cleaned/BOOK_cleaned.md --output data/chunks/BOOK_chunks.json

# Step 4: Vectorize (interactive - prompts per book)
python project/vectorize_chunks.py

# Step 5: Interactive Q&A (type 'test' for built-in test cases)
python project/rag_engine.py

# Run RAGAS evaluation (RAG + no-RAG baseline comparison)
python project/ragas_evaluation.py
```

## Key Architecture Decisions

**Book identifiers** (used as ChromaDB collection suffixes) — defined in `vectorize_chunks.py::BOOK_NAME_MAP` and referenced in `rag_engine.py` test queries:

| 教材 | Docling 标识 | MinerU 标识 |
|------|-------------|-------------|
| 操作系统 | `os` | `os_mineru` |
| 计算机组成原理 | `computer_organization` | `computer_organization_mineru` |
| 计算机网络 | `computer_network` | `computer_network_mineru` |
| 数据结构 | `data_structure` | `data_structure_mineru` |
| 数据库原理及应用教程 | `database` | `database_mineru` |

The mapping lives in `vectorize_chunks.py::BOOK_NAME_MAP`. ChromaDB enforces `[a-zA-Z0-9._-]` for collection names.

**ChromaDB collections** follow the naming pattern `textbook_{book_name}`. The RAGEngine and MultiBookVectorizer both rely on this convention.

**Hybrid retrieval scoring** in `rag_engine.py`: embedding similarity weight 1.0, BM25 score weight 0.3 (BM25 raw scores are first scaled by 0.05). Sections containing "小结", "习题", "思考题" and chunks with <100 chars are filtered from semantic results.

**Embedding model** in `rag_engine.py` and `vectorize_chunks.py`: `BAAI/bge-large-zh-v1.5`. When HyDE is disabled, queries are prefixed with `"为这个句子生成表示以用于检索相关文章："` before encoding (as recommended for BGE models).

**HyDE (Hypothetical Document Embeddings)** in `rag_engine.py`: controlled by `RAGEngine(enable_hyde=True/False)`, default `True`. When enabled, the LLM generates a ~100-char hypothetical textbook passage from the query, which is then embedded and used for vector retrieval instead of the raw query. This improves semantic alignment between query and document vectors (context_precision +6%, context_recall +4.5% vs. disabled).

**Cross-Encoder reranking** in `rag_engine.py`: enabled by default (`enable_reranker=True`) using `BAAI/bge-reranker-base`. First-pass retrieval is widened to `top_k * 3` candidates; the reranker scores each `(query, content)` pair and returns the true top_k. Controlled by `RAGEngine(enable_reranker=True/False)`. Falls back gracefully if model unavailable.

**LLM client** (`llm_client.py`) uses an OpenAI-compatible API configured via `project/.env` (`LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL`). Per-script overrides: `RAG_API_KEY/BASE/MODEL` for rag_engine, `RAGAS_API_KEY/BASE/MODEL` for ragas_evaluation. Default model: `gemini-3.1-flash-lite-preview`.

**Evaluation dataset** is `data/evaluation/test_questions.json` — a list of `{"question", "ground_truth", "book_name"}` objects used by `ragas_evaluation.py`. Falls back to the built-in `create_test_dataset()` if the file is missing.

**`ragas_evaluation.py`** uses LangChain's `ChatOpenAI` to drive RAGAS metrics (faithfulness, answer relevancy, context precision, context recall). Controlled by top-level constants: `RUN_BASELINE` enables an optional no-RAG comparison that calls the LLM directly without retrieval. Results are saved under `artifacts/evaluations/`. Embeddings fall back to a local `paraphrase-multilingual-MiniLM-L12-v2` model if the API embedding endpoint is unavailable.

## Key Files

| File | Purpose |
|------|---------|
| `rag_engine.py` | Core RAGEngine class - entry point for Q&A |
| `llm_client.py` | LLMClient wrapping OpenAI-compatible API |
| `vectorize_chunks.py` | MultiBookVectorizer - ChromaDB + sentence-transformers |
| `chunk_textbooks.py` | SmartTextbookChunker - Markdown to JSON chunks |
| `clean_markdown.py` | SmartMarkdownCleaner - heading normalization |
| `parsingPDF_mineru.py` | MinerU-based PDF parser (alternative to parsingPDF.py); outputs `*_mineru.md` |
| `ragas_evaluation.py` | RAGAS evaluation (faithfulness, relevancy, precision, recall) + optional no-RAG baseline (`RUN_BASELINE`) |
| `data/evaluation/test_questions.json` | Ground-truth Q&A pairs for RAGAS evaluation |
| `data/parsed/` | Parsed Markdown from PDF backends |
| `data/cleaned/` | Cleaned Markdown |
| `data/chunks/` | Chunk JSON and previews |
| `artifacts/vector_db/` | Rebuildable local ChromaDB store |
| `artifacts/evaluations/` | RAGAS and baseline outputs |

### Streamlit UI (`app.py` + submodules)

`app.py` is the entry point (~55 lines, pure assembly). Functionality is split across:

| File | Purpose |
|------|---------|
| `app.py` | Entry point: page config, session state, tab layout |
| `config/constants.py` | Path constants (`BASE_DIR`, `VECTOR_DB_PATH`, `RAGAS_RESULTS_PATH`, `TEST_QUESTIONS_PATH`), `BOOK_NAME_LABELS` (book_id → Chinese name), `RAGAS_METRIC_LABELS` (metric key → Chinese label) |
| `services/app_services.py` | `load_available_books()` (reads ChromaDB SQLite), `load_engine()` (`@cache_resource`), `load_ragas_results()`, `run_ragas_evaluation()` |
| `ui/styles.py` | `inject_custom_styles()` — hero banner, status grid, source card CSS |
| `ui/layout.py` | `render_sidebar()` (returns dict of params), `render_hero()` (3-cell status grid) |
| `ui/chat_page.py` | `render_chat_tab()` — chat input, session_state message history |
| `ui/eval_page.py` | `render_eval_tab()` — bar chart, sortable/searchable results table |
| `ui/helpers.py` | `format_book_label()`, `format_section_label()`, `render_answer_block()`, `render_sources_expander()`, `render_source_preview()` |

## Utility Scripts

| File | Purpose |
|------|---------|
| `get_models.py` | List available models from the configured API endpoint |
| `extract_images.py` | Extract embedded images from the default `data/raw/` PDF into `artifacts/images/` |
| `clean_db.py` | Interactive ChromaDB manager: list collections and delete by name or all |

## Diagnostic / Test Scripts

| File | Purpose |
|------|---------|
| `check_env.py` | Verify PyTorch, CUDA, and GPU are working |
| `check_parsing_quality.py` | Compatibility wrapper that checks an explicit Markdown path |
| `check_quality.py` | Compatibility wrapper that checks an explicit chunk JSON path |
| `test_pdf_parser.py` | Test Docling PDF→Markdown conversion; set `PDF_PATH` and `MAX_PAGES` at the bottom |
| `test_vector_db.py` | Inspect ChromaDB collections and sample records |
| `test_llm_api.py` | Test LLM API connectivity; `quick_test()` for daily use, `discover_endpoint()` when debugging a new API provider |

## Dependencies

Key Python packages: `docling`, `mineru`, `chromadb`, `sentence-transformers`, `rank-bm25`, `jieba`, `openai`, `pandas`, `openpyxl`, `tqdm`, `ragas`, `langchain-openai`, `langchain-community`, `datasets`

Use the dedicated Conda environment from `environment.yml`. After activation, synchronize the lockfile into that environment without creating `.venv`:

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --inexact
```

Do not install heavyweight extras unless the task requires them. Development tools are declared in the standard `dependency-groups.dev` group; model, UI, parser, and evaluation runtimes remain optional extras.

Embedding and reranking are selected through `rag_textbook_qa.providers`. `local` loads sentence-transformers lazily; `remote` calls only `/health`, `/v1/embeddings`, and `/v1/rerank` on the optional Worker. Keep ChromaDB and BM25 in the caller process. Vectorization must use one provider for the whole job; query fallback may catch only transient connection/server failures, never authentication or model-fingerprint errors.

When exposing `rag-qa worker serve` beyond loopback, bind to the machine's Tailscale IP and require `RAG_QA_WORKER_TOKEN`. Do not add public port-forwarding instructions or log the token.

MinerU configuration and model-cache locations are machine-specific. Do not commit absolute Windows, macOS, or Linux paths; use the tool's environment/configuration on each machine.
