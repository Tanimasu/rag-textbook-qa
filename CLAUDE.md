# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese-language **RAG (Retrieval-Augmented Generation) Q&A system** for computer science textbooks. It parses PDFs, chunks text, vectorizes content, and uses hybrid retrieval (BM25 + semantic embeddings) to answer student questions via an LLM.

## Pipeline Steps

The processing pipeline runs in sequence:

1. **PDF to Markdown** — two options (edit hardcoded paths in script before running):
   - `parsingPDF.py`: Uses Docling + EasyOCR (GPU). Forces OCR on all pages. Known issue: misses content on scanned pages.
   - `parsingPDF_mineru.py`: Uses MinerU 2.7.6 (pipeline backend). `parse_method="auto"` detects per page whether OCR is needed. Produces ~50% more content than Docling on mixed digital/scanned PDFs. Output files are named `*_mineru.md` to distinguish from Docling output. Always uses `formula_enable=True`; for long PDFs that would OOM, automatically splits into `CHUNK_PAGES`-page segments (default 120) using pymupdf, processes each segment separately, then merges the Markdown. **Do NOT set `formula_enable=False`** — MinerU will treat formula regions as images (`![](images/...)`) which are useless for RAG.
2. **Clean Markdown** — `clean_markdown.py`: Normalizes heading hierarchy (SmartMarkdownCleaner), outputs `*_cleaned.md`
3. **Chunk** — `chunk_textbooks.py`: Splits cleaned Markdown into JSON chunks (`*_chunks.json`) using SmartTextbookChunker (max 800 chars, min 100 chars, 50 char overlap). HTML tables (MinerU output) are kept as single chunks regardless of size to preserve table integrity.
4. **Vectorize** — `vectorize_chunks.py`: Embeds chunks with `BAAI/bge-large-zh-v1.5` and stores in ChromaDB at `project/vector_db/`. Each book gets its own collection named `textbook_{book_name}`.
5. **Query/RAG** — `rag_engine.py`: Hybrid retrieval (embedding + BM25/jieba) → Cross-Encoder reranking (`BAAI/bge-reranker-base`) → prompt construction → LLM call via `llm_client.py`. Optional HyDE (`enable_hyde=True`) generates a hypothetical document via LLM before embedding the query.
6. **Evaluate** — `ragas_evaluation.py`: full RAGAS metrics (faithfulness, answer relevancy, context precision/recall) via LangChain + LLM. Also runs a **no-RAG baseline** (direct LLM, no retrieval) and prints a side-by-side comparison. Falls back to local HuggingFace embeddings if API embeddings are unavailable.

## Running Scripts

All scripts are run from `project/` as the working directory:

```bash
cd project/

# Step 1: Parse PDF (edit hardcoded paths in script first)
python parsingPDF.py          # Docling version
python parsingPDF_mineru.py   # MinerU version (recommended for scanned PDFs)

# Step 2: Clean markdown
python clean_markdown.py

# Step 3: Chunk a textbook
python chunk_textbooks.py

# Step 4: Vectorize (interactive - prompts per book)
python vectorize_chunks.py

# Step 5: Interactive Q&A (type 'test' for built-in test cases)
python rag_engine.py

# Run RAGAS evaluation (RAG + no-RAG baseline comparison)
python ragas_evaluation.py
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

**Evaluation dataset** is `project/test_questions.json` — a list of `{"question", "ground_truth", "book_name"}` objects used by `ragas_evaluation.py`. Falls back to the built-in `create_test_dataset()` if the file is missing.

**`ragas_evaluation.py`** uses LangChain's `ChatOpenAI` to drive RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall). Controlled by top-level constants: `RUN_BASELINE` (default `False`) enables an optional no-RAG baseline comparison that calls the LLM directly without retrieval — set to `True` only when needed to avoid wasting tokens. Results saved to `ragas_evaluation_results.csv` (RAG) and `ragas_baseline_results.csv` (baseline). Embeddings fall back to a local `paraphrase-multilingual-MiniLM-L12-v2` model if the API embedding endpoint is unavailable.

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
| `test_questions.json` | Ground-truth Q&A pairs for RAGAS evaluation |
| `project/vector_db/` | Persisted ChromaDB vector store |
| `project/output/` | Intermediate Markdown and chunk files |
| `project/data/` | Source PDF textbooks |

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
| `extract_images.py` | Extract embedded images from a PDF to PNG files; set `PDF_PATH` / `OUTPUT_DIR` at the bottom |
| `clean_db.py` | Interactive ChromaDB manager: list collections and delete by name or all |

## Diagnostic / Test Scripts

| File | Purpose |
|------|---------|
| `check_env.py` | Verify PyTorch, CUDA, and GPU are working |
| `check_parsing_quality.py` | Check Markdown quality after PDF parsing; set `MD_PATH` at the bottom |
| `check_quality.py` | Check chunk JSON quality after chunking; set `json_path` at the bottom |
| `test_pdf_parser.py` | Test Docling PDF→Markdown conversion; set `PDF_PATH` and `MAX_PAGES` at the bottom |
| `test_vector_db.py` | Inspect ChromaDB collections and sample records |
| `test_llm_api.py` | Test LLM API connectivity; `quick_test()` for daily use, `discover_endpoint()` when debugging a new API provider |

## Dependencies

Key Python packages: `docling`, `mineru`, `chromadb`, `sentence-transformers`, `rank-bm25`, `jieba`, `openai`, `pandas`, `openpyxl`, `tqdm`, `ragas`, `langchain-openai`, `langchain-community`, `datasets`

MinerU requires `C:\Users\Trasky\mineru.json` (auto-generated on first model download) and models cached at `D:\HuggingFaceCache\hub`. Windows Developer Mode must be enabled for HuggingFace Hub symlinks to work.
