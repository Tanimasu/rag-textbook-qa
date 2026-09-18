# Public question-answering service: chat page, REST API and /docs on port 7860.
#
# The image carries everything the service reads at runtime — CPU-only PyTorch, both
# retrieval models at pinned revisions, and the vector index — so it starts without
# network access to model hubs. The LLM is configured at run time through
# LLM_API_KEY / LLM_API_BASE / LLM_MODEL; without them the service still runs and
# answers from retrieval only. Never bake project/.env into an image.
#
#   docker build -t rag-textbook-qa .
#   docker run -p 7860:7860 -e LLM_API_KEY -e LLM_API_BASE -e LLM_MODEL rag-textbook-qa

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# Dependencies first, so code edits do not reinstall them. The lockfile's pins are
# installed as-is except for PyTorch: its PyPI Linux wheel pulls in several GB of
# CUDA libraries that a CPU host never loads, so the same torch version comes from
# the CPU-only index instead. --no-deps keeps any resolver from pulling them back.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-header --no-annotate \
        --extra api --extra local-models --output-file /tmp/locked.txt \
    && torch_pin="$(grep -E '^torch==' /tmp/locked.txt | cut -d' ' -f1)" \
    && test -n "$torch_pin" \
    && grep -vE '^(torch|triton|nvidia-|cuda-)' /tmp/locked.txt > /tmp/requirements.txt \
    && uv pip install --system --no-deps -r /tmp/requirements.txt \
    && uv pip install --system --no-deps --index-url https://download.pytorch.org/whl/cpu "$torch_pin" \
    && rm /tmp/locked.txt /tmp/requirements.txt

COPY scripts/fetch_models.py /tmp/fetch_models.py
RUN python /tmp/fetch_models.py \
    && HF_HUB_OFFLINE=1 python /tmp/fetch_models.py --verify \
    && rm /tmp/fetch_models.py

COPY src ./src
RUN uv pip install --system --no-deps . \
    && uv pip check --system

# The index is copied, never rebuilt: re-chunking changes chunk ids, which silently
# kills the pinned conflict quotes and invalidates every published number.
COPY artifacts/vector_db ./artifacts/vector_db

# Hugging Face runs Docker Spaces as uid 1000. Chroma writes sqlite lock files next
# to the index, and answer feedback lands in artifacts/product/.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/artifacts/product \
    && chown -R app:app /app "$HF_HOME"
USER app

ENV RAG_QA_HOME=/app \
    RAG_QA_COMPUTE_BACKEND=local \
    RAG_QA_DEVICE=cpu \
    RAG_QA_TRUST_PROXY=true \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False

EXPOSE 7860
# --public: this is a portfolio demo meant to open without a password. Cost stays
# bounded by the per-IP rate limit and the daily generation budget, and setting
# RAG_QA_ACCESS_CODE still enforces a code.
CMD ["rag-qa", "serve", "--host", "0.0.0.0", "--port", "7860", "--db-path", "/app/artifacts/vector_db", "--public"]
