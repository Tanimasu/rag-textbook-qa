"""GPU worker exposing only embedding and reranking over HTTP."""

import platform
import secrets
from collections.abc import Mapping
from typing import Any

from rag_textbook_qa.providers.base import (
    AuthenticationError,
    EmbeddingProvider,
    MissingOptionalDependencyError,
    ModelMismatchError,
    ProviderError,
    RerankerProvider,
)
from rag_textbook_qa.providers.config import (
    is_loopback_host,
    validate_worker_token,
)
from rag_textbook_qa.providers.local import LocalEmbeddingProvider, LocalRerankerProvider

MAX_BATCH_ITEMS = 128
MAX_BATCH_CHARACTERS = 250_000


def _validated_worker_token(token: str | None) -> str | None:
    try:
        return validate_worker_token(token)
    except ProviderError as exc:
        raise AuthenticationError(str(exc)) from exc


def validate_worker_bind(host: str, token: str | None) -> None:
    """Require application authentication whenever the worker leaves loopback."""

    validated_token = _validated_worker_token(token)
    if not is_loopback_host(host) and validated_token is None:
        raise AuthenticationError("Worker 监听非本机地址时必须设置 RAG_QA_WORKER_TOKEN")


class WorkerRuntime:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        *,
        token: str | None,
        device: str,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.token = _validated_worker_token(token)
        self.device = device

    def authorize(self, authorization: str | None) -> None:
        if self.token is None:
            return
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise AuthenticationError("缺少 Bearer token")
        supplied = authorization[len(prefix) :]
        try:
            valid_supplied = validate_worker_token(supplied)
        except ProviderError as exc:
            raise AuthenticationError("Bearer token 无效") from exc
        if valid_supplied is None or not secrets.compare_digest(valid_supplied, self.token):
            raise AuthenticationError("Bearer token 无效")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol_version": "1",
            "device": self.device,
            "platform": platform.system(),
            "models": {
                "embedding": self.embedding_provider.identity.as_dict(),
                "reranker": self.reranker_provider.identity.as_dict(),
            },
        }

    def embeddings(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        if model != self.embedding_provider.identity.model:
            raise ModelMismatchError(
                f"请求模型 {model!r}，Worker 模型 {self.embedding_provider.identity.model!r}"
            )
        texts = _validated_texts(payload.get("texts"), field="texts")
        input_type = payload.get("input_type")
        if input_type == "document":
            embeddings = self.embedding_provider.embed_documents(texts)
        elif input_type == "query":
            embeddings = self.embedding_provider.embed_queries(texts)
        else:
            raise ValueError("input_type 必须是 document 或 query")
        return {
            "model": self.embedding_provider.identity.model,
            "fingerprint": self.embedding_provider.identity.fingerprint,
            "embeddings": embeddings,
        }

    def rerank(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        if model != self.reranker_provider.identity.model:
            raise ModelMismatchError(
                f"请求模型 {model!r}，Worker 模型 {self.reranker_provider.identity.model!r}"
            )
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        documents = _validated_texts(payload.get("documents"), field="documents")
        return {
            "model": self.reranker_provider.identity.model,
            "fingerprint": self.reranker_provider.identity.fingerprint,
            "scores": self.reranker_provider.rerank(query, documents),
        }


def _validated_texts(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} 必须是非空字符串数组")
    if len(value) > MAX_BATCH_ITEMS:
        raise ValueError(f"{field} 每批最多 {MAX_BATCH_ITEMS} 条")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} 必须是非空字符串数组")
    if sum(len(item) for item in value) > MAX_BATCH_CHARACTERS:
        raise ValueError(f"{field} 每批字符总数不能超过 {MAX_BATCH_CHARACTERS}")
    return value


def create_worker_app(runtime: WorkerRuntime):
    """Create the optional FastAPI app without importing it in normal clients."""

    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "启动 Worker 需要安装：uv sync --extra worker"
        ) from exc

    app = FastAPI(title="rag-textbook-qa model worker", version="1")

    def authorize(request: Request) -> None:
        try:
            runtime.authorize(request.headers.get("Authorization"))
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def execute(operation):
        try:
            return operation()
        except ModelMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MissingOptionalDependencyError as exc:
            raise HTTPException(status_code=424, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health")
    def health(request: Request):
        authorize(request)
        return runtime.health()

    @app.post("/v1/embeddings")
    def embeddings(payload: dict[str, Any], request: Request):
        authorize(request)
        return execute(lambda: runtime.embeddings(payload))

    @app.post("/v1/rerank")
    def rerank(payload: dict[str, Any], request: Request):
        authorize(request)
        return execute(lambda: runtime.rerank(payload))

    return app


def run_worker_server(
    *,
    host: str,
    port: int,
    embedding_model: str,
    reranker_model: str,
    device: str,
    token: str | None,
) -> None:
    validate_worker_bind(host, token)
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "启动 Worker 需要安装：uv sync --extra worker"
        ) from exc

    runtime = WorkerRuntime(
        LocalEmbeddingProvider(embedding_model, device=device),
        LocalRerankerProvider(reranker_model, device=device),
        token=token,
        device=device,
    )
    uvicorn.run(create_worker_app(runtime), host=host, port=port)
