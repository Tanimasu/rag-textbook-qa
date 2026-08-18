"""HTTP clients for a Tailscale-reachable model worker."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_textbook_qa.providers.base import (
    DEFAULT_QUERY_INSTRUCTION,
    AuthenticationError,
    ModelIdentity,
    ModelMismatchError,
    ProviderProtocolError,
    TransientProviderError,
    validate_embeddings,
    validate_scores,
)


class RemoteWorkerClient:
    """Small JSON client with explicit error categories for safe fallback."""

    def __init__(self, base_url: str, *, token: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": "rag-textbook-qa/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = _http_error_detail(exc)
            if exc.code in {401, 403}:
                raise AuthenticationError(f"远程 Worker 认证失败: {detail}") from exc
            if exc.code == 409:
                raise ModelMismatchError(f"远程 Worker 模型不一致: {detail}") from exc
            if exc.code >= 500 or exc.code in {408, 429}:
                raise TransientProviderError(
                    f"远程 Worker 暂时不可用（HTTP {exc.code}）: {detail}"
                ) from exc
            raise ProviderProtocolError(
                f"远程 Worker 拒绝请求（HTTP {exc.code}）: {detail}"
            ) from exc
        except (TimeoutError, URLError) as exc:
            raise TransientProviderError(f"无法连接远程 Worker: {exc}") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError("远程 Worker 返回了无效 JSON") from exc
        if not isinstance(decoded, dict):
            raise ProviderProtocolError("远程 Worker 响应必须是 JSON object")
        return decoded


def _http_error_detail(error: HTTPError) -> str:
    try:
        raw = error.read().decode("utf-8")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return str(payload.get("detail", payload))
        return str(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return error.reason or "未知错误"


class _RemoteProvider:
    def __init__(self, client: RemoteWorkerClient, identity: ModelIdentity) -> None:
        self.client = client
        self._identity = identity
        self._health_verified = False

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def _ensure_compatible(self) -> None:
        if self._health_verified:
            return
        health = self.client.request("/health")
        models = health.get("models")
        if not isinstance(models, dict):
            raise ProviderProtocolError("远程 Worker /health 缺少 models")
        remote = models.get(self.identity.task)
        if not isinstance(remote, dict):
            raise ModelMismatchError(f"远程 Worker 未提供 {self.identity.task} 模型")
        if remote.get("fingerprint") != self.identity.fingerprint:
            raise ModelMismatchError(
                f"本地期望 {self.identity.model}，远程配置为 {remote.get('model', '未知')}"
            )
        self._health_verified = True


class RemoteEmbeddingProvider(_RemoteProvider):
    def __init__(self, client: RemoteWorkerClient, model: str) -> None:
        super().__init__(
            client,
            ModelIdentity(
                task="embedding",
                model=model,
                normalized=True,
                query_instruction=DEFAULT_QUERY_INSTRUCTION,
            ),
        )

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        self._ensure_compatible()
        response = self.client.request(
            "/v1/embeddings",
            method="POST",
            payload={
                "model": self.identity.model,
                "input_type": input_type,
                "texts": values,
            },
        )
        if response.get("fingerprint") != self.identity.fingerprint:
            raise ModelMismatchError("远程 embedding 响应指纹与配置不一致")
        return validate_embeddings(response.get("embeddings"), len(values))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "query")


class RemoteRerankerProvider(_RemoteProvider):
    def __init__(self, client: RemoteWorkerClient, model: str) -> None:
        super().__init__(client, ModelIdentity(task="reranker", model=model))

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        values = list(documents)
        if not values:
            return []
        self._ensure_compatible()
        response = self.client.request(
            "/v1/rerank",
            method="POST",
            payload={
                "model": self.identity.model,
                "query": query,
                "documents": values,
            },
        )
        if response.get("fingerprint") != self.identity.fingerprint:
            raise ModelMismatchError("远程 reranker 响应指纹与配置不一致")
        return validate_scores(response.get("scores"), len(values))


class FallbackEmbeddingProvider:
    """Fallback only on transient remote failures, never on auth/model errors."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        if primary.identity.fingerprint != fallback.identity.fingerprint:
            raise ModelMismatchError("embedding 回退模型必须与远程模型完全一致")
        self.primary = primary
        self.fallback = fallback

    @property
    def identity(self) -> ModelIdentity:
        return self.primary.identity

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts)
        except TransientProviderError:
            return self.fallback.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self.primary.embed_queries(texts)
        except TransientProviderError:
            return self.fallback.embed_queries(texts)


class FallbackRerankerProvider:
    def __init__(self, primary: Any, fallback: Any) -> None:
        if primary.identity.fingerprint != fallback.identity.fingerprint:
            raise ModelMismatchError("reranker 回退模型必须与远程模型完全一致")
        self.primary = primary
        self.fallback = fallback

    @property
    def identity(self) -> ModelIdentity:
        return self.primary.identity

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        try:
            return self.primary.rerank(query, documents)
        except TransientProviderError:
            return self.fallback.rerank(query, documents)
