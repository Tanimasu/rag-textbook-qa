"""HTTP clients for a Tailscale-reachable model worker."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_textbook_qa.providers.base import (
    DEFAULT_QUERY_INSTRUCTION,
    AuthenticationError,
    ModelIdentity,
    ModelMismatchError,
    ProviderCall,
    ProviderProtocolError,
    ProviderTelemetry,
    TransientProviderError,
    validate_embeddings,
    validate_scores,
)
from rag_textbook_qa.providers.config import validate_worker_token


class RemoteWorkerClient:
    """Small JSON client with explicit error categories for safe fallback."""

    def __init__(self, base_url: str, *, token: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = validate_worker_token(token)
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
        self.remote_device = "remote"
        self.remote_platform: str | None = None
        self.telemetry = ProviderTelemetry()

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
        device = health.get("device")
        if isinstance(device, str) and device.strip():
            self.remote_device = device.strip().lower()
        remote_platform = health.get("platform")
        if isinstance(remote_platform, str) and remote_platform.strip():
            self.remote_platform = remote_platform.strip()
        self._health_verified = True

    def _record_call(
        self,
        started: float,
        *,
        success: bool,
        error_category: str | None = None,
    ) -> None:
        self.telemetry.record(
            ProviderCall(
                task=self.identity.task,
                backend="remote",
                model=self.identity.model,
                device=self.remote_device,
                platform=self.remote_platform,
                elapsed_seconds=time.monotonic() - started,
                success=success,
                error_category=error_category,
            )
        )


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
        started = time.monotonic()
        try:
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
            result = validate_embeddings(response.get("embeddings"), len(values))
        except Exception as exc:
            self._record_call(started, success=False, error_category=type(exc).__name__)
            raise
        self._record_call(started, success=True)
        return result

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
        started = time.monotonic()
        try:
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
            result = validate_scores(response.get("scores"), len(values))
        except Exception as exc:
            self._record_call(started, success=False, error_category=type(exc).__name__)
            raise
        self._record_call(started, success=True)
        return result


class FallbackEmbeddingProvider:
    """Fallback only on transient remote failures, never on auth/model errors."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        if primary.identity.fingerprint != fallback.identity.fingerprint:
            raise ModelMismatchError("embedding 回退模型必须与远程模型完全一致")
        self.primary = primary
        self.fallback = fallback
        self.telemetry = ProviderTelemetry()

    @property
    def identity(self) -> ModelIdentity:
        return self.primary.identity

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._run("embed_documents", texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._run("embed_queries", texts)

    def _run(self, method: str, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        started = time.monotonic()
        primary_marker = _telemetry_marker(self.primary)
        try:
            result = getattr(self.primary, method)(values)
        except TransientProviderError as remote_error:
            fallback_marker = _telemetry_marker(self.fallback)
            try:
                result = getattr(self.fallback, method)(values)
            except Exception as exc:
                self._record_effective_call(
                    started,
                    _latest_call(self.fallback, fallback_marker),
                    backend="local",
                    fallback_used=True,
                    success=False,
                    error_category=type(exc).__name__,
                )
                raise
            self._record_effective_call(
                started,
                _latest_call(self.fallback, fallback_marker),
                backend="local",
                fallback_used=True,
                success=True,
                error_category=type(remote_error).__name__,
            )
            return result
        except Exception as exc:
            self._record_effective_call(
                started,
                _latest_call(self.primary, primary_marker),
                backend="remote",
                fallback_used=False,
                success=False,
                error_category=type(exc).__name__,
            )
            raise
        self._record_effective_call(
            started,
            _latest_call(self.primary, primary_marker),
            backend="remote",
            fallback_used=False,
            success=True,
        )
        return result

    def _record_effective_call(
        self,
        started: float,
        effective: ProviderCall | None,
        *,
        backend: str,
        fallback_used: bool,
        success: bool,
        error_category: str | None = None,
    ) -> None:
        self.telemetry.record(
            ProviderCall(
                task="embedding",
                backend=backend,
                model=self.identity.model,
                device=effective.device if effective else backend,
                platform=effective.platform if effective else None,
                elapsed_seconds=time.monotonic() - started,
                success=success,
                fallback_used=fallback_used,
                error_category=error_category,
            )
        )


class FallbackRerankerProvider:
    def __init__(self, primary: Any, fallback: Any) -> None:
        if primary.identity.fingerprint != fallback.identity.fingerprint:
            raise ModelMismatchError("reranker 回退模型必须与远程模型完全一致")
        self.primary = primary
        self.fallback = fallback
        self.telemetry = ProviderTelemetry()

    @property
    def identity(self) -> ModelIdentity:
        return self.primary.identity

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        values = list(documents)
        if not values:
            return []
        started = time.monotonic()
        primary_marker = _telemetry_marker(self.primary)
        try:
            result = self.primary.rerank(query, values)
        except TransientProviderError as remote_error:
            fallback_marker = _telemetry_marker(self.fallback)
            try:
                result = self.fallback.rerank(query, values)
            except Exception as exc:
                self._record_effective_call(
                    started,
                    _latest_call(self.fallback, fallback_marker),
                    backend="local",
                    fallback_used=True,
                    success=False,
                    error_category=type(exc).__name__,
                )
                raise
            self._record_effective_call(
                started,
                _latest_call(self.fallback, fallback_marker),
                backend="local",
                fallback_used=True,
                success=True,
                error_category=type(remote_error).__name__,
            )
            return result
        except Exception as exc:
            self._record_effective_call(
                started,
                _latest_call(self.primary, primary_marker),
                backend="remote",
                fallback_used=False,
                success=False,
                error_category=type(exc).__name__,
            )
            raise
        self._record_effective_call(
            started,
            _latest_call(self.primary, primary_marker),
            backend="remote",
            fallback_used=False,
            success=True,
        )
        return result

    def _record_effective_call(
        self,
        started: float,
        effective: ProviderCall | None,
        *,
        backend: str,
        fallback_used: bool,
        success: bool,
        error_category: str | None = None,
    ) -> None:
        self.telemetry.record(
            ProviderCall(
                task="reranker",
                backend=backend,
                model=self.identity.model,
                device=effective.device if effective else backend,
                platform=effective.platform if effective else None,
                elapsed_seconds=time.monotonic() - started,
                success=success,
                fallback_used=fallback_used,
                error_category=error_category,
            )
        )


def _telemetry_marker(provider: Any) -> int | None:
    telemetry = getattr(provider, "telemetry", None)
    return telemetry.mark() if isinstance(telemetry, ProviderTelemetry) else None


def _latest_call(provider: Any, marker: int | None) -> ProviderCall | None:
    if marker is None:
        return None
    telemetry = getattr(provider, "telemetry", None)
    if not isinstance(telemetry, ProviderTelemetry):
        return None
    calls = telemetry.since(marker)
    return calls[-1] if calls else None
