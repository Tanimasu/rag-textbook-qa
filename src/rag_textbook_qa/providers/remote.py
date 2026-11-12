"""HTTP clients for a Tailscale-reachable model worker."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from functools import cache
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from rag_textbook_qa.providers.base import (
    DEFAULT_QUERY_INSTRUCTION,
    WORKER_MAX_BATCH_CHARACTERS,
    WORKER_MAX_BATCH_ITEMS,
    AuthenticationError,
    ModelIdentity,
    ModelMismatchError,
    ProviderCall,
    ProviderError,
    ProviderProtocolError,
    ProviderTelemetry,
    TransientProviderError,
    validate_embeddings,
    validate_scores,
    validate_worker_health_status,
)
from rag_textbook_qa.providers.config import (
    DEFAULT_REMOTE_CONNECT_TIMEOUT,
    validate_worker_token,
)


def _worker_batches(values: list[str]) -> list[list[str]]:
    """Preserve whole inputs and their order within the worker protocol limits."""

    # Validate all inputs before the first request, so a later oversized document
    # does not waste earlier inference or silently lose text by truncation.
    if any(not isinstance(value, str) or not value for value in values):
        raise ProviderProtocolError("远程模型输入必须是非空字符串")
    if any(len(value) > WORKER_MAX_BATCH_CHARACTERS for value in values):
        raise ProviderProtocolError(f"单条远程模型输入不能超过 {WORKER_MAX_BATCH_CHARACTERS} 字符")
    batches, batch, characters = [], [], 0
    for value in values:
        if batch and (len(batch) >= WORKER_MAX_BATCH_ITEMS
                      or characters + len(value) > WORKER_MAX_BATCH_CHARACTERS):
            batches.append(batch)
            batch, characters = [], 0
        batch.append(value)
        characters += len(value)
    if batch:
        batches.append(batch)
    return batches


class _NoWorkerRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        # A Worker is an explicitly configured authenticated endpoint. urllib's
        # default redirect handler can forward its bearer token to another host.
        return None


@cache
def _bounded_connection(http_class: type, connect_timeout: float) -> type:
    """A connection class that gives up on an unreachable host early."""

    class BoundedConnection(http_class):
        def connect(self) -> None:
            # urllib hands one timeout to the socket for its whole life. Model
            # inference can legitimately take minutes; reaching the host cannot.
            response_timeout = self.timeout
            self.timeout = connect_timeout
            try:
                super().connect()
            finally:
                self.timeout = response_timeout
            self.sock.settimeout(response_timeout)

    return BoundedConnection


class _BoundedConnect:
    def __init__(self, connect_timeout: float) -> None:
        super().__init__()
        self._connect_timeout = connect_timeout

    def do_open(self, http_class, req, **http_conn_args):
        return super().do_open(
            _bounded_connection(http_class, self._connect_timeout), req, **http_conn_args
        )


class _WorkerHTTPHandler(_BoundedConnect, HTTPHandler):
    pass


class _WorkerHTTPSHandler(_BoundedConnect, HTTPSHandler):
    pass


def _positive_seconds(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


class RemoteWorkerClient:
    """Small JSON client with explicit error categories for safe fallback."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None,
        timeout: float,
        connect_timeout: float = DEFAULT_REMOTE_CONNECT_TIMEOUT,
    ) -> None:
        if not _positive_seconds(timeout):
            raise ProviderError("远程 Worker timeout 必须是有限正数")
        if not _positive_seconds(connect_timeout):
            raise ProviderError("远程 Worker connect timeout 必须是有限正数")
        self.base_url = base_url.rstrip("/")
        self.token = validate_worker_token(token)
        self.timeout = timeout
        # Establishing the connection never gets longer than the whole request.
        self.connect_timeout = min(connect_timeout, timeout)
        self._opener = build_opener(
            _NoWorkerRedirects(),
            _WorkerHTTPHandler(self.connect_timeout),
            _WorkerHTTPSHandler(self.connect_timeout),
        )

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
            with self._opener.open(request, timeout=self.timeout) as response:
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
        except (TimeoutError, URLError, ConnectionError, IncompleteRead) as exc:
            # A successful status line does not guarantee that the response body
            # arrived. Discard partial output and let query fallback recompute it.
            raise TransientProviderError("远程 Worker 连接失败或响应未完整接收") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError("远程 Worker 返回了无效 JSON") from exc
        if not isinstance(decoded, dict):
            raise ProviderProtocolError("远程 Worker 响应必须是 JSON object")
        return decoded


def _http_error_detail(error: HTTPError) -> str:
    try:
        with error:
            raw = error.read().decode("utf-8")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return str(payload.get("detail", payload))
        return str(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, IncompleteRead):
        # Authentication/model errors retain their status classification even if
        # the explanatory body was cut short; they must never trigger fallback.
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
        validate_worker_health_status(health)
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
        self.remote_device = "remote"
        self.remote_platform = None
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
            batches = _worker_batches(values)
            self._ensure_compatible()
            embeddings = []
            for batch in batches:
                response = self.client.request(
                    "/v1/embeddings",
                    method="POST",
                    payload={"model": self.identity.model, "input_type": input_type, "texts": batch},
                )
                if response.get("fingerprint") != self.identity.fingerprint:
                    raise ModelMismatchError("远程 embedding 响应指纹与配置不一致")
                embeddings.extend(validate_embeddings(response.get("embeddings"), len(batch)))
            # Also check that dimensions agree across separate worker responses.
            result = validate_embeddings(embeddings, len(values))
        except Exception as exc:
            self._record_call(started, success=False, error_category=type(exc).__name__)
            if isinstance(exc, TransientProviderError):
                self._health_verified = False
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
            batches = _worker_batches(values)
            self._ensure_compatible()
            result = []
            for batch in batches:
                response = self.client.request(
                    "/v1/rerank",
                    method="POST",
                    payload={"model": self.identity.model, "query": query, "documents": batch},
                )
                if response.get("fingerprint") != self.identity.fingerprint:
                    raise ModelMismatchError("远程 reranker 响应指纹与配置不一致")
                result.extend(validate_scores(response.get("scores"), len(batch)))
        except Exception as exc:
            self._record_call(started, success=False, error_category=type(exc).__name__)
            if isinstance(exc, TransientProviderError):
                self._health_verified = False
            raise
        self._record_call(started, success=True)
        return result


class RemoteCooldown:
    """Remembers a transient Worker failure so later queries skip the wait.

    Only the query fallback wrappers consult it. A provider without a fallback
    still contacts the Worker on every call and reports each failure.
    """

    def __init__(
        self, seconds: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ProviderError("远程 Worker cooldown 必须是有限非负数")
        self.seconds = seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._resume_at: float | None = None

    def active(self) -> bool:
        with self._lock:
            if self._resume_at is None:
                return False
            if self._clock() < self._resume_at:
                return True
            self._resume_at = None
            return False

    def start(self) -> None:
        if self.seconds <= 0:
            return
        with self._lock:
            self._resume_at = self._clock() + self.seconds


class _QueryFallback:
    """Fallback only on transient remote failures, never on auth/model errors."""

    _task: str

    def __init__(
        self, primary: Any, fallback: Any, *, cooldown: RemoteCooldown | None = None
    ) -> None:
        if primary.identity.fingerprint != fallback.identity.fingerprint:
            raise ModelMismatchError(f"{self._task} 回退模型必须与远程模型完全一致")
        self.primary = primary
        self.fallback = fallback
        self.cooldown = cooldown
        self.telemetry = ProviderTelemetry()

    @property
    def identity(self) -> ModelIdentity:
        return self.primary.identity

    def _call(self, invoke: Callable[[Any], Any]) -> Any:
        started = time.monotonic()
        if self.cooldown is not None and self.cooldown.active():
            # The Worker failed moments ago. The fallback is the same model, so
            # answer from it now instead of waiting for the same failure again.
            return self._call_fallback(invoke, started, TransientProviderError.__name__)
        primary_marker = _telemetry_marker(self.primary)
        try:
            result = invoke(self.primary)
        except TransientProviderError as remote_error:
            if self.cooldown is not None:
                self.cooldown.start()
            return self._call_fallback(invoke, started, type(remote_error).__name__)
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

    def _call_fallback(
        self, invoke: Callable[[Any], Any], started: float, remote_category: str
    ) -> Any:
        fallback_marker = _telemetry_marker(self.fallback)
        try:
            result = invoke(self.fallback)
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
            error_category=remote_category,
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
                task=self._task,
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


class FallbackEmbeddingProvider(_QueryFallback):
    _task = "embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._run("embed_documents", texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._run("embed_queries", texts)

    def _run(self, method: str, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        return self._call(lambda provider: getattr(provider, method)(values))


class FallbackRerankerProvider(_QueryFallback):
    _task = "reranker"

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        values = list(documents)
        if not values:
            return []
        return self._call(lambda provider: provider.rerank(query, values))


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
