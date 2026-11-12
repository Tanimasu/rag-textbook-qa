"""Environment-backed configuration for local and remote compute."""

from __future__ import annotations

import ipaddress
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from rag_textbook_qa.providers.base import ProviderError

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
# An unreachable Worker drops packets instead of refusing them, so without its own
# bound a connection attempt waits for the operating system (about 75 s on macOS).
DEFAULT_REMOTE_CONNECT_TIMEOUT = 5.0
# How long query fallback keeps answering locally after a transient Worker failure
# before it tries the Worker again.
DEFAULT_REMOTE_COOLDOWN = 60.0


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ProviderError(f"无法识别的布尔值: {value}")


def _parse_seconds(
    values: Mapping[str, str], name: str, default: float, *, allow_zero: bool = False
) -> float:
    try:
        seconds = float(values.get(name, str(default)))
    except ValueError as exc:
        raise ProviderError(f"{name} 必须是数字") from exc
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        requirement = "有限非负数" if allow_zero else "有限正数"
        raise ProviderError(f"{name} 必须是{requirement}")
    return seconds


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_remote_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderError("RAG_QA_REMOTE_URL 必须是有效的 http:// 或 https:// 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderError("RAG_QA_REMOTE_URL 不应包含凭据、查询参数或 fragment")
    return url


def validate_worker_token(value: str | None) -> str | None:
    """Validate a Worker bearer token without ever including it in errors."""

    if value is None or value == "":
        return None
    if value != value.strip():
        raise ProviderError("RAG_QA_WORKER_TOKEN 不能包含首尾空白")
    if not value.isascii():
        raise ProviderError("RAG_QA_WORKER_TOKEN 必须只包含 ASCII 字符")
    if any(
        character.isspace() or ord(character) < 0x21 or ord(character) == 0x7F
        for character in value
    ):
        raise ProviderError("RAG_QA_WORKER_TOKEN 不能包含空白或控制字符")
    return value


@dataclass(frozen=True)
class ComputeSettings:
    backend: str = "local"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    device: str = "auto"
    remote_url: str | None = None
    remote_token: str | None = None
    remote_timeout_seconds: float = 120.0
    remote_connect_timeout_seconds: float = DEFAULT_REMOTE_CONNECT_TIMEOUT
    remote_cooldown_seconds: float = DEFAULT_REMOTE_COOLDOWN
    query_fallback_to_local: bool = False
    reranker_batch_size: int | None = None
    reranker_max_length: int | None = None
    reranker_dtype: str = "float32"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ComputeSettings:
        values = os.environ if environ is None else environ
        backend = values.get("RAG_QA_COMPUTE_BACKEND", "local").strip().lower()
        if backend not in {"local", "remote"}:
            raise ProviderError("RAG_QA_COMPUTE_BACKEND 只能是 local 或 remote")

        remote_url_value = values.get("RAG_QA_REMOTE_URL", "").strip()
        remote_url = validate_remote_url(remote_url_value) if remote_url_value else None
        token = validate_worker_token(values.get("RAG_QA_WORKER_TOKEN"))
        timeout = _parse_seconds(values, "RAG_QA_REMOTE_TIMEOUT", 120.0)
        connect_timeout = _parse_seconds(
            values, "RAG_QA_REMOTE_CONNECT_TIMEOUT", DEFAULT_REMOTE_CONNECT_TIMEOUT
        )
        cooldown = _parse_seconds(
            values, "RAG_QA_REMOTE_COOLDOWN", DEFAULT_REMOTE_COOLDOWN, allow_zero=True
        )
        try:
            raw_batch = values.get("RAG_QA_RERANK_BATCH_SIZE")
            reranker_batch_size = int(raw_batch) if raw_batch is not None else None
        except ValueError as exc:
            raise ProviderError("RAG_QA_RERANK_BATCH_SIZE 必须是正整数") from exc
        if reranker_batch_size is not None and reranker_batch_size <= 0:
            raise ProviderError("RAG_QA_RERANK_BATCH_SIZE 必须是正整数")
        raw_length = values.get("RAG_QA_RERANK_MAX_LENGTH", "").strip()
        try:
            reranker_max_length = int(raw_length) if raw_length else None
        except ValueError as exc:
            raise ProviderError("RAG_QA_RERANK_MAX_LENGTH 必须是正整数") from exc
        if reranker_max_length is not None and reranker_max_length <= 0:
            raise ProviderError("RAG_QA_RERANK_MAX_LENGTH 必须是正整数")
        reranker_dtype = values.get("RAG_QA_RERANK_DTYPE", "float32").strip().lower()
        if reranker_dtype not in {"float32", "float16"}:
            raise ProviderError("RAG_QA_RERANK_DTYPE 只能是 float32 或 float16")

        if backend == "remote":
            if remote_url is None:
                raise ProviderError("remote 模式必须设置 RAG_QA_REMOTE_URL")
            if not is_loopback_host(urlsplit(remote_url).hostname) and token is None:
                raise ProviderError("非本机 remote 地址必须设置 RAG_QA_WORKER_TOKEN")

        return cls(
            backend=backend,
            embedding_model=values.get("RAG_QA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
            or DEFAULT_EMBEDDING_MODEL,
            reranker_model=values.get("RAG_QA_RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()
            or DEFAULT_RERANKER_MODEL,
            device=values.get("RAG_QA_DEVICE", "auto").strip().lower() or "auto",
            remote_url=remote_url,
            remote_token=token,
            remote_timeout_seconds=timeout,
            remote_connect_timeout_seconds=connect_timeout,
            remote_cooldown_seconds=cooldown,
            query_fallback_to_local=_parse_bool(
                values.get("RAG_QA_QUERY_FALLBACK_TO_LOCAL"), default=False
            ),
            reranker_batch_size=reranker_batch_size,
            reranker_max_length=reranker_max_length,
            reranker_dtype=reranker_dtype,
        )

    def safe_summary(self) -> dict[str, str | float | bool | None]:
        return {
            "backend": self.backend,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "device": self.device,
            "reranker_batch_size": self.reranker_batch_size,
            "reranker_max_length": self.reranker_max_length,
            "reranker_dtype": self.reranker_dtype,
            "remote_url": self.remote_url,
            "remote_token_configured": self.remote_token is not None,
            "remote_timeout_seconds": self.remote_timeout_seconds,
            "remote_connect_timeout_seconds": self.remote_connect_timeout_seconds,
            "remote_cooldown_seconds": self.remote_cooldown_seconds,
            "query_fallback_to_local": self.query_fallback_to_local,
        }
