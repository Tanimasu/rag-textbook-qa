"""Environment-backed configuration for local and remote compute."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from rag_textbook_qa.providers.base import ProviderError

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ProviderError(f"无法识别的布尔值: {value}")


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


@dataclass(frozen=True)
class ComputeSettings:
    backend: str = "local"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    device: str = "auto"
    remote_url: str | None = None
    remote_token: str | None = None
    remote_timeout_seconds: float = 120.0
    query_fallback_to_local: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ComputeSettings:
        values = os.environ if environ is None else environ
        backend = values.get("RAG_QA_COMPUTE_BACKEND", "local").strip().lower()
        if backend not in {"local", "remote"}:
            raise ProviderError("RAG_QA_COMPUTE_BACKEND 只能是 local 或 remote")

        remote_url_value = values.get("RAG_QA_REMOTE_URL", "").strip()
        remote_url = validate_remote_url(remote_url_value) if remote_url_value else None
        token = values.get("RAG_QA_WORKER_TOKEN", "").strip() or None
        try:
            timeout = float(values.get("RAG_QA_REMOTE_TIMEOUT", "120"))
        except ValueError as exc:
            raise ProviderError("RAG_QA_REMOTE_TIMEOUT 必须是数字") from exc
        if timeout <= 0:
            raise ProviderError("RAG_QA_REMOTE_TIMEOUT 必须大于 0")

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
            query_fallback_to_local=_parse_bool(
                values.get("RAG_QA_QUERY_FALLBACK_TO_LOCAL"), default=False
            ),
        )

    def safe_summary(self) -> dict[str, str | float | bool | None]:
        return {
            "backend": self.backend,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "device": self.device,
            "remote_url": self.remote_url,
            "remote_token_configured": self.remote_token is not None,
            "remote_timeout_seconds": self.remote_timeout_seconds,
            "query_fallback_to_local": self.query_fallback_to_local,
        }
