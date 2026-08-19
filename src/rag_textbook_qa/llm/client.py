"""Small OpenAI-compatible LLM client used by the RAG service."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from openai import OpenAI

DEFAULT_LLM_BASE_URL = "https://api.ohmygpt.com/v1"
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite-preview"


class LLMConfigurationError(RuntimeError):
    """The LLM client cannot be created from the supplied configuration."""


def _validated_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMConfigurationError("LLM_API_BASE 必须是有效的 http:// 或 https:// 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LLMConfigurationError("LLM_API_BASE 不应包含凭据、查询参数或 fragment")
    return f"{normalized}/"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMSettings:
        values = os.environ if environ is None else environ
        return cls(
            api_key=values.get("LLM_API_KEY", ""),
            base_url=values.get("LLM_API_BASE", DEFAULT_LLM_BASE_URL).strip()
            or DEFAULT_LLM_BASE_URL,
            model=values.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip()
            or DEFAULT_LLM_MODEL,
        )


class LLMClient:
    """Synchronous client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = DEFAULT_LLM_MODEL,
        verbose: bool = True,
        *,
        sdk_client: Any | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("必须设置 LLM_API_KEY 或传入 api_key")
        if api_key != api_key.strip():
            raise LLMConfigurationError("LLM API key 不能包含首尾空白")
        if not model or not model.strip():
            raise LLMConfigurationError("LLM model 不能为空")

        self.base_url = _validated_base_url(base_url)
        self.default_model = model.strip()
        self.verbose = verbose
        self.client = (
            sdk_client
            if sdk_client is not None
            else OpenAI(api_key=api_key, base_url=self.base_url)
        )

        if self.verbose:
            print("LLM 客户端初始化:")
            print(f"Base URL: {self.base_url}")
            print(f"默认模型: {self.default_model}")

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        return {
            "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion": int(getattr(usage, "completion_tokens", 0) or 0),
            "total": int(getattr(usage, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _failure(error: Exception, *, model: str, label: str) -> dict[str, Any]:
        message = str(error)
        return {
            "success": False,
            "error": message,
            "answer": f"❌ {label}：{message}",
            "model": model,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "time": 0,
        }

    def generate_answer(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retry: int = 2,
    ) -> dict[str, Any]:
        selected_model = model or self.default_model
        if retry < 0:
            raise ValueError("retry 不能小于 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")

        last_error: Exception | None = None
        for attempt in range(retry + 1):
            try:
                if self.verbose:
                    if attempt:
                        print(f"重试 ({attempt}/{retry})...")
                    else:
                        print(f"调用 LLM API: {selected_model}")
                started = time.monotonic()
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                elapsed = round(time.monotonic() - started, 2)
                choice = response.choices[0]
                answer = choice.message.content or ""
                usage = self._usage(response)
                if self.verbose:
                    print(f"成功（{elapsed} 秒，{usage['total']} tokens）")
                return {
                    "success": True,
                    "answer": answer,
                    "model": selected_model,
                    "tokens": usage,
                    "time": elapsed,
                    "finish_reason": choice.finish_reason,
                }
            except Exception as exc:  # noqa: BLE001 - normalize third-party SDK errors
                last_error = exc
                if self.verbose:
                    print(f"调用失败: {str(exc)[:100]}")
                if attempt < retry:
                    time.sleep(1)

        assert last_error is not None
        return self._failure(last_error, model=selected_model, label="生成答案时出错")

    def stream_answer(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Iterator[str]:
        selected_model = model or self.default_model
        try:
            stream = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                content = getattr(choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001 - generator exposes SDK failures as text
            message = f"\n\n❌ 流式生成错误：{exc}"
            if self.verbose:
                print(message)
            yield message

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        selected_model = model or self.default_model
        try:
            started = time.monotonic()
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = round(time.monotonic() - started, 2)
            return {
                "success": True,
                "answer": response.choices[0].message.content or "",
                "model": selected_model,
                "tokens": self._usage(response),
                "time": elapsed,
            }
        except Exception as exc:  # noqa: BLE001 - normalize third-party SDK errors
            if self.verbose:
                print(f"对话失败: {str(exc)[:100]}")
            return self._failure(exc, model=selected_model, label="对话失败")


def create_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    verbose: bool = True,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_client: Any | None = None,
) -> LLMClient:
    """Create an LLM client, resolving omitted values at call time."""

    settings = LLMSettings.from_env(environ)
    return LLMClient(
        api_key=settings.api_key if api_key is None else api_key,
        base_url=settings.base_url if base_url is None else base_url,
        model=settings.model if model is None else model,
        verbose=verbose,
        sdk_client=sdk_client,
    )
