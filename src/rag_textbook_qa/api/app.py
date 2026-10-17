"""Public question-answering API and the chat page served beside it.

The deployed demo is one process: this app answers `/v1/ask`, streams answers as
server-sent events on `/v1/ask/stream`, documents itself at `/docs`, and serves a
single-page chat UI at `/`. It exposes a deliberately small slice of the engine.
Query decomposition, citation checking and HyDE stay off because none of them has
passed acceptance and each one adds model calls, and no provider error text is ever
returned, because it can carry upstream detail.
"""

import asyncio
import contextlib
import hashlib
import json
import math
import re
import sys
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from rag_textbook_qa import __version__
from rag_textbook_qa.api.feedback import (
    AnswerRecordExpiredError,
    AnswerRegistry,
    FeedbackStore,
)
from rag_textbook_qa.api.guard import (
    AccessDenied,
    AccessGuard,
    Busy,
    GuardSettings,
    RateLimited,
    RequestAdmission,
)
from rag_textbook_qa.api.imports import MAX_UPLOAD_BYTES, ImportBusyError, ImportManager
from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.llm import GenerationCancelled
from rag_textbook_qa.providers.base import AuthenticationError, MissingOptionalDependencyError
from rag_textbook_qa.providers.config import is_loopback_host
from rag_textbook_qa.rag.references import answer_body, render_source_sections
from rag_textbook_qa.timing import finite_seconds

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - only without the api extra
    raise MissingOptionalDependencyError("启动问答服务需要安装：uv sync --extra api") from exc

MAX_QUERY_CHARACTERS = 500
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
PUBLIC_FAILURE = "暂时无法完成回答，请稍后再试。"
NO_EVIDENCE = "没有在教材中找到能用的相关内容，可以换个问法，或者换一本教材试试。"
RETRIEVAL_ONLY_NOTICES = {
    "budget": "今日的生成额度已经用完，下面只列出检索到的教材原文，没有调用大模型。",
    "llm_unavailable": "当前没有配置可用的大模型，下面只列出检索到的教材原文。",
}
_HEADING_FIELDS = ("chapter", "section_h2", "section_h3", "section_h4")
_TIMING_FIELDS = ("retrieval_seconds", "generation_seconds", "first_token_seconds", "total_seconds")
_CITATION_REFERENCE = re.compile(r"【参考资料\s*(\d+)】")
# Long enough to cost nothing, short enough that no proxy idles out the connection
# while a reasoning model has produced no visible text yet.
_KEEPALIVE_SECONDS = 10.0
_QUEUE_POLL_SECONDS = 0.05


class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARACTERS,
        description="学生的问题，最多 500 字",
    )
    book_id: str | None = Field(
        default=None,
        description="教材标识，取值见 /v1/books；省略则检索全部教材",
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="检索片段数")


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    rating: Literal["helpful", "needs_improvement"]
    reason: Literal[
        "not_answered",
        "irrelevant_sources",
        "unsupported_answer",
        "incomplete",
        "too_slow",
        "other",
    ] | None = None
    comment: str = Field(default="", max_length=500)


class ImportStartRequest(BaseModel):
    confirm_replace: Literal[True]


def public_result(result: Mapping[str, Any], *, retrieval_only: str | None) -> dict[str, Any]:
    """Reduce an engine result to what a public client may see.

    The prompt and the raw ranked candidates are internals; provider error text can
    carry upstream detail. None of the three leaves this function.
    """

    sources = [_public_source(source) for source in result.get("context_sources") or []]
    if not sources:
        status, message = "no_evidence", NO_EVIDENCE
    elif retrieval_only is not None:
        status, message = "retrieval_only", RETRIEVAL_ONLY_NOTICES[retrieval_only]
    elif result.get("success"):
        status, message = "answered", None
    else:
        status, message = "failed", PUBLIC_FAILURE
    answer = result.get("answer") if status == "answered" else None
    if answer:
        answer = render_source_sections(str(answer), result.get("context_sources") or [])
    if status == "answered" and not answer_body(str(answer or "")).strip():
        status, message, answer = "failed", PUBLIC_FAILURE, None
    execution = result.get("execution") or {}
    compute = {
        name: stage
        for name in ("embedding", "reranker")
        if (stage := _public_compute_stage(execution.get(name))) is not None
    }
    return {
        "status": status,
        "answer": answer,
        "message": message,
        "sources": sources,
        "conflicts": [conflict.get("topic") for conflict in result.get("source_conflicts") or []],
        "timing": {
            field: seconds for field in _TIMING_FIELDS
            if (seconds := finite_seconds(execution.get(field))) is not None
        },
        "compute": compute,
        "citation_integrity": _citation_integrity(str(answer or ""), sources)
        if status == "answered"
        else None,
    }


def _public_compute_stage(stage: Any) -> dict[str, Any] | None:
    """Keep only bounded, display-safe provider telemetry at the public boundary."""

    if not isinstance(stage, Mapping):
        return None
    backend = str(stage.get("backend") or "unknown").lower()
    if backend not in {"local", "remote", "mixed"}:
        backend = "unknown"
    device = str(stage.get("device") or "unknown").strip().lower()
    if re.fullmatch(r"cuda:[0-9]+", device):
        device = "cuda"
    if device not in {"cpu", "cuda", "mps", "mixed"}:
        device = "unknown"
    platform = str(stage.get("platform") or "unknown")
    if platform not in {"Windows", "Darwin", "Linux", "mixed"}:
        platform = "unknown"
    elapsed = finite_seconds(stage.get("elapsed_seconds"))
    return {
        "backend": backend,
        "device": device,
        "platform": platform,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "fallback_used": bool(stage.get("fallback_used")),
    }


def _citation_integrity(answer: str, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check citation links only; this deliberately makes no grounding claim."""

    cited = sorted({int(match) for match in _CITATION_REFERENCE.findall(answer_body(answer))})
    available = {
        source["citation_id"]
        for source in sources
        if type(source.get("citation_id")) is int
    }
    unknown = [identifier for identifier in cited if identifier not in available]
    if not cited:
        status = "missing"
    elif unknown:
        status = "invalid"
    else:
        status = "linked"
    return {"status": status, "cited": cited, "unknown": unknown}


def _public_source(source: Mapping[str, Any]) -> dict[str, Any]:
    book_id = str(source.get("book_name") or "")
    fingerprint = source.get("source_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        fingerprint = None
        if not source.get("truncated") and not source.get("table_compacted"):
            fingerprint = hashlib.sha256(str(source.get("content") or "").strip().encode()).hexdigest()
    return {
        "citation_id": source.get("citation_id"),
        "book_id": book_id,
        "chunk_id": str(source.get("chunk_id") or ""),
        "source_sha256": fingerprint,
        "book": BOOK_LABELS.get(book_id, book_id),
        "section": " > ".join(str(source[field]) for field in _HEADING_FIELDS if source.get(field)),
        "excerpt": str(source.get("content") or ""),
        "truncated": bool(source.get("truncated")),
        "table_compacted": bool(source.get("table_compacted")),
    }


def _log_failure(exc: BaseException) -> None:
    # The type name only: the message can carry provider detail or the question itself.
    print(f"问答失败: {type(exc).__name__}", file=sys.stderr, flush=True)


def public_retrieval(result: Mapping[str, Any]) -> dict[str, Any]:
    """Show bounded, source-only stage snapshots without engine internals."""
    stages = []
    for key, label in (("keyword", "关键词"), ("embedding", "向量"), ("fusion", "融合"),
                       ("rerank", "重排"), ("context", "最终上下文")):
        rows = result.get("stages", {}).get(key, [])
        items = []
        for position, row in enumerate(rows[:30], 1):
            source = _public_source(row)
            source["preview_truncated"] = len(source["excerpt"]) > 12000
            source["excerpt"] = source["excerpt"][:12000]
            scores = {
                name: float(value) for name in ("similarity", "bm25_score", "rrf_score", "rerank_score")
                if isinstance(value := row.get(name), (int, float))
                and not isinstance(value, bool) and math.isfinite(value)
            }
            ranks = {
                name: rank for name in ("bm25", "embedding")
                if type(rank := row.get("source_ranks", {}).get(name)) is int and rank > 0
            }
            items.append({**source, "position": position, "scores": scores, "source_ranks": ranks})
        stages.append({"id": key, "label": label, "count": len(rows), "items": items})
    return {
        "stages": stages,
        "reranker_enabled": bool(result.get("reranker_enabled")),
        "context_budget": result.get("context_budget"),
        "elapsed_seconds": finite_seconds(result.get("elapsed_seconds")),
        "generation_calls": 0,
        "fusion_weights": {
            key: float(value) for key in ("embedding", "bm25")
            if isinstance(value := result.get("fusion_weights", {}).get(key), (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        },
    }


def _client_address(request: Request, *, trust_proxy: bool) -> str:
    """Identify a caller for rate limiting.

    Behind exactly one trusted proxy, the rightmost X-Forwarded-For entry is the one
    that proxy appended; everything to its left came from the client and is forgeable.
    """

    if trust_proxy:
        forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",")]
        if forwarded[-1]:
            return forwarded[-1]
    return request.client.host if request.client else "unknown"


class _AnswerJob:
    """Start a thread only after an admitted coroutine has the execution slot."""

    def __init__(
        self,
        produce: Callable[[Callable[[str], None], Callable[[], bool]], dict[str, Any]],
        admission: RequestAdmission,
    ) -> None:
        self._produce = produce
        self.admission = admission
        self.events: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

    def _put(self, item: tuple[str, dict[str, Any]] | None) -> None:
        # The loop is gone if the server shut down while this answer was running.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self.events.put_nowait, item)

    def start(self) -> None:
        def work() -> None:
            try:
                if self.admission.is_cancelled():
                    return
                result = self._produce(
                    lambda chunk: self._put(("chunk", {"text": chunk})),
                    self.admission.is_cancelled,
                )
                self._put(("result", result))
            except GenerationCancelled:
                pass  # The reader stopped the answer. This is not a server error.
            except Exception as exc:  # noqa: BLE001 - hide provider text at the public boundary
                _log_failure(exc)
                self._put(("error", {"status": "failed", "message": PUBLIC_FAILURE}))
            finally:
                self.admission.release()
                self._put(None)

        try:
            threading.Thread(target=work, name="rag-qa-answer", daemon=True).start()
        except BaseException:
            self.admission.release()
            raise


class _AnswerStreamResponse(StreamingResponse):
    """Release an admission even if disconnect happens before iteration begins."""

    def __init__(self, *args: Any, admission: RequestAdmission, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._admission = admission

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._admission.cancel()


async def _stream(
    produce: Callable[[Callable[[str], None], Callable[[], bool]], dict[str, Any]],
    guard: AccessGuard,
    *,
    keepalive_seconds: float = _KEEPALIVE_SECONDS,
    admission: RequestAdmission | None = None,
) -> AsyncIterator[str]:
    """Relay one answer as server-sent events.

    Waiting requests hold only bounded admissions and coroutines. One answer thread
    starts after acquiring the execution slot. Starlette cancels this relay when a
    client leaves, releasing a queued admission immediately or signalling the active
    producer to stop. An active worker keeps its slot until it really exits.
    """

    admission = admission or guard.admit_request()
    job = _AnswerJob(produce, admission)
    loop = asyncio.get_running_loop()
    next_keepalive = loop.time() + keepalive_seconds
    try:
        while not admission.try_start():
            if admission.is_cancelled():
                return
            if loop.time() >= next_keepalive:
                yield ": keepalive\n\n"
                next_keepalive = loop.time() + keepalive_seconds
            await asyncio.sleep(min(_QUEUE_POLL_SECONDS, keepalive_seconds))
        job.start()
        while True:
            try:
                item = await asyncio.wait_for(job.events.get(), keepalive_seconds)
            except TimeoutError:
                # An SSE comment: the page ignores it, but the connection stays warm.
                yield ": keepalive\n\n"
                continue
            if item is None:
                return
            name, data = item
            yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    except Busy as exc:
        yield f"event: error\ndata: {json.dumps({'status': 'busy', 'message': str(exc)}, ensure_ascii=False)}\n\n"
    except Exception as exc:  # noqa: BLE001 - thread startup failures are public-safe too
        _log_failure(exc)
        yield f"event: error\ndata: {json.dumps({'status': 'failed', 'message': PUBLIC_FAILURE}, ensure_ascii=False)}\n\n"
    finally:
        admission.cancel()


async def _answer_once(
    produce: Callable[[Callable[[str], None], Callable[[], bool]], dict[str, Any]],
    admission: RequestAdmission,
    request: Request,
) -> dict[str, Any]:
    """Wait for a REST answer without occupying the shared FastAPI threadpool."""

    job = _AnswerJob(produce, admission)
    try:
        while True:
            if admission.is_cancelled() or await request.is_disconnected():
                raise HTTPException(status_code=499, detail="请求已取消")
            if admission.try_start():
                break
            await asyncio.sleep(_QUEUE_POLL_SECONDS)
        job.start()
        while True:
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="请求已取消")
            try:
                item = await asyncio.wait_for(job.events.get(), _QUEUE_POLL_SECONDS)
            except TimeoutError:
                continue
            if item is None:
                raise HTTPException(status_code=499, detail="请求已取消")
            name, data = item
            if name == "result":
                return data
            if name == "error":
                raise HTTPException(status_code=500, detail=PUBLIC_FAILURE)
    finally:
        admission.cancel()


def create_api_app(
    engine: Any,
    guard: AccessGuard,
    books: Sequence[Mapping[str, Any]],
    *,
    feedback_store: FeedbackStore,
    import_manager: ImportManager | None = None,
) -> FastAPI:
    """Build the public app around one shared engine and one guard."""

    book_list = [dict(book) for book in books]
    answers = AnswerRegistry()
    page = (
        resources.files("rag_textbook_qa.api")
        .joinpath("static/index.html")
        .read_text(encoding="utf-8")
    )
    retrieval_page = resources.files("rag_textbook_qa.api").joinpath("static/retrieval.html").read_text(encoding="utf-8")
    import_page = resources.files("rag_textbook_qa.api").joinpath("static/imports.html").read_text(encoding="utf-8")
    backup_script = resources.files("rag_textbook_qa.api").joinpath("static/history-backup.js").read_text(encoding="utf-8")

    app = FastAPI(
        title="计算机教材 RAG 问答",
        version=__version__,
        description=(
            "基于五本计算机教材的检索增强问答。回答依据检索到的教材片段生成，并附带引用的原文。"
            "公开版关闭了尚未通过验收的实验功能，并对调用频率和每日生成次数设了上限。"
        ),
    )

    def current_books() -> list[dict[str, Any]]:
        indexed_books = getattr(engine, "list_indexed_books", None)
        if not callable(indexed_books):
            return book_list
        return [
            {
                "book_id": book["book_name"],
                "label": BOOK_LABELS.get(book["book_name"], book["book_name"]),
                "chunks": book["count"],
            }
            for book in indexed_books()
        ]

    def authorize(request: Request, *, rate_scope: str = "question") -> None:
        client = _client_address(request, trust_proxy=guard.settings.trust_proxy)
        try:
            # Rate first, so guessing the access code is throttled like any other call.
            guard.check_rate(client, scope=rate_scope)
            guard.check_access(request.headers.get("X-Access-Code"))
        except RateLimited as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc
        except AccessDenied as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def local_imports(request: Request, *, rate_scope: str = "imports") -> ImportManager:
        if import_manager is None:
            raise HTTPException(status_code=404, detail="教材导入只在本机服务提供")
        if request.client is None or not is_loopback_host(request.client.host):
            raise HTTPException(status_code=403, detail="教材导入只接受本机连接")
        origin = request.headers.get("Origin")
        if (not is_loopback_host(request.url.hostname or "")
                or (origin is not None and origin != str(request.base_url).rstrip("/"))):
            raise HTTPException(status_code=403, detail="请从本机服务页面操作教材导入")
        authorize(request, rate_scope=rate_scope)
        return import_manager

    async def admit(payload: AskRequest, request: Request) -> tuple[str, str | None, int]:
        authorize(request)
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="问题不能为空")
        if payload.book_id is not None:
            # Catalog reads include native Chroma calls and publication retry
            # sleeps; neither may hold up streaming or disconnect handling.
            books = await run_in_threadpool(current_books)
            if payload.book_id not in {book["book_id"] for book in books}:
                raise HTTPException(status_code=404, detail="没有这本教材的索引")
        return query, payload.book_id, payload.top_k

    def answer(
        query: str,
        book_id: str | None,
        top_k: int,
        sink: Callable[[str], None] | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        reserved_on = None
        if not getattr(engine, "enable_llm", False):
            retrieval_only = "llm_unavailable"
        else:
            reserved_on = guard.reserve_generation()
            retrieval_only = None if reserved_on is not None else "budget"
        generation_started = False

        def mark_generation_started() -> None:
            nonlocal generation_started
            generation_started = True

        try:
            result = engine.ask(
                query=query,
                book_name=book_id,
                top_k=top_k,
                use_llm=retrieval_only is None,
                use_hyde=False,
                use_decomposition=False,
                verify_citations=False,
                on_answer_chunk=sink if retrieval_only is None else None,
                should_stop=should_stop,
                on_generation_start=mark_generation_started,
            )
        except GenerationCancelled as exc:
            # The client can stop between the engine's start signal and its HTTP
            # request, so its request_sent flag is authoritative for cancellation.
            generation_started = exc.request_sent
            raise
        finally:
            if reserved_on is not None and not generation_started:
                # No evidence, retrieval failure, or a stop before the model call.
                guard.refund_generation(reserved_on)
        response = public_result(result, retrieval_only=retrieval_only)
        response["answer_id"] = answers.remember(
            query=query,
            book_id=book_id,
            result=response,
        )
        return response

    # HEAD as well as GET: uptime probes and some proxies check the root without a body.
    @app.api_route(
        "/",
        methods=["GET", "HEAD"],
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/retrieval", response_class=HTMLResponse, include_in_schema=False)
    def retrieval_index() -> HTMLResponse:
        return HTMLResponse(retrieval_page)

    @app.get("/imports", response_class=HTMLResponse, include_in_schema=False)
    def imports_index() -> HTMLResponse:
        return HTMLResponse(import_page)

    @app.get("/assets/history-backup.js", include_in_schema=False)
    def history_backup_script() -> Response:
        return Response(backup_script, media_type="application/javascript")

    @app.get("/health", summary="服务状态")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "books": len(current_books()),
            "imports_enabled": import_manager is not None,
            **guard.status(),
        }

    @app.get("/v1/books", summary="已索引的教材")
    def list_books() -> list[dict[str, Any]]:
        return current_books()

    @app.get("/v1/imports", summary="本机教材导入能力")
    def import_capabilities(request: Request) -> dict[str, Any]:
        manager = local_imports(request, rate_scope="import_status")
        return {"max_upload_bytes": MAX_UPLOAD_BYTES, "formats": [".md", ".json"],
                "storage_path": str(manager.root),
                "books": [{"book_id": key, "label": label} for key, label in BOOK_LABELS.items()]}

    @app.post("/v1/imports/prepare", summary="上传教材并解析预览，不建立索引")
    async def prepare_import(book_id: str, filename: str, request: Request) -> dict[str, Any]:
        manager = local_imports(request)
        if len(filename) > 1024:
            raise HTTPException(status_code=422, detail="文件名太长")
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="教材文件不能超过 8 MB")
            content.extend(part)
        try:
            return await run_in_threadpool(manager.prepare, book_id, filename, bytes(content))
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="教材解析失败，请检查文件格式、UTF-8 编码和分块字段") from None
        except Exception as exc:  # noqa: BLE001 - do not expose local paths
            _log_failure(exc)
            raise HTTPException(status_code=503, detail="教材暂时无法解析，请检查本机存储后重试") from None

    @app.get("/v1/imports/{job_id}", summary="本机教材解析和索引进度")
    def import_status(job_id: str, request: Request) -> dict[str, Any]:
        manager = local_imports(request, rate_scope="import_status")
        try:
            return manager.status(job_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="没有这个导入任务") from None
        except Exception as exc:  # noqa: BLE001 - do not expose local paths
            _log_failure(exc)
            raise HTTPException(status_code=503, detail="导入进度暂时无法读取") from None

    @app.post("/v1/imports/{job_id}/start", summary="确认建立或替换本机教材索引")
    def start_import(job_id: str, payload: ImportStartRequest, request: Request) -> dict[str, Any]:
        manager = local_imports(request)
        try:
            return manager.start(job_id)
        except ImportBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except LookupError:
            raise HTTPException(status_code=404, detail="没有这个导入任务") from None
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=422, detail="导入预览文件已改变或丢失，请重新上传") from None
        except Exception as exc:  # noqa: BLE001 - do not expose local paths
            _log_failure(exc)
            raise HTTPException(status_code=503, detail="导入暂时无法启动") from None

    @app.get("/v1/books/{book_id}/passages", summary="按引用位置阅读教材及相邻片段")
    def read_passages(book_id: str, chunk_id: str, request: Request) -> dict[str, Any]:
        authorize(request, rate_scope="reading")
        if not chunk_id.strip() or len(chunk_id) > 512 or len(book_id) > 503:
            raise HTTPException(status_code=422, detail="教材或片段编号格式不正确")
        read = getattr(engine, "read_passages", None)
        if not callable(read):
            raise HTTPException(status_code=404, detail="当前服务不提供原文阅读")
        try:
            window = read(book_id, chunk_id)
            return {
                "book_id": book_id,
                "anchor_chunk_id": window["anchor_chunk_id"],
                "source_sha256": window["source_sha256"],
                "previous_chunk_id": window["previous_chunk_id"],
                "next_chunk_id": window["next_chunk_id"],
                "passages": [
                    {**_public_source(row), "is_anchor": bool(row.get("is_anchor"))}
                    for row in window["passages"]
                ],
            }
        except LookupError:
            raise HTTPException(status_code=404, detail="原文位置已不存在，可能是教材索引已经更新") from None
        except Exception as exc:  # noqa: BLE001 - keep provider details private
            _log_failure(exc)
            raise HTTPException(status_code=503, detail="原文暂时无法读取，请稍后重试") from None

    @app.post("/v1/ask", summary="提问，一次性返回回答与引用")
    async def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
        query, book_id, top_k = await admit(payload, request)
        try:
            admission = guard.admit_request()
            return await _answer_once(
                lambda sink, should_stop: answer(query, book_id, top_k, None, should_stop),
                admission,
                request,
            )
        except Busy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - public responses never carry provider text
            _log_failure(exc)
            raise HTTPException(status_code=500, detail=PUBLIC_FAILURE) from None

    @app.post("/v1/retrieval/inspect", summary="观察关键词、向量、融合、重排和上下文")
    async def inspect_retrieval(payload: AskRequest, request: Request) -> dict[str, Any]:
        query, book_id, top_k = await admit(payload, request)
        if book_id is None:
            raise HTTPException(status_code=422, detail="观察检索过程需要先选择一本教材")
        inspect = getattr(engine, "inspect_retrieval", None)
        if not callable(inspect):
            raise HTTPException(status_code=503, detail="当前服务不提供检索观察")
        try:
            admission = guard.admit_request()
            return await _answer_once(
                lambda sink, should_stop: public_retrieval(
                    inspect(query, book_id, top_k, should_stop=should_stop)
                ), admission, request,
            )
        except Busy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - keep provider details private
            _log_failure(exc)
            raise HTTPException(status_code=503, detail="检索观察暂时失败，请稍后重试") from None

    @app.post("/v1/ask/stream", summary="提问，以服务器推送事件（SSE）流式返回")
    async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
        query, book_id, top_k = await admit(payload, request)
        try:
            admission = guard.admit_request()
        except Busy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _AnswerStreamResponse(
            _stream(
                lambda sink, should_stop: answer(query, book_id, top_k, sink, should_stop),
                guard,
                admission=admission,
            ),
            admission=admission,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/feedback", summary="提交对某次回答的反馈")
    def submit_feedback(payload: FeedbackRequest, request: Request) -> dict[str, str]:
        authorize(request, rate_scope="feedback")
        comment = payload.comment.strip()
        if payload.rating == "helpful" and payload.reason is not None:
            raise HTTPException(status_code=422, detail="正向反馈不需要问题分类")
        if payload.rating == "needs_improvement" and payload.reason is None:
            raise HTTPException(status_code=422, detail="请选择需要改进的原因")
        if payload.reason == "other" and not comment:
            raise HTTPException(status_code=422, detail="选择其他时请填写补充说明")
        try:
            snapshot = answers.resolve(payload.answer_id)
            feedback_store.save(
                snapshot,
                rating=payload.rating,
                reason=payload.reason,
                comment=comment,
            )
        except AnswerRecordExpiredError:
            raise HTTPException(
                status_code=404,
                detail="回答记录已过期，请重新提问后再反馈",
            ) from None
        except Exception as exc:  # noqa: BLE001 - public responses never carry storage detail
            _log_failure(exc)
            raise HTTPException(status_code=500, detail="反馈暂时无法保存，请稍后再试") from None
        return {"status": "saved"}

    return app


def run_api_server(
    *,
    host: str,
    port: int,
    db_path: str | Path,
    public: bool = False,
    warmup: bool = True,
) -> None:
    settings = GuardSettings.from_env()
    if not is_loopback_host(host) and settings.access_code is None and not public:
        raise AuthenticationError(
            "对外监听时必须设置 RAG_QA_ACCESS_CODE；确认要无口令开放时请显式传入 --public"
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingOptionalDependencyError("启动问答服务需要安装：uv sync --extra api") from exc

    from rag_textbook_qa.indexing import list_indexed_books
    from rag_textbook_qa.rag import RAGEngine

    books = [
        {
            "book_id": book["book_name"],
            "label": BOOK_LABELS.get(book["book_name"], book["book_name"]),
            "chunks": book["count"],
        }
        for book in list_indexed_books(db_path)
    ]
    feedback_store = FeedbackStore(
        Path(db_path).expanduser().resolve().parent / "product" / "feedback.sqlite3"
    )
    engine = RAGEngine(db_path=db_path, verbose=False, enable_hyde=False)
    importer = None
    if is_loopback_host(host) and not public:
        def build_import(chunks_file: Path, book_id: str, progress: Callable[[int, int, str], None]) -> None:
            engine.vectorizer.vectorize_book(chunks_file, book_id, on_progress=progress)

        importer = ImportManager(Path(db_path).expanduser().resolve().parent / "product" / "imports", build_import)
    try:
        if warmup and books:
            # Load both retrieval models now rather than inside the first visitor's request.
            print("正在预热检索模型...", flush=True)
            engine.search_single_book(books[0]["book_id"], "模型预热", 1, use_hyde=False)
        uvicorn.run(
            create_api_app(
                engine,
                AccessGuard(settings),
                books,
                feedback_store=feedback_store,
                import_manager=importer,
            ),
            host=host,
            port=port,
        )
    finally:
        if importer is not None:
            importer.close()
        engine.close()
