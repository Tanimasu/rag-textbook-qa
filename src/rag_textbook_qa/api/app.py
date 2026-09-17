"""Public question-answering API and the chat page served beside it.

The deployed demo is one process: this app answers `/v1/ask`, streams answers as
server-sent events on `/v1/ask/stream`, documents itself at `/docs`, and serves a
single-page chat UI at `/`. It exposes a deliberately small slice of the engine.
Query decomposition, citation checking and HyDE stay off because none of them has
passed acceptance and each one adds model calls, and no provider error text is ever
returned, because it can carry upstream detail.
"""

import json
import math
import queue
import re
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
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
)
from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.providers.base import AuthenticationError, MissingOptionalDependencyError
from rag_textbook_qa.providers.config import is_loopback_host

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
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


class _StreamCancelled(Exception):
    """Stop the producer after its client closes the streaming response."""


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
        "timing": {field: execution[field] for field in _TIMING_FIELDS if field in execution},
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
    device = str(stage.get("device") or "unknown").lower()
    if device not in {"cpu", "cuda", "mps", "mixed"}:
        device = "unknown"
    platform = str(stage.get("platform") or "unknown")
    if platform not in {"Windows", "Darwin", "Linux", "mixed"}:
        platform = "unknown"
    try:
        elapsed = float(stage.get("elapsed_seconds", 0))
    except (TypeError, ValueError):
        elapsed = 0.0
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
    return {
        "backend": backend,
        "device": device,
        "platform": platform,
        "elapsed_seconds": round(elapsed, 3),
        "fallback_used": bool(stage.get("fallback_used")),
    }


def _citation_integrity(answer: str, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check citation links only; this deliberately makes no grounding claim."""

    cited = sorted({int(match) for match in _CITATION_REFERENCE.findall(answer)})
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
    return {
        "citation_id": source.get("citation_id"),
        "book_id": book_id,
        "book": BOOK_LABELS.get(book_id, book_id),
        "section": " > ".join(str(source[field]) for field in _HEADING_FIELDS if source.get(field)),
        "excerpt": str(source.get("content") or ""),
        "truncated": bool(source.get("truncated")),
    }


def _log_failure(exc: BaseException) -> None:
    # The type name only: the message can carry provider detail or the question itself.
    print(f"问答失败: {type(exc).__name__}", file=sys.stderr, flush=True)


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


def _stream(
    produce: Callable[[Callable[[str], None]], dict[str, Any]],
    guard: AccessGuard,
) -> Iterator[str]:
    """Relay one answer as server-sent events.

    The answer runs on its own thread, and only that thread holds the generation
    slot. A client that disconnects mid-answer therefore cannot strand the slot and
    wedge every later request: the thread finishes and releases it either way.
    """

    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
    cancelled = threading.Event()

    def emit(chunk: str) -> None:
        if cancelled.is_set():
            raise _StreamCancelled
        events.put(("chunk", {"text": chunk}))

    def work() -> None:
        try:
            with guard.generation_slot():
                result = produce(emit)
            events.put(("result", result))
        except _StreamCancelled:
            # The browser intentionally stopped or left. This is not a server error.
            pass
        except Busy as exc:
            events.put(("error", {"status": "busy", "message": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - public responses never carry provider text
            _log_failure(exc)
            events.put(("error", {"status": "failed", "message": PUBLIC_FAILURE}))
        finally:
            events.put(None)

    threading.Thread(target=work, name="rag-qa-answer", daemon=True).start()
    try:
        while (item := events.get()) is not None:
            name, data = item
            yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    finally:
        cancelled.set()


def create_api_app(
    engine: Any,
    guard: AccessGuard,
    books: Sequence[Mapping[str, Any]],
    *,
    feedback_store: FeedbackStore,
) -> FastAPI:
    """Build the public app around one shared engine and one guard."""

    book_list = [dict(book) for book in books]
    known_books = {book["book_id"] for book in book_list}
    answers = AnswerRegistry()
    page = (
        resources.files("rag_textbook_qa.api")
        .joinpath("static/index.html")
        .read_text(encoding="utf-8")
    )

    app = FastAPI(
        title="计算机教材 RAG 问答",
        version=__version__,
        description=(
            "基于五本计算机教材的检索增强问答。回答依据检索到的教材片段生成，并附带引用的原文。"
            "公开版关闭了尚未通过验收的实验功能，并对调用频率和每日生成次数设了上限。"
        ),
    )

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

    def admit(payload: AskRequest, request: Request) -> tuple[str, str | None, int]:
        authorize(request)
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="问题不能为空")
        if payload.book_id is not None and payload.book_id not in known_books:
            raise HTTPException(status_code=404, detail="没有这本教材的索引")
        return query, payload.book_id, payload.top_k

    def answer(
        query: str,
        book_id: str | None,
        top_k: int,
        sink: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        if not getattr(engine, "enable_llm", False):
            retrieval_only = "llm_unavailable"
        elif guard.reserve_generation():
            retrieval_only = None
        else:
            retrieval_only = "budget"
        result = engine.ask(
            query=query,
            book_name=book_id,
            top_k=top_k,
            use_llm=retrieval_only is None,
            use_hyde=False,
            use_decomposition=False,
            verify_citations=False,
            on_answer_chunk=sink if retrieval_only is None else None,
        )
        if retrieval_only is None and not result.get("context_sources"):
            # Nothing was packed, so the engine never called the model.
            guard.refund_generation()
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

    @app.get("/health", summary="服务状态")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "books": len(book_list),
            **guard.status(),
        }

    @app.get("/v1/books", summary="已索引的教材")
    def list_books() -> list[dict[str, Any]]:
        return book_list

    @app.post("/v1/ask", summary="提问，一次性返回回答与引用")
    def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
        query, book_id, top_k = admit(payload, request)
        try:
            with guard.generation_slot():
                return answer(query, book_id, top_k, None)
        except Busy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - public responses never carry provider text
            _log_failure(exc)
            raise HTTPException(status_code=500, detail=PUBLIC_FAILURE) from None

    @app.post("/v1/ask/stream", summary="提问，以服务器推送事件（SSE）流式返回")
    def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
        query, book_id, top_k = admit(payload, request)
        return StreamingResponse(
            _stream(lambda sink: answer(query, book_id, top_k, sink), guard),
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
            ),
            host=host,
            port=port,
        )
    finally:
        engine.close()
