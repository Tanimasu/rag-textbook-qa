import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import warnings
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`.*")
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from rag_textbook_qa.api.app import (
    PUBLIC_FAILURE,
    _stream,
    create_api_app,
    public_result,
    run_api_server,
)
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import (
    AccessDenied,
    AccessGuard,
    Busy,
    GuardSettings,
    RateLimited,
)
from rag_textbook_qa.cli import main
from rag_textbook_qa.llm import GenerationCancelled
from rag_textbook_qa.providers.base import AuthenticationError

SOURCE = {
    "book_name": "os",
    "chapter": "第二章",
    "section_h2": "2.1 进程",
    "content": "进程是程序的一次执行过程。",
    "citation_id": 1,
    "truncated": False,
}
BOOKS = [{"book_id": "os", "label": "操作系统", "chunks": 12}]
SECRET = "sk-live-upstream-detail"


class FakeEngine:
    def __init__(self, *, sources=(SOURCE,), success=True, error=None, enable_llm=True, raises=None):
        self.sources = sources
        self.success = success
        self.error = error
        self.enable_llm = enable_llm
        self.raises = raises
        self.calls = []

    def ask(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        generating = kwargs["use_llm"] and bool(self.sources)
        sink = kwargs.get("on_answer_chunk")
        if generating and sink is not None:
            for piece in ("进程是", "程序的执行【参考资料 1】"):
                sink(piece)
        return {
            "success": self.success and generating,
            "answer": "进程是程序的执行【参考资料 1】" if generating else None,
            "error": self.error,
            "context_sources": [dict(source) for source in self.sources],
            "results": [{"internal": "candidate"}],
            "prompt": "SYSTEM PROMPT WITH FULL CONTEXT",
            "source_conflicts": [],
            "execution": {
                "embedding": {
                    "backend": "remote",
                    "device": "cuda",
                    "platform": "Windows",
                    "model": "private-model-name",
                    "remote_url": "http://private-worker",
                    "elapsed_seconds": 0.1234,
                    "fallback_used": False,
                },
                "reranker": {
                    "backend": "local",
                    "device": "mps",
                    "platform": "Darwin",
                    "elapsed_seconds": 0.5,
                    "fallback_used": True,
                },
                "retrieval_seconds": 0.1,
                "total_seconds": 0.4,
            },
        }


def sse_events(body):
    parsed = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        name = next(line[6:].strip() for line in lines if line.startswith("event:"))
        data = next(line[5:].strip() for line in lines if line.startswith("data:"))
        parsed.append((name, json.loads(data)))
    return parsed


class GuardTests(unittest.TestCase):
    def test_access_code_is_optional_and_must_match_exactly(self):
        AccessGuard(GuardSettings()).check_access(None)
        guard = AccessGuard(GuardSettings(access_code="demo-2026"))
        for supplied in (None, "", "demo-2025", "DEMO-2026"):
            with self.subTest(supplied=supplied), self.assertRaises(AccessDenied):
                guard.check_access(supplied)
        guard.check_access("demo-2026")

    def test_rate_limit_is_a_sliding_window_per_client(self):
        now = [1000.0]
        guard = AccessGuard(
            GuardSettings(requests_per_window=2, window_seconds=60),
            clock=lambda: now[0],
        )
        guard.check_rate("a")
        guard.check_rate("a")
        guard.check_rate("b")
        with self.assertRaises(RateLimited) as raised:
            guard.check_rate("a")
        self.assertGreaterEqual(raised.exception.retry_after_seconds, 1)
        now[0] += 61
        guard.check_rate("a")

    def test_feedback_has_an_independent_rate_limit(self):
        guard = AccessGuard(
            GuardSettings(
                requests_per_window=1,
                feedback_requests_per_window=2,
                window_seconds=60,
            )
        )
        guard.check_rate("a")
        guard.check_rate("a", scope="feedback")
        guard.check_rate("a", scope="feedback")
        with self.assertRaises(RateLimited):
            guard.check_rate("a")
        with self.assertRaises(RateLimited):
            guard.check_rate("a", scope="feedback")
        with self.assertRaises(ValueError):
            guard.check_rate("a", scope="unknown")

    def test_rate_limit_state_stays_bounded_for_many_clients(self):
        guard = AccessGuard(GuardSettings(requests_per_window=2, window_seconds=60))

        with patch("rag_textbook_qa.api.guard._MAX_RATE_BUCKETS", 2):
            guard.check_rate("first")
            guard.check_rate("second")
            guard.check_rate("third")

        self.assertEqual(len(guard._hits), 2)
        self.assertNotIn(("question", "first"), guard._hits)

    def test_daily_budget_degrades_then_resets_on_a_new_day(self):
        day = [date(2026, 9, 15)]
        guard = AccessGuard(GuardSettings(daily_generations=2), today=lambda: day[0])

        self.assertEqual([guard.reserve_generation() for _ in range(3)], [True, True, False])
        self.assertEqual(guard.status()["generations_remaining_today"], 0)
        day[0] = date(2026, 9, 16)
        self.assertTrue(guard.reserve_generation())

    def test_a_held_generation_slot_times_out_as_busy(self):
        guard = AccessGuard(GuardSettings(queue_timeout_seconds=0.05))
        held, release = threading.Event(), threading.Event()

        def hold():
            with guard.generation_slot():
                held.set()
                release.wait(2)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(2)
        try:
            with self.assertRaises(Busy), guard.generation_slot():
                pass
        finally:
            release.set()
            holder.join(2)
        with guard.generation_slot():
            pass

    def test_settings_reject_unsafe_values_without_echoing_them(self):
        for code in (" padded", "two words", "中文口令"):
            with self.subTest(code=code), self.assertRaises(ValueError) as raised:
                GuardSettings.from_env({"RAG_QA_ACCESS_CODE": code})
            self.assertNotIn(code.strip(), str(raised.exception))
        with self.assertRaises(ValueError):
            GuardSettings.from_env({"RAG_QA_DAILY_GENERATIONS": "0"})
        with self.assertRaises(ValueError):
            GuardSettings.from_env({"RAG_QA_FEEDBACK_RATE_LIMIT": "0"})

        settings = GuardSettings.from_env(
            {
                "RAG_QA_ACCESS_CODE": "",
                "RAG_QA_TRUST_PROXY": "true",
                "RAG_QA_FEEDBACK_RATE_LIMIT": "7",
            }
        )
        self.assertIsNone(settings.access_code)
        self.assertTrue(settings.trust_proxy)
        self.assertEqual(settings.feedback_requests_per_window, 7)


class PublicResultTests(unittest.TestCase):
    def test_internals_and_provider_error_text_never_leave(self):
        raw = FakeEngine(success=False, error=SECRET).ask(query="q", use_llm=True)
        payload = public_result(raw, retrieval_only=None)

        self.assertEqual(payload["status"], "failed")
        self.assertIsNone(payload["answer"])
        rendered = json.dumps(payload, ensure_ascii=False)
        for leaked in (SECRET, "SYSTEM PROMPT", "candidate"):
            self.assertNotIn(leaked, rendered)
        self.assertEqual(payload["sources"][0]["book"], "操作系统")
        self.assertEqual(payload["sources"][0]["section"], "第二章 > 2.1 进程")
        self.assertEqual(
            payload["compute"]["embedding"],
            {
                "backend": "remote",
                "device": "cuda",
                "platform": "Windows",
                "elapsed_seconds": 0.123,
                "fallback_used": False,
            },
        )
        self.assertNotIn("private-model-name", json.dumps(payload))
        self.assertNotIn("private-worker", json.dumps(payload))
        self.assertIsNone(payload["citation_integrity"])

    def test_citation_integrity_reports_missing_and_unknown_links(self):
        linked = FakeEngine().ask(query="q", use_llm=True)
        missing = FakeEngine().ask(query="q", use_llm=True)
        missing["answer"] = "回答没有引用编号。"
        invalid = FakeEngine().ask(query="q", use_llm=True)
        invalid["answer"] = "一条有效引用【参考资料 1】，一条无效引用【参考资料 9】。"

        self.assertEqual(
            public_result(linked, retrieval_only=None)["citation_integrity"],
            {"status": "linked", "cited": [1], "unknown": []},
        )
        self.assertEqual(
            public_result(missing, retrieval_only=None)["citation_integrity"],
            {"status": "missing", "cited": [], "unknown": []},
        )
        self.assertEqual(
            public_result(invalid, retrieval_only=None)["citation_integrity"],
            {"status": "invalid", "cited": [1, 9], "unknown": [9]},
        )

    def test_retrieval_only_and_missing_evidence_are_not_failures(self):
        budget = public_result(FakeEngine().ask(query="q", use_llm=False), retrieval_only="budget")
        empty = public_result(FakeEngine(sources=()).ask(query="q", use_llm=True), retrieval_only=None)

        self.assertEqual(budget["status"], "retrieval_only")
        self.assertTrue(budget["sources"])
        self.assertEqual(empty["status"], "no_evidence")


class ApiAppTests(unittest.TestCase):
    def client(self, engine=None, **settings):
        self.engine = engine or FakeEngine()
        self.guard = AccessGuard(GuardSettings(**settings))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.feedback_store = FeedbackStore(Path(temporary.name) / "feedback.sqlite3")
        return TestClient(
            create_api_app(
                self.engine,
                self.guard,
                BOOKS,
                feedback_store=self.feedback_store,
            )
        )

    def test_page_books_docs_and_health_expose_no_secrets(self):
        client = self.client(access_code="demo-2026")

        page = client.get("/")
        health = client.get("/health").json()

        self.assertEqual(page.status_code, 200)
        self.assertIn("计算机教材问答", page.text)
        self.assertIn("请选择教材", page.text)
        self.assertNotIn("全部教材", page.text)
        self.assertIn('data-book="os" disabled', page.text)
        self.assertIn('data-book="database" disabled', page.text)
        self.assertIn('data-book="computer_network" disabled', page.text)
        self.assertIn("example.dataset.book", page.text)
        self.assertIn("sessionSet(BOOK_KEY, bookSelect.value)", page.text)
        self.assertIn("这本教材当前没有可用索引", page.text)
        self.assertIn("远程 Worker", page.text)
        self.assertIn("已回退到", page.text)
        self.assertIn("computeChip", page.text)
        self.assertIn("正在检索教材", page.text)
        self.assertIn("正在组织答案", page.text)
        self.assertIn("重试本题", page.text)
        self.assertIn("addRetry(card, query, bookId)", page.text)
        self.assertIn("bookSelect.value = bookId", page.text)
        self.assertIn("停止生成", page.text)
        self.assertIn("清空对话", page.text)
        self.assertIn("new AbortController()", page.text)
        self.assertIn('thread.setAttribute("aria-busy", "true")', page.text)
        self.assertIn("这份回答没有标出对应的资料编号", page.text)
        self.assertIn("回答引用了下方不存在的资料编号", page.text)
        self.assertIn("trackScrollIntent", page.text)
        self.assertIn("window.setTimeout(paint, 125)", page.text)
        self.assertIn("👍 有帮助", page.text)
        self.assertIn("👎 需要改进", page.text)
        self.assertIn("/v1/feedback", page.text)
        self.assertIn('explainRefusal(card, response, "反馈提交")', page.text)
        self.assertEqual(client.head("/").status_code, 200)
        self.assertEqual(client.get("/v1/books").json(), BOOKS)
        self.assertEqual(client.get("/docs").status_code, 200)
        self.assertTrue(health["access_code_required"])
        self.assertNotIn("demo-2026", json.dumps(health))

    def test_ask_requires_the_access_code_when_one_is_configured(self):
        client = self.client(access_code="demo-2026")

        denied = client.post("/v1/ask", json={"query": "什么是进程？"})
        allowed = client.post(
            "/v1/ask",
            json={"query": "什么是进程？"},
            headers={"X-Access-Code": "demo-2026"},
        )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["status"], "answered")
        self.assertRegex(allowed.json()["answer_id"], r"^[0-9a-f]{32}$")

    def test_feedback_is_saved_only_after_an_explicit_valid_submission(self):
        client = self.client()
        answer = client.post(
            "/v1/ask",
            json={"query": "什么是进程？", "book_id": "os"},
        ).json()

        self.assertEqual(self.feedback_store.records(), [])
        saved = client.post(
            "/v1/feedback",
            json={"answer_id": answer["answer_id"], "rating": "helpful"},
        )
        records = self.feedback_store.records()

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json(), {"status": "saved"})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query"], "什么是进程？")
        self.assertEqual(records[0]["book_id"], "os")
        self.assertEqual(records[0]["rating"], "helpful")
        self.assertIsNone(records[0]["reason"])
        self.assertNotIn("client", records[0])
        persisted = json.dumps(records[0], ensure_ascii=False)
        self.assertNotIn("SYSTEM PROMPT", persisted)
        self.assertNotIn("candidate", persisted)

    def test_negative_feedback_requires_a_reason_and_known_recent_answer(self):
        client = self.client()
        answer_id = client.post("/v1/ask", json={"query": "问题", "book_id": "os"}).json()[
            "answer_id"
        ]

        missing_reason = client.post(
            "/v1/feedback",
            json={"answer_id": answer_id, "rating": "needs_improvement"},
        )
        missing_other_note = client.post(
            "/v1/feedback",
            json={
                "answer_id": answer_id,
                "rating": "needs_improvement",
                "reason": "other",
            },
        )
        unknown = client.post(
            "/v1/feedback",
            json={"answer_id": "0" * 32, "rating": "helpful"},
        )

        self.assertEqual(missing_reason.status_code, 422)
        self.assertEqual(missing_other_note.status_code, 422)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(self.feedback_store.records(), [])

    def test_feedback_uses_the_same_access_code_as_questions(self):
        client = self.client(access_code="demo-2026")
        answer_id = client.post(
            "/v1/ask",
            headers={"X-Access-Code": "demo-2026"},
            json={"query": "问题", "book_id": "os"},
        ).json()["answer_id"]

        denied = client.post(
            "/v1/feedback",
            json={"answer_id": answer_id, "rating": "helpful"},
        )
        saved = client.post(
            "/v1/feedback",
            headers={"X-Access-Code": "demo-2026"},
            json={"answer_id": answer_id, "rating": "helpful"},
        )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(len(self.feedback_store.records()), 1)

    def test_public_calls_keep_unaccepted_paths_off(self):
        client = self.client()
        client.post("/v1/ask", json={"query": " 什么是进程？ ", "book_id": "os", "top_k": 3})

        call = self.engine.calls[0]
        self.assertEqual(call["query"], "什么是进程？")
        self.assertEqual((call["book_name"], call["top_k"]), ("os", 3))
        self.assertFalse(call["use_hyde"])
        self.assertFalse(call["use_decomposition"])
        self.assertFalse(call["verify_citations"])

    def test_invalid_requests_never_reach_the_engine(self):
        client = self.client()
        cases = [
            ({"query": "   "}, 422),
            ({"query": "问题", "book_id": "missing"}, 404),
            ({"query": "问" * 501}, 422),
            ({"query": "问题", "top_k": 11}, 422),
        ]
        for body, status in cases:
            with self.subTest(body=body):
                self.assertEqual(client.post("/v1/ask", json=body).status_code, status)
        self.assertEqual(self.engine.calls, [])

    def test_rate_limited_callers_are_told_when_to_retry(self):
        client = self.client(requests_per_window=1, window_seconds=60)
        client.post("/v1/ask", json={"query": "问题"})
        limited = client.post("/v1/ask", json={"query": "问题"})

        self.assertEqual(limited.status_code, 429)
        self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)

    def test_feedback_does_not_consume_the_question_rate_limit(self):
        client = self.client(requests_per_window=2, feedback_requests_per_window=1)
        answer_id = client.post(
            "/v1/ask",
            json={"query": "问题1", "book_id": "os"},
        ).json()["answer_id"]

        first_feedback = client.post(
            "/v1/feedback",
            json={"answer_id": answer_id, "rating": "helpful"},
        )
        limited_feedback = client.post(
            "/v1/feedback",
            json={"answer_id": answer_id, "rating": "helpful"},
        )
        second_question = client.post(
            "/v1/ask",
            json={"query": "问题2", "book_id": "os"},
        )

        self.assertEqual(first_feedback.status_code, 200)
        self.assertEqual(limited_feedback.status_code, 429)
        self.assertEqual(second_question.status_code, 200)

    def test_exhausted_budget_answers_from_retrieval_alone(self):
        client = self.client(daily_generations=1)
        client.post("/v1/ask", json={"query": "问题"})
        degraded = client.post("/v1/ask", json={"query": "问题"}).json()

        self.assertFalse(self.engine.calls[-1]["use_llm"])
        self.assertEqual(degraded["status"], "retrieval_only")
        self.assertIsNone(degraded["answer"])
        self.assertTrue(degraded["sources"])

    def test_questions_without_evidence_do_not_spend_the_budget(self):
        client = self.client(engine=FakeEngine(sources=()), daily_generations=5)
        client.post("/v1/ask", json={"query": "问题"})

        self.assertEqual(client.get("/health").json()["generations_remaining_today"], 5)

    def test_an_engine_without_a_model_answers_from_retrieval_for_free(self):
        client = self.client(engine=FakeEngine(enable_llm=False), daily_generations=5)
        payload = client.post("/v1/ask", json={"query": "问题"}).json()

        self.assertEqual(payload["status"], "retrieval_only")
        self.assertIn("没有配置可用的大模型", payload["message"])
        self.assertEqual(client.get("/health").json()["generations_remaining_today"], 5)

    def test_stream_relays_chunks_then_the_result(self):
        client = self.client()
        response = client.post("/v1/ask/stream", json={"query": "问题"})
        events = sse_events(response.text)

        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertEqual([name for name, _ in events], ["chunk", "chunk", "result"])
        self.assertEqual(
            "".join(data["text"] for name, data in events if name == "chunk"),
            "进程是程序的执行【参考资料 1】",
        )
        self.assertEqual(events[-1][1]["status"], "answered")
        self.assertRegex(events[-1][1]["answer_id"], r"^[0-9a-f]{32}$")

    def test_engine_failures_return_generic_errors_on_both_routes(self):
        client = self.client(engine=FakeEngine(raises=RuntimeError(SECRET)))
        with patch("sys.stderr"):
            blocking = client.post("/v1/ask", json={"query": "问题"})
            streamed = client.post("/v1/ask/stream", json={"query": "问题"})

        self.assertEqual(blocking.status_code, 500)
        self.assertNotIn(SECRET, blocking.text)
        self.assertEqual(
            sse_events(streamed.text),
            [("error", {"status": "failed", "message": PUBLIC_FAILURE})],
        )
        self.assertNotIn(SECRET, streamed.text)

    def test_stopping_before_the_model_is_called_refunds_the_budget(self):
        for request_sent, remaining in ((False, 5), (True, 4)):
            with self.subTest(request_sent=request_sent):
                engine = FakeEngine(raises=GenerationCancelled(request_sent=request_sent))
                client = self.client(engine=engine, daily_generations=5)

                response = client.post("/v1/ask/stream", json={"query": "问题"})

                # A stop is the reader's choice: no result, and no error either.
                self.assertEqual(response.text, "")
                self.assertIsNotNone(engine.calls[0]["should_stop"])
                health = client.get("/health").json()
                self.assertEqual(health["generations_remaining_today"], remaining)


def answer_threads_finish():
    for thread in threading.enumerate():
        if thread.name == "rag-qa-answer":
            thread.join(2)


def disconnect_after_first_body(stream):
    """Drive a streaming response the way uvicorn does, then hang up."""

    async def scenario():
        first_body = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()

        async def receive():
            await first_body.wait()
            return {"type": "http.disconnect"}

        response = StreamingResponse(stream, media_type="text/event-stream")
        # uvicorn advertises ASGI 2.3, where Starlette cancels the response task on
        # disconnect instead of waiting for a write to fail.
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await asyncio.wait_for(response(scope, receive, send), 2)

    asyncio.run(scenario())


class StreamStopTests(unittest.TestCase):
    def test_a_disconnect_during_silent_reasoning_stops_the_producer(self):
        guard = AccessGuard(GuardSettings(queue_timeout_seconds=1))
        producer_stopped = threading.Event()

        def produce(sink, should_stop):
            sink("第一段")
            # Hidden reasoning: nothing reaches the relay, so only the stop flag,
            # not a failed write, can end this.
            deadline = time.monotonic() + 2
            while not should_stop():
                if time.monotonic() > deadline:
                    return {"status": "answered"}
                time.sleep(0.01)
            producer_stopped.set()
            raise GenerationCancelled(request_sent=True)

        disconnect_after_first_body(_stream(produce, guard, keepalive_seconds=60))

        self.assertTrue(producer_stopped.wait(1))
        answer_threads_finish()
        with guard.generation_slot():
            pass

    def test_a_reader_who_leaves_while_queued_never_starts_an_answer(self):
        guard = AccessGuard(GuardSettings(queue_timeout_seconds=2))
        produced = threading.Event()
        bodies = []

        def produce(sink, should_stop):
            produced.set()
            return {"status": "answered"}

        async def scenario():
            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    bodies.append(message["body"])

            async def receive():
                await asyncio.sleep(0.1)
                return {"type": "http.disconnect"}

            response = StreamingResponse(
                _stream(produce, guard, keepalive_seconds=0.02),
                media_type="text/event-stream",
            )
            scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
            await asyncio.wait_for(response(scope, receive, send), 2)

        # Someone else is answering, so this request waits for the slot.
        with guard.generation_slot():
            asyncio.run(scenario())
        answer_threads_finish()

        self.assertFalse(produced.is_set())
        # While it waited, the relay kept the connection warm with SSE comments only.
        self.assertTrue(bodies)
        self.assertEqual(set(bodies), {b": keepalive\n\n"})


class ServerTests(unittest.TestCase):
    def test_an_open_public_bind_is_refused_before_any_model_loads(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rag_textbook_qa.rag.RAGEngine") as engine_class,
            self.assertRaises(AuthenticationError),
        ):
            run_api_server(host="0.0.0.0", port=7860, db_path="unused")
        engine_class.assert_not_called()

    def test_serving_builds_one_engine_warms_it_and_always_closes_it(self):
        fake_uvicorn = SimpleNamespace(run=MagicMock())
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "artifacts" / "vector_db"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
                patch(
                    "rag_textbook_qa.indexing.list_indexed_books",
                    return_value=[{"book_name": "os", "count": 12}],
                ),
                patch("rag_textbook_qa.rag.RAGEngine") as engine_class,
            ):
                run_api_server(host="0.0.0.0", port=7860, db_path=db_path, public=True)

            self.assertTrue(
                (Path(directory) / "artifacts" / "product" / "feedback.sqlite3").is_file()
            )

        engine = engine_class.return_value
        engine_class.assert_called_once_with(db_path=db_path, verbose=False, enable_hyde=False)
        engine.search_single_book.assert_called_once()
        fake_uvicorn.run.assert_called_once()
        engine.close.assert_called_once_with()

    def test_serve_command_delegates_to_the_api_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "rag_textbook_qa").mkdir(parents=True)
            (root / "project").mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            (root / "project" / ".env").write_text("", encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("rag_textbook_qa.api.app.run_api_server") as serve,
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "serve",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "7860",
                        "--public",
                        "--no-warmup",
                    ]
                )

        self.assertEqual(exit_code, 0)
        serve.assert_called_once_with(
            host="0.0.0.0",
            port=7860,
            db_path=root.resolve() / "artifacts" / "vector_db",
            public=True,
            warmup=False,
        )


if __name__ == "__main__":
    unittest.main()
