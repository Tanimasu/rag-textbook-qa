import json
import os
import sys
import tempfile
import threading
import unittest
import warnings
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`.*")
from fastapi.testclient import TestClient

from rag_textbook_qa.api.app import PUBLIC_FAILURE, create_api_app, public_result, run_api_server
from rag_textbook_qa.api.guard import (
    AccessDenied,
    AccessGuard,
    Busy,
    GuardSettings,
    RateLimited,
)
from rag_textbook_qa.cli import main
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
            "execution": {"retrieval_seconds": 0.1, "total_seconds": 0.4},
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

        settings = GuardSettings.from_env({"RAG_QA_ACCESS_CODE": "", "RAG_QA_TRUST_PROXY": "true"})
        self.assertIsNone(settings.access_code)
        self.assertTrue(settings.trust_proxy)


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
        return TestClient(create_api_app(self.engine, self.guard, BOOKS))

    def test_page_books_docs_and_health_expose_no_secrets(self):
        client = self.client(access_code="demo-2026")

        page = client.get("/")
        health = client.get("/health").json()

        self.assertEqual(page.status_code, 200)
        self.assertIn("计算机教材问答", page.text)
        self.assertIn("请选择教材", page.text)
        self.assertNotIn("全部教材", page.text)
        self.assertIn("正在检索教材", page.text)
        self.assertIn("正在组织答案", page.text)
        self.assertIn("trackScrollIntent", page.text)
        self.assertIn("window.setTimeout(paint, 125)", page.text)
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
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
            patch(
                "rag_textbook_qa.indexing.list_indexed_books",
                return_value=[{"book_name": "os", "count": 12}],
            ),
            patch("rag_textbook_qa.rag.RAGEngine") as engine_class,
        ):
            run_api_server(host="0.0.0.0", port=7860, db_path="db", public=True)

        engine = engine_class.return_value
        engine_class.assert_called_once_with(db_path="db", verbose=False, enable_hyde=False)
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
