import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from test_api import BOOKS, FakeEngine

from rag_textbook_qa.api.app import _answer_once, _AnswerStreamResponse, _stream, create_api_app
from rag_textbook_qa.api.feedback import FeedbackStore
from rag_textbook_qa.api.guard import AccessGuard, Busy, GuardSettings


class AdmissionTests(unittest.TestCase):
    def test_capacity_fifo_cancellation_and_idempotent_release(self):
        guard = AccessGuard(GuardSettings(max_pending_requests=3))
        active, first, second = [guard.admit_request() for _ in range(3)]
        self.assertTrue(active.try_start())
        with self.assertRaises(Busy):
            guard.admit_request()
        self.assertFalse(second.try_start())
        first.cancel()
        first.cancel()
        replacement = guard.admit_request()
        self.assertEqual(guard.status()["requests_in_flight"], 3)
        active.cancel()
        self.assertFalse(second.try_start())  # Running work still owns its slot.
        active.release()
        active.release()
        self.assertFalse(replacement.try_start())
        self.assertTrue(second.try_start())
        second.release()
        self.assertTrue(replacement.try_start())
        replacement.release()
        self.assertEqual(guard.status()["requests_in_flight"], 0)

    def test_queue_timeout_returns_capacity_without_reserving_generation(self):
        now = [0.0]
        guard = AccessGuard(GuardSettings(max_pending_requests=2, queue_timeout_seconds=1),
                            clock=lambda: now[0])
        active = guard.admit_request()
        self.assertTrue(active.try_start())
        waiting = guard.admit_request()
        now[0] = 2
        with self.assertRaisesRegex(Busy, "超时"):
            waiting.try_start()
        self.assertEqual(guard.status()["requests_in_flight"], 1)
        self.assertEqual(guard.status()["generations_remaining_today"], 200)
        active.release()

    def test_capacity_is_configurable_and_must_be_positive(self):
        self.assertEqual(GuardSettings.from_env({"RAG_QA_MAX_PENDING_REQUESTS": "2"})
                         .max_pending_requests, 2)
        with self.assertRaises(ValueError):
            GuardSettings.from_env({"RAG_QA_MAX_PENDING_REQUESTS": "0"})


class RequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_book_validation_keeps_the_event_loop_available_on_both_routes(self):
        loop = asyncio.get_running_loop()
        for endpoint in ("/v1/ask", "/v1/ask/stream"):
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as directory:
                engine = FakeEngine(enable_llm=False)
                released = threading.Event()
                catalog_yielded = []

                def catalog(released=released, catalog_yielded=catalog_yielded):
                    # Publication retries and Chroma reads are synchronous. The
                    # event loop must still run while this catalog call waits.
                    loop.call_soon_threadsafe(released.set)
                    catalog_yielded.append(released.wait(1))
                    return [{"book_name": "os", "count": 2}]

                engine.list_indexed_books = catalog
                guard = AccessGuard(GuardSettings())
                app = create_api_app(
                    engine, guard, BOOKS,
                    feedback_store=FeedbackStore(Path(directory) / "feedback.sqlite3"),
                )
                async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                    response = await client.post(endpoint, json={"query": "问题", "book_id": "os"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(catalog_yielded, [True], "Catalog validation blocked the event loop")
                self.assertEqual(guard.status()["requests_in_flight"], 0)

    async def test_queued_stream_disconnect_releases_capacity_without_a_thread(self):
        guard = AccessGuard(GuardSettings(max_pending_requests=2))
        active = guard.admit_request()
        active.try_start()
        admission = guard.admit_request()
        stream = _stream(lambda *_: self.fail("Queued request executed"), guard,
                         admission=admission, keepalive_seconds=0.001)
        with patch("rag_textbook_qa.api.app.threading.Thread") as answer_thread:
            self.assertEqual(await anext(stream), ": keepalive\n\n")
            await stream.aclose()
        answer_thread.assert_not_called()
        self.assertEqual(guard.status()["requests_in_flight"], 1)
        active.release()

    async def test_rest_disconnect_while_waiting_never_starts_work(self):
        guard = AccessGuard(GuardSettings(max_pending_requests=2))
        active = guard.admit_request()
        active.try_start()
        admission = guard.admit_request()

        async def disconnected():
            return True

        request = SimpleNamespace(is_disconnected=disconnected)
        with (
            patch("rag_textbook_qa.api.app.threading.Thread") as answer_thread,
            self.assertRaises(HTTPException) as caught,
        ):
            await _answer_once(lambda *_: self.fail("Queued request executed"), admission, request)
        self.assertEqual(caught.exception.status_code, 499)
        answer_thread.assert_not_called()
        self.assertEqual(guard.status()["requests_in_flight"], 1)
        active.release()

    async def test_active_disconnect_holds_slot_until_worker_finishes(self):
        guard = AccessGuard(GuardSettings(max_pending_requests=2))
        admission = guard.admit_request()
        worker_done = threading.Event()
        finish_worker = threading.Event()
        self.addCleanup(finish_worker.set)

        def produce(sink, should_stop):
            sink("第一段")
            try:
                finish_worker.wait(2)
                self.assertTrue(should_stop())
                return {"status": "answered"}
            finally:
                worker_done.set()

        stream = _stream(produce, guard, admission=admission)
        self.assertIn("chunk", await anext(stream))
        await stream.aclose()
        next_request = guard.admit_request()
        self.assertFalse(next_request.try_start())
        finish_worker.set()
        self.assertTrue(await asyncio.to_thread(worker_done.wait, 2))
        # The worker's finally releases after signalling the fake producer's exit.
        for _ in range(100):
            if next_request.try_start():
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("Slot was not released after worker exit")
        next_request.release()
        self.assertEqual(guard.status()["requests_in_flight"], 0)

    async def test_response_disconnect_before_body_iteration_returns_admission(self):
        guard = AccessGuard(GuardSettings())
        admission = guard.admit_request()

        async def fail_before_iteration(*args):
            raise RuntimeError("transport closed")

        response = _AnswerStreamResponse(_stream(lambda *_: {}, guard, admission=admission),
                                         admission=admission)
        with (
            patch("fastapi.responses.StreamingResponse.__call__", fail_before_iteration),
            self.assertRaises(RuntimeError),
        ):
            await response({}, None, None)
        self.assertEqual(guard.status()["requests_in_flight"], 0)


class QueueEndpointTests(unittest.TestCase):
    def test_books_and_validation_follow_the_engine_catalog_after_startup(self):
        engine = FakeEngine(enable_llm=False)
        catalog = [{"book_name": "os", "count": 2}]
        engine.list_indexed_books = lambda: catalog
        guard = AccessGuard(GuardSettings())
        with tempfile.TemporaryDirectory() as directory:
            app = create_api_app(engine, guard, BOOKS,
                                 feedback_store=FeedbackStore(Path(directory) / "feedback.sqlite3"))
            with TestClient(app) as client:
                catalog[:] = [{"book_name": "database", "count": 3}]
                self.assertEqual(client.get("/v1/books").json()[0]["book_id"], "database")
                self.assertEqual(client.post("/v1/ask", json={"query": "问题", "book_id": "os"})
                                 .status_code, 404)
                self.assertEqual(client.post("/v1/ask", json={"query": "问题", "book_id": "database"})
                                 .status_code, 200)
        self.assertEqual(engine.calls[0]["book_name"], "database")

    def test_full_queue_refuses_both_endpoints_before_sse_or_worker_start(self):
        guard = AccessGuard(GuardSettings(max_pending_requests=1))
        active = guard.admit_request()
        active.try_start()
        self.addCleanup(active.release)
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            app = create_api_app(engine, guard, BOOKS,
                                 feedback_store=FeedbackStore(Path(directory) / "feedback.sqlite3"))
            with TestClient(app) as client:
                for endpoint in ("/v1/ask", "/v1/ask/stream"):
                    response = client.post(endpoint, json={"query": "进程是什么？"})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.headers["content-type"], "application/json")
                self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(engine.calls, [])
        self.assertEqual(guard.status()["generations_remaining_today"], 200)


if __name__ == "__main__":
    unittest.main()
