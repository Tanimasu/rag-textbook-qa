"""Run the shipped public page in Chromium against an isolated local fake backend.

Run explicitly: python tests/browser/public_chat_browser.py -v. This file is not
part of ordinary unittest discovery and needs the optional ``browser`` group.
Only the backend is fake: the browser executes the unchanged HTML and JavaScript.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unittest
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/rag_textbook_qa/api/static/index.html"
RETRIEVAL_PAGE = ROOT / "src/rag_textbook_qa/api/static/retrieval.html"
IMPORT_PAGE = ROOT / "src/rag_textbook_qa/api/static/imports.html"


def answer_result(answer_id: str = "answer-1") -> dict[str, Any]:
    return {
        "status": "answered",
        "answer_id": answer_id,
        "answer": "## 进程\n**进程**是程序的一次执行。【参考资料 1】\n\n<img src=x>",
        "sources": [
            {
                "citation_id": 1,
                "book_id": "os",
                "chunk_id": "chapter_p1",
                "source_sha256": hashlib.sha256("进程是程序在一个数据集合上的一次执行。".encode()).hexdigest(),
                "book": "操作系统",
                "section": "第二章 进程管理",
                "excerpt": "进程是程序在一个数据集合上的一次执行。",
                "truncated": False,
            }
        ],
        "citation_integrity": {"status": "linked"},
        "timing": {"retrieval_seconds": 0.1, "total_seconds": 0.2},
    }


def retrieval_result() -> dict[str, Any]:
    first = {**answer_result()["sources"][0], "position": 1, "scores": {"rrf_score": .02},
             "source_ranks": {"embedding": 1, "bm25": 2}, "preview_truncated": False}
    other = {**first, "chunk_id": "chapter_p2", "excerpt": "程序是静态代码。", "position": 2}
    return {
        "context_budget": 4000, "elapsed_seconds": .2, "reranker_enabled": True,
        "generation_calls": 0, "fusion_weights": {"embedding": 1, "bm25": .5},
        "stages": [
            {"id": key, "label": label, "count": len(items), "items": items}
            for key, label, items in (
                ("keyword", "关键词", [first, other]), ("embedding", "向量", [first, other]),
                ("fusion", "融合", [first, other]), ("rerank", "重排", [first]),
                ("context", "最终上下文", [first]),
            )
        ],
    }


def import_result(state: str = "prepared") -> dict[str, Any]:
    return {
        "job_id": "a" * 32, "book_id": "os", "book": "操作系统", "filename": "教材.md",
        "state": state, "stage": "embedding" if state == "building" else state,
        "completed_chunks": 1 if state == "building" else 2 if state == "complete" else 0,
        "total_chunks": 2,
        "message": {"prepared": "解析完成，确认预览后可建立索引。", "building": "正在向量化教材片段…",
                    "complete": "索引已建立，可以返回聊天选择这本教材。", "failed": "建索引失败，可以重试。"}[state],
        "preview": [{"section": "第一章 · 进程", "excerpt": "进程是程序的一次执行。<script>neverRun()</script>"}],
    }


@dataclass
class Reply:
    initial: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    final: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    gate: threading.Event | None = None
    http_status: int = 200
    detail: str = ""
    retry_after: str | None = None
    line_ending: str = "\n"


class FakeBackend(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.replies: deque[Reply] = deque()
        self.asks: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []
        self.access_code_required = False
        self.lock = threading.Lock()
        self.gates: list[threading.Event] = []
        self.book_failures_remaining = 0
        self.reading_changed = False
        self.reading_failure = False
        self.reads: list[dict[str, Any]] = []
        self.inspections: list[dict[str, Any]] = []
        self.inspection_gate: threading.Event | None = None
        self.inspection_status = 200
        self.imports_enabled = False
        self.import_state = "prepared"
        self.uploads: list[bytes] = []
        self.import_starts: list[dict[str, Any]] = []

    def enqueue(self, reply: Reply) -> None:
        with self.lock:
            self.replies.append(reply)
        if reply.gate is not None:
            self.gates.append(reply.gate)


class FakeHandler(BaseHTTPRequestHandler):
    server: FakeBackend
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args: Any) -> None:
        pass

    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, value: Any, status: int = 200) -> None:
        self.respond(status, json.dumps(value).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path == "/":
            self.respond(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/retrieval":
            self.respond(200, RETRIEVAL_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/imports":
            self.respond(200, IMPORT_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/v1/imports":
            if not self.server.imports_enabled:
                self.json_response({"detail": "教材导入只在本机服务提供"}, 404)
            else:
                self.json_response({"books": [{"book_id": "os", "label": "操作系统"}],
                                    "formats": [".md", ".json"], "max_upload_bytes": 8388608,
                                    "storage_path": "/tmp/browser-imports"})
        elif self.path == "/v1/imports/" + "a" * 32:
            self.json_response(import_result(self.server.import_state))
        elif self.path.startswith("/v1/books/os/passages?"):
            chunk_id = parse_qs(urlsplit(self.path).query)["chunk_id"][0]
            self.server.reads.append({"chunk_id": chunk_id, "access_code": self.headers.get("X-Access-Code")})
            if self.server.reading_failure:
                self.json_response({"detail": "原文暂时无法读取，请稍后重试"}, 503)
                return
            source = dict(answer_result()["sources"][0])
            if self.server.reading_changed:
                source.update(excerpt="更新后的教材原文。", source_sha256="b" * 64)
            source.update(chunk_id=chunk_id, is_anchor=True)
            neighbor = {**source, "chunk_id": "chapter_p2", "excerpt": "相邻原文：进程具有动态性。", "is_anchor": False}
            self.json_response({
                "book_id": "os", "anchor_chunk_id": chunk_id,
                "source_sha256": source["source_sha256"],
                "previous_chunk_id": "chapter_p1" if chunk_id == "chapter_p2" else None,
                "next_chunk_id": "chapter_p2" if chunk_id == "chapter_p1" else None,
                "passages": [source, neighbor],
            })
        elif self.path == "/v1/books":
            with self.server.lock:
                if self.server.book_failures_remaining:
                    self.server.book_failures_remaining -= 1
                    self.json_response({"detail": "temporarily unavailable"}, 503)
                    return
            self.json_response(
                [
                    {"book_id": "os", "label": "操作系统"},
                    {"book_id": "database", "label": "数据库原理及应用"},
                ]
            )
        elif self.path == "/health":
            self.json_response(
                {
                    "access_code_required": self.server.access_code_required,
                    "generations_remaining_today": 17,
                    "imports_enabled": self.server.imports_enabled,
                }
            )
        else:
            self.json_response({"detail": "not found"}, 404)

    def write_event(self, name: str, payload: dict[str, Any], line_ending: str) -> None:
        frame = line_ending.join(
            [f"event: {name}", f"data: {json.dumps(payload, ensure_ascii=False)}", "", ""]
        ).encode()
        # Split in the middle of a Chinese UTF-8 character as well as an SSE frame.
        # TCP may coalesce writes; the test also holds the final event behind a gate.
        split = next((i + 1 for i, value in enumerate(frame) if value >= 0xE0), 8)
        self.wfile.write(frame[:split])
        self.wfile.flush()
        self.wfile.write(frame[split:])
        self.wfile.flush()

    def do_POST(self) -> None:
        content = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.startswith("/v1/imports/prepare?"):
            self.server.uploads.append(content)
            self.server.import_state = "prepared"
            self.json_response(import_result())
            return
        payload = json.loads(content)
        if self.path == "/v1/imports/" + "a" * 32 + "/start":
            self.server.import_starts.append(payload)
            self.server.import_state = "building"
            self.json_response(import_result("building"))
            return
        if self.path == "/v1/retrieval/inspect":
            self.server.inspections.append({"payload": payload, "access_code": self.headers.get("X-Access-Code")})
            if self.server.inspection_gate:
                self.server.inspection_gate.wait(timeout=10)
            try:
                self.json_response(retrieval_result() if self.server.inspection_status == 200
                                   else {"detail": "需要正确的访问口令"}, self.server.inspection_status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == "/v1/feedback":
            with self.server.lock:
                self.server.feedback.append(payload)
            self.json_response({"status": "saved"})
            return
        if self.path != "/v1/ask/stream":
            self.json_response({"detail": "not found"}, 404)
            return
        with self.server.lock:
            self.server.asks.append(
                {"payload": payload, "access_code": self.headers.get("X-Access-Code")}
            )
            reply = self.server.replies.popleft() if self.server.replies else Reply(http_status=500)
        if reply.http_status != 200:
            body = json.dumps({"detail": reply.detail}).encode()
            self.send_response(reply.http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if reply.retry_after:
                self.send_header("Retry-After", reply.retry_after)
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for name, data in reply.initial:
                self.write_event(name, data, reply.line_ending)
            if reply.gate is not None and not reply.gate.wait(timeout=10):
                return
            for name, data in reply.final:
                self.write_event(name, data, reply.line_ending)
        except (BrokenPipeError, ConnectionResetError):
            # Stopping generation aborts the browser's fetch and closes this socket.
            pass


class PublicChatBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from playwright.sync_api import expect, sync_playwright

        cls.expect = staticmethod(expect)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        channel = os.environ.get("RAG_QA_BROWSER_CHANNEL")
        cls.browser = cls.playwright.chromium.launch(headless=True, channel=channel or None)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.server_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.stop_backend)
        self.base_url = f"http://127.0.0.1:{self.backend.server_port}"
        self.context = self.browser.new_context()
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.set_default_timeout(5000)
        self.errors: list[str] = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.external_requests: list[str] = []

        def local_only(route: Any) -> None:
            if route.request.url.startswith(self.base_url + "/"):
                route.continue_()
            else:
                self.external_requests.append(route.request.url)
                route.abort()

        self.context.route("**/*", local_only)

    def tearDown(self) -> None:
        self.assertEqual(self.errors, [], "The shipped page raised a JavaScript error")
        self.assertEqual(self.external_requests, [], "The page tried to contact an external service")

    def stop_backend(self) -> None:
        for gate in self.backend.gates:
            gate.set()
        self.backend.shutdown()
        self.backend.server_close()
        self.server_thread.join(timeout=5)

    def open_page(self, book_id: str = "os") -> None:
        self.page.goto(self.base_url)
        self.expect(self.page.locator("#book option")).to_have_count(3)
        self.expect(self.page.locator("#quota")).to_have_text("今日剩余生成次数：17")
        if book_id:
            self.page.locator("#book").select_option(book_id)

    def submit(self, query: str = "什么是进程？") -> None:
        self.page.locator("#input").fill(query)
        self.page.locator("#send").click()

    def expect_ready(self, focus: str = "#input") -> None:
        self.expect(self.page.locator("#send")).to_be_enabled()
        self.expect(self.page.locator("#cancel")).to_be_hidden()
        self.expect(self.page.locator("#thread")).not_to_have_attribute("aria-busy", "true")
        self.expect(self.page.locator(focus)).to_be_focused()

    def isolated_clipboard(self, *, api: bool, command: str) -> None:
        # Never read or overwrite the user's system clipboard in this test.
        config = json.dumps({"api": api, "command": command})
        self.context.add_init_script("""(() => {
          const config = CONFIG;
          window.copyAttempts = [];
          Object.defineProperty(navigator, 'clipboard', { configurable: true,
            value: config.api ? { writeText: async () => {
              throw new Error('isolated clipboard permission refusal');
            } } : undefined });
          document.execCommand = (command) => {
            window.copyAttempts.push({ command, text: document.activeElement.value });
            if (config.command === 'throw') throw new Error('isolated copy failure');
            return config.command === 'success';
          };
        })();""".replace("CONFIG", config))

    def history_records(self) -> list[dict[str, Any]]:
        return self.page.evaluate("""() => new Promise((resolve, reject) => {
          const request = indexedDB.open('rag-qa-history', 1);
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction('records', 'readonly');
            const rows = transaction.objectStore('records').getAll();
            transaction.oncomplete = () => { db.close(); resolve(rows.result); };
          };
        })""")

    def saved_answer(self, query: str = "什么是进程？", book_id: str = "os",
                     result: dict[str, Any] | None = None) -> None:
        self.backend.enqueue(Reply(final=[("result", result or answer_result())]))
        self.page.locator("#book").select_option(book_id)
        self.submit(query)
        self.expect(self.page.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.expect_ready()

    def capture_page(self, name: str) -> None:
        screenshot_dir = os.environ.get("RAG_QA_BROWSER_SCREENSHOTS")
        if screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(Path(screenshot_dir) / name), full_page=True)

    def test_history_restores_snapshot_citations_and_export_without_new_request(self) -> None:
        self.backend.access_code_required = True
        self.open_page()
        self.page.locator("#code").fill("never-persist-this-code")
        result = answer_result()
        result["conflicts"] = ["教材中的两种定义"]
        self.saved_answer(result=result)
        rows = self.history_records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"]["sources"], result["sources"])
        self.assertNotIn("answer_id", rows[0]["result"])
        self.assertNotIn("never-persist-this-code", json.dumps(rows))
        self.backend.book_failures_remaining = 1
        self.page.reload()
        card = self.page.locator(".answer")
        self.expect(card).to_contain_text("已保存的回答")
        self.expect(card).to_contain_text("教材中的两种定义")
        self.expect(card.get_by_role("button", name="👍 有帮助")).to_have_count(0)
        card.get_by_role("button", name="资料 1", exact=True).click()
        self.expect(card.locator(".source .excerpt")).to_have_text(result["sources"][0]["excerpt"])
        with self.page.expect_download() as download_info:
            card.get_by_role("button", name="保存问答（含来源）", exact=True).click()
        exported = Path(download_info.value.path()).read_text(encoding="utf-8")
        self.assertIn(result["answer"], exported)
        self.assertIn(result["sources"][0]["excerpt"], exported)
        self.assertEqual(len(self.backend.asks), 1)
        self.assertEqual(self.backend.feedback, [])

    def test_history_search_book_and_favorite_filters_and_view(self) -> None:
        self.open_page()
        self.saved_answer("进程与程序有什么区别？")
        result = answer_result("db-answer")
        result["answer"] = "候选键保证记录唯一。【参考资料 1】"
        result["sources"][0].update(book="数据库原理及应用", section="第三章 关系模型",
                                    excerpt="复合属性也可以构成候选键。")
        self.saved_answer("主键如何选择？", "database", result)
        self.page.locator("#toggle-history").click()
        items = self.page.locator(".history-item")
        self.expect(items).to_have_count(2)
        self.page.locator("#history-search").fill("复合属性")
        self.expect(items).to_have_count(1)
        self.expect(items.locator("h3")).to_have_text("主键如何选择？")
        self.page.locator("#history-search").fill("候选键 唯一")
        self.expect(items).to_have_count(1)
        items.get_by_role("button", name="☆ 收藏", exact=True).click()
        self.expect(items.get_by_role("button", name="★ 已收藏", exact=True)).to_have_attribute(
            "aria-pressed", "true"
        )
        self.page.locator("#history-favorites").check()
        self.page.locator("#history-book").select_option("os")
        self.expect(items).to_have_count(0)
        self.expect(self.page.locator("#history-list")).to_contain_text("没有匹配的记录")
        self.page.locator("#history-book").select_option("database")
        self.expect(items).to_have_count(1)
        self.capture_page("history-desktop.png")
        items.get_by_role("button", name="查看原回答", exact=True).click()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.expect(self.page.locator(".answer .body")).to_contain_text("候选键")
        self.expect(self.page.locator("#history-panel")).to_be_hidden()
        self.page.reload()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="★ 已收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.assertFalse(next(row for row in self.history_records()
                              if row["query"] == "主键如何选择？")["favorite"])
        self.assertEqual(len(self.backend.asks), 2)

    def test_clear_view_and_delete_record_have_distinct_persistent_effects(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator("#clear").click()
        self.page.reload()
        self.expect(self.page.locator("#thread .msg")).to_have_count(0)
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator(".answer").get_by_role("button", name="删除记录").click()
        self.assertEqual(len(self.history_records()), 1)
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator(".answer").get_by_role("button", name="删除记录").click()
        self.expect(self.page.locator("#thread .msg")).to_have_count(0)
        self.page.reload()
        self.expect(self.page.locator("#empty")).to_be_visible()
        self.assertEqual(self.history_records(), [])

    def test_bulk_cleanup_preserves_favorites_then_deletes_all(self) -> None:
        self.open_page()
        self.saved_answer("要收藏的问题")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.saved_answer("临时问题", "database")
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-cleanup summary").click()
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator("#history-clean").click()
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-item h3")).to_have_text("要收藏的问题")
        self.page.reload()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-cleanup summary").click()
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator("#history-delete-all").click()
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator("#history-delete-all").click()
        self.expect(self.page.locator(".history-item")).to_have_count(0)
        self.expect(self.page.locator("#empty")).to_be_visible()
        self.assertEqual(self.history_records(), [])

    def test_storage_blocked_keeps_chat_and_export_available(self) -> None:
        self.context.add_init_script("""Object.defineProperty(window, 'indexedDB', {
          get() { throw new DOMException('Storage disabled', 'SecurityError'); }
        });""")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page()
        self.expect(self.page.locator("#history-status")).to_contain_text("无法读取或保存历史记录")
        self.submit()
        self.expect_ready()
        self.expect(self.page.locator(".answer")).to_contain_text("未能保存历史记录")
        self.expect(self.page.get_by_role("button", name="保存问答（含来源）")).to_be_enabled()
        self.expect(self.page.get_by_role("button", name="重试保存")).to_be_enabled()

    def test_failed_storage_write_can_retry_without_duplicating_record(self) -> None:
        self.context.add_init_script("""(() => {
          const put = IDBObjectStore.prototype.put;
          let fail = true;
          IDBObjectStore.prototype.put = function(...args) {
            if (fail) { fail = false; throw new DOMException('Full', 'QuotaExceededError'); }
            return put.apply(this, args);
          };
        })();""")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page()
        self.submit()
        self.expect_ready()
        self.expect(self.page.locator(".answer")).to_contain_text("未能保存历史记录")
        self.assertEqual(self.history_records(), [])
        self.page.get_by_role("button", name="重试保存").click()
        self.expect(self.page.locator(".answer")).to_contain_text("已保存到此浏览器")
        self.page.reload()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.assertEqual(len(self.history_records()), 1)
        self.assertEqual(len(self.backend.asks), 1)

    def test_mobile_history_layout_and_citation_snapshot(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        self.saved_answer("进程和程序有什么区别？" * 5)
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-search").fill("进程")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.assertTrue(self.page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ))
        self.capture_page("history-mobile.png")
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        self.page.locator(".answer .cite").click()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()
        self.assertTrue(self.page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ))

    def test_partial_stream_is_not_saved_and_retrieval_only_snapshot_is_saved(self) -> None:
        self.backend.enqueue(Reply(initial=[("chunk", {"text": "没有完成的回答"})]))
        self.open_page()
        self.submit()
        self.expect_ready()
        self.assertEqual(self.history_records(), [])
        result = answer_result()
        result.update(status="retrieval_only", answer=None, message="本次仅展示教材原文。")
        self.saved_answer("只查看资料", result=result)
        self.page.reload()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.expect(self.page.locator(".answer")).to_contain_text("本次仅展示教材原文")
        self.expect(self.page.locator(".source .excerpt")).to_be_visible()
        self.expect(self.page.locator("#thread")).not_to_contain_text("没有完成的回答")

    def test_library_pagination_and_search_include_records_outside_first_page(self) -> None:
        self.open_page()
        # Seed a larger, deterministic personal library without making 25 requests.
        self.page.evaluate("""result => new Promise(resolve => {
          const request = indexedDB.open('rag-qa-history', 1);
          request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction('records', 'readwrite');
            for (let i = 0; i < 25; i++) transaction.objectStore('records').put({
              id: 'seed-' + i, created_at: new Date(1700000000000 + i * 1000).toISOString(),
              query: i === 0 ? '历史问题 第零条' : '历史问题 ' + i,
              book_id: 'os', book_label: '操作系统', favorite: false,
              result: { status: result.status, answer: result.answer, sources: result.sources }
            });
            transaction.oncomplete = () => { db.close(); resolve(); };
          };
        })""", answer_result())
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator(".history-item")).to_have_count(20)
        self.page.locator("#history-more").click()
        self.expect(self.page.locator(".history-item")).to_have_count(25)
        self.expect(self.page.locator("#history-more")).to_be_hidden()
        self.page.locator("#history-search").fill("第零条")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.assertEqual(self.backend.asks, [])

    def test_multiple_tabs_keep_each_others_records_and_new_tab_restores_history(self) -> None:
        self.open_page()
        self.saved_answer("第一个标签的问题")
        second = self.context.new_page()
        second.on("pageerror", lambda error: self.errors.append(str(error)))
        second.goto(self.base_url)
        self.expect(second.locator(".answer")).to_have_count(1)
        self.backend.enqueue(Reply(final=[("result", answer_result("second-tab"))]))
        second.locator("#book").select_option("database")
        second.locator("#input").fill("第二个标签的问题")
        second.locator("#send").click()
        self.expect(second.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator(".history-item")).to_have_count(2)
        self.assertEqual(len(self.history_records()), 2)
        other = self.browser.new_context()
        try:
            other.route("**/*", lambda route: route.continue_()
                        if route.request.url.startswith(self.base_url + "/") else route.abort())
            private_page = other.new_page()
            private_page.goto(self.base_url)
            self.expect(private_page.locator("#send")).to_be_enabled()
            self.expect(private_page.locator(".answer")).to_have_count(0)
        finally:
            other.close()

    def test_reader_pairs_snapshot_with_answer_and_reads_neighbors_only_on_request(self) -> None:
        self.backend.access_code_required = True
        self.open_page()
        self.page.locator("#code").fill("reader-test-code")
        self.saved_answer()
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.expect(self.page.locator(".reader-quote")).to_have_text(
            answer_result()["sources"][0]["excerpt"]
        )
        left = self.page.locator(".answer-content").bounding_box()
        right = self.page.locator(".reader").bounding_box()
        self.assertGreater(right["x"], left["x"] + left["width"])
        self.assertEqual(self.backend.reads, [])
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("定位片段与引用时的原文一致")
        self.expect(self.page.locator(".reader-current")).to_contain_text("进程具有动态性")
        self.expect(self.page.get_by_role("button", name="向前阅读", exact=True)).to_be_disabled()
        self.page.get_by_role("button", name="向后阅读", exact=True).click()
        self.expect(self.page.get_by_role("button", name="向后阅读", exact=True)).to_be_disabled()
        self.expect(self.page.get_by_role("button", name="向前阅读", exact=True)).to_be_enabled()
        self.assertEqual(self.backend.reads, [
            {"chunk_id": "chapter_p1", "access_code": "reader-test-code"},
            {"chunk_id": "chapter_p2", "access_code": "reader-test-code"},
        ])
        self.capture_page("reader-desktop.png")
        self.page.get_by_role("button", name="收起对照", exact=True).click()
        self.expect(self.page.locator(".reader")).to_have_count(0)
        self.expect(self.page.locator(".app")).not_to_have_class("app has-reader")
        self.assertEqual(len(self.backend.asks), 1)

    def test_reader_marks_changed_text_and_keeps_old_snapshot_when_read_fails(self) -> None:
        self.open_page()
        self.saved_answer()
        self.backend.reading_changed = True
        self.page.reload()
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("教材原文已更新")
        self.expect(self.page.locator(".reader-current")).to_contain_text("更新后的教材原文")
        self.expect(self.page.locator(".reader-quote")).not_to_contain_text("更新后的教材原文")
        self.backend.reading_failure = True
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("原文暂时无法读取")
        self.expect(self.page.locator(".reader-quote")).to_have_text(
            answer_result()["sources"][0]["excerpt"]
        )
        self.page.locator("#clear").click()
        self.expect(self.page.locator(".reader")).to_have_count(0)
        self.expect(self.page.locator(".app")).not_to_have_class("app has-reader")

    def test_mobile_reader_switches_between_answer_and_source_without_overflow(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer .cite").click()
        self.expect(self.page.locator(".reader")).to_be_visible()
        self.expect(self.page.locator(".answer-content")).to_be_hidden()
        self.page.locator(".reader-tabs").get_by_role("button", name="回答", exact=True).click()
        self.expect(self.page.locator(".answer-content")).to_be_visible()
        self.expect(self.page.locator(".reader")).to_be_hidden()
        self.page.locator(".reader-tabs").get_by_role("button", name="原文", exact=True).click()
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("进程具有动态性")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("reader-mobile.png")

    def test_legacy_history_without_position_still_opens_snapshot(self) -> None:
        self.open_page()
        result = answer_result()
        for key in ("chunk_id", "source_sha256"):
            result["sources"][0].pop(key)
        self.saved_answer(result=result)
        self.page.reload()
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()
        self.expect(self.page.get_by_role("button", name="读取相邻原文", exact=True)).to_be_disabled()
        self.assertEqual(self.backend.reads, [])

    def test_retrieval_observatory_shows_stages_rank_changes_and_actual_context(self) -> None:
        self.open_page()
        self.page.get_by_role("link", name="检索观察台", exact=True).click()
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("进程和程序的区别")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-status")).to_contain_text("检索观察完成")
        self.expect(self.page.locator("#inspect-stages button")).to_have_count(5)
        self.expect(self.page.locator(".candidate")).to_have_count(1)
        self.expect(self.page.locator(".excerpt")).to_have_text(answer_result()["sources"][0]["excerpt"])
        self.page.get_by_role("button", name="关键词 2", exact=True).click()
        self.expect(self.page.locator(".candidate")).to_have_count(2)
        self.expect(self.page.locator(".candidate").last).to_contain_text("未入选上下文")
        self.page.get_by_role("button", name="融合 2", exact=True).click()
        self.expect(self.page.locator("#inspect-explanation")).to_contain_text("关键词权重 0.5")
        self.page.get_by_text("同一片段的排名变化", exact=True).click()
        self.expect(self.page.locator("#inspect-ranks table tr")).to_have_count(3)
        self.capture_page("retrieval-desktop.png")
        self.assertEqual(self.backend.inspections[0]["payload"],
                         {"query": "进程和程序的区别", "book_id": "os", "top_k": 5})
        self.assertEqual(self.backend.asks, [])
        self.assertEqual(self.backend.feedback, [])

    def test_retrieval_observatory_keeps_request_book_when_next_draft_changes(self) -> None:
        gate = threading.Event()
        self.backend.inspection_gate = gate
        self.backend.gates.append(gate)
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("进程")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-send")).to_be_disabled()
        self.page.locator("#inspect-book").select_option("database")
        self.page.locator("#inspect-query").fill("下一次准备问主键")
        gate.set()
        self.expect(self.page.locator("#inspect-title")).to_have_text("进程")
        self.expect(self.page.locator("#inspect-summary")).to_contain_text("操作系统")
        self.expect(self.page.locator("#inspect-query")).to_have_value("下一次准备问主键")

    def test_retrieval_mobile_access_code_retry_and_stop(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.backend.inspection_status = 401
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("进程")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-code")).to_be_visible()
        self.expect(self.page.locator("#inspect-status")).to_contain_text("访问口令")
        self.page.locator("#inspect-code").fill("inspect-test-code")
        self.backend.inspection_status = 200
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-result")).to_be_visible()
        self.assertEqual(self.backend.inspections[-1]["access_code"], "inspect-test-code")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("retrieval-mobile.png")
        gate = threading.Event()
        self.backend.inspection_gate = gate
        self.backend.gates.append(gate)
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-stop")).to_be_visible()
        self.page.locator("#inspect-stop").click()
        self.expect(self.page.locator("#inspect-status")).to_contain_text("已停止等待")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()

    def test_import_guide_previews_before_build_and_restores_progress_after_refresh(self) -> None:
        self.backend.imports_enabled = True
        self.open_page()
        self.page.get_by_role("link", name="导入教材", exact=True).click()
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        content = "# 第一章\n进程是程序的一次执行。".encode()
        self.page.locator("#import-file").set_input_files({"name": "教材.md", "mimeType": "text/markdown", "buffer": content})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.expect(self.page.locator("#import-start")).to_be_disabled()
        self.expect(self.page.locator("#import-preview script")).to_have_count(0)
        self.expect(self.page.locator("#import-preview")).to_contain_text("<script>neverRun()</script>")
        self.assertEqual(self.backend.import_starts, [])
        self.assertEqual(self.backend.uploads, [content])
        self.capture_page("imports-desktop.png")
        self.page.locator("#import-confirm").check()
        self.page.locator("#import-start").click()
        self.expect(self.page.locator("#import-progress-text")).to_contain_text("1 / 2")
        self.expect(self.page.locator("#import-file")).to_be_disabled()
        self.page.reload()
        self.expect(self.page.locator("#import-progress-text")).to_contain_text("1 / 2")
        self.backend.import_state = "complete"
        self.page.locator("#import-refresh").click()
        self.expect(self.page.locator("#import-return")).to_be_visible()
        self.expect(self.page.locator("#import-progress")).to_have_attribute("value", "2")
        self.assertEqual(self.backend.import_starts, [{"confirm_replace": True}])
        self.assertEqual(self.backend.asks, [])

    def test_import_mobile_failed_job_can_retry_with_preserved_preview(self) -> None:
        self.backend.imports_enabled = True
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        self.page.locator("#import-file").set_input_files({"name": "教材.json", "mimeType": "application/json", "buffer": b"[]"})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.page.locator("#import-confirm").check()
        self.page.locator("#import-start").click()
        self.expect(self.page.locator("#import-progress-text")).to_contain_text("1 / 2")
        self.backend.import_state = "failed"
        self.page.locator("#import-refresh").click()
        self.expect(self.page.locator("#import-start")).to_have_text("重试建立索引")
        self.expect(self.page.locator("#import-preview")).to_contain_text("进程是程序")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("imports-mobile.png")
        self.page.locator("#import-start").click()
        self.expect(self.page.locator("#import-progress-text")).to_contain_text("1 / 2")
        self.assertEqual(len(self.backend.import_starts), 2)
        self.assertEqual(len(self.backend.uploads), 1)

    def test_import_disabled_service_and_oversized_file_do_not_upload(self) -> None:
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-status")).to_contain_text("只在本机服务提供")
        self.expect(self.page.locator("#import-prepare")).to_be_disabled()
        self.backend.imports_enabled = True
        self.page.locator("#import-connect").click()
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        self.page.locator("#import-file").set_input_files({"name": "big.md", "mimeType": "text/markdown", "buffer": b"a" * (8 * 1024 * 1024 + 1)})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-status")).to_contain_text("不超过 8 MB")
        self.assertEqual(self.backend.uploads, [])
        self.assertEqual(self.backend.import_starts, [])

    def test_unknown_compute_time_keeps_answer_actions_available(self) -> None:
        result = answer_result()
        result["compute"] = {
            "embedding": {"backend": "local", "platform": "Darwin", "device": "mps",
                          "fallback_used": False, "elapsed_seconds": None},
        }
        result["timing"] = {"total_seconds": True, "retrieval_seconds": -1,
                            "first_token_seconds": 0}
        self.backend.enqueue(Reply(final=[("result", result)]))
        self.open_page()
        self.submit()
        self.expect_ready()
        card = self.page.locator(".answer")
        self.expect(card).to_contain_text("Embedding · 本地（macOS） · MPS · 耗时未知")
        self.expect(card.locator(".body")).to_contain_text("进程")
        self.expect(card.locator(":scope > .meta")).to_have_text("首字 0.00 秒")
        self.expect(card.get_by_role("button", name="保存问答（含来源）")).to_be_enabled()
        self.assertEqual(len(self.backend.asks), 1)

    def test_copy_fallback_preserves_focus_and_next_question(self) -> None:
        self.isolated_clipboard(api=False, command="success")
        result = answer_result()
        self.backend.enqueue(Reply(final=[("result", result)]))
        self.open_page()
        self.submit()
        self.expect_ready()
        self.page.locator("#input").fill("下一题草稿")
        self.page.locator("#book").select_option("database")
        copy = self.page.get_by_role("button", name=re.compile(r"^(复制答案|已复制)$"))
        copy.focus()
        copy.press("Enter")
        self.expect(copy).to_have_text("已复制")
        self.expect(copy).to_be_focused()
        self.expect(self.page.locator("textarea[readonly]")).to_have_count(0)
        self.expect(self.page.locator("#input")).to_have_value("下一题草稿")
        self.expect(self.page.locator("#book")).to_have_value("database")
        self.assertEqual(self.page.evaluate("window.copyAttempts"),
                         [{"command": "copy", "text": result["answer"]}])
        self.assertEqual(len(self.backend.asks), 1)
        self.assertEqual(self.backend.feedback, [])

    def test_clipboard_api_refusal_can_use_local_fallback(self) -> None:
        self.isolated_clipboard(api=True, command="success")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page()
        self.submit()
        copy = self.page.get_by_role("button", name=re.compile(r"^(复制答案|已复制|复制失败.*)$"))
        copy.click()
        self.expect(copy).to_have_text("已复制")
        self.expect(copy).to_be_focused()
        self.assertEqual(len(self.page.evaluate("window.copyAttempts")), 1)
        self.assertEqual(len(self.backend.asks), 1)

    def test_copy_failure_removes_temporary_input_and_explains_recovery(self) -> None:
        self.isolated_clipboard(api=False, command="throw")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page()
        self.submit()
        copy = self.page.get_by_role("button", name=re.compile(r"^(复制答案|复制失败.*)$"))
        copy.click()
        self.expect(copy).to_have_text("复制失败，请手动复制")
        self.expect(copy).to_be_focused()
        self.expect(self.page.locator("textarea[readonly]")).to_have_count(0)
        self.page.get_by_role("button", name="保存问答（含来源）").wait_for(state="visible")
        self.assertEqual(len(self.backend.asks), 1)

    def test_submit_stream_final_sources_and_clear(self) -> None:
        gate = threading.Event()
        result = answer_result()
        result["compute"] = {
            "embedding": {"backend": "remote", "platform": "Windows", "device": "cuda",
                          "fallback_used": False, "elapsed_seconds": .05},
            "reranker": {"backend": "remote", "platform": "Windows", "device": "cuda",
                         "fallback_used": False, "elapsed_seconds": .05},
        }
        self.backend.enqueue(
            Reply(
                initial=[("chunk", {"text": "流式片段：正在形成答案"})],
                final=[("result", result)],
                gate=gate,
            )
        )
        self.open_page()
        self.submit()
        body = self.page.locator(".answer .body")
        self.expect(body).to_have_text("流式片段：正在形成答案")
        self.expect(self.page.locator("#send")).to_be_disabled()
        self.expect(self.page.locator("#clear")).to_be_disabled()
        self.expect(self.page.locator("#thread")).to_have_attribute("aria-busy", "true")
        self.assertEqual(
            self.backend.asks[0]["payload"], {"query": "什么是进程？", "book_id": "os"}
        )
        gate.set()
        self.expect(body.locator("strong")).to_have_text("进程")
        self.expect(body).not_to_contain_text("流式片段")
        self.expect(body.locator("img")).to_have_count(0)
        self.expect(body).to_contain_text("<img src=x>")
        self.expect_ready()
        runtime = self.page.locator("details.runtime")
        self.expect(runtime).not_to_have_attribute("open", "")
        self.expect(runtime.locator(".meta")).not_to_be_visible()
        self.page.get_by_text("运行详情", exact=True).click()
        self.expect(runtime.locator(".meta")).to_contain_text("远程 Worker（Windows）")
        self.expect(self.page.locator("details.sources")).not_to_have_attribute("open", "")
        body.get_by_role("button", name="资料 1").click()
        self.expect(self.page.locator("details.sources")).to_have_attribute("open", "")
        self.expect(self.page.locator(".source")).to_be_focused()
        self.expect(self.page.locator(".source .excerpt")).to_be_visible()
        self.page.locator("#clear").click()
        self.expect(self.page.locator("#thread .msg")).to_have_count(0)
        self.expect(self.page.locator("#empty")).to_be_visible()

    def test_citation_navigation_allows_keyboard_reading_of_long_original(self) -> None:
        result = answer_result()
        result["sources"][0]["excerpt"] = "\n".join(
            [f"原文第 {number} 段：程序在数据集合上的一次执行。" for number in range(50)]
            + ["原文结尾已核对"]
        )
        self.backend.enqueue(Reply(final=[("result", result)]))
        self.open_page()
        self.submit()
        self.page.get_by_role("button", name="资料 1", exact=True).click()
        self.expect(self.page.locator(".source")).to_be_focused()
        self.page.locator(".source").press("Tab")
        excerpt = self.page.get_by_role("region", name="资料 1 原文", exact=True)
        self.expect(excerpt).to_be_focused()
        self.assertTrue(excerpt.evaluate("node => node.scrollHeight > node.clientHeight"))
        excerpt.press("End")
        self.expect(excerpt).not_to_have_js_property("scrollTop", 0)
        self.expect(excerpt).to_contain_text("原文结尾已核对")

    def test_download_preserves_original_question_sources_and_notices_without_new_requests(self) -> None:
        self.open_page()
        for width in (1280, 390):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 844})
                result = answer_result()
                result["answer"] = "完整回答【参考资料 7】【参考资料 99】"
                result["citation_integrity"] = {"status": "invalid", "unknown": [99]}
                result["conflicts"] = ["不同定义"]
                result["sources"][0].update({"citation_id": 7, "truncated": True, "table_compacted": True,
                    "section": "第二章 > 第四级标题", "excerpt": "原文\n```python\nprint(7)\n```\n原文末尾"})
                result["internal_metadata"] = "private-do-not-export"
                self.backend.enqueue(Reply(final=[("result", result)]))
                question = f"原问题{width}：教材如何定义进程？"
                self.submit(question)
                card = self.page.locator(".answer").last
                self.expect(card.locator(".body")).to_contain_text("完整回答")
                self.expect(card.locator(".source .section")).to_have_text(
                    "第二章 > 第四级标题 · 片段已截断 · 表格按行整理"
                )
                self.expect_ready()
                self.page.locator("#input").fill("尚未发送的新问题")
                self.page.locator("#book").select_option("database")
                requests_before = len(self.backend.asks)
                with self.page.expect_download(timeout=5000) as download_info:
                    card.get_by_role("button", name="保存问答（含来源）", exact=True).click()
                download = download_info.value
                self.assertEqual(download.suggested_filename, "教材问答.md")
                text = Path(download.path()).read_text(encoding="utf-8")
                for expected in (question, result["answer"], "参考资料 7", "第四级标题",
                                 result["sources"][0]["excerpt"], "````text", "片段已截断",
                                 "表格按行整理", "不存在的资料编号（99）", "不同定义"):
                    self.assertIn(expected, text)
                self.assertNotIn("尚未发送的新问题", text)
                self.assertNotIn("private-do-not-export", text)
                self.assertEqual(len(self.backend.asks), requests_before)
                self.expect(self.page.locator("#input")).to_have_value("尚未发送的新问题")
                self.expect(self.page.locator("#book")).to_have_value("database")
                self.expect(card.get_by_role("button", name="保存问答（含来源）")).to_be_focused()

    def test_completed_chapters_copy_export_and_source_navigation_agree(self) -> None:
        self.isolated_clipboard(api=False, command="success")
        raw = "正文【参考资料 1】\n\n## 参考章节\n第99章 错误章节【参考资料 1】"
        result = answer_result()
        # API chapter mapping has its own unit tests. Keep this fixture independent
        # of the project package, which the browser-only CI environment omits.
        result["answer"] = "正文【参考资料 1】\n\n## 本次来源章节\n\n- 【参考资料 1】操作系统：第二章 > 2.1 进程"
        result["sources"][0]["section"] = "第二章 > 2.1 进程"
        self.backend.enqueue(Reply(initial=[("chunk", {"text": raw})], final=[("result", result)]))
        self.open_page()
        self.submit()
        self.expect_ready()
        card = self.page.locator(".answer")
        self.expect(card.locator(".body")).to_contain_text("第二章 > 2.1 进程")
        self.expect(card.locator(".body")).not_to_contain_text("错误章节")
        card.get_by_role("button", name="资料 1", exact=True).last.click()
        self.expect(card.locator(".source")).to_be_focused()
        self.expect(card.locator(".source .section")).to_have_text("第二章 > 2.1 进程")
        card.get_by_role("button", name="复制答案", exact=True).click()
        self.assertEqual(self.page.evaluate("window.copyAttempts[0].text"), result["answer"])
        with self.page.expect_download(timeout=5000) as download_info:
            card.get_by_role("button", name="保存问答（含来源）", exact=True).click()
        saved = Path(download_info.value.path()).read_text(encoding="utf-8")
        self.assertIn(result["answer"], saved)
        self.assertNotIn("错误章节", saved)
        self.assertEqual(len(self.backend.asks), 1)
        self.assertEqual(self.backend.feedback, [])

    def test_book_loading_failure_recovers_without_losing_question_draft(self) -> None:
        self.backend.book_failures_remaining = 1
        self.page.goto(self.base_url)
        self.expect(self.page.locator("#book-status")).to_have_text("教材暂时加载失败，请重试。")
        self.expect(self.page.locator("#send")).to_be_disabled()
        self.page.locator("#input").fill("已经写好的问题")
        self.page.locator("#input").press("Enter")
        self.assertEqual(self.backend.asks, [])
        self.page.get_by_role("button", name="重新加载教材", exact=True).click()
        self.expect(self.page.locator("#book option")).to_have_count(3)
        self.expect(self.page.locator("#book")).to_be_enabled()
        self.expect(self.page.locator("#send")).to_be_enabled()
        self.expect(self.page.locator("#input")).to_have_value("已经写好的问题")
        self.expect(self.page.locator("#book-status")).to_be_hidden()
        self.expect(self.page.locator("#reload-books")).to_be_hidden()
        self.assertEqual(self.backend.asks, [])

    def test_http_refusal_retry_restores_original_query_and_book(self) -> None:
        self.backend.enqueue(Reply(http_status=401, detail="Unauthorized"))
        self.backend.enqueue(Reply(final=[("result", answer_result("answer-retry"))]))
        self.open_page()
        self.submit("程序和进程有什么区别？")
        self.expect(self.page.locator(".notice")).to_contain_text("需要正确的访问口令")
        self.expect(self.page.locator("#code-field")).to_be_visible()
        self.expect_ready("#code")
        self.page.locator("#code").fill("browser-test-code")
        self.page.locator("#book").select_option("database")
        self.page.locator("#input").fill("这是编辑后的其他问题")
        self.page.get_by_role("button", name="重试本题").click()
        self.expect(self.page.locator(".answer").last.locator("strong")).to_have_text("进程")
        self.assertEqual(self.backend.asks[0]["payload"], self.backend.asks[1]["payload"])
        self.assertEqual(self.backend.asks[1]["access_code"], "browser-test-code")
        self.expect(self.page.locator("#book")).to_have_value("os")
        self.expect_ready()

    def test_finishing_answer_preserves_focus_on_next_question_controls(self) -> None:
        gate = threading.Event()
        self.backend.enqueue(
            Reply(initial=[("chunk", {"text": "正在回答"})],
                  final=[("result", answer_result())], gate=gate)
        )
        self.open_page()
        self.submit()
        self.expect(self.page.locator(".answer .body")).to_have_text("正在回答")
        self.page.locator("#book").select_option("database")
        self.page.locator("#book").focus()
        gate.set()
        self.expect(self.page.locator(".answer strong")).to_have_text("进程")
        self.expect_ready("#book")
        self.assertEqual(self.backend.asks[0]["payload"]["book_id"], "os")

    def test_old_retry_cannot_replace_draft_while_another_answer_is_running(self) -> None:
        self.backend.enqueue(Reply(http_status=500, detail="暂时失败"))
        gate = threading.Event()
        self.backend.enqueue(
            Reply(initial=[("chunk", {"text": "第二题正在回答"})],
                  final=[("result", answer_result())], gate=gate)
        )
        self.open_page()
        self.submit("第一次失败的问题")
        retry = self.page.get_by_role("button", name="重试本题")
        self.expect(retry).to_be_enabled()
        self.submit("第二个问题")
        self.expect(self.page.locator(".answer").last.locator(".body")).to_have_text(
            "第二题正在回答"
        )
        self.page.locator("#book").select_option("database")
        self.page.locator("#input").fill("正在准备第三个问题")
        self.expect(retry).to_be_disabled()
        gate.set()
        self.expect(self.page.locator(".answer strong")).to_have_text("进程")
        self.expect(retry).to_be_enabled()
        self.expect(self.page.locator("#input")).to_have_value("正在准备第三个问题")
        self.expect(self.page.locator("#book")).to_have_value("database")
        self.assertEqual(len(self.backend.asks), 2)

    def test_valid_crlf_and_cr_streams_render_complete_answers(self) -> None:
        self.open_page()
        for ending in ("\r\n", "\r"):
            with self.subTest(line_ending=repr(ending)):
                self.backend.enqueue(
                    Reply(final=[("result", answer_result())], line_ending=ending)
                )
                self.submit()
                card = self.page.locator(".answer").last
                self.expect(card.locator("strong")).to_have_text("进程")
                self.expect(card.locator(".notice.error")).to_have_count(0)
                self.expect_ready()

    def test_crlf_stream_split_into_single_bytes(self) -> None:
        # A Response body preserves these byte boundaries, unlike TCP, which may
        # coalesce writes. Exercise both CR/LF and Chinese UTF-8 splits in-browser.
        frame = "event: result\r\ndata: " + json.dumps(answer_result(), ensure_ascii=False) + "\r\n\r\n"
        self.page.add_init_script(
            """(frame => {
              const original = window.fetch.bind(window);
              window.fetch = (url, options) => {
                if (url !== '/v1/ask/stream') return original(url, options);
                const bytes = new TextEncoder().encode(frame);
                let offset = 0;
                return Promise.resolve(new Response(new ReadableStream({
                  pull(controller) {
                    if (offset === bytes.length) controller.close();
                    else controller.enqueue(bytes.slice(offset, ++offset));
                  }
                }), { headers: { 'Content-Type': 'text/event-stream' } }));
              };
            })(""" + json.dumps(frame) + ");"
        )
        self.open_page()
        self.submit()
        self.expect(self.page.locator(".answer strong")).to_have_text("进程")
        self.expect_ready()

    def test_terminal_result_finishes_before_connection_closes(self) -> None:
        gate = threading.Event()
        self.backend.enqueue(
            Reply(
                initial=[("chunk", {"text": "临时片段"}), ("result", answer_result())],
                final=[("chunk", {"text": "结果后的无效片段"})],
                gate=gate,
            )
        )
        self.open_page()
        self.submit()
        card = self.page.locator(".answer")
        self.expect(card.locator("strong")).to_have_text("进程")
        self.expect_ready()
        self.assertFalse(gate.is_set(), "The terminal result must not wait for socket EOF")
        gate.set()
        self.expect(card.locator(".body")).not_to_contain_text("临时片段")
        self.expect(card.locator(".body")).not_to_contain_text("无效片段")
        self.expect(card.locator(".sources")).to_have_count(1)
        self.expect(card.get_by_role("button", name="重试本题")).to_have_count(0)

    def test_terminal_error_finishes_before_connection_closes(self) -> None:
        gate = threading.Event()
        self.backend.enqueue(
            Reply(
                initial=[("error", {"status": "busy", "message": "服务繁忙，请稍后重试。"})],
                gate=gate,
            )
        )
        self.open_page()
        self.submit()
        self.expect(self.page.locator(".notice.warn")).to_have_text("服务繁忙，请稍后重试。")
        self.expect_ready()
        self.assertFalse(gate.is_set(), "A terminal error must release the form immediately")
        self.expect(self.page.get_by_role("button", name="重试本题")).to_have_count(1)

    def test_stream_error_and_incomplete_connection_offer_retry(self) -> None:
        self.backend.enqueue(
            Reply(
                initial=[("chunk", {"text": "保留部分回答"})],
                final=[("error", {"status": "busy", "message": "服务繁忙，请稍后重试。"})],
            )
        )
        self.backend.enqueue(Reply(initial=[("chunk", {"text": "连接关闭前的片段"})]))
        self.open_page()
        self.submit()
        self.expect(self.page.locator(".answer .body")).to_have_text("保留部分回答")
        self.expect(self.page.locator(".notice.warn")).to_have_text("服务繁忙，请稍后重试。")
        self.expect_ready()
        self.page.get_by_role("button", name="重试本题").click()
        last = self.page.locator(".answer").last
        self.expect(last.locator(".body")).to_have_text("连接关闭前的片段")
        self.expect(last.locator(".notice.error")).to_have_text("连接中断了，回答没有完整返回。")
        self.expect(last.get_by_role("button", name="重试本题")).to_be_visible()
        self.expect_ready()

    def test_stop_aborts_pending_stream_and_allows_next_question(self) -> None:
        gate = threading.Event()
        self.backend.enqueue(
            Reply(
                initial=[("chunk", {"text": "已生成的部分内容"})],
                final=[("result", answer_result("cancelled-answer"))],
                gate=gate,
            )
        )
        self.backend.enqueue(Reply(final=[("result", answer_result("next-answer"))]))
        self.open_page()
        self.submit()
        self.expect(self.page.locator(".answer .body")).to_have_text("已生成的部分内容")
        self.page.locator("#cancel").click()
        self.expect(self.page.locator(".notice.warn")).to_have_text("已停止生成。")
        self.expect(self.page.get_by_role("button", name="保存问答（含来源）")).to_have_count(0)
        self.expect_ready()
        gate.set()
        self.submit("停止后再问一个问题")
        self.expect(self.page.locator(".answer").last.locator("strong")).to_have_text("进程")
        self.expect(self.page.locator(".answer").first.locator(".body")).to_have_text(
            "已生成的部分内容"
        )
        self.expect(self.page.locator(".answer").first.locator(".sources")).to_have_count(0)
        self.expect_ready()

    def test_feedback_requires_reason_and_comment_then_sends_selected_answer(self) -> None:
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page()
        self.submit()
        self.page.get_by_role("button", name="👎 需要改进").click()
        panel = self.page.locator(".feedback-panel")
        self.expect(panel).to_be_visible()
        panel.get_by_role("button", name="提交反馈").click()
        self.assertEqual(self.backend.feedback, [])
        panel.locator("select").select_option("other")
        panel.get_by_role("button", name="提交反馈").click()
        self.assertEqual(self.backend.feedback, [])
        self.expect(panel.locator("textarea")).to_have_js_property(
            "validationMessage", "选择其他时请填写补充说明"
        )
        panel.locator("textarea").fill("  定义还需要举一个例子。  ")
        panel.get_by_role("button", name="提交反馈").click()
        self.expect(self.page.locator(".feedback-status")).to_have_text("感谢反馈，已经记录。")
        self.assertEqual(
            self.backend.feedback,
            [
                {
                    "answer_id": "answer-1",
                    "rating": "needs_improvement",
                    "reason": "other",
                    "comment": "定义还需要举一个例子。",
                }
            ],
        )
        self.expect(panel).to_have_count(0)
        self.expect(self.page.locator("[data-feedback]")).to_have_count(0)

    def test_feedback_retry_replaces_its_error_and_preserves_answer_warnings(self) -> None:
        result = answer_result()
        result["citation_integrity"] = {"status": "missing"}
        self.backend.enqueue(Reply(final=[("result", result)]))
        requests = []

        def feedback(route: Any) -> None:
            requests.append(route.request.post_data_json)
            if len(requests) == 1:
                route.fulfill(status=422, content_type="application/json",
                              body=json.dumps({"detail": [{"type": "invalid"}]}))
            elif len(requests) == 2:
                route.fulfill(status=503, content_type="application/json",
                              body=json.dumps({"detail": "反馈暂时无法保存，请稍后再试"}))
            elif len(requests) == 3:
                route.fulfill(status=200, content_type="application/json", body='{"status":"failed"}')
            elif len(requests) == 4:
                route.fulfill(status=200, content_type="text/html", body="<h1>Unavailable</h1>")
            else:
                route.fulfill(status=200, content_type="application/json", body='{"status":"saved"}')

        self.page.route("**/v1/feedback", feedback)
        self.open_page()
        self.submit()
        self.page.get_by_role("button", name="👎 需要改进").click()
        panel = self.page.locator(".feedback-panel")
        panel.locator("select").select_option("other")
        panel.locator("textarea").fill("需要补充例子")
        panel.get_by_role("button", name="提交反馈").click()
        self.expect(self.page.locator(".feedback-notice")).to_contain_text("反馈格式不正确")
        self.expect(panel.locator("textarea")).to_have_value("需要补充例子")
        self.expect(panel.get_by_role("button", name="提交反馈")).to_be_enabled()
        panel.get_by_role("button", name="提交反馈").click()
        self.expect(self.page.locator(".feedback-notice")).to_contain_text("反馈暂时无法保存")
        self.expect(self.page.locator(".feedback-notice")).to_have_count(1)
        self.expect(self.page.locator(".answer")).not_to_contain_text("反馈格式不正确")
        for _ in range(2):
            panel.get_by_role("button", name="提交反馈").click()
            self.expect(self.page.locator(".feedback-notice")).to_contain_text("反馈暂时没有保存成功")
            self.expect(panel.locator("textarea")).to_have_value("需要补充例子")
            self.expect(self.page.locator(".feedback-status")).to_have_count(0)
        panel.get_by_role("button", name="提交反馈").click()
        self.expect(self.page.locator(".feedback-status")).to_have_text("感谢反馈，已经记录。")
        self.expect(self.page.locator(".feedback-notice")).to_have_count(0)
        self.expect(self.page.locator(".answer")).not_to_contain_text("反馈暂时无法保存")
        self.expect(self.page.locator(".answer .notice.warn")).to_have_count(1)
        self.assertEqual([row["comment"] for row in requests], ["需要补充例子"] * 5)
        self.assertEqual(len(self.backend.asks), 1)

    def test_helpful_feedback_and_keyboard_submission(self) -> None:
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.open_page(book_id="")
        self.page.locator("#input").fill("什么是进程？")
        self.page.locator("#input").press("Enter")
        self.assertEqual(self.backend.asks, [])
        self.expect(self.page.locator("#book")).to_have_js_property(
            "validationMessage", "请先选择一本教材"
        )
        self.page.locator("#book").select_option("os")
        self.page.locator("#input").press("Shift+Enter")
        self.page.locator("#input").dispatch_event("keydown", {"key": "Enter", "isComposing": True})
        self.assertEqual(self.backend.asks, [])
        self.page.locator("#input").press("Enter")
        self.page.get_by_role("button", name="👍 有帮助").click()
        self.expect(self.page.locator(".feedback-status")).to_have_text("感谢反馈，已经记录。")
        self.assertEqual(
            self.backend.feedback,
            [{"answer_id": "answer-1", "rating": "helpful", "reason": None, "comment": ""}],
        )


if __name__ == "__main__":
    unittest.main()
