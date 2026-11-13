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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/rag_textbook_qa/api/static/index.html"
RETRIEVAL_PAGE = ROOT / "src/rag_textbook_qa/api/static/retrieval.html"
IMPORT_PAGE = ROOT / "src/rag_textbook_qa/api/static/imports.html"
BACKUP_SCRIPT = ROOT / "src/rag_textbook_qa/api/static/history-backup.js"
MARKDOWN_SCRIPT = ROOT / "src/rag_textbook_qa/api/static/markdown.js"


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


def all_books_result() -> dict[str, Any]:
    """An answer whose sources come from two textbooks, as an all-book search returns."""
    result = answer_result("answer-all")
    result["answer"] = "进程是程序的一次执行。【参考资料 1】事务是不可分割的工作单位。【参考资料 2】"
    excerpt = "事务是一个不可分割的工作单位。"
    result["sources"].append({
        "citation_id": 2, "book_id": "database", "chunk_id": "chapter_p9",
        "source_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "book": "数据库原理及应用", "section": "第十章 事务", "excerpt": excerpt, "truncated": False,
    })
    return result


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
        "created_at": "2026-10-17T00:00:00+00:00",
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
        self.reading_unordered = False
        self.reading_failure = False
        self.reading_gate: threading.Event | None = None
        self.reads: list[dict[str, Any]] = []
        self.read_books: list[str] = []
        self.inspections: list[dict[str, Any]] = []
        self.inspection_gate: threading.Event | None = None
        self.inspection_status = 200
        self.inspection_payload: dict[str, Any] | None = None
        self.imports_enabled = False
        self.import_state = "prepared"
        self.uploads: list[bytes] = []
        self.import_starts: list[dict[str, Any]] = []
        self.import_deleted = False
        self.import_read_gate: threading.Event | None = None

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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json_response(self, value: Any, status: int = 200) -> None:
        self.respond(status, json.dumps(value).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path == "/":
            self.respond(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/retrieval":
            self.respond(200, RETRIEVAL_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/imports":
            self.respond(200, IMPORT_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/assets/history-backup.js":
            self.respond(200, BACKUP_SCRIPT.read_bytes(), "application/javascript; charset=utf-8")
        elif self.path == "/assets/markdown.js":
            self.respond(200, MARKDOWN_SCRIPT.read_bytes(), "application/javascript; charset=utf-8")
        elif self.path == "/v1/imports":
            if not self.server.imports_enabled:
                self.json_response({"detail": "教材导入只在本机服务提供"}, 404)
            else:
                self.json_response({"books": [{"book_id": "os", "label": "操作系统"}],
                                    "formats": [".md", ".json"], "max_upload_bytes": 8388608,
                                    "storage_path": "/tmp/browser-imports"})
        elif self.path == "/v1/imports/jobs":
            self.json_response({"jobs": [import_result(self.server.import_state)] if self.server.uploads
                                and not self.server.import_deleted else [], "unreadable_count": 0})
        elif self.path == "/v1/imports/" + "a" * 32:
            if self.server.import_read_gate:
                self.server.import_read_gate.wait(10)
            self.json_response({"detail": "没有这个导入任务"}, 404) if self.server.import_deleted else self.json_response(import_result(self.server.import_state))
        elif (read := re.fullmatch(r"/v1/books/(os|database)/passages", urlsplit(self.path).path)):
            book_id = read.group(1)
            chunk_id = parse_qs(urlsplit(self.path).query)["chunk_id"][0]
            self.server.reads.append({"chunk_id": chunk_id, "access_code": self.headers.get("X-Access-Code")})
            self.server.read_books.append(book_id)
            if self.server.reading_gate:
                self.server.reading_gate.wait(10)
            if self.server.reading_failure:
                self.json_response({"detail": "原文暂时无法读取，请稍后重试"}, 503)
                return
            source = dict(all_books_result()["sources"][book_id == "database"])
            if self.server.reading_changed:
                source.update(excerpt="更新后的教材原文。", source_sha256="b" * 64)
            source.update(chunk_id=chunk_id, is_anchor=True)
            neighbor = {**source, "chunk_id": "chapter_p2", "excerpt": "相邻原文：进程具有动态性。", "is_anchor": False}
            self.json_response({
                "book_id": book_id, "anchor_chunk_id": chunk_id,
                "source_sha256": source["source_sha256"],
                "previous_chunk_id": "chapter_p1" if chunk_id == "chapter_p2" else None,
                "next_chunk_id": "chapter_p2" if chunk_id == "chapter_p1" else None,
                "passages": [source, neighbor],
                **({"navigation_available": False, "previous_chunk_id": None, "next_chunk_id": None,
                    "passages": [source]} if self.server.reading_unordered else {}),
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
            self.server.import_deleted = False
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
                self.json_response((self.server.inspection_payload if self.server.inspection_payload is not None
                                    else retrieval_result()) if self.server.inspection_status == 200
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

    def do_DELETE(self) -> None:
        if self.path == "/v1/imports/" + "a" * 32 and self.server.import_state != "building":
            self.server.import_deleted = True
            self.json_response({"status": "deleted"})
        else:
            self.json_response({"detail": "任务仍在建立索引"}, 409)


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

        self.local_only = local_only
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
        # The placeholder, the all-textbooks scope and the two indexed books.
        self.expect(self.page.locator("#book option")).to_have_count(4)
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

    def fail_first_library_read_after_write(self) -> None:
        self.page.evaluate("""() => {
          let written = false, failed = false;
          for (const name of ['put', 'add', 'delete', 'clear']) {
            const method = IDBObjectStore.prototype[name];
            IDBObjectStore.prototype[name] = function(...args) {
              const result = method.apply(this, args); written = true; return result;
            };
          }
          const getAll = IDBObjectStore.prototype.getAll;
          IDBObjectStore.prototype.getAll = function(...args) {
            if (written && !failed) {
              failed = true;
              throw new DOMException('fixture list read failure after commit', 'UnknownError');
            }
            return getAll.apply(this, args);
          };
        }""")

    def conflicting_notes(self, draft: str = "本页要保留的草稿", latest: str = "另一页最新保存的笔记", *, live: bool = True):
        if not live:
            self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("整理有冲突的笔记")
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill(draft)
        other = self.context.new_page()
        other.on("pageerror", lambda error: self.errors.append(str(error)))
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="学习笔记")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="学习笔记").click()
        other.get_by_role("textbox", name="编辑学习笔记").fill(latest)
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(other.locator(".note-editor")).to_contain_text("笔记已保存")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿未覆盖最新版本")
        return other

    def block_question_draft_cache(self) -> None:
        self.context.add_init_script("""(() => {
          window.questionCacheBroken = true;
          const set = Storage.prototype.setItem;
          Storage.prototype.setItem = function(key, value) {
            if (window.questionCacheBroken && String(key) === 'rag-qa-question-draft') {
              throw new DOMException('fixture question cache full', 'QuotaExceededError');
            }
            return set.call(this, key, value);
          };
        })();""")

    def block_note_draft_cache(self, *, blocked: bool = True) -> None:
        self.context.add_init_script("""(() => {
          window.noteCacheBroken = BLOCKED;
          const set = Storage.prototype.setItem;
          Storage.prototype.setItem = function(key, value) {
            if (window.noteCacheBroken && String(key).startsWith('rag-qa-note-draft-')) {
              throw new DOMException('fixture note cache unavailable', 'QuotaExceededError');
            }
            return set.call(this, key, value);
          };
        })();""".replace("BLOCKED", json.dumps(blocked)))

    def uncached_note(self, *, review: bool = False) -> None:
        self.block_note_draft_cache()
        self.open_page()
        self.saved_answer("暂存失败时保护笔记")
        if review:
            self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
            self.page.locator("#toggle-history").click()
            self.page.locator("#history-review-start").click()
            self.page.locator("#review-reveal").click()
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("没有暂存成功的唯一草稿")
        self.expect(self.page.locator(".note-draft-storage-warning")).to_be_visible()

    def capture_page(self, name: str) -> None:
        screenshot_dir = os.environ.get("RAG_QA_BROWSER_SCREENSHOTS")
        if screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(Path(screenshot_dir) / name), full_page=True)

    def deleted_note_fixture(self, *, review: bool = False, live: bool = True, another: bool = False) -> str:
        if not live:
            self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("待保护的草稿问答")
        identifier = self.history_records()[0]["id"]
        if another:
            self.saved_answer("可以继续查看的问答")
        if review:
            self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
            self.page.locator("#toggle-history").click()
            self.page.locator("#history-review-start").click()
            self.page.locator("#review-reveal").click()
        self.page.locator(".answer").first.get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("仍需导出的笔记草稿")
        other = self.context.new_page()
        other.on("pageerror", lambda error: self.errors.append(str(error)))
        other.goto(self.base_url)
        self.expect(other.locator(".answer")).to_have_count(2 if another else 1)
        other.locator("#toggle-history").click()
        other.once("dialog", lambda dialog: dialog.accept())
        other.locator(".history-item").filter(has_text="待保护的草稿问答").get_by_role("button", name="删除记录").click()
        self.expect(other.locator(".history-item").filter(has_text="待保护的草稿问答")).to_have_count(0)
        if live:
            self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        return identifier

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
        self.expect(self.page.locator("#history-export-md")).to_be_disabled()

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
        with self.page.expect_download() as info:
            self.page.get_by_role("button", name="导出本次记录", exact=True).click()
        exported = Path(info.value.path()).read_text()
        self.assertIn("本次没有完整的生成回答", exported)
        self.assertIn(result["sources"][0]["excerpt"], exported)
        self.assertIn(result["message"], exported)
        self.assertEqual(len(self.backend.asks), 2)

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

    def test_all_book_observatory_tracks_and_exports_same_chunk_id_in_different_books(self) -> None:
        data = retrieval_result()
        data["scope"] = "all"
        first = data["stages"][0]["items"][0]
        other = {**first, "book_id": "database", "book": "数据库原理及应用",
                 "position": 2, "excerpt": "数据库中的进程。", "citation_id": None}
        other["source_sha256"] = hashlib.sha256(other["excerpt"].encode()).hexdigest()
        for stage in data["stages"]:
            stage["items"] = [dict(first)] if stage["id"] == "context" else [dict(first), dict(other)]
            stage["count"] = len(stage["items"])
            if stage["id"] in {"keyword", "embedding"}:
                for item in stage["items"]:
                    item["book_position"] = 1
        self.backend.inspection_payload = data
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.page.emulate_media(color_scheme="dark")
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.expect(self.page.locator("#inspect-book")).to_have_value("")
        self.expect(self.page.locator('#inspect-book option[value="*"]')).to_have_count(1)
        self.expect(self.page.locator("#inspect-scope-hint")).to_be_hidden()
        self.page.locator("#inspect-book").select_option("*")
        self.expect(self.page.locator("#inspect-scope-hint")).to_have_text(
            "将在全部教材中检索，每条资料标明所属教材；跨教材回答质量尚未单独验收。")
        self.page.locator("#inspect-query").fill("跨教材观察")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-status")).to_contain_text("检索观察完成")
        self.expect(self.page.locator("#inspect-summary")).to_contain_text("全部教材")
        self.page.get_by_role("button", name="关键词 2", exact=True).click()
        self.expect(self.page.locator("#inspect-explanation")).to_contain_text("按本内排名交错展示")
        self.expect(self.page.locator(".candidate")).to_have_count(2)
        self.expect(self.page.locator(".candidate").first).to_contain_text("入选上下文")
        self.expect(self.page.locator(".candidate").last).to_contain_text("未入选上下文")
        self.expect(self.page.locator(".candidate").last).to_contain_text("本教材第 1 名")
        self.page.get_by_text("同一片段的排名变化", exact=True).click()
        rows = self.page.locator("#inspect-ranks table tr")
        self.expect(rows).to_have_count(3)
        self.expect(rows.nth(1)).to_contain_text("操作系统（os）")
        self.expect(rows.nth(2)).to_contain_text("数据库原理及应用（database）")
        self.expect(rows.nth(1).locator("td").last).to_have_text("1")
        self.expect(rows.nth(2).locator("td").last).to_have_text("—")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("retrieval-all-books-dark-320.png")
        self.page.locator("#inspect-book").select_option("os")
        with self.page.expect_download() as download:
            self.page.locator("#inspect-export").click()
        exported = Path(download.value.path()).read_text()
        for text in ("教材：全部教材", "教材编号：*", "教材编号：os", "教材编号：database",
                     "操作系统（os） / chapter_p1", "数据库原理及应用（database） / chapter_p1",
                     "本教材排名：1", "生成模型调用：0"):
            self.assertIn(text, exported)
        self.assertEqual(self.backend.inspections[0]["payload"], {"query": "跨教材观察", "top_k": 5})
        self.assertEqual(len(self.backend.inspections), 1)
        self.assertEqual(self.backend.asks, [])
        self.assertEqual(self.backend.feedback, [])

    def test_single_indexed_book_does_not_offer_or_restore_all_book_inspection(self) -> None:
        self.context.add_init_script("sessionStorage.setItem('rag-qa-selected-book', '*')")
        self.page.route("**/v1/books", lambda route: route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps([{"book_id": "os", "label": "操作系统"}]),
        ))
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.expect(self.page.locator('#inspect-book option[value="*"]')).to_have_count(0)
        self.expect(self.page.locator("#inspect-book")).to_have_value("")
        self.expect(self.page.locator("#inspect-scope-hint")).to_be_hidden()
        self.assertEqual(self.backend.inspections, [])

    def test_retrieval_export_keeps_completed_request_and_public_stage_snapshot(self) -> None:
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.page.emulate_media(color_scheme="dark")
        self.backend.access_code_required = True
        data = retrieval_result()
        data.update(system_prompt="private-hidden-prompt", api_base="private-hidden-provider")
        first = data["stages"][0]["items"][0]
        first.update(chunk_id="id|with`punctuation", excerpt="原文包含 ``` 和 <img src=never>，请保留原样。",
                     truncated=True, preview_truncated=True, table_compacted=True, api_key="private-hidden-field")
        first["scores"]["api_key"] = "private-hidden-score"
        data["stages"][0]["count"] = 42
        self.backend.inspection_payload = data
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.expect(self.page.locator("#inspect-code")).to_be_visible()
        self.page.locator("#inspect-code").fill("private-inspection-code")
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-top-k").select_option("3")
        self.page.locator("#inspect-query").fill("本次进程观察问题")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-result")).to_be_visible()
        self.page.locator("#inspect-book").select_option("database")
        self.page.locator("#inspect-top-k").select_option("10")
        self.page.locator("#inspect-query").fill("下一次尚未观察的问题")
        self.page.get_by_role("button", name="关键词 42", exact=True).click()
        with self.page.expect_download() as download:
            self.page.locator("#inspect-export").click()
        exported = Path(download.value.path()).read_text()
        for text in ("# 检索观察记录", "本次进程观察问题", "教材：操作系统", "最终候选数（请求）：3",
                     "## 同一片段的排名变化", "## 关键词", "## 向量", "## 融合", "## 重排", "## 最终上下文",
                     "本阶段共 42 个候选，本文件保留展示的 2 个", "长片段预览已截断", "交付片段已截断",
                     "表格按行整理", "id|with`punctuation", "原文包含 ``` 和 <img src=never>", "生成模型调用：0"):
            self.assertIn(text, exported)
        for text in ("下一次尚未观察的问题", "private-hidden", "private-inspection-code", "教材：数据库"):
            self.assertNotIn(text, exported)
        self.assertTrue(download.value.suggested_filename.endswith(".md"))
        self.expect(self.page.locator("#inspect-export-status")).to_contain_text("没有重新运行检索")
        self.expect(self.page.locator("#inspect-query")).to_have_value("下一次尚未观察的问题")
        self.assertEqual(len(self.backend.inspections), 1)
        self.assertEqual(self.backend.asks, [])
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("retrieval-export-dark-320.png")

    def test_retrieval_export_is_unavailable_for_cancelled_or_failed_new_request(self) -> None:
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("第一次已完成")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-export")).to_be_enabled()
        gate = threading.Event()
        self.backend.gates.append(gate)
        self.backend.inspection_gate = gate
        self.page.locator("#inspect-query").fill("第二次仍在等待")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-stop")).to_be_visible()
        self.expect(self.page.locator("#inspect-export")).to_be_disabled()
        downloads = []
        self.page.on("download", lambda download: downloads.append(download))
        self.page.locator("#inspect-export").dispatch_event("click")
        self.page.locator("#inspect-stop").click()
        self.expect(self.page.locator("#inspect-status")).to_contain_text("已停止等待")
        self.expect(self.page.locator("#inspect-export")).to_be_disabled()
        gate.set()
        self.backend.inspection_status = 503
        self.backend.inspection_gate = None
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.expect(self.page.locator("#inspect-result")).to_be_hidden()
        self.expect(self.page.locator("#inspect-export")).to_be_disabled()
        self.page.locator("#inspect-export").dispatch_event("click")
        self.assertEqual(downloads, [])
        self.assertEqual(self.backend.asks, [])

    def test_malformed_retrieval_stages_are_rejected_before_display_or_export(self) -> None:
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("检查完整观察结构")
        malformed = []
        data = retrieval_result()
        data["stages"][4]["items"] = None
        malformed.append(data)
        data = retrieval_result()
        data["stages"][4]["id"] = "keyword"
        malformed.append(data)
        data = retrieval_result()
        data["stages"][0]["items"][0]["scores"] = None
        malformed.append(data)
        for data in malformed:
            with self.subTest(stage=data["stages"][4]["id"]):
                self.backend.inspection_payload = data
                self.page.locator("#inspect-send").click()
                self.expect(self.page.locator("#inspect-status")).to_contain_text("检索结果格式不正确")
                self.expect(self.page.locator("#inspect-send")).to_be_enabled()
                self.expect(self.page.locator("#inspect-result")).to_be_hidden()
                self.expect(self.page.locator("#inspect-export")).to_be_disabled()
        self.assertEqual(len(self.backend.inspections), 3)
        self.assertEqual(self.backend.asks, [])

    def test_retrieval_export_handles_empty_stages_without_fabricating_candidates(self) -> None:
        data = retrieval_result()
        data["reranker_enabled"] = False
        for stage in data["stages"]:
            stage.update(count=0, items=[])
        self.backend.inspection_payload = data
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("无候选的观察")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-result")).to_be_visible()
        with self.page.expect_download() as download:
            self.page.locator("#inspect-export").click()
        exported = Path(download.value.path()).read_text()
        self.assertEqual(exported.count("这个阶段没有候选片段。"), 5)
        self.assertIn("本次没有展示候选", exported)
        self.assertIn("未配置，沿用融合排名", exported)
        self.assertNotIn("### 候选", exported)
        self.assertEqual(len(self.backend.inspections), 1)

    def test_retrieval_export_failure_can_retry_without_repeating_inspection(self) -> None:
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("文件创建故障后重试导出")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-export")).to_be_enabled()
        self.page.evaluate("""() => {
          const create = URL.createObjectURL;
          URL.createObjectURL = function() {
            URL.createObjectURL = create;
            throw new Error('fixture blob failure');
          };
        }""")
        self.page.locator("#inspect-export").click()
        self.expect(self.page.locator("#inspect-export-status")).to_contain_text("本次结果仍保留")
        self.expect(self.page.locator("#inspect-result")).to_be_visible()
        with self.page.expect_download() as download:
            self.page.locator("#inspect-export").click()
        self.assertIn("文件创建故障后重试导出", Path(download.value.path()).read_text())
        self.assertEqual(len(self.backend.inspections), 1)
        self.assertEqual(self.backend.asks, [])

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

    def backup_file(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"name": "教材问答备份.json", "mimeType": "application/json",
                "buffer": json.dumps({"format": "rag-qa-history", "version": 1, "records": records}).encode()}

    def test_history_backup_round_trip_keeps_sources_favorites_and_existing_versions(self) -> None:
        self.backend.access_code_required = True
        self.open_page()
        self.page.locator("#code").fill("never-export-this-code")
        self.saved_answer("第一条记录")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.saved_answer("另一条记录", "database")
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        with self.page.expect_download() as info:
            self.page.locator("#history-backup-all").click()
        data = Path(info.value.path()).read_text()
        backup = json.loads(data)
        self.assertEqual(backup["format"], "rag-qa-history")
        self.assertEqual(len(backup["records"]), 2)
        self.assertNotIn("never-export-this-code", data)
        self.assertNotIn("answer_id", data)
        self.page.locator(".history-cleanup summary").click()
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator("#history-delete-all").click()
        self.expect(self.page.locator(".history-item")).to_have_count(0)
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(backup["records"]))
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("预计新增 2 条")
        self.assertEqual(self.history_records(), [])
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("恢复完成：新增 2 条")
        self.assertEqual(len(self.history_records()), 2)
        self.assertEqual(sum(row["favorite"] for row in self.history_records()), 1)
        self.assertEqual(self.history_records()[0]["result"]["sources"], backup["records"][0]["result"]["sources"])
        favorite = self.page.locator(".history-item").filter(has=self.page.get_by_role("heading", name="第一条记录"))
        favorite.get_by_role("button", name="★ 已收藏").click()
        self.expect(favorite.get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(backup["records"]))
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("相同编号 2 条")
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("新增 0 条")
        self.assertEqual(sum(row["favorite"] for row in self.history_records()), 0)
        self.assertEqual(len(self.backend.asks), 2)

    def test_filtered_backup_and_invalid_backup_never_modify_current_library(self) -> None:
        self.open_page()
        self.saved_answer("操作系统的问题")
        self.saved_answer("数据库的问题", "database")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-book").select_option("database")
        self.page.locator(".history-backup summary").click()
        with self.page.expect_download() as info:
            self.page.locator("#history-backup-filtered").click()
        backup = json.loads(Path(info.value.path()).read_text())
        self.assertEqual([row["query"] for row in backup["records"]], ["数据库的问题"])
        original = self.history_records()
        incoming = [dict(original[0], id="valid-incoming"), dict(original[0], id="bad-incoming", favorite="yes")]
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(incoming))
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("无效记录")
        self.expect(self.page.locator("#history-restore")).to_be_disabled()
        self.assertEqual(self.history_records(), original)
        incoming = [dict(original[0], id="repeated-id")] * 2
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(incoming))
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("无效记录")
        self.assertEqual(self.history_records(), original)

    def test_filtered_markdown_export_keeps_saved_notes_and_citations_only(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.backend.access_code_required = True
        self.open_page()
        self.page.locator("#code").fill("bulk-export-private-code")
        self.saved_answer("需要整理的操作系统题目")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("整理关键词\n````\n这是已保存的笔记")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("不能混入导出的未保存草稿")
        self.saved_answer("不匹配的数据库题目", "database")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-book").select_option("os")
        self.page.locator("#history-favorites").check()
        self.page.locator("#history-notes").check()
        self.page.locator("#history-search").fill("整理关键词 进程")
        original = self.history_records()
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").click()
        exported = Path(download.value.path()).read_text()
        self.assertTrue(download.value.suggested_filename.endswith(".md"))
        for text in ("# 学习问答整理", "共 1 条记录", "需要整理的操作系统题目", "这是已保存的笔记",
                     "收藏：是", "操作系统", "进程是程序在一个数据集合上的一次执行。", "### 我的学习笔记"):
            self.assertIn(text, exported)
        for text in ("不能混入导出的未保存草稿", "不匹配的数据库题目", "answer_id", "bulk-export-private-code"):
            self.assertNotIn(text, exported)
        self.assertEqual(self.history_records(), original)
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("不能混入导出的未保存草稿")
        self.expect(self.page.locator("#history-export-status")).to_contain_text("1 条记录交给浏览器下载")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("history-export-mobile.png")
        self.assertEqual(len(self.backend.asks), 2)

    def test_history_search_shows_a_deep_citation_match_as_safe_text(self) -> None:
        self.open_page()
        result = answer_result()
        result["answer"] = "这里是回答开头，不含所查术语。" * 15
        result["sources"][0]["excerpt"] = "前面的材料。" * 80 + "唯一术语回收机制 <img src=never>" + "后续材料。" * 80
        self.saved_answer("查阅教材中提到的术语", result=result)
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-search").fill("唯一术语回收机制")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-item")).to_contain_text("唯一术语回收机制")
        self.expect(self.page.locator(".history-match")).to_contain_text("引用原文命中")
        self.expect(self.page.locator(".history-match img")).to_have_count(0)
        self.assertEqual(self.history_records()[0]["result"]["sources"][0]["excerpt"], result["sources"][0]["excerpt"])
        self.assertEqual(len(self.backend.asks), 1)

    def test_history_search_shows_saved_note_context_and_handles_case_expansion(self) -> None:
        self.page.emulate_media(color_scheme="dark")
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.open_page()
        self.saved_answer("查找以前写的笔记")
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("İ" * 200 + "🧠UNIQUE_needle🧠" + "后面的笔记。" * 80)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.page.locator("#clear").click()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-search").fill("unique_needle")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-match")).to_contain_text("笔记命中")
        self.expect(self.page.locator(".history-match")).to_contain_text("🧠UNIQUE_needle🧠")
        self.assertNotIn("\ufffd", self.page.locator(".history-match").inner_text())
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.page.locator(".history-match").evaluate("element => element.scrollIntoView({block:'start'})")
        self.capture_page("history-search-context-dark-320.png")
        self.assertEqual(len(self.backend.asks), 1)

    def test_history_search_labels_answer_chapter_and_book_matches_without_changing_filters(self) -> None:
        self.open_page()
        result = answer_result()
        result["answer"] = "前文。" * 80 + "答案专有关键词" + "后文。" * 80
        result["sources"][0]["book"] = "资料书名专有关键词"
        result["sources"][0]["section"] = "引用章节专有关键词"
        self.saved_answer("题目标题专有关键词", result=result)
        original = self.history_records()
        self.page.locator("#toggle-history").click()
        for query, label in (("答案专有关键词", "回答命中"), ("引用章节专有关键词", "引用章节命中"),
                             ("资料书名专有关键词", "引用教材命中"), ("操作系统", "教材命中")):
            with self.subTest(query=query):
                self.page.locator("#history-search").fill(query)
                self.expect(self.page.locator(".history-item")).to_have_count(1)
                self.expect(self.page.locator(".history-match")).to_contain_text(label)
                self.expect(self.page.locator(".history-match")).to_contain_text(query)
        self.page.locator("#history-search").fill("题目标题专有关键词")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-match")).to_have_count(0)
        self.page.locator("#history-search").fill("答案专有关键词 引用章节专有关键词")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.page.locator("#history-search").fill("答案专有关键词 不存在的词语")
        self.expect(self.page.locator(".history-item")).to_have_count(0)
        self.expect(self.page.locator("#history-export-md")).to_be_disabled()
        self.page.locator("#history-search").fill("")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-match")).to_have_count(0)
        self.assertEqual(self.history_records(), original)
        self.assertEqual(len(self.backend.asks), 1)

    def test_markdown_export_includes_all_matches_beyond_the_visible_page(self) -> None:
        self.open_page()
        self.saved_answer("原始种子问答")
        seed = self.history_records()[0]
        incoming = [dict(seed, id=f"batch-{index}", query=f"批量题目 {index:02}",
                         created_at=f"2026-10-{index + 1:02}T00:00:00Z") for index in range(21)]
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(incoming))
        self.expect(self.page.locator("#history-restore")).to_be_enabled()
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("新增 21 条")
        self.page.locator("#history-search").fill("批量题目")
        self.expect(self.page.locator(".history-item")).to_have_count(20)
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").click()
        exported = Path(download.value.path()).read_text()
        self.assertIn("共 21 条记录", exported)
        self.assertEqual(exported.count(" · 教材问答记录"), 21)
        self.assertLess(exported.index("批量题目 20"), exported.index("批量题目 00"))
        self.assertNotIn("原始种子问答", exported)
        self.page.locator("#history-export-practice").check()
        with self.page.expect_download() as practice:
            self.page.locator("#history-export-md").click()
        question_part = Path(practice.value.path()).read_text().split("\n## 已保存回答与来源\n")[0]
        self.assertEqual(question_part.count("### 第 "), 21)
        self.assertIn("批量题目 00", question_part)
        self.expect(self.page.locator(".history-item")).to_have_count(20)
        self.assertEqual(len(self.backend.asks), 1)

    def test_markdown_export_reads_latest_saved_note_without_replacing_current_draft(self) -> None:
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("跨页导出已保存版本")
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("本页未保存的内容")
        self.page.locator("#input").fill("下一题的问题草稿")
        self.page.locator("#toggle-history").click()
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="学习笔记")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="学习笔记").click()
        other.get_by_role("textbox", name="编辑学习笔记").fill("另一页刚保存的最新笔记")
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(other.locator(".note-editor")).to_contain_text("笔记已保存")
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").dispatch_event("click")
        exported = Path(download.value.path()).read_text()
        self.assertIn("另一页刚保存的最新笔记", exported)
        self.assertNotIn("本页未保存的内容", exported)
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("本页未保存的内容")
        self.expect(self.page.locator("#input")).to_have_value("下一题的问题草稿")
        self.assertEqual(len(self.backend.asks), 1)

    def test_markdown_export_read_failure_leaves_library_unchanged_and_allows_retry(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator("#toggle-history").click()
        original = self.history_records()
        self.page.evaluate("""() => {
          const getAll = IDBObjectStore.prototype.getAll;
          IDBObjectStore.prototype.getAll = function() {
            IDBObjectStore.prototype.getAll = getAll;
            throw new DOMException('fixture read failure', 'UnknownError');
          };
        }""")
        downloads = []
        self.page.on("download", lambda download: downloads.append(download))
        self.page.locator("#history-export-md").click()
        self.expect(self.page.locator("#history-export-status")).to_contain_text("导出未完成")
        self.assertEqual(downloads, [])
        self.assertEqual(self.history_records(), original)
        with self.page.expect_download():
            self.page.locator("#history-export-md").click()
        self.expect(self.page.locator("#history-export-status")).to_contain_text("1 条记录交给浏览器下载")
        self.assertEqual(self.history_records(), original)

    def test_markdown_export_keeps_clicked_filters_when_the_ui_changes_during_read(self) -> None:
        self.open_page()
        self.saved_answer("最初选择的操作系统题目")
        self.saved_answer("后来切换到的数据库题目", "database")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-book").select_option("os")
        self.page.locator("#history-export-practice").check()
        self.page.evaluate("""() => {
          const getAll = IDBObjectStore.prototype.getAll;
          IDBObjectStore.prototype.getAll = function(...args) {
            IDBObjectStore.prototype.getAll = getAll;
            const request = getAll.apply(this, args);
            request.addEventListener('success', () => {
              const select = document.querySelector('#history-book');
              select.value = 'database'; select.dispatchEvent(new Event('change'));
              const practice = document.querySelector('#history-export-practice');
              practice.checked = false; practice.dispatchEvent(new Event('change'));
            });
            return request;
          };
        }""")
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").click()
        exported = Path(download.value.path()).read_text()
        self.assertTrue(exported.startswith("# 学习问答练习"))
        self.assertIn("最初选择的操作系统题目", exported)
        self.assertNotIn("后来切换到的数据库题目", exported)
        self.expect(self.page.locator("#history-book")).to_have_value("database")
        self.expect(self.page.locator("#history-export-practice")).not_to_be_checked()
        self.assertEqual(len(self.history_records()), 2)

    def test_markdown_export_with_many_records_requires_filtering_then_can_continue(self) -> None:
        self.open_page()
        self.page.evaluate("""() => new Promise(resolve => {
          const request = indexedDB.open('rag-qa-history', 1);
          request.onsuccess = () => {
            const db = request.result, transaction = db.transaction('records', 'readwrite');
            const store = transaction.objectStore('records');
            for (let i = 0; i < 5001; i++) store.put({
              id:'export-limit-' + i, created_at:'2026-10-18T00:00:00.000Z',
              query:'批量编号 ' + i, book_id:'os', book_label:'操作系统', favorite:false, note:'',
              result:{status:'retrieval_only', answer:null, sources:[]}
            });
            transaction.oncomplete = () => { db.close(); resolve(); };
          };
        })""")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-export-practice").check()
        self.expect(self.page.locator("#history-count")).to_contain_text("5001 条记录")
        downloads = []
        self.page.on("download", lambda download: downloads.append(download))
        self.page.locator("#history-export-md").click()
        self.expect(self.page.locator("#history-export-status")).to_contain_text("单次最多导出 5000 条")
        self.assertEqual(downloads, [])
        self.assertEqual(len(self.history_records()), 5001)
        self.page.locator("#history-search").fill("批量编号 5000")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").click()
        exported = Path(download.value.path()).read_text()
        self.assertIn("共 1 条记录", exported)
        self.assertIn("本次没有完整的生成回答。", exported)
        self.assertIn("本次记录未包含教材片段。", exported)
        self.assertEqual(len(self.history_records()), 5001)
        self.assertEqual(self.backend.asks, [])

    def test_valid_dense_backtick_source_exports_without_argument_stack_overflow(self) -> None:
        self.open_page()
        excerpt = "`x" * 140000 + "\n``````\n原文末尾🧠"
        result = answer_result()
        result["sources"][0]["excerpt"] = excerpt
        row = {"id":"dense-code-source", "created_at":"2026-10-18T00:00:00Z", "query":"保留完整代码片段",
               "book_id":"os", "book_label":"操作系统", "favorite":False, "note":"", "result":result}
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file([row]))
        self.expect(self.page.locator("#history-restore")).to_be_enabled()
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("恢复完成")
        original = self.history_records()
        with self.page.expect_download() as collection:
            self.page.locator("#history-export-md").click()
        self.expect(self.page.locator("#history-export-status")).to_contain_text("已将 1 条记录")
        self.assertIn(excerpt, Path(collection.value.path()).read_text())
        self.page.locator("#history-export-practice").check()
        with self.page.expect_download() as practice:
            self.page.locator("#history-export-md").click()
        self.assertIn(excerpt, Path(practice.value.path()).read_text())
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        with self.page.expect_download() as single:
            self.page.locator(".answer").get_by_role("button", name="保存问答（含来源）", exact=True).click()
        self.assertIn(excerpt, Path(single.value.path()).read_text())
        self.assertEqual(self.history_records(), original)
        self.assertEqual(self.backend.asks, [])

    def test_practice_markdown_keeps_answers_and_saved_notes_after_all_questions(self) -> None:
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.page.emulate_media(color_scheme="dark")
        self.open_page()
        result = answer_result()
        result["answer"] = "ANSWER_ONLY_TOKEN：这是当时保存的回答。"
        self.saved_answer("练习题甲", result=result)
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        area = self.page.get_by_role("textbox", name="编辑学习笔记")
        area.fill("SAVED_NOTE_ONLY_TOKEN")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        area.fill("UNSAVED_DRAFT_ONLY_TOKEN")
        missing = {"status":"no_evidence", "answer":None, "sources":[], "message":"未找到教材证据。"}
        self.saved_answer("练习题乙", result=missing)
        self.page.locator("#input").fill("下一题尚未发送的草稿")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-export-practice").check()
        original = self.history_records()
        with self.page.expect_download() as download:
            self.page.locator("#history-export-md").click()
        exported = Path(download.value.path()).read_text()
        questions, answers = exported.split("\n## 已保存回答与来源\n", 1)
        self.assertTrue(exported.startswith("# 学习问答练习"))
        self.assertTrue(download.value.suggested_filename.startswith("教材问答练习-"))
        for query in ("练习题甲", "练习题乙"):
            self.assertIn(query, questions)
            self.assertIn(query, answers)
        self.assertEqual(questions.count("我的回答："), 2)
        for text in ("ANSWER_ONLY_TOKEN", "SAVED_NOTE_ONLY_TOKEN", "进程是程序在一个数据集合上的一次执行。"):
            self.assertNotIn(text, questions)
            self.assertIn(text, answers)
        self.assertNotIn("UNSAVED_DRAFT_ONLY_TOKEN", exported)
        self.assertIn("本次没有完整的生成回答。", answers)
        self.assertIn("未找到教材证据。", answers)
        self.assertEqual(self.history_records(), original)
        self.expect(area).to_have_value("UNSAVED_DRAFT_ONLY_TOKEN")
        self.expect(self.page.locator("#input")).to_have_value("下一题尚未发送的草稿")
        self.expect(self.page.locator("#history-export-status")).to_contain_text("练习版")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("history-practice-export-dark-320.png")
        self.assertEqual(len(self.backend.asks), 2)

    def test_practice_export_choice_restores_in_current_tab_and_can_be_cleared(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-export-practice")).not_to_be_checked()
        self.page.locator("#history-export-practice").check()
        self.page.reload()
        self.expect(self.page.locator("#history-export-practice")).to_be_checked()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-export-practice").uncheck()
        self.page.reload()
        self.expect(self.page.locator("#history-export-practice")).not_to_be_checked()
        self.assertEqual(len(self.backend.asks), 1)

    def test_markdown_export_does_not_download_records_deleted_by_another_tab(self) -> None:
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer()
        self.page.locator("#toggle-history").click()
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="删除记录")).to_be_visible()
        other.once("dialog", lambda dialog: dialog.accept())
        other.locator(".answer").get_by_role("button", name="删除记录").click()
        self.expect(other.locator(".answer")).to_have_count(0)
        downloads = []
        self.page.on("download", lambda download: downloads.append(download))
        self.page.locator("#history-export-md").dispatch_event("click")
        self.expect(self.page.locator("#history-export-status")).to_contain_text("当前筛选已没有记录")
        self.assertEqual(downloads, [])
        self.assertEqual(self.history_records(), [])

    def test_history_restore_transaction_aborts_all_new_rows_when_storage_write_fails(self) -> None:
        self.open_page()
        self.saved_answer()
        original = self.history_records()
        incoming = [dict(original[0], id="incoming-one"), dict(original[0], id="incoming-two")]
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(incoming))
        self.expect(self.page.locator("#history-restore")).to_be_enabled()
        self.page.evaluate("""() => {
          const add = IDBObjectStore.prototype.add; let count = 0;
          IDBObjectStore.prototype.add = function(...args) {
            if (++count === 2) throw new DOMException('Full', 'QuotaExceededError');
            return add.apply(this, args);
          };
        }""")
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("恢复失败")
        self.assertEqual(self.history_records(), original)
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("新增 2 条")
        self.assertEqual(len(self.history_records()), 3)

    def test_backup_parser_rejects_unknown_versions_bad_sources_and_private_fields(self) -> None:
        self.open_page()
        self.saved_answer()
        original = self.history_records()[0]
        result = self.page.evaluate("""record => {
          const attempt = value => { try { RagHistoryBackup.parse(JSON.stringify(value)); return false; } catch { return true; } };
          const envelope = records => ({ format:'rag-qa-history', version:1, records });
          const badCitation = structuredClone(record); badCitation.result.sources[0].citation_id = 0;
          const badTiming = structuredClone(record); badTiming.result.timing.total_seconds = -1;
          const secret = structuredClone(record); secret.result.answer_id = 'private-answer-id'; secret.result.prompt = 'private-prompt';
          return { rejected:[attempt({ ...envelope([record]), version:3 }), attempt(envelope([badCitation])), attempt(envelope([badTiming]))],
            exported:RagHistoryBackup.serialize([secret]) };
        }""", original)
        self.assertEqual(result["rejected"], [True, True, True])
        self.assertNotIn("private-prompt", result["exported"])
        self.assertNotIn("private-answer-id", result["exported"])

    def test_backup_rejects_conflicting_terminal_states_and_backwards_review_dates(self) -> None:
        self.open_page()
        self.saved_answer("备份一致性检查")
        original = self.history_records()[0]
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        bad_records = []
        for status in ("retrieval_only", "no_evidence", "failed"):
            row = json.loads(json.dumps(original))
            row["id"] = "conflicting-" + status
            row["result"]["status"] = status
            bad_records.append(row)
        backwards = json.loads(json.dumps(original))
        backwards["id"] = "backwards-review"
        backwards["review"] = {"count": 1, "last_rating": "again",
                               "last_reviewed_at": "2026-10-18T08:00:00Z",
                               "next_review_at": "2026-10-18T07:00:00Z"}
        bad_records.append(backwards)
        equal = json.loads(json.dumps(backwards))
        equal["id"] = "equal-review-times"
        equal["review"]["next_review_at"] = equal["review"]["last_reviewed_at"]
        bad_records.append(equal)
        for row in bad_records:
            with self.subTest(record=row["id"]):
                self.page.locator("#history-backup-file").set_input_files(self.backup_file([row]))
                self.expect(self.page.locator("#history-backup-preview")).to_contain_text("无效记录")
                self.expect(self.page.locator("#history-restore")).to_be_disabled()
                self.assertEqual(self.history_records(), [original])
        legacy = json.loads(json.dumps(original))
        legacy["result"].update(status="retrieval_only", answer="")
        normalized = self.page.evaluate("""record => RagHistoryBackup.parse(JSON.stringify({
          format:'rag-qa-history', version:1, records:[record]
        }))[0]""", legacy)
        self.assertIsNone(normalized["result"]["answer"])

    def test_saved_notes_restore_search_export_and_backup_without_changing_answer(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        note = "复习要点：程序是静态的，进程是动态的。\n<script>neverRun()</script>"
        self.page.get_by_role("textbox", name="编辑学习笔记", exact=True).fill(note)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.expect(self.page.locator(".saved-note .note-text")).to_have_text(note)
        self.expect(self.page.locator(".saved-note script")).to_have_count(0)
        self.page.reload()
        self.expect(self.page.locator(".saved-note .note-text")).to_have_text(note)
        with self.page.expect_download() as info:
            self.page.get_by_role("button", name="保存问答（含来源）", exact=True).click()
        exported = Path(info.value.path()).read_text()
        self.assertIn("## 我的学习笔记", exported)
        self.assertIn(note, exported)
        self.assertIn(answer_result()["answer"], exported)
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-notes").check()
        self.page.locator("#history-search").fill("复习要点")
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator(".history-match")).to_contain_text("笔记命中：复习要点")
        rows = self.history_records()
        canonical = self.page.evaluate("rows => JSON.parse(RagHistoryBackup.serialize(rows)).records", rows)
        self.assertEqual(canonical[0]["note"], note)
        self.assertEqual(rows[0]["result"]["answer"], answer_result()["answer"])
        self.assertEqual(len(self.backend.asks), 1)

    def test_note_draft_survives_refresh_and_save_keeps_favorite_then_can_clear_note(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        self.page.get_by_role("textbox", name="编辑学习笔记", exact=True).fill("没有保存的学习草稿")
        self.page.reload()
        self.expect(self.page.locator(".answer")).to_have_count(1)
        self.page.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        area = self.page.get_by_role("textbox", name="编辑学习笔记", exact=True)
        self.expect(area).to_have_value("没有保存的学习草稿")
        self.assertEqual(self.history_records()[0]["note"], "")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏", exact=True).click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏", exact=True)).to_be_visible()
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertTrue(self.history_records()[0]["favorite"])
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("notes-mobile.png")
        area.fill("")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已清空")
        self.expect(self.page.locator(".saved-note")).to_be_hidden()
        self.assertEqual(self.history_records()[0]["note"], "")

    def test_concurrent_note_edits_keep_latest_saved_note_and_preserve_other_draft(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        self.page.get_by_role("textbox", name="编辑学习笔记", exact=True).fill("第一个标签页的草稿")
        other = self.context.new_page()
        other.on("pageerror", lambda error: self.errors.append(str(error)))
        other.goto(self.base_url)
        self.expect(other.locator(".answer")).to_have_count(1)
        other.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        other.get_by_role("textbox", name="编辑学习笔记", exact=True).fill("另一个标签页已经保存的笔记")
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(other.locator(".note-editor")).to_contain_text("笔记已保存")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("其他标签页已修改")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记", exact=True)).to_have_value("第一个标签页的草稿")
        self.assertEqual(self.history_records()[0]["note"], "另一个标签页已经保存的笔记")
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.page.locator(".answer").get_by_role("button", name="编辑笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("核对上方最新笔记")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记", exact=True)).to_have_value("第一个标签页的草稿")
        merged = "另一个标签页已经保存的笔记\n  第一个标签页的补充  \n"
        self.page.get_by_role("textbox", name="编辑学习笔记", exact=True).fill(merged)
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记", exact=True)).to_have_value(merged)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], merged)

    def test_mobile_late_citation_opens_reader_in_view(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        result = answer_result()
        source = result["sources"][0]
        result["sources"] = [{**source, "citation_id": i, "section": f"第 {i} 节", "excerpt": "原文。" * 100} for i in range(1, 11)]
        result["answer"] = "见第十份资料。【参考资料 10】"
        self.saved_answer(result=result)
        self.page.locator(".answer .cite").click()
        self.expect(self.page.locator(".reader select")).to_have_value("10")
        self.expect(self.page.locator(".reader-heading")).to_be_in_viewport()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()

    def test_review_hides_answer_until_reveal_then_schedules_next_review_without_model(self) -> None:
        self.open_page()
        self.saved_answer("需要复习的进程问题")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.expect(self.page.locator("#review-question")).to_have_text("需要复习的进程问题")
        self.expect(self.page.locator(".answer")).to_have_count(0)
        self.expect(self.page.locator(".composer")).to_be_hidden()
        self.page.locator("#review-reveal").click()
        self.expect(self.page.locator(".answer .body")).to_contain_text("进程")
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 1 条")
        self.expect(self.page.locator(".composer")).to_be_visible()
        saved = self.history_records()[0]
        self.assertEqual(saved["review"]["last_rating"], "again")
        self.assertEqual(saved["review"]["count"], 1)
        self.assertTrue(self.page.evaluate("value => Date.parse(value.next_review_at) > Date.parse(value.last_reviewed_at)", saved["review"]))
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-review-start")).to_be_disabled()
        self.page.locator("#history-review-all").check()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.locator("#review-familiar").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertEqual(self.history_records()[0]["review"]["count"], 2)
        self.assertEqual(self.history_records()[0]["review"]["last_rating"], "familiar")
        self.assertEqual(len(self.backend.asks), 1)

    def test_conflicting_notes_can_be_joined_without_writing_until_explicit_save(self) -> None:
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.page.emulate_media(color_scheme="dark")
        other = self.conflicting_notes()
        other.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        original = self.history_records()
        self.page.locator(".note-editor").evaluate("element => element.scrollIntoView({block:'start'})")
        self.capture_page("note-conflict-action-dark-320.png")
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        area = self.page.get_by_role("textbox", name="编辑学习笔记")
        combined = area.input_value()
        self.assertIn("本页要保留的草稿", combined)
        self.assertEqual(combined.count("另一页最新保存的笔记"), 1)
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.assertEqual(self.history_records(), original)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        saved = self.history_records()[0]
        self.assertEqual(saved["note"], combined)
        self.assertTrue(saved["favorite"])
        self.assertEqual(saved["result"], original[0]["result"])
        self.page.reload()
        self.expect(self.page.locator(".saved-note")).to_contain_text("本页要保留的草稿")
        self.expect(self.page.locator(".saved-note")).to_contain_text("另一页最新保存的笔记")
        self.assertEqual(len(self.backend.asks), 1)

    def test_recovered_conflict_draft_can_join_latest_and_survive_another_refresh(self) -> None:
        self.conflicting_notes()
        self.page.reload()
        self.expect(self.page.locator(".answer").get_by_role("button", name="编辑笔记")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="编辑笔记").click()
        self.expect(self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True)).to_be_visible()
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        combined = self.page.get_by_role("textbox", name="编辑学习笔记").input_value()
        self.assertEqual(self.history_records()[0]["note"], "另一页最新保存的笔记")
        self.page.reload()
        self.expect(self.page.locator(".answer").get_by_role("button", name="编辑笔记")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="编辑笔记").click()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value(combined)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], combined)
        self.assertEqual(len(self.backend.asks), 1)

    def test_refreshed_conflict_draft_keeps_its_original_version_until_joined(self) -> None:
        self.conflicting_notes()
        original = self.history_records()
        self.page.reload()
        self.expect(self.page.locator(".answer").get_by_role("button", name="编辑笔记")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="编辑笔记").click()
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.get_by_role("button", name="保存笔记", exact=True)).to_be_enabled()
        self.assertEqual(self.history_records(), original)
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿未覆盖最新版本")
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertIn("本页要保留的草稿", self.history_records()[0]["note"])
        self.assertIn("另一页最新保存的笔记", self.history_records()[0]["note"])

    def test_joined_note_still_detects_another_update_before_save(self) -> None:
        other = self.conflicting_notes(latest="另一页第一次保存")
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        other.get_by_role("textbox", name="编辑学习笔记").fill("另一页第二次保存")
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(other.locator(".note-editor")).to_contain_text("笔记已保存")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿未覆盖最新版本")
        self.assertEqual(self.history_records()[0]["note"], "另一页第二次保存")
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        area = self.page.get_by_role("textbox", name="编辑学习笔记")
        for text in ("本页要保留的草稿", "另一页第一次保存", "另一页第二次保存"):
            self.assertEqual(area.input_value().count(text), 1)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], area.input_value())

    def test_joining_notes_over_limit_keeps_both_originals_and_can_retry_after_edit(self) -> None:
        self.conflicting_notes(draft="甲" * 2400, latest="乙" * 2400)
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("超过 4000 字，原草稿未改动")
        area = self.page.get_by_role("textbox", name="编辑学习笔记")
        self.expect(area).to_have_value("甲" * 2400)
        self.assertEqual(self.history_records()[0]["note"], "乙" * 2400)
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿未覆盖最新版本")
        self.assertEqual(self.history_records()[0]["note"], "乙" * 2400)
        area.fill("甲" * 1000)
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertIn("甲" * 1000, self.history_records()[0]["note"])
        self.assertIn("乙" * 2400, self.history_records()[0]["note"])

    def test_join_note_read_failure_keeps_draft_and_allows_retry(self) -> None:
        self.conflicting_notes()
        original = self.history_records()
        self.page.evaluate("""() => {
          const get = IDBObjectStore.prototype.get;
          IDBObjectStore.prototype.get = function() {
            IDBObjectStore.prototype.get = get;
            throw new DOMException('fixture get failure', 'UnknownError');
          };
        }""")
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("最新笔记暂时无法读取")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("本页要保留的草稿")
        self.assertEqual(self.history_records(), original)
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("请整理重复内容后再保存")
        self.assertEqual(self.history_records(), original)

    def test_join_note_detects_deletion_and_keeps_exportable_original_draft(self) -> None:
        other = self.conflicting_notes(live=False)
        other.once("dialog", lambda dialog: dialog.accept())
        other.locator(".answer").get_by_role("button", name="删除记录", exact=True).click()
        self.expect(other.locator(".answer")).to_have_count(0)
        self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True).dispatch_event("click")
        self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("本页要保留的草稿")
        self.expect(self.page.get_by_role("button", name="把最新笔记加入草稿", exact=True)).to_be_disabled()
        self.assertEqual(self.history_records(), [])
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.assertIn("本页要保留的草稿", Path(download.value.path()).read_text())

    def test_review_session_restores_next_question_after_refresh_and_skip_does_not_grade(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        for question in ("第一条收藏", "第二条收藏"):
            self.saved_answer(question)
            self.page.locator(".answer").last.get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").last.get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-meta")).to_contain_text("2 / 2")
        question = self.page.locator("#review-question").inner_text()
        self.page.reload()
        self.expect(self.page.locator("#review-question")).to_have_text(question)
        self.expect(self.page.locator(".answer")).to_have_count(0)
        self.expect(self.page.locator("#review-meta")).to_contain_text("已复习 1 条")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("review-mobile.png")
        self.page.locator("#review-skip").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertEqual(sum(bool(row.get("review")) for row in self.history_records()), 1)
        canonical = self.page.evaluate("rows => JSON.parse(RagHistoryBackup.serialize(rows))", self.history_records())
        self.assertEqual(canonical["version"], 2)
        self.assertEqual(sum(bool(row.get("review")) for row in canonical["records"]), 1)

    def test_review_respects_book_filter_and_end_does_not_mark_question(self) -> None:
        self.open_page()
        for query, book in (("操作系统收藏", "os"), ("数据库收藏", "database")):
            self.saved_answer(query, book)
            self.page.locator(".answer").last.get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").last.get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-book").select_option("database")
        self.page.locator("#history-review-start").click()
        self.expect(self.page.locator("#review-question")).to_have_text("数据库收藏")
        self.expect(self.page.locator("#review-meta")).to_contain_text("1 / 1")
        self.page.locator("#review-end").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertFalse(any(row.get("review") for row in self.history_records()))

    def test_open_review_panel_updates_due_count_when_local_midnight_passes(self) -> None:
        self.context.close()
        self.context = self.browser.new_context(timezone_id="Australia/Perth")
        self.addCleanup(self.context.close)
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.context.route("**/*", self.local_only)
        self.page = self.context.new_page()
        self.page.set_default_timeout(5000)
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.clock.install(time=datetime.fromisoformat("2026-10-18T23:59:40+08:00"))
        self.open_page()
        self.saved_answer("跨午夜仍打开的复习面板")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        original = self.history_records()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-review-count")).to_contain_text("0 条待复习")
        self.expect(self.page.locator("#history-review-start")).to_be_disabled()
        self.page.wait_for_timeout(100)  # Let the explicit panel read/focus finish before time advances.
        self.page.clock.fast_forward(30_000)
        self.expect(self.page.locator("#history-review-count")).to_contain_text("1 条待复习")
        self.expect(self.page.locator("#history-review-start")).to_be_enabled()
        self.assertEqual(self.history_records(), original)
        self.assertEqual(len(self.backend.asks), 1)

    def test_distant_review_date_remains_not_due_after_a_day_without_changing_drafts(self) -> None:
        self.page.clock.install(time=datetime.fromisoformat("2026-10-18T00:00:00+00:00"))
        self.open_page()
        self.saved_answer("未来复习日期的原问答")
        incoming = dict(self.history_records()[0], id="distant-review", favorite=True,
                        review={"count": 1, "last_rating": "familiar",
                                "last_reviewed_at": "2026-10-18T00:00:00Z",
                                "next_review_at": "2099-10-18T00:00:00Z"})
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file([incoming]))
        self.expect(self.page.locator("#history-restore")).to_be_enabled()
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("新增 1 条")
        self.page.locator("#input").fill("保持下一题的提问草稿")
        original = self.history_records()
        self.page.clock.fast_forward(24 * 60 * 60 * 1000)
        self.expect(self.page.locator("#history-review-count")).to_contain_text("0 条待复习")
        self.expect(self.page.locator("#history-review-start")).to_be_disabled()
        self.expect(self.page.locator("#input")).to_have_value("保持下一题的提问草稿")
        self.assertEqual(self.history_records(), original)
        self.assertEqual(len(self.backend.asks), 1)

    def test_opening_history_keeps_focus_the_user_moved_during_a_slow_read(self) -> None:
        self.open_page()
        self.saved_answer("慢读取时的焦点")
        # Hold read-only history transactions until the test releases them, so the
        # panel's asynchronous refresh finishes at a chosen moment.
        self.page.evaluate("""() => {
          const original = IDBDatabase.prototype.transaction;
          const completed = Object.getOwnPropertyDescriptor(IDBTransaction.prototype, 'oncomplete').set;
          window.__historyReadGate = Promise.resolve();
          window.__holdHistoryReads = () => {
            window.__historyReadGate = new Promise(resolve => { window.__releaseHistoryReads = resolve; });
          };
          IDBDatabase.prototype.transaction = function (store, mode, ...rest) {
            const transaction = original.call(this, store, mode, ...rest);
            if (mode === 'readonly') {
              Object.defineProperty(transaction, 'oncomplete', { configurable: true, set(handler) {
                completed.call(transaction, event => window.__historyReadGate.then(() => handler.call(transaction, event)));
              } });
            }
            return transaction;
          };
        }""")
        release = """async () => {
          window.__releaseHistoryReads();
          for (let turn = 0; turn < 3; turn++) await new Promise(resolve => setTimeout(resolve, 0));
        }"""
        # Without other input, the panel still moves focus to its search box once loaded.
        self.page.evaluate("window.__holdHistoryReads()")
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-panel")).to_be_visible()
        self.page.evaluate(release)
        self.expect(self.page.locator("#history-search")).to_be_focused()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-panel")).to_be_hidden()
        # A control the user reached before the read finished keeps its focus.
        self.page.evaluate("window.__holdHistoryReads()")
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-panel")).to_be_visible()
        self.page.locator("#history-favorites").focus()
        self.page.evaluate(release)
        self.expect(self.page.locator("#history-favorites")).to_be_focused()
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "history-favorites")

    def test_keyboard_review_and_history_reading_move_focus_to_current_content(self) -> None:
        self.open_page()
        for question in ("键盘复习第一题", "键盘复习第二题"):
            self.saved_answer(question)
            self.page.locator(".answer").last.get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").last.get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#input").fill("复习结束后继续编辑的草稿")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").focus()
        self.page.locator("#history-review-start").press("Enter")
        self.expect(self.page.locator("#review-reveal")).to_be_focused()
        self.page.locator("#review-reveal").press("Enter")
        self.expect(self.page.locator(".answer .body")).to_be_focused()
        self.page.locator("#review-skip").focus()
        self.page.locator("#review-skip").press("Enter")
        self.expect(self.page.locator("#review-question")).to_have_text("键盘复习第二题")
        self.expect(self.page.locator("#review-reveal")).to_be_focused()
        self.page.locator("#review-reveal").press("Enter")
        self.page.locator("#review-familiar").focus()
        self.page.locator("#review-familiar").press("Enter")
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#input")).to_be_focused()
        self.expect(self.page.locator("#input")).to_have_value("复习结束后继续编辑的草稿")
        self.page.locator("#toggle-history").click()
        view = self.page.locator("[data-history-view]").first
        view.focus()
        view.press("Enter")
        self.expect(self.page.locator(".answer .body")).to_be_focused()
        self.assertEqual(sum(bool(row.get("review")) for row in self.history_records()), 1)
        self.assertEqual(len(self.backend.asks), 2)

    def test_review_is_due_from_start_of_target_local_day_including_dst_change(self) -> None:
        for zone, days, rating, initial, target in (
            ("Australia/Perth", 1, "#review-again", "2026-10-18T22:15:00+08:00", "2026-10-19T00:00:00+08:00"),
            ("America/New_York", 7, "#review-familiar", "2026-10-31T22:15:00-04:00", "2026-11-07T00:00:00-05:00"),
        ):
            with self.subTest(zone=zone, days=days):
                context = self.browser.new_context(timezone_id=zone)
                context.route("**/*", self.local_only)
                original_page = self.page
                try:
                    self.page = context.new_page()
                    self.page.on("pageerror", lambda error: self.errors.append(str(error)))
                    start = datetime.fromisoformat(initial)
                    self.page.clock.set_fixed_time(start)
                    self.open_page()
                    self.saved_answer("晚上复习后，目标日期早上应已到期")
                    self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
                    self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
                    self.page.locator("#toggle-history").click()
                    self.page.locator("#history-review-start").click()
                    self.page.locator("#review-reveal").click()
                    self.page.locator(rating).click()
                    self.expect(self.page.locator("#review-panel")).to_be_hidden()
                    scheduled = self.history_records()[0]["review"]["next_review_at"]
                    expected = datetime.fromisoformat(target)
                    self.assertEqual(datetime.fromisoformat(scheduled), expected)
                    self.page.clock.set_fixed_time(expected.replace(hour=8))
                    self.page.locator("#toggle-history").click()
                    self.expect(self.page.locator("#history-review-start")).to_be_enabled()
                finally:
                    self.page = original_page
                    context.close()

    def test_opening_another_history_record_ends_review_before_showing_its_answer(self) -> None:
        self.open_page()
        self.saved_answer("复习中的问题")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.saved_answer("只想查看的另一条记录")
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-item").filter(has_text="只想查看的另一条记录").get_by_role("button", name="查看原回答").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator(".composer")).to_be_visible()
        self.expect(self.page.locator(".user")).to_have_text("只想查看的另一条记录")
        self.assertFalse(any(row.get("review") for row in self.history_records()))

    def test_unfavoriting_current_review_skips_it_without_grading(self) -> None:
        self.open_page()
        for question in ("当前复习问题", "下一条复习问题"):
            self.saved_answer(question)
            self.page.locator(".answer").last.get_by_role("button", name="☆ 收藏").click()
            self.expect(self.page.locator(".answer").last.get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.expect(self.page.locator("#review-question")).to_have_text("当前复习问题")
        self.page.locator("#review-reveal").click()
        self.page.locator(".answer").get_by_role("button", name="★ 已收藏").click()
        self.expect(self.page.locator("#review-question")).to_have_text("下一条复习问题")
        self.expect(self.page.locator(".answer")).to_have_count(0)
        self.page.locator("#review-end").click()
        self.assertFalse(any(row.get("review") for row in self.history_records()))

    def test_other_tab_unfavorite_is_checked_before_review_rating_is_saved(self) -> None:
        # Exercise the transaction guard even when cross-tab notifications are
        # unavailable and the old tab has not regained foreground focus.
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("另一标签页取消的收藏")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="★ 已收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.page.evaluate("document.querySelector('#review-again').click()")
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 0 条")
        self.assertFalse(self.history_records()[0].get("review"))

    def test_cross_tab_updates_refresh_library_and_keep_unsaved_note_draft(self) -> None:
        self.open_page()
        self.saved_answer("跨页同步的记录")
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("尚未保存的草稿")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="学习笔记").click()
        other.get_by_role("textbox", name="编辑学习笔记").fill("另一页已保存的笔记")
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".saved-note")).to_contain_text("另一页已保存的笔记")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("尚未保存的草稿")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿未覆盖最新版本")
        self.assertEqual(self.history_records()[0]["note"], "另一页已保存的笔记")
        self.assertEqual(len(self.backend.asks), 1)
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        other.locator("#book").select_option("os")
        other.locator("#input").fill("另一页的新问答")
        other.locator("#send").click()
        self.expect(other.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.expect(self.page.locator("#history-count")).to_contain_text("2 条记录")
        self.expect(self.page.locator(".msg.user")).to_have_count(1)
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("尚未保存的草稿")

    def test_cross_tab_refresh_falls_back_to_focus_when_channel_is_unavailable(self) -> None:
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("回到标签页时更新")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.evaluate("window.dispatchEvent(new Event('focus'))")
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.assertEqual(len(self.backend.asks), 1)

    def test_cross_tab_delete_keeps_open_note_draft_until_export_and_close(self) -> None:
        self.open_page()
        self.saved_answer("被另一页删除的问答")
        identifier = self.history_records()[0]["id"]
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("不能消失的未保存笔记")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="删除记录")).to_be_visible()
        other.once("dialog", lambda dialog: dialog.accept())
        other.locator(".answer").get_by_role("button", name="删除记录").click()
        self.expect(self.page.locator(".note-record-missing")).to_contain_text("当前草稿仍保留")
        self.expect(self.page.locator(".answer")).not_to_contain_text("已保存到此浏览器")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("不能消失的未保存笔记")
        self.assertEqual(self.history_records(), [])
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        exported = Path(download.value.path()).read_text()
        self.assertIn("不能消失的未保存笔记", exported)
        self.assertIn("被另一页删除的问答", exported)
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.expect(self.page.locator(".answer")).to_have_count(0)
        self.assertIsNone(self.page.evaluate("id => sessionStorage.getItem('rag-qa-note-draft-' + id)", identifier))

    def test_leaving_deleted_note_can_be_cancelled_before_switching_history(self) -> None:
        identifier = self.deleted_note_fixture(another=True)
        self.page.locator("#toggle-history").click()
        dialogs = []
        def dismiss(dialog):
            dialogs.append(dialog.message)
            dialog.dismiss()
        self.page.once("dialog", dismiss)
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        self.assertEqual(len(dialogs), 1)
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("仍需导出的笔记草稿")
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        self.expect(self.page.locator(".note-editor")).to_have_count(0)
        self.expect(self.page.locator(".msg.user")).to_have_text("可以继续查看的问答")
        self.assertIsNone(self.page.evaluate("id => sessionStorage.getItem('rag-qa-note-draft-' + id)", identifier))

    def test_reloading_unexported_deleted_note_warns_and_can_keep_current_draft(self) -> None:
        from playwright.sync_api import Error
        self.deleted_note_fixture()
        dialogs = []
        def dismiss(dialog):
            dialogs.append(dialog.type)
            dialog.dismiss()
        self.page.once("dialog", dismiss)
        try:
            self.page.reload()
        except Error:
            pass  # Dismissing Chromium's beforeunload cancels navigation.
        self.assertEqual(dialogs, ["beforeunload"])
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("仍需导出的笔记草稿")

    def test_note_save_detects_deletion_without_channel_and_keeps_exportable_draft(self) -> None:
        self.deleted_note_fixture(live=False)
        self.page.evaluate("""() => {
          const getAll = IDBObjectStore.prototype.getAll;
          IDBObjectStore.prototype.getAll = function() {
            IDBObjectStore.prototype.getAll = getAll;
            throw new DOMException('fixture list read failure', 'UnknownError');
          };
        }""")
        self.page.get_by_role("button", name="保存笔记", exact=True).dispatch_event("click")
        self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("仍需导出的笔记草稿")
        self.assertEqual(self.history_records(), [])

    def test_review_rating_race_with_deleted_record_keeps_open_note_draft(self) -> None:
        self.deleted_note_fixture(review=True, live=False)
        self.page.evaluate("""() => {
          const getAll = IDBObjectStore.prototype.getAll;
          IDBObjectStore.prototype.getAll = function() {
            IDBObjectStore.prototype.getAll = getAll;
            throw new DOMException('fixture list read failure', 'UnknownError');
          };
        }""")
        self.page.locator("#review-again").dispatch_event("click")
        self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("仍需导出的笔记草稿")
        self.expect(self.page.locator("#review-again")).to_be_disabled()
        self.assertEqual(self.history_records(), [])
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator("#review-end").click()
        self.expect(self.page.locator("#review-panel")).to_be_visible()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("仍需导出的笔记草稿")
        with self.page.expect_download():
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.page.locator("#review-end").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 0 条")

    def test_editing_after_draft_export_requires_confirm_or_a_new_export_before_clear(self) -> None:
        self.deleted_note_fixture()
        with self.page.expect_download():
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("导出后新增的草稿内容")
        dialogs = []
        def dismiss(dialog):
            dialogs.append(dialog.message)
            dialog.dismiss()
        self.page.once("dialog", dismiss)
        self.page.locator("#clear").click()
        self.assertEqual(len(dialogs), 1)
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("导出后新增的草稿内容")
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.assertIn("导出后新增的草稿内容", Path(download.value.path()).read_text())
        self.page.locator("#clear").click()
        self.expect(self.page.locator(".answer")).to_have_count(0)

    def test_deleted_note_warning_and_export_fit_dark_narrow_screen(self) -> None:
        self.page.set_viewport_size({"width": 320, "height": 844})
        self.page.emulate_media(color_scheme="dark")
        self.deleted_note_fixture()
        self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).scroll_into_view_if_needed()
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.assertIn("仍需导出的笔记草稿", Path(download.value.path()).read_text())
        self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        self.capture_page("deleted-note-dark-320.png")

    def test_deleting_review_question_elsewhere_preserves_open_draft_and_disables_grading(self) -> None:
        self.open_page()
        self.saved_answer("复习中保留草稿")
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("复习中尚未保存的笔记")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="删除记录")).to_be_visible()
        other.once("dialog", lambda dialog: dialog.accept())
        other.locator(".answer").get_by_role("button", name="删除记录").click()
        self.expect(self.page.locator(".note-record-missing")).to_be_visible()
        self.expect(self.page.locator("#review-again")).to_be_disabled()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("复习中尚未保存的笔记")
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 0 条")

    def test_other_tab_updates_wait_for_active_answer_and_keep_next_question_draft(self) -> None:
        self.open_page()
        self.saved_answer("已有的记录")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer")).to_have_count(1)
        gate = threading.Event()
        self.backend.enqueue(Reply(initial=[("chunk", {"text": "当前答案正在返回"})],
                                   final=[("result", answer_result())], gate=gate))
        self.submit("还在返回的问题")
        self.expect(self.page.locator(".answer").last).to_contain_text("当前答案正在返回")
        self.page.locator("#input").fill("下一题尚未发送的草稿")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        other.locator("#book").select_option("os")
        other.locator("#input").fill("另一页新保存的题目")
        other.locator("#send").click()
        self.expect(other.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.expect(self.page.locator("#thread")).to_have_attribute("aria-busy", "true")
        self.expect(self.page.locator(".answer").last).to_contain_text("当前答案正在返回")
        gate.set()
        self.expect(self.page.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.expect(self.page.locator("#history-count")).to_contain_text("3 条记录")
        self.expect(self.page.locator("#input")).to_have_value("下一题尚未发送的草稿")
        self.assertEqual(len(self.backend.asks), 3)

    def test_recent_imports_can_reopen_and_discard_copy_after_session_pointer_is_lost(self) -> None:
        self.backend.imports_enabled = True
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        self.page.locator("#import-file").set_input_files({"name": "教材.md", "mimeType": "text/markdown", "buffer": b"textbook"})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.page.evaluate("sessionStorage.removeItem('rag-qa-import-job')")
        self.page.reload()
        self.expect(self.page.locator("#import-recent")).to_be_visible()
        self.page.locator("#import-recent summary").click()
        self.expect(self.page.locator(".recent-job")).to_have_count(1)
        self.page.locator(".recent-job").get_by_role("button", name="查看进度").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator(".recent-job").get_by_role("button", name="清理导入副本").click()
        self.assertFalse(self.backend.import_deleted)
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator(".recent-job").get_by_role("button", name="清理导入副本").click()
        self.expect(self.page.locator(".recent-job")).to_have_count(0)
        self.expect(self.page.locator("#import-job")).to_be_hidden()
        self.expect(self.page.locator("#import-status")).to_contain_text("已发布索引保留")
        self.assertEqual(len(self.backend.uploads), 1)

    def test_changed_import_file_invalidates_old_preview_and_running_copy_cannot_be_cleaned(self) -> None:
        self.backend.imports_enabled = True
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        self.page.locator("#import-file").set_input_files({"name": "one.md", "mimeType": "text/markdown", "buffer": b"first"})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.page.locator("#import-confirm").check()
        self.page.locator("#import-file").set_input_files({"name": "two.md", "mimeType": "text/markdown", "buffer": b"second"})
        self.expect(self.page.locator("#import-job")).to_be_hidden()
        self.expect(self.page.locator("#import-start")).to_be_disabled()
        self.assertEqual(self.backend.import_starts, [])
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        self.page.locator("#import-confirm").check()
        self.page.locator("#import-start").click()
        self.expect(self.page.locator("#import-progress-text")).to_contain_text("1 / 2")
        self.page.locator("#import-recent summary").click()
        self.page.locator("#import-recent-reload").click()
        self.expect(self.page.locator(".recent-job").get_by_role("button", name="清理导入副本")).to_be_disabled()
        self.assertEqual(self.backend.uploads, [b"first", b"second"])

    def test_old_import_progress_response_cannot_restore_invalidated_file_preview(self) -> None:
        self.backend.imports_enabled = True
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-prepare")).to_be_enabled()
        self.page.locator("#import-book").select_option("os")
        self.page.locator("#import-file").set_input_files({"name": "one.md", "mimeType": "text/markdown", "buffer": b"first"})
        self.page.locator("#import-prepare").click()
        self.expect(self.page.locator("#import-job")).to_be_visible()
        gate = threading.Event()
        self.backend.import_read_gate = gate
        self.backend.gates.append(gate)
        self.page.locator("#import-refresh").click()
        self.page.locator("#import-file").set_input_files({"name": "two.md", "mimeType": "text/markdown", "buffer": b"second"})
        self.expect(self.page.locator("#import-job")).to_be_hidden()
        gate.set()
        self.page.wait_for_load_state("networkidle")
        self.expect(self.page.locator("#import-job")).to_be_hidden()
        self.expect(self.page.locator("#import-start")).to_be_disabled()
        self.assertEqual(self.backend.import_starts, [])

    def test_markdown_comparison_table_keeps_alignment_citations_and_literal_html(self) -> None:
        self.open_page()
        result = answer_result()
        result["answer"] = "3. 比较下面的概念。\n\n| 项目 | 说明 |\n| :--- | ---: |\n| 程序 | 静态代码【参考资料 1】 |\n| 转义\\|竖线 | <img src=x onerror=neverRun()> |\n\n4. 请结合原文核对。"
        self.saved_answer(result=result)
        body = self.page.locator(".answer .body")
        self.expect(body.locator("table")).to_have_count(1)
        self.expect(body.locator("th").last).to_have_css("text-align", "right")
        self.expect(body.locator("tbody tr")).to_have_count(2)
        self.expect(body.locator("tbody tr").last.locator("td").first).to_have_text("转义|竖线")
        self.expect(body.locator("img")).to_have_count(0)
        self.expect(body.locator("tbody")).to_contain_text("<img src=x onerror=neverRun()>")
        self.expect(body.locator("ol").first).to_have_attribute("start", "3")
        self.expect(body.locator("ol").last).to_have_attribute("start", "4")
        body.locator(".cite").click()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()
        self.assertEqual(self.history_records()[0]["result"]["answer"], result["answer"])

    def test_code_copy_preserves_nested_fences_unicode_and_literal_citation(self) -> None:
        self.isolated_clipboard(api=False, command="success")
        self.open_page()
        code = 'print("中文 <script>neverRun()</script>")\n```\n【参考资料 1】'
        result = answer_result()
        result["answer"] = "````python\n" + code + "\n````\n\n代码之后的说明。`**原样**【参考资料 9】`"
        self.saved_answer(result=result)
        body = self.page.locator(".answer .body")
        self.expect(body.locator("pre code")).to_have_text(code)
        self.expect(body.locator("pre .cite")).to_have_count(0)
        self.expect(body.locator("script")).to_have_count(0)
        self.expect(body.locator(".cite")).to_have_count(0)
        self.expect(body.locator("p code")).to_have_text("**原样**【参考资料 9】")
        body.get_by_role("button", name="复制代码", exact=True).click()
        self.expect(body.locator("[data-copy-code]")).to_have_text("已复制")
        self.assertEqual(self.page.evaluate("window.copyAttempts[0].text"), code)
        self.expect(body).to_contain_text("代码之后的说明")

    def test_wide_tables_and_tilde_code_scroll_inside_mobile_answer(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        result = answer_result()
        result["answer"] = "| 项目 | 特性一 | 特性二 | 特性三 | 特性四 | 特性五 |\n| --- | --- | --- | --- | --- | --- |\n| 进程 | 动态 | 并发 | 独立 | 异步 | 资源 |\n\n~~~text\n" + "long_code_" * 30 + "\n~~~"
        self.saved_answer(result=result)
        body = self.page.locator(".answer .body")
        self.expect(body.locator("table")).to_have_count(1)
        self.expect(body.locator("pre code")).to_have_text("long_code_" * 30)
        self.assertTrue(body.locator(".table-scroll").evaluate("el => el.scrollWidth > el.clientWidth"))
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("answer-table-mobile.png")

    def test_clearing_thread_aborts_pending_source_read(self) -> None:
        self.open_page()
        self.saved_answer()
        gate = threading.Event()
        self.backend.reading_gate = gate
        self.backend.gates.append(gate)
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        with self.page.expect_request(lambda request: "/passages?" in request.url):
            self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        with self.page.expect_event("requestfailed", predicate=lambda request: "/passages?" in request.url):
            self.page.locator("#clear").click()
        self.expect(self.page.locator(".reader")).to_have_count(0)
        self.expect(self.page.locator("#empty")).to_be_visible()
        self.assertEqual(len(self.backend.reads), 1)

    def test_question_cache_failure_warns_and_cancelled_reload_keeps_text(self) -> None:
        from playwright.sync_api import Error
        self.block_question_draft_cache()
        self.open_page()
        self.page.locator("#input").fill("暂存失败时仍在本页的问题")
        self.expect(self.page.locator("#draft-status")).to_be_visible()
        self.expect(self.page.locator("#draft-status")).to_contain_text("提问草稿暂存失败")
        dialogs = []
        self.page.once("dialog", lambda dialog: (dialogs.append(dialog.type), dialog.dismiss()))
        try:
            self.page.reload()
        except Error:
            pass
        self.assertEqual(dialogs, ["beforeunload"])
        self.expect(self.page.locator("#input")).to_have_value("暂存失败时仍在本页的问题")
        self.assertEqual(self.backend.asks, [])

    def test_question_cache_recovery_and_manual_clear_remove_refresh_warning(self) -> None:
        self.block_question_draft_cache()
        self.open_page()
        self.page.locator("#input").fill("暂存失败的问题")
        self.expect(self.page.locator("#draft-status")).to_contain_text("暂存失败")
        self.page.evaluate("window.questionCacheBroken = false")
        self.page.locator("#input").fill("恢复正常暂存的问题")
        self.expect(self.page.locator("#draft-status")).to_be_hidden()
        dialogs = []
        self.page.on("dialog", lambda dialog: (dialogs.append(dialog.type), dialog.dismiss()))
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("恢复正常暂存的问题")
        self.page.locator("#input").fill("再次暂存失败的问题")
        self.expect(self.page.locator("#draft-status")).to_contain_text("暂存失败")
        self.page.locator("#input").fill("")
        self.expect(self.page.locator("#draft-status")).to_be_hidden()
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("")
        self.assertEqual(dialogs, [])
        self.assertEqual(self.backend.asks, [])

    def test_sending_uncached_question_preserves_next_uncached_draft_after_completion(self) -> None:
        from playwright.sync_api import Error
        gate = threading.Event()
        self.backend.enqueue(Reply(initial=[("chunk", {"text": "等待上一题"})],
                                   final=[("result", answer_result())], gate=gate))
        self.block_question_draft_cache()
        self.open_page()
        self.page.locator("#input").fill("准备发送的暂存失败问题")
        self.page.locator("#send").click()
        self.expect(self.page.locator(".answer .body")).to_have_text("等待上一题")
        self.expect(self.page.locator("#input")).to_have_value("")
        self.expect(self.page.locator("#draft-status")).to_be_hidden()
        self.page.locator("#input").fill("回答期间写的下一题")
        gate.set()
        self.expect(self.page.locator(".answer")).to_contain_text("已保存到此浏览器")
        self.expect(self.page.locator("#draft-status")).to_contain_text("提问草稿暂存失败")
        dialogs = []
        self.page.once("dialog", lambda dialog: (dialogs.append(dialog.type), dialog.dismiss()))
        try:
            self.page.reload()
        except Error:
            pass
        self.assertEqual(dialogs, ["beforeunload"])
        self.expect(self.page.locator("#input")).to_have_value("回答期间写的下一题")
        self.assertEqual(len(self.backend.asks), 1)

    def test_edit_original_question_keeps_context_and_warns_if_draft_cache_fails(self) -> None:
        self.block_question_draft_cache()
        self.open_page()
        self.saved_answer("仍然可以编辑的原问题")
        self.page.locator(".answer").get_by_role("button", name="编辑原问题", exact=True).click()
        self.expect(self.page.locator("#draft-status")).to_contain_text("已载入")
        self.expect(self.page.locator("#draft-status")).to_contain_text("提问草稿暂存失败")
        self.expect(self.page.locator("#input")).to_have_value("仍然可以编辑的原问题")
        self.assertEqual(len(self.backend.asks), 1)

    def test_question_draft_and_book_restore_on_refresh_then_clear_after_submission(self) -> None:
        self.open_page("database")
        self.page.locator("#input").fill("还没有发送的问题\n第二行")
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("还没有发送的问题\n第二行")
        self.expect(self.page.locator("#book")).to_have_value("database")
        self.expect(self.page.locator("#draft-status")).to_contain_text("已恢复")
        self.assertEqual(self.backend.asks, [])
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.page.locator("#send").click()
        self.expect(self.page.locator(".answer")).to_contain_text("已保存到此浏览器")
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("")
        self.assertEqual(len(self.backend.asks), 1)

    def test_clearing_access_code_forgets_previous_value_on_all_three_pages(self) -> None:
        self.backend.access_code_required = self.backend.imports_enabled = True
        self.open_page()
        for path, code_selector in (("/", "#code"), ("/retrieval", "#inspect-code"), ("/imports", "#import-code")):
            with self.subTest(path=path):
                self.page.evaluate("sessionStorage.setItem('rag-qa-access-code', 'fixture-code-to-forget')")
                self.page.goto(self.base_url + path)
                self.expect(self.page.locator(code_selector)).to_be_visible()
                self.expect(self.page.locator(code_selector)).to_have_value("fixture-code-to-forget")
                self.page.locator(code_selector).fill("")
                self.page.reload()
                self.expect(self.page.locator(code_selector)).to_have_value("")
                self.assertNotEqual(self.page.evaluate("sessionStorage.getItem('rag-qa-access-code')"), "fixture-code-to-forget")
        self.assertEqual(self.backend.asks, [])

    def test_edit_original_question_restores_book_and_never_submits_automatically(self) -> None:
        self.open_page()
        self.saved_answer("进程是什么？")
        self.page.locator("#book").select_option("database")
        self.page.locator("#input").fill("当前的另一个草稿")
        self.page.locator(".answer").get_by_role("button", name="编辑原问题", exact=True).click()
        self.expect(self.page.locator("#input")).to_have_value("进程是什么？")
        self.expect(self.page.locator("#book")).to_have_value("os")
        self.expect(self.page.locator("#draft-status")).to_contain_text("独立检索")
        self.assertEqual(len(self.backend.asks), 1)
        self.page.locator("#input").fill("进程和线程有什么区别？")
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("进程和线程有什么区别？")
        self.expect(self.page.locator("#book")).to_have_value("os")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.page.locator("#send").click()
        self.expect(self.page.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.assertEqual(self.backend.asks[-1]["payload"], {"query": "进程和线程有什么区别？", "book_id": "os"})

    def test_next_question_draft_survives_previous_answer_completion_and_refresh(self) -> None:
        gate = threading.Event()
        self.backend.enqueue(Reply(initial=[("chunk", {"text": "正在回答上一题"})],
                                   final=[("result", answer_result())], gate=gate))
        self.open_page()
        self.submit("上一题")
        self.expect(self.page.locator(".answer .body")).to_have_text("正在回答上一题")
        self.page.locator("#book").select_option("database")
        self.page.locator("#input").fill("准备好的下一题")
        gate.set()
        self.expect(self.page.locator(".answer")).to_contain_text("已保存到此浏览器")
        self.page.reload()
        self.expect(self.page.locator("#input")).to_have_value("准备好的下一题")
        self.expect(self.page.locator("#book")).to_have_value("database")
        self.assertEqual(len(self.backend.asks), 1)

    def test_editing_record_without_current_book_keeps_question_and_requires_selection(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page()
        self.saved_answer("旧教材的原问题")
        self.page.route("**/v1/books", lambda route: route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps([{"book_id":"database", "label":"数据库原理及应用"}]),
        ))
        self.page.reload()
        self.expect(self.page.locator("#book option")).to_have_count(2)
        self.page.locator(".answer").get_by_role("button", name="编辑原问题", exact=True).click()
        self.expect(self.page.locator("#input")).to_have_value("旧教材的原问题")
        self.expect(self.page.locator("#book")).to_have_value("")
        self.expect(self.page.locator("#book-status")).to_contain_text("原教材当前不可用")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.page.locator("#send").click()
        self.assertEqual(len(self.backend.asks), 1)

    def test_history_filters_restore_in_current_tab_and_reset_shows_all_records(self) -> None:
        self.open_page()
        self.saved_answer("操作系统题目")
        self.saved_answer("数据库收藏题目", "database")
        self.page.locator(".answer").last.get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").last.get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-book").select_option("database")
        self.page.locator("#history-search").fill("数据库")
        self.page.locator("#history-favorites").check()
        self.page.reload()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator("#history-book")).to_have_value("database")
        self.expect(self.page.locator("#history-search")).to_have_value("数据库")
        self.expect(self.page.locator("#history-favorites")).to_be_checked()
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.page.locator("#history-reset-filters").click()
        self.expect(self.page.locator(".history-item")).to_have_count(2)
        self.expect(self.page.locator("#history-search")).to_have_value("")

    def test_deleting_record_also_removes_its_unsaved_note_draft(self) -> None:
        self.open_page()
        self.saved_answer()
        record_id = self.history_records()[0]["id"]
        self.page.locator(".answer").get_by_role("button", name="学习笔记", exact=True).click()
        self.page.get_by_role("textbox", name="编辑学习笔记", exact=True).fill("跟着记录一起清理的草稿")
        key = "rag-qa-note-draft-" + record_id
        self.assertIsNotNone(self.page.evaluate("key => sessionStorage.getItem(key)", key))
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.locator(".answer").get_by_role("button", name="删除记录", exact=True).click()
        self.expect(self.page.locator(".answer")).to_have_count(0)
        self.assertIsNone(self.page.evaluate("key => sessionStorage.getItem(key)", key))

    def test_committed_note_remains_editable_when_library_refresh_fails(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        area = self.page.get_by_role("textbox", name="编辑学习笔记")
        area.fill("第一次已经提交的笔记")
        self.fail_first_library_read_after_write()
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], "第一次已经提交的笔记")
        self.expect(self.page.locator(".saved-note")).to_contain_text("第一次已经提交的笔记")
        area.fill("在同一编辑器继续修改")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], "在同一编辑器继续修改")

    def test_note_write_failure_keeps_draft_and_allows_retry(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="学习笔记").click()
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("写入被拒绝时应保留的草稿")
        self.page.evaluate("""() => {
          const put = IDBObjectStore.prototype.put;
          IDBObjectStore.prototype.put = function(...args) {
            IDBObjectStore.prototype.put = put;
            throw new DOMException('fixture full storage', 'QuotaExceededError');
          };
        }""")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记没有保存成功")
        self.assertEqual(self.history_records()[0]["note"], "")
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("写入被拒绝时应保留的草稿")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.assertEqual(self.history_records()[0]["note"], "写入被拒绝时应保留的草稿")

    def test_failed_tab_note_cache_keeps_draft_when_clear_close_or_history_switch_is_cancelled(self) -> None:
        self.uncached_note()
        dialogs = []
        self.page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        self.page.locator("#clear").click()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("没有暂存成功的唯一草稿")
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-item").get_by_role("button", name="查看原回答").click()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("没有暂存成功的唯一草稿")
        self.assertEqual(len(dialogs), 3)
        self.assertTrue(all("暂存失败" in message for message in dialogs))
        self.assertEqual(self.history_records()[0]["note"], "")
        self.assertEqual(len(self.backend.asks), 1)

    def test_uncached_note_reload_can_be_cancelled_and_export_allows_close(self) -> None:
        from playwright.sync_api import Error
        self.uncached_note()
        dialogs = []
        self.page.once("dialog", lambda dialog: (dialogs.append(dialog.type), dialog.dismiss()))
        try:
            self.page.reload()
        except Error:
            pass
        self.assertEqual(dialogs, ["beforeunload"])
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("没有暂存成功的唯一草稿")
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.assertIn("没有暂存成功的唯一草稿", Path(download.value.path()).read_text())
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_have_count(0)
        self.assertEqual(self.history_records()[0]["note"], "")

    def test_saving_uncached_note_makes_clear_safe_without_an_extra_confirmation(self) -> None:
        self.uncached_note()
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("笔记已保存")
        self.expect(self.page.locator(".note-draft-storage-warning")).to_be_hidden()
        dialogs = []
        self.page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        self.page.locator("#clear").click()
        self.expect(self.page.locator(".note-editor")).to_have_count(0)
        self.assertEqual(dialogs, [])
        self.assertEqual(self.history_records()[0]["note"], "没有暂存成功的唯一草稿")

    def test_note_cache_recovery_restores_normal_clear_and_draft_reopening(self) -> None:
        self.uncached_note()
        self.page.evaluate("window.noteCacheBroken = false")
        self.page.get_by_role("textbox", name="编辑学习笔记").fill("缓存恢复后完整暂存的草稿")
        self.expect(self.page.locator(".note-draft-storage-warning")).to_be_hidden()
        dialogs = []
        self.page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        self.page.locator("#clear").click()
        self.expect(self.page.locator(".note-editor")).to_have_count(0)
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-item").get_by_role("button", name="学习笔记", exact=True).click()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("缓存恢复后完整暂存的草稿")
        self.assertEqual(dialogs, [])
        self.assertEqual(self.history_records()[0]["note"], "")

    def test_review_with_uncached_note_does_not_count_until_draft_is_preserved(self) -> None:
        self.uncached_note(review=True)
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-meta")).to_contain_text("本题尚未记次")
        self.expect(self.page.locator("#review-panel")).to_be_visible()
        self.assertIsNone(self.history_records()[0].get("review"))
        with self.page.expect_download():
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertEqual(self.history_records()[0]["review"]["count"], 1)
        self.assertEqual(self.history_records()[0]["note"], "")
        self.assertEqual(len(self.backend.asks), 1)

    def test_other_tab_unfavorite_keeps_uncached_review_draft_until_note_is_saved(self) -> None:
        self.uncached_note(review=True)
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="★ 已收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.expect(self.page.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.expect(self.page.locator("#review-panel")).to_be_visible()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("没有暂存成功的唯一草稿")
        self.page.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertEqual(self.history_records()[0]["note"], "没有暂存成功的唯一草稿")
        self.assertIsNone(self.history_records()[0].get("review"))

    def test_failed_draft_download_does_not_remove_uncached_note_protection(self) -> None:
        self.uncached_note()
        self.page.evaluate("""() => {
          const create = URL.createObjectURL;
          URL.createObjectURL = function() { URL.createObjectURL = create; throw new Error('fixture blob failure'); };
        }""")
        self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_contain_text("草稿导出未完成")
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.locator("#clear").click()
        self.expect(self.page.get_by_role("textbox", name="编辑学习笔记")).to_have_value("没有暂存成功的唯一草稿")
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="导出草稿（含原问答）", exact=True).click()
        self.assertIn("没有暂存成功的唯一草稿", Path(download.value.path()).read_text())
        self.page.get_by_role("button", name="关闭编辑", exact=True).click()
        self.expect(self.page.locator(".note-editor")).to_have_count(0)

    def test_committed_review_advances_once_when_library_refresh_fails(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.fail_first_library_read_after_write()
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 1 条")
        self.assertEqual(self.history_records()[0]["review"]["count"], 1)

    def test_review_write_failure_does_not_advance_or_count_before_retry(self) -> None:
        self.open_page()
        self.saved_answer()
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.page.locator("#toggle-history").click()
        self.page.locator("#history-review-start").click()
        self.page.locator("#review-reveal").click()
        self.page.evaluate("""() => {
          const put = IDBObjectStore.prototype.put;
          IDBObjectStore.prototype.put = function(...args) {
            IDBObjectStore.prototype.put = put;
            throw new DOMException('fixture full storage', 'QuotaExceededError');
          };
        }""")
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-meta")).to_contain_text("复习进度未保存")
        self.expect(self.page.locator("#review-panel")).to_be_visible()
        self.assertIsNone(self.history_records()[0].get("review"))
        self.page.locator("#review-again").click()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.expect(self.page.locator("#history-status")).to_contain_text("已复习 1 条")
        self.assertEqual(self.history_records()[0]["review"]["count"], 1)
        self.page.reload()
        self.expect(self.page.locator("#review-panel")).to_be_hidden()
        self.assertEqual(self.history_records()[0]["review"]["count"], 1)

    def test_committed_restore_is_not_offered_again_when_library_refresh_fails(self) -> None:
        self.open_page()
        self.saved_answer()
        incoming = [dict(self.history_records()[0], id="committed-restore")]
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.page.locator("#history-backup-file").set_input_files(self.backup_file(incoming))
        self.expect(self.page.locator("#history-restore")).to_be_enabled()
        self.fail_first_library_read_after_write()
        self.page.locator("#history-restore").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("恢复完成：新增 1 条")
        self.expect(self.page.locator("#history-restore")).to_be_disabled()
        self.assertEqual(len(self.history_records()), 2)
        self.page.locator("#toggle-history").click()
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator(".history-item")).to_have_count(2)

    def test_committed_favorite_reports_only_read_failure_and_refresh_recovers(self) -> None:
        self.open_page()
        self.saved_answer()
        self.fail_first_library_read_after_write()
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator("#history-status")).to_contain_text("更新已保存")
        self.assertTrue(self.history_records()[0]["favorite"])
        self.page.locator("#toggle-history").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.expect(self.page.locator("#history-status")).to_be_hidden()

    def test_committed_answer_is_not_reported_unsaved_when_library_refresh_fails(self) -> None:
        self.open_page()
        self.expect(self.page.locator("#history-count")).to_contain_text("0 条记录")
        self.page.evaluate("""() => {
          const put = IDBObjectStore.prototype.put, getAll = IDBObjectStore.prototype.getAll;
          window.historyPutAttempts = 0; let failedRead = false;
          IDBObjectStore.prototype.put = function(...args) {
            window.historyPutAttempts++;
            return put.apply(this,args);
          };
          IDBObjectStore.prototype.getAll = function(...args) {
            if (window.historyPutAttempts && !failedRead) {
              failedRead = true;
              throw new DOMException('fixture read failure after commit','UnknownError');
            }
            return getAll.apply(this,args);
          };
        }""")
        self.backend.enqueue(Reply(final=[("result", answer_result())]))
        self.submit("写入成功但列表读取失败的问答")
        self.expect(self.page.locator("#send")).to_be_enabled()
        self.assertEqual(len(self.history_records()), 1)
        self.expect(self.page.locator(".answer")).to_contain_text("已保存到此浏览器")
        self.expect(self.page.get_by_role("button", name="重试保存", exact=True)).to_have_count(0)
        self.assertEqual(self.page.evaluate("window.historyPutAttempts"), 1)
        self.assertEqual(len(self.history_records()), 1)
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        self.assertEqual(self.page.evaluate("window.historyPutAttempts"), 1)
        self.assertTrue(self.history_records()[0]["favorite"])
        self.assertEqual(len(self.backend.asks), 1)

    def test_stale_favorite_buttons_apply_the_visible_intent_and_keep_other_notes(self) -> None:
        self.context.add_init_script("window.BroadcastChannel = undefined")
        self.open_page()
        self.saved_answer("按按钮显示的意图收藏")
        other = self.context.new_page()
        other.goto(self.base_url)
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="学习笔记").click()
        other.get_by_role("textbox", name="编辑学习笔记").fill("另一页保存的笔记不能覆盖")
        other.get_by_role("button", name="保存笔记", exact=True).click()
        self.expect(other.locator(".note-editor")).to_contain_text("笔记已保存")
        other.locator(".answer").get_by_role("button", name="☆ 收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        # The inactive page still offers add-to-favorites. It must not cancel
        # another tab's already-completed add when its click finally arrives.
        self.page.locator(".answer").get_by_role("button", name="☆ 收藏").dispatch_event("click")
        self.expect(self.page.locator(".answer").get_by_role("button", name="★ 已收藏")).to_be_visible()
        other.locator(".answer").get_by_role("button", name="★ 已收藏").click()
        self.expect(other.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.page.locator(".answer").get_by_role("button", name="★ 已收藏").dispatch_event("click")
        self.expect(self.page.locator(".answer").get_by_role("button", name="☆ 收藏")).to_be_visible()
        self.assertEqual(self.history_records()[0]["note"], "另一页保存的笔记不能覆盖")
        self.assertEqual(len(self.backend.asks), 1)

    def test_missing_markdown_asset_falls_back_to_safe_readable_text(self) -> None:
        self.page.route("**/assets/markdown.js", lambda route: route.fulfill(status=503, body=""))
        self.open_page()
        self.saved_answer()
        body = self.page.locator(".answer .body")
        self.expect(body).to_contain_text("暂按文本展示")
        self.expect(body).to_contain_text("<img src=x>")
        self.expect(body.locator("img")).to_have_count(0)
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()
        self.assertEqual(self.history_records()[0]["result"]["answer"], answer_result()["answer"])

    def test_missing_backup_asset_disables_restore_but_keeps_history_and_markdown_export(self) -> None:
        self.page.route("**/assets/history-backup.js", lambda route: route.fulfill(status=503, body=""))
        self.open_page()
        self.saved_answer("备份组件不可用时仍保留问答")
        self.page.locator("#toggle-history").click()
        self.page.locator(".history-backup summary").click()
        self.expect(self.page.locator("#history-backup-preview")).to_contain_text("备份组件未能加载")
        for selector in ("#history-backup-all", "#history-backup-filtered", "#history-backup-file", "#history-restore"):
            self.expect(self.page.locator(selector)).to_be_disabled()
        self.expect(self.page.locator(".history-item")).to_have_count(1)
        self.expect(self.page.locator("#history-export-md")).to_be_enabled()
        with self.page.expect_download() as collected:
            self.page.locator("#history-export-md").click()
        self.assertIn("备份组件不可用时仍保留问答", Path(collected.value.path()).read_text())
        self.page.locator("#toggle-history").click()
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="保存问答（含来源）", exact=True).click()
        self.assertIn("备份组件不可用时仍保留问答", Path(download.value.path()).read_text())
        self.page.reload()
        self.expect(self.page.locator(".answer .body")).to_contain_text("进程")
        self.assertEqual(len(self.backend.asks), 1)

    def test_no_evidence_record_exports_question_and_status_without_fake_answer(self) -> None:
        self.open_page()
        result = answer_result()
        result.update(status="no_evidence", answer=None, sources=[], citation_integrity=None,
                      message="本次未找到教材证据，请换个问题。")
        self.saved_answer("没有检索资料的问题", result=result)
        self.page.reload()
        self.expect(self.page.get_by_role("button", name="复制答案", exact=True)).to_have_count(0)
        with self.page.expect_download() as info:
            self.page.get_by_role("button", name="导出本次记录", exact=True).click()
        exported = Path(info.value.path()).read_text()
        self.assertIn("没有检索资料的问题", exported)
        self.assertIn(result["message"], exported)
        self.assertIn("本次记录未包含教材片段", exported)
        self.assertIn("本次没有完整的生成回答", exported)
        self.assertEqual(len(self.backend.asks), 1)

    def test_dark_narrow_screens_keep_controls_and_error_messages_readable(self) -> None:
        self.page.emulate_media(color_scheme="dark")
        self.page.set_viewport_size({"width": 320, "height": 800})
        self.backend.imports_enabled = True
        self.backend.access_code_required = True
        self.open_page()
        self.saved_answer()
        self.expect(self.page.locator("#send")).to_have_css("background-color", "rgb(59, 111, 216)")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.expect(self.page.locator(".reader-quote")).to_be_visible()
        self.capture_page("reader-dark-320.png")
        self.backend.inspection_status = 401
        self.page.goto(self.base_url + "/retrieval")
        self.expect(self.page.locator("#inspect-send")).to_be_enabled()
        self.page.locator("#inspect-book").select_option("os")
        self.page.locator("#inspect-query").fill("进程")
        self.page.locator("#inspect-send").click()
        self.expect(self.page.locator("#inspect-status")).to_have_css("color", "rgb(244, 163, 163)")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.backend.imports_enabled = False
        self.page.goto(self.base_url + "/imports")
        self.expect(self.page.locator("#import-status")).to_contain_text("只在本机服务提供")
        self.expect(self.page.locator("#import-status")).to_have_css("color", "rgb(244, 163, 163)")
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
        self.capture_page("imports-dark-320.png")

    def test_reader_explains_missing_source_order_and_keeps_exact_anchor(self) -> None:
        self.backend.reading_unordered = True
        self.open_page()
        self.saved_answer()
        self.page.get_by_role("button", name="对照原文", exact=True).click()
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("未提供顺序编号")
        self.expect(self.page.locator(".reader-passage")).to_have_count(1)
        self.expect(self.page.get_by_role("button", name="向前阅读", exact=True)).to_be_disabled()
        self.expect(self.page.get_by_role("button", name="向后阅读", exact=True)).to_be_disabled()
        self.expect(self.page.locator(".reader-quote")).to_have_text(answer_result()["sources"][0]["excerpt"])

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
        self.expect(self.page.locator("#book option")).to_have_count(4)
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

    def test_all_books_scope_asks_without_a_book_and_reads_each_source_from_its_own(self) -> None:
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_page(book_id="")
        self.assertEqual(
            self.page.locator("#book option").all_text_contents(),
            ["请选择教材", "全部教材", "操作系统", "数据库原理及应用"],
        )
        # Nothing is preselected: the wider scope is the reader's explicit choice.
        self.expect(self.page.locator("#book")).to_have_value("")
        self.expect(self.page.locator("#book-status")).to_be_hidden()
        self.page.locator("#book").select_option("*")
        self.expect(self.page.locator("#book-status")).to_have_text(
            "将在全部教材中检索，每条资料标明所属教材；跨教材回答质量尚未单独验收。"
        )
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))

        self.backend.enqueue(Reply(final=[("result", all_books_result())]))
        self.submit("进程和事务分别是什么？")
        answer = self.page.locator(".answer").last
        self.expect(answer).to_contain_text("已保存到此浏览器")
        self.expect_ready()
        # Leaving the book out is how the service is asked to search every textbook.
        self.assertEqual(self.backend.asks[-1]["payload"], {"query": "进程和事务分别是什么？"})
        self.assertEqual(
            answer.locator(".source").evaluate_all(
                "nodes => nodes.map(node => node.getAttribute('aria-label'))"
            ),
            ["资料 1 · 操作系统", "资料 2 · 数据库原理及应用"],
        )
        record = self.history_records()[0]
        self.assertEqual((record["book_id"], record["book_label"]), ("*", "全部教材"))
        self.assertEqual(
            [source["book_id"] for source in record["result"]["sources"]], ["os", "database"]
        )

        answer.get_by_role("button", name="对照原文", exact=True).click()
        self.page.get_by_label("选择引用原文", exact=True).select_option("2")
        self.page.get_by_role("button", name="读取相邻原文", exact=True).click()
        self.expect(self.page.locator(".reader-current")).to_contain_text("定位片段与引用时的原文一致")
        self.assertEqual(self.backend.read_books, ["database"])

        self.page.locator("#book").select_option("os")
        self.expect(self.page.locator("#book-status")).to_be_hidden()

    def test_all_books_records_filter_reopen_and_back_up_like_other_records(self) -> None:
        self.open_page()
        self.saved_answer("什么是进程？")
        self.saved_answer("进程和事务分别是什么？", "*", all_books_result())
        self.page.locator("#toggle-history").click()
        items = self.page.locator(".history-item")
        self.expect(items).to_have_count(2)
        # The first entry already means every record, so the scope gets its own name.
        self.assertEqual(
            sorted(self.page.locator("#history-book option").evaluate_all(
                "nodes => nodes.map(node => [node.value, node.textContent])"
            )),
            [["", "全部教材"], ["*", "跨教材提问"], ["os", "操作系统"]],
        )
        self.page.locator("#history-book").select_option("*")
        self.expect(items).to_have_count(1)
        self.expect(items.locator("h3")).to_have_text("进程和事务分别是什么？")
        self.expect(items).to_contain_text("全部教材")
        self.page.locator("#history-book").select_option("os")
        self.expect(items).to_have_count(1)
        self.expect(items.locator("h3")).to_have_text("什么是进程？")

        restored = self.page.evaluate(
            "records => RagHistoryBackup.parse(RagHistoryBackup.serialize(records))"
            ".map(record => [record.book_id, record.book_label])",
            self.history_records(),
        )
        self.assertEqual(sorted(restored), [["*", "全部教材"], ["os", "操作系统"]])

        self.page.locator("#toggle-history").click()
        self.page.locator("#book").select_option("os")
        self.page.locator(".answer").last.get_by_role("button", name="编辑原问题", exact=True).click()
        self.expect(self.page.locator("#input")).to_have_value("进程和事务分别是什么？")
        self.expect(self.page.locator("#book")).to_have_value("*")
        self.expect(self.page.locator("#book-status")).to_contain_text("将在全部教材中检索")
        self.backend.enqueue(Reply(final=[("result", all_books_result())]))
        self.page.locator("#send").click()
        self.expect(self.page.locator(".answer")).to_have_count(3)
        self.expect(self.page.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.assertEqual(self.backend.asks[-1]["payload"], {"query": "进程和事务分别是什么？"})

    def test_failed_all_books_question_retries_in_the_same_scope(self) -> None:
        self.backend.enqueue(Reply(final=[("error", {"status": "busy", "message": "服务繁忙，请稍后重试。"})]))
        self.backend.enqueue(Reply(final=[("result", all_books_result())]))
        self.open_page(book_id="*")
        self.submit("进程和事务分别是什么？")
        self.expect(self.page.locator(".notice.warn")).to_have_text("服务繁忙，请稍后重试。")
        self.expect_ready()
        self.page.locator("#book").select_option("os")
        self.page.get_by_role("button", name="重试本题").click()
        self.expect(self.page.locator(".answer").last).to_contain_text("已保存到此浏览器")
        self.expect(self.page.locator("#book")).to_have_value("*")
        self.expect(self.page.locator("#book-status")).to_contain_text("将在全部教材中检索")
        self.assertEqual(
            [ask["payload"] for ask in self.backend.asks], [{"query": "进程和事务分别是什么？"}] * 2
        )

    def test_single_textbook_offers_no_wider_scope_and_does_not_restore_one(self) -> None:
        self.open_page(book_id="*")
        self.saved_answer("进程和事务分别是什么？", "*", all_books_result())
        self.page.route("**/v1/books", lambda route: route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps([{"book_id": "database", "label": "数据库原理及应用"}]),
        ))
        self.page.reload()
        self.assertEqual(
            self.page.locator("#book option").all_text_contents(), ["请选择教材", "数据库原理及应用"]
        )
        self.expect(self.page.locator("#book")).to_have_value("")
        self.expect(self.page.locator("#book-status")).to_be_hidden()
        self.page.locator(".answer").get_by_role("button", name="编辑原问题", exact=True).click()
        self.expect(self.page.locator("#book")).to_have_value("")
        self.expect(self.page.locator("#book-status")).to_contain_text("原教材当前不可用")
        self.page.locator("#send").click()
        self.assertEqual(len(self.backend.asks), 1)


if __name__ == "__main__":
    unittest.main()
