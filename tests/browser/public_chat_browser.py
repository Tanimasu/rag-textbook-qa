"""Run the shipped public page in Chromium against an isolated local fake backend.

Run explicitly: python tests/browser/public_chat_browser.py -v. This file is not
part of ordinary unittest discovery and needs the optional ``browser`` group.
Only the backend is fake: the browser executes the unchanged HTML and JavaScript.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/rag_textbook_qa/api/static/index.html"


def answer_result(answer_id: str = "answer-1") -> dict[str, Any]:
    return {
        "status": "answered",
        "answer_id": answer_id,
        "answer": "## 进程\n**进程**是程序的一次执行。【参考资料 1】\n\n<img src=x>",
        "sources": [
            {
                "citation_id": 1,
                "book": "操作系统",
                "section": "第二章 进程管理",
                "excerpt": "进程是程序在一个数据集合上的一次执行。",
                "truncated": False,
            }
        ],
        "citation_integrity": {"status": "linked"},
        "timing": {"retrieval_seconds": 0.1, "total_seconds": 0.2},
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
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
