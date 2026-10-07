"""Check the real SDK against a loopback completion server without a model."""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from openai import OpenAI

from rag_textbook_qa.llm import LLMClient, LLMGenerationIncompleteError


def run() -> dict:
    replies: deque[tuple[str | None, threading.Event]] = deque()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "stream": payload.get("stream")})
            reason, gate = replies.popleft()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            frame = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                     "model": "fixture", "choices": [{"index": 0,
                     "delta": {"content": "测试片段"}, "finish_reason": reason}]}
            try:
                self.wfile.write(("data: " + json.dumps(frame) + "\r\n\r\n").encode())
                self.wfile.flush()
                # No [DONE] or EOF until the client has finished processing the
                # terminal marker. A missing marker instead gets a normal EOF.
                if reason is not None:
                    gate.wait(10)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    gates = []
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with OpenAI(
            api_key="fixture-key", base_url=base_url, timeout=3, max_retries=0,
            http_client=httpx.Client(trust_env=False),
        ) as sdk:
            client = LLMClient(api_key="fixture-key", base_url=base_url, sdk_client=sdk, verbose=False)
            for reason in ("stop", "length", None):
                gate = threading.Event()
                gates.append(gate)
                replies.append((reason, gate))
                received = []
                try:
                    try:
                        received.extend(client.stream_answer("fixture question", raise_on_error=True))
                    except LLMGenerationIncompleteError:
                        assert reason != "stop", "A complete answer was rejected"
                    else:
                        assert reason == "stop", "An incomplete answer was accepted"
                    assert received == ["测试片段"]
                    assert not gate.is_set(), "The client waited for upstream EOF"
                finally:
                    gate.set()
        assert requests == [{"path": "/v1/chat/completions", "stream": True}] * 3
    finally:
        for gate in gates:
            gate.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        if thread.is_alive():
            raise RuntimeError("Loopback completion server did not stop")
    return {"complete": True, "backend": "loopback fixture with real OpenAI SDK; no model",
            "checks": {"stop_finishes_before_eof": True, "length_fails_before_eof": True,
                       "eof_without_terminal_marker_is_incomplete": True,
                       "one_upstream_request_per_case": True}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出文件已存在，请换一个路径")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
