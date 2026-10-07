"""Check truncated Worker responses over real loopback HTTP without models."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

from rag_textbook_qa.providers.base import (
    AuthenticationError,
    ModelIdentity,
    ModelMismatchError,
    ProviderProtocolError,
)
from rag_textbook_qa.providers.remote import (
    FallbackRerankerProvider,
    RemoteRerankerProvider,
    RemoteWorkerClient,
)


def run() -> dict:
    identity = ModelIdentity(task="reranker", model="fixture-model")
    state = {"mode": "cut_second_batch", "batches": [], "health_checks": 0,
             "redirect_destination_requests": 0, "redirect_destination_received_auth": False}

    class DestinationHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["redirect_destination_requests"] += 1
            state["redirect_destination_received_auth"] |= self.headers.get("Authorization") is not None
            body = json.dumps({"models": {"reranker": identity.as_dict()}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, payload, *, truncate=False):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:len(body) // 2] if truncate else body)
            self.wfile.flush()
            self.close_connection = True

        def do_GET(self):
            state["health_checks"] += 1
            if state["mode"] == "redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{destination.server_address[1]}/health")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if state["mode"] in {"auth", "model"}:
                self.respond(401 if state["mode"] == "auth" else 409, {"detail": "fixture rejection"}, truncate=True)
            else:
                restarting = state["mode"] == "cut_second_batch"
                self.respond(200, {"status": "ok", "protocol_version": "1",
                                   "device": "cuda" if restarting else "cpu",
                                   "platform": "Windows" if restarting else "Linux",
                                   "models": {"reranker": identity.as_dict()}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            documents = payload["documents"]
            state["batches"].append(list(documents))
            if state["mode"] == "malformed":
                # Complete HTTP framing, invalid provider payload: not a transport failure.
                self.respond(200, ["not an object"])
                return
            self.respond(200, {"fingerprint": identity.fingerprint, "scores": [0.1] * len(documents)},
                         truncate=state["mode"] == "cut_second_batch" and len(documents) < 128)

    destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)
    destination.daemon_threads = True
    destination_thread = threading.Thread(target=destination.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    destination_thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        client = RemoteWorkerClient(f"http://127.0.0.1:{server.server_address[1]}", token=None, timeout=2)
        primary = RemoteRerankerProvider(client, identity.model)
        fallback = MagicMock()
        fallback.identity = identity
        fallback.rerank.return_value = [0.9] * 150
        provider = FallbackRerankerProvider(primary, fallback)
        documents = [f"document-{index}" for index in range(150)]
        assert provider.rerank("question", documents) == [0.9] * 150
        assert [len(batch) for batch in state["batches"]] == [128, 22]
        assert [value for batch in state["batches"] for value in batch] == documents
        fallback.rerank.assert_called_once_with("question", documents)
        assert provider.telemetry.since(0)[-1].fallback_used
        state["mode"] = "healthy"
        assert provider.rerank("question", documents) == [0.1] * 150
        assert not provider.telemetry.since(0)[-1].fallback_used
        assert provider.telemetry.since(0)[-1].device == "cpu"
        assert provider.telemetry.since(0)[-1].platform == "Linux"
        assert state["health_checks"] == 2
        assert fallback.rerank.call_count == 1
        for mode, expected in (("auth", AuthenticationError), ("model", ModelMismatchError),
                               ("malformed", ProviderProtocolError)):
            state["mode"] = mode
            provider = FallbackRerankerProvider(RemoteRerankerProvider(client, identity.model), fallback)
            try:
                provider.rerank("question", documents)
            except expected:
                pass
            else:
                raise AssertionError(f"{mode} did not fail closed")
            assert fallback.rerank.call_count == 1
        state["mode"] = "redirect"
        authenticated = RemoteWorkerClient(client.base_url, token="fixture-token", timeout=2)
        provider = FallbackRerankerProvider(RemoteRerankerProvider(authenticated, identity.model), fallback)
        try:
            provider.rerank("question", documents)
        except ProviderProtocolError:
            pass
        else:
            raise AssertionError(
                f"redirect followed: destination_requests={state['redirect_destination_requests']}, "
                f"auth_forwarded={state['redirect_destination_received_auth']}"
            )
        assert state["redirect_destination_requests"] == 0
        assert not state["redirect_destination_received_auth"]
        assert fallback.rerank.call_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        destination.shutdown()
        destination.server_close()
        destination_thread.join(5)
        if thread.is_alive():
            raise RuntimeError("Loopback Worker did not stop")
        if destination_thread.is_alive():
            raise RuntimeError("Redirect destination did not stop")
    return {"complete": True, "backend": "local fake HTTP Worker; no models", "checks": {
        "truncated_second_batch_recomputes_full_input": True, "healthy_remote_recovers": True,
        "recovered_worker_health_and_device_are_refreshed": True,
        "truncated_401_does_not_fallback": True, "truncated_409_does_not_fallback": True,
        "malformed_provider_payload_does_not_fallback": True,
        "redirect_does_not_forward_auth_or_trigger_fallback": True,
    }}


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
