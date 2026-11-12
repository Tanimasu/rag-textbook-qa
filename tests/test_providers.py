import io
import unittest
from http.client import IncompleteRead, RemoteDisconnected
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from rag_textbook_qa.providers import (
    AuthenticationError,
    ModelIdentity,
    ModelMismatchError,
    ProviderCall,
    ProviderError,
    ProviderProtocolError,
    ProviderTelemetry,
    TransientProviderError,
    provider_trace,
)
from rag_textbook_qa.providers.base import DEFAULT_QUERY_INSTRUCTION, validate_embeddings
from rag_textbook_qa.providers.config import ComputeSettings
from rag_textbook_qa.providers.factory import (
    create_embedding_provider,
    create_remote_cooldown,
    create_reranker_provider,
)
from rag_textbook_qa.providers.remote import (
    FallbackEmbeddingProvider,
    FallbackRerankerProvider,
    RemoteCooldown,
    RemoteEmbeddingProvider,
    RemoteRerankerProvider,
    RemoteWorkerClient,
)


class FakeRemoteClient:
    def __init__(self, model="embedding-model"):
        self.identity = ModelIdentity(
            task="embedding",
            model=model,
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        self.requests = []

    def request(self, path, *, method="GET", payload=None):
        self.requests.append((path, method, payload))
        if path == "/health":
            return {
                "status": "ok", "protocol_version": "1",
                "device": "cuda",
                "platform": "Windows",
                "models": {"embedding": self.identity.as_dict()},
            }
        return {
            "fingerprint": self.identity.fingerprint,
            "embeddings": [[float(index), 1.0] for index, _ in enumerate(payload["texts"])],
        }


class StubEmbeddingProvider:
    def __init__(self, identity, outcome):
        self._identity = identity
        self.outcome = outcome
        self.calls = 0

    @property
    def identity(self):
        return self._identity

    def _run(self):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def embed_documents(self, texts):
        return self._run()

    def embed_queries(self, texts):
        return self._run()


class ProviderTests(unittest.TestCase):
    def test_unhealthy_or_incompatible_health_stops_before_inference_or_fallback(self):
        invalid_states = [
            {"status": "starting", "protocol_version": "1"},
            {"protocol_version": "1"},
            {"status": "ok", "protocol_version": "2"},
            {"status": "ok"},
            {"status": "ok", "protocol_version": 1},
        ]
        for primary_type, fallback_type in (
            (RemoteEmbeddingProvider, FallbackEmbeddingProvider),
            (RemoteRerankerProvider, FallbackRerankerProvider),
        ):
            for health in invalid_states:
                with self.subTest(provider=primary_type.__name__, health=health):
                    client = MagicMock()
                    primary = primary_type(client, "model")
                    client.request.return_value = {
                        **health, "device": "cuda",
                        "models": {primary.identity.task: primary.identity.as_dict()},
                        "fingerprint": primary.identity.fingerprint,
                        "embeddings": [[1.0, 0.0]], "scores": [0.5],
                    }
                    fallback = MagicMock(identity=primary.identity)
                    provider = fallback_type(primary, fallback)
                    with self.assertRaises(ProviderProtocolError):
                        if primary.identity.task == "embedding":
                            provider.embed_queries(["question"])
                        else:
                            provider.rerank("question", ["document"])
                    client.request.assert_called_once_with("/health")
                    fallback.embed_queries.assert_not_called()
                    fallback.rerank.assert_not_called()
                    self.assertFalse(primary._health_verified)

    def test_remote_health_is_rechecked_after_an_interrupted_worker(self):
        for provider_type in (RemoteEmbeddingProvider, RemoteRerankerProvider):
            with self.subTest(provider=provider_type.__name__):
                client = MagicMock()
                provider = provider_type(client, "model")
                payload = {"fingerprint": provider.identity.fingerprint,
                           "embeddings": [[1.0, 0.0]], "scores": [0.9]}
                def health(device, platform, provider=provider):
                    return {"status": "ok", "protocol_version": "1",
                            "device": device, "platform": platform,
                            "models": {provider.identity.task: provider.identity.as_dict()}}
                client.request.side_effect = [
                    health("cuda", "Windows"), payload,
                    TransientProviderError("worker restarted"),
                    health("cpu", "Linux"), payload,
                ]
                def invoke(provider=provider):
                    if provider.identity.task == "embedding":
                        return provider.embed_queries(["question"])
                    return provider.rerank("question", ["document"])
                invoke()
                with self.assertRaises(TransientProviderError):
                    invoke()
                invoke()
                calls = provider.telemetry.since(0)
                self.assertEqual(calls[0].device, "cuda")
                self.assertEqual(calls[-1].device, "cpu")
                self.assertEqual(calls[-1].platform, "Linux")
                self.assertEqual(sum(call.args[0] == "/health" for call in client.request.call_args_list), 2)

    def test_local_reranker_batches_without_reordering_documents(self):
        provider = create_reranker_provider(ComputeSettings(device="cpu", reranker_batch_size=8))
        scores = [0.2, 0.9, 0.1]
        with patch.object(provider, "_load") as load:
            load.return_value.predict.return_value = scores
            self.assertEqual(provider.rerank("query", ["first", "second", "third"]), scores)
        load.return_value.predict.assert_called_once_with(
            [("query", "first"), ("query", "second"), ("query", "third")],
            batch_size=8, show_progress_bar=False,
        )
        self.assertEqual(len(provider.telemetry.since(0)), 1)

    def test_provider_telemetry_is_isolated_by_request_trace(self):
        telemetry = ProviderTelemetry()
        call = ProviderCall(
            task="embedding",
            backend="remote",
            model="model",
            device="cuda",
            platform="Windows",
            elapsed_seconds=0.1,
            success=True,
        )

        with provider_trace() as first_trace:
            telemetry.record(call)
        with provider_trace() as second_trace:
            telemetry.record(call)

        self.assertEqual(telemetry.for_trace(first_trace), [call])
        self.assertEqual(telemetry.for_trace(second_trace), [call])

    def test_non_finite_embeddings_are_rejected(self):
        with self.assertRaisesRegex(ProviderProtocolError, "NaN"):
            validate_embeddings([[float("nan")]], 1)

    def test_http_client_classifies_auth_and_network_errors(self):
        client = RemoteWorkerClient("http://worker", token="secret", timeout=1)
        unauthorized = HTTPError(
            "http://worker/health",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"detail":"bad token"}'),
        )
        with (
            patch.object(client._opener, "open", side_effect=unauthorized),
            self.assertRaises(AuthenticationError),
        ):
            client.request("/health")

        with (
            patch.object(
                client._opener, "open",
                side_effect=URLError("offline"),
            ),
            self.assertRaises(TransientProviderError),
        ):
            client.request("/health")

    def test_http_client_rejects_non_ascii_token_before_building_request(self):
        with self.assertRaisesRegex(ProviderError, "ASCII"):
            RemoteWorkerClient("http://worker", token="中文-token", timeout=1)

    def test_interrupted_response_is_transient_and_triggers_query_fallback(self):
        client = RemoteWorkerClient("http://worker", token=None, timeout=1)
        primary = RemoteEmbeddingProvider(client, "embedding-model")
        primary._health_verified = True
        fallback = StubEmbeddingProvider(primary.identity, [[1.0, 2.0]])
        provider = FallbackEmbeddingProvider(primary, fallback)
        for interruption in (IncompleteRead(b'{"embeddings":', 20),
                             RemoteDisconnected("closed"), ConnectionResetError("reset")):
            response = MagicMock()
            response.__enter__.return_value.read.side_effect = interruption
            with self.subTest(interruption=type(interruption).__name__), patch.object(
                client._opener, "open", return_value=response,
            ):
                self.assertEqual(provider.embed_queries(["question"]), [[1.0, 2.0]])
                event = provider.telemetry.since(0)[-1]
                self.assertTrue(event.fallback_used)
                self.assertEqual(event.error_category, "TransientProviderError")
        self.assertEqual(fallback.calls, 3)

    def test_truncated_auth_error_body_still_fails_closed_without_fallback(self):
        client = RemoteWorkerClient("http://worker", token="secret", timeout=1)
        primary = RemoteEmbeddingProvider(client, "embedding-model")
        fallback = StubEmbeddingProvider(primary.identity, [[1.0, 2.0]])
        provider = FallbackEmbeddingProvider(primary, fallback)
        error = HTTPError("http://worker/health", 401, "Unauthorized", {}, None)
        error.read = MagicMock(side_effect=IncompleteRead(b'{"detail":', 10))
        error.close = MagicMock()
        with (
            patch.object(client._opener, "open", side_effect=error),
            self.assertRaises(AuthenticationError),
        ):
            provider.embed_queries(["question"])
        self.assertEqual(fallback.calls, 0)
        error.close.assert_called_once()

    def test_client_timeout_must_be_finite_and_positive_before_connecting(self):
        for timeout in (float("nan"), float("inf"), -1, 0):
            with self.subTest(timeout=timeout), self.assertRaises(ProviderError):
                RemoteWorkerClient("http://worker", token=None, timeout=timeout)
            with self.subTest(connect_timeout=timeout), self.assertRaises(ProviderError):
                RemoteWorkerClient(
                    "http://worker", token=None, timeout=1, connect_timeout=timeout
                )

    def test_remote_embedding_checks_health_once_and_preserves_input_type(self):
        client = FakeRemoteClient()
        provider = RemoteEmbeddingProvider(client, "embedding-model")

        self.assertEqual(provider.embed_queries(["q"]), [[0.0, 1.0]])
        self.assertEqual(provider.embed_documents(["a", "b"]), [[0.0, 1.0], [1.0, 1.0]])

        self.assertEqual([request[0] for request in client.requests].count("/health"), 1)
        self.assertEqual(client.requests[1][2]["input_type"], "query")
        self.assertEqual(client.requests[2][2]["input_type"], "document")
        events = provider.telemetry.since(0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1].backend, "remote")
        self.assertEqual(events[-1].device, "cuda")
        self.assertEqual(events[-1].platform, "Windows")
        self.assertTrue(events[-1].success)

    def test_remote_embedding_fails_fast_on_model_mismatch(self):
        provider = RemoteEmbeddingProvider(FakeRemoteClient("other-model"), "expected-model")

        with self.assertRaises(ModelMismatchError):
            provider.embed_queries(["q"])

    def test_fallback_only_handles_transient_failures(self):
        identity = ModelIdentity(
            task="embedding",
            model="model",
            normalized=True,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        primary = StubEmbeddingProvider(identity, TransientProviderError("offline"))
        fallback = StubEmbeddingProvider(identity, [[1.0, 2.0]])
        provider = FallbackEmbeddingProvider(primary, fallback)

        self.assertEqual(provider.embed_queries(["q"]), [[1.0, 2.0]])
        self.assertEqual(fallback.calls, 1)
        self.assertIsInstance(provider.telemetry, ProviderTelemetry)
        event = provider.telemetry.since(0)[-1]
        self.assertEqual(event.backend, "local")
        self.assertTrue(event.fallback_used)
        self.assertEqual(event.error_category, "TransientProviderError")
        self.assertTrue(event.success)

        auth_primary = StubEmbeddingProvider(identity, AuthenticationError("bad token"))
        provider = FallbackEmbeddingProvider(auth_primary, fallback)
        with self.assertRaises(AuthenticationError):
            provider.embed_queries(["q"])


class FakeSocket:
    """Just enough of a socket for http.client to send a request and read a reply."""

    def __init__(self, reply=b'{"status":"ok"}'):
        self.reply = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(reply)}\r\nConnection: close\r\n\r\n".encode()
            + reply
        )
        self.timeouts = []

    def setsockopt(self, *args):
        pass

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendall(self, data):
        pass

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.reply)

    def close(self):
        pass


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class StubRerankerProvider:
    def __init__(self, identity, outcome):
        self.identity = identity
        self.outcome = outcome
        self.calls = 0

    def rerank(self, query, documents):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


EMBEDDING_IDENTITY = ModelIdentity(
    task="embedding", model="model", normalized=True,
    query_instruction=DEFAULT_QUERY_INSTRUCTION,
)
RERANKER_IDENTITY = ModelIdentity(task="reranker", model="model")


class RemoteAvailabilityTests(unittest.TestCase):
    def test_reaching_the_worker_is_bounded_separately_from_waiting_for_its_answer(self):
        for scheme in ("http", "https"):
            with self.subTest(scheme=scheme):
                client = RemoteWorkerClient(
                    f"{scheme}://worker.test:8765", token=None, timeout=120, connect_timeout=3
                )
                connected = FakeSocket()
                with (
                    patch("socket.create_connection", return_value=connected) as connect,
                    patch("ssl.SSLContext.wrap_socket", side_effect=lambda sock, **_: sock),
                ):
                    self.assertEqual(client.request("/health"), {"status": "ok"})
                # http.client passes (address, timeout, source_address) positionally.
                self.assertEqual(connect.call_args.args[1], 3)
                # Inference may take minutes, so the reply keeps the long limit.
                self.assertEqual(connected.timeouts, [120])

    def test_unreachable_worker_is_reported_after_the_connect_timeout(self):
        for timeout, connect_timeout, expected in ((120, 3, 3), (2, 5, 2)):
            with self.subTest(timeout=timeout, connect_timeout=connect_timeout):
                client = RemoteWorkerClient(
                    "http://worker.test:8765", token="secret", timeout=timeout,
                    connect_timeout=connect_timeout,
                )
                self.assertEqual(client.connect_timeout, expected)
                with (
                    patch("socket.create_connection", side_effect=TimeoutError("timed out")) as connect,
                    self.assertRaises(TransientProviderError) as raised,
                ):
                    client.request("/health")
                self.assertEqual(connect.call_args.args[1], expected)
                self.assertNotIn("secret", str(raised.exception))
                self.assertNotIn("worker.test", str(raised.exception))

    def test_cooldown_answers_locally_without_asking_the_failed_worker_again(self):
        clock = FakeClock()
        primary = StubEmbeddingProvider(EMBEDDING_IDENTITY, TransientProviderError("offline"))
        fallback = StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0, 2.0]])
        provider = FallbackEmbeddingProvider(
            primary, fallback, cooldown=RemoteCooldown(60, clock=clock)
        )

        for _ in range(3):
            self.assertEqual(provider.embed_queries(["q"]), [[1.0, 2.0]])
            clock.now += 10
        self.assertEqual(provider.embed_documents(["hypothetical passage"]), [[1.0, 2.0]])
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 4)
        for event in provider.telemetry.since(0):
            self.assertEqual(event.backend, "local")
            self.assertTrue(event.fallback_used)
            self.assertTrue(event.success)
            self.assertEqual(event.error_category, "TransientProviderError")

        # The wait is counted from the failure, not from the last skipped call.
        clock.now = 1000.0 + 60
        provider.embed_queries(["q"])
        self.assertEqual(primary.calls, 2)
        provider.embed_queries(["q"])
        self.assertEqual(primary.calls, 2)

    def test_recovered_worker_is_used_again_once_the_cooldown_ends(self):
        clock = FakeClock()
        primary = StubRerankerProvider(RERANKER_IDENTITY, TransientProviderError("offline"))
        fallback = StubRerankerProvider(RERANKER_IDENTITY, [0.1])
        provider = FallbackRerankerProvider(
            primary, fallback, cooldown=RemoteCooldown(60, clock=clock)
        )
        self.assertEqual(provider.rerank("q", ["document"]), [0.1])

        primary.outcome = [0.9]
        clock.now += 59
        self.assertEqual(provider.rerank("q", ["document"]), [0.1])
        clock.now += 1
        for _ in range(2):
            self.assertEqual(provider.rerank("q", ["document"]), [0.9])
        self.assertEqual(primary.calls, 3)
        self.assertEqual(fallback.calls, 2)
        event = provider.telemetry.since(0)[-1]
        self.assertEqual(event.backend, "remote")
        self.assertFalse(event.fallback_used)
        self.assertIsNone(event.error_category)

    def test_one_cooldown_covers_embedding_and_reranking_on_the_same_worker(self):
        cooldown = RemoteCooldown(60, clock=FakeClock())
        embedding_primary = StubEmbeddingProvider(
            EMBEDDING_IDENTITY, TransientProviderError("offline")
        )
        reranker_primary = StubRerankerProvider(
            RERANKER_IDENTITY, TransientProviderError("offline")
        )
        embedding = FallbackEmbeddingProvider(
            embedding_primary, StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0]]),
            cooldown=cooldown,
        )
        reranker = FallbackRerankerProvider(
            reranker_primary, StubRerankerProvider(RERANKER_IDENTITY, [0.5]),
            cooldown=cooldown,
        )

        self.assertEqual(embedding.embed_queries(["q"]), [[1.0]])
        self.assertEqual(reranker.rerank("q", ["document"]), [0.5])
        self.assertEqual(embedding_primary.calls, 1)
        self.assertEqual(reranker_primary.calls, 0)
        self.assertTrue(reranker.telemetry.since(0)[-1].fallback_used)

    def test_cooldown_starts_only_for_transient_failures(self):
        for error in (
            AuthenticationError("bad token"),
            ModelMismatchError("other model"),
            ProviderProtocolError("malformed"),
        ):
            with self.subTest(error=type(error).__name__):
                cooldown = RemoteCooldown(60, clock=FakeClock())
                primary = StubEmbeddingProvider(EMBEDDING_IDENTITY, error)
                fallback = StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0]])
                provider = FallbackEmbeddingProvider(primary, fallback, cooldown=cooldown)
                for _ in range(2):
                    with self.assertRaises(type(error)):
                        provider.embed_queries(["q"])
                self.assertEqual(primary.calls, 2)
                self.assertEqual(fallback.calls, 0)
                self.assertFalse(cooldown.active())

    def test_rejection_after_a_cooldown_is_reported_instead_of_answered_locally(self):
        clock = FakeClock()
        primary = StubEmbeddingProvider(EMBEDDING_IDENTITY, TransientProviderError("offline"))
        fallback = StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0]])
        provider = FallbackEmbeddingProvider(
            primary, fallback, cooldown=RemoteCooldown(60, clock=clock)
        )
        provider.embed_queries(["q"])

        # The Worker came back with a different token while queries ran locally.
        primary.outcome = AuthenticationError("bad token")
        clock.now += 60
        for _ in range(2):
            with self.assertRaises(AuthenticationError):
                provider.embed_queries(["q"])
        self.assertEqual(primary.calls, 3)
        self.assertEqual(fallback.calls, 1)

    def test_failing_fallback_is_reported_during_a_cooldown(self):
        primary = StubEmbeddingProvider(EMBEDDING_IDENTITY, TransientProviderError("offline"))
        fallback = StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0]])
        provider = FallbackEmbeddingProvider(
            primary, fallback, cooldown=RemoteCooldown(60, clock=FakeClock())
        )
        provider.embed_queries(["q"])
        fallback.outcome = RuntimeError("local model failed")
        with self.assertRaisesRegex(RuntimeError, "local model failed"):
            provider.embed_queries(["q"])
        event = provider.telemetry.since(0)[-1]
        self.assertFalse(event.success)
        self.assertTrue(event.fallback_used)
        self.assertEqual(event.error_category, "RuntimeError")

    def test_zero_cooldown_and_no_cooldown_ask_the_worker_every_time(self):
        for cooldown in (None, RemoteCooldown(0, clock=FakeClock())):
            with self.subTest(cooldown=cooldown is not None):
                primary = StubEmbeddingProvider(
                    EMBEDDING_IDENTITY, TransientProviderError("offline")
                )
                fallback = StubEmbeddingProvider(EMBEDDING_IDENTITY, [[1.0]])
                provider = FallbackEmbeddingProvider(primary, fallback, cooldown=cooldown)
                for _ in range(3):
                    provider.embed_queries(["q"])
                self.assertEqual(primary.calls, 3)

    def test_cooldown_length_must_be_finite_and_not_negative(self):
        for seconds in (float("nan"), float("inf"), -1, True, "60"):
            with self.subTest(seconds=seconds), self.assertRaises(ProviderError):
                RemoteCooldown(seconds)

    def test_only_query_fallback_remembers_worker_failures(self):
        remote = ComputeSettings(
            backend="remote", remote_url="http://127.0.0.1:9", device="cpu",
            query_fallback_to_local=True, remote_connect_timeout_seconds=2,
            remote_cooldown_seconds=45,
        )
        self.assertIsNone(create_remote_cooldown(ComputeSettings(device="cpu")))
        self.assertIsNone(
            create_remote_cooldown(
                ComputeSettings(backend="remote", remote_url="http://127.0.0.1:9")
            )
        )

        # Index builds never fall back, so they get no memory of earlier failures.
        for factory, plain in (
            (create_embedding_provider, RemoteEmbeddingProvider),
            (create_reranker_provider, RemoteRerankerProvider),
        ):
            with self.subTest(factory=factory.__name__):
                building = factory(remote)
                self.assertIsInstance(building, plain)
                self.assertEqual(building.client.connect_timeout, 2)
                self.assertFalse(hasattr(building, "cooldown"))

                querying = factory(remote, allow_query_fallback=True)
                self.assertEqual(querying.cooldown.seconds, 45)
                self.assertEqual(querying.primary.client.connect_timeout, 2)

        shared = create_remote_cooldown(remote)
        embedding = create_embedding_provider(remote, allow_query_fallback=True, cooldown=shared)
        reranker = create_reranker_provider(remote, allow_query_fallback=True, cooldown=shared)
        self.assertIs(embedding.cooldown, shared)
        self.assertIs(reranker.cooldown, shared)


if __name__ == "__main__":
    unittest.main()
