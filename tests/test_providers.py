import io
import unittest
from unittest.mock import patch
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
from rag_textbook_qa.providers.remote import (
    FallbackEmbeddingProvider,
    RemoteEmbeddingProvider,
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
            patch("rag_textbook_qa.providers.remote.urlopen", side_effect=unauthorized),
            self.assertRaises(AuthenticationError),
        ):
            client.request("/health")

        with (
            patch(
                "rag_textbook_qa.providers.remote.urlopen",
                side_effect=URLError("offline"),
            ),
            self.assertRaises(TransientProviderError),
        ):
            client.request("/health")

    def test_http_client_rejects_non_ascii_token_before_building_request(self):
        with self.assertRaisesRegex(ProviderError, "ASCII"):
            RemoteWorkerClient("http://worker", token="中文-token", timeout=1)

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


if __name__ == "__main__":
    unittest.main()
