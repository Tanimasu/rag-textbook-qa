import unittest

from rag_textbook_qa.providers import ComputeSettings, ProviderError


class ComputeSettingsTests(unittest.TestCase):
    def test_reranker_experimental_configuration_is_validated_and_reported(self):
        defaults = ComputeSettings.from_env({})
        self.assertIsNone(defaults.reranker_max_length)
        self.assertEqual(defaults.reranker_dtype, "float32")
        settings = ComputeSettings.from_env({"RAG_QA_RERANK_MAX_LENGTH": "384",
                                             "RAG_QA_RERANK_DTYPE": "FLOAT16"})
        self.assertEqual(settings.safe_summary()["reranker_max_length"], 384)
        self.assertEqual(settings.safe_summary()["reranker_dtype"], "float16")
        for value in ("0", "-1", "1.5", "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "MAX_LENGTH"):
                ComputeSettings.from_env({"RAG_QA_RERANK_MAX_LENGTH": value})
        with self.assertRaisesRegex(ProviderError, "DTYPE"):
            ComputeSettings.from_env({"RAG_QA_RERANK_DTYPE": "int8"})

    def test_remote_timeout_rejects_non_finite_values(self):
        for value in ("nan", "inf", "-inf", "1e999", "0", "-1"):
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "REMOTE_TIMEOUT"):
                ComputeSettings.from_env({"RAG_QA_REMOTE_TIMEOUT": value})

    def test_connect_timeout_and_cooldown_are_validated_and_reported(self):
        defaults = ComputeSettings.from_env({})
        self.assertEqual(defaults.remote_connect_timeout_seconds, 5)
        self.assertEqual(defaults.remote_cooldown_seconds, 60)

        settings = ComputeSettings.from_env(
            {"RAG_QA_REMOTE_CONNECT_TIMEOUT": "2.5", "RAG_QA_REMOTE_COOLDOWN": "0"}
        )
        self.assertEqual(settings.remote_connect_timeout_seconds, 2.5)
        # Zero is the documented way to ask the Worker again on every query.
        self.assertEqual(settings.remote_cooldown_seconds, 0)
        self.assertEqual(settings.safe_summary()["remote_connect_timeout_seconds"], 2.5)
        self.assertEqual(settings.safe_summary()["remote_cooldown_seconds"], 0)

        for value in ("nan", "inf", "0", "-1", "soon"):
            with self.subTest(connect_timeout=value), self.assertRaisesRegex(
                ProviderError, "REMOTE_CONNECT_TIMEOUT"
            ):
                ComputeSettings.from_env({"RAG_QA_REMOTE_CONNECT_TIMEOUT": value})
        for value in ("nan", "inf", "-1", "later"):
            with self.subTest(cooldown=value), self.assertRaisesRegex(
                ProviderError, "REMOTE_COOLDOWN"
            ):
                ComputeSettings.from_env({"RAG_QA_REMOTE_COOLDOWN": value})

    def test_default_is_local_and_does_not_require_remote_configuration(self):
        settings = ComputeSettings.from_env({})

        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.device, "auto")
        self.assertIsNone(settings.remote_url)
        self.assertFalse(settings.query_fallback_to_local)
        self.assertIsNone(settings.reranker_batch_size)

    def test_reranker_batch_size_is_positive_and_reported(self):
        settings = ComputeSettings.from_env({"RAG_QA_RERANK_BATCH_SIZE": "8"})
        self.assertEqual(settings.reranker_batch_size, 8)
        self.assertEqual(settings.safe_summary()["reranker_batch_size"], 8)
        for value in ("0", "-1", "1.5", "", "many"):
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "RERANK_BATCH_SIZE"):
                ComputeSettings.from_env({"RAG_QA_RERANK_BATCH_SIZE": value})

    def test_remote_tailscale_address_requires_token(self):
        with self.assertRaisesRegex(ProviderError, "RAG_QA_WORKER_TOKEN"):
            ComputeSettings.from_env(
                {
                    "RAG_QA_COMPUTE_BACKEND": "remote",
                    "RAG_QA_REMOTE_URL": "http://100.64.0.10:8765",
                }
            )

    def test_remote_configuration_is_normalized_without_exposing_token(self):
        settings = ComputeSettings.from_env(
            {
                "RAG_QA_COMPUTE_BACKEND": "REMOTE",
                "RAG_QA_REMOTE_URL": "http://100.64.0.10:8765/",
                "RAG_QA_WORKER_TOKEN": "secret-token",
                "RAG_QA_REMOTE_TIMEOUT": "45",
                "RAG_QA_QUERY_FALLBACK_TO_LOCAL": "yes",
            }
        )

        self.assertEqual(settings.remote_url, "http://100.64.0.10:8765")
        self.assertEqual(settings.remote_timeout_seconds, 45)
        self.assertTrue(settings.query_fallback_to_local)
        self.assertNotIn("secret-token", repr(settings.safe_summary()))

    def test_localhost_remote_worker_can_run_without_token(self):
        settings = ComputeSettings.from_env(
            {
                "RAG_QA_COMPUTE_BACKEND": "remote",
                "RAG_QA_REMOTE_URL": "http://127.0.0.1:8765",
            }
        )

        self.assertIsNone(settings.remote_token)

    def test_worker_token_rejects_non_ascii_and_whitespace(self):
        base = {
            "RAG_QA_COMPUTE_BACKEND": "remote",
            "RAG_QA_REMOTE_URL": "http://100.64.0.10:8765",
        }
        for token in ("中文-token", " leading", "trailing ", "two words"):
            with self.subTest(token=token), self.assertRaisesRegex(
                ProviderError, "RAG_QA_WORKER_TOKEN"
            ):
                ComputeSettings.from_env({**base, "RAG_QA_WORKER_TOKEN": token})


if __name__ == "__main__":
    unittest.main()
