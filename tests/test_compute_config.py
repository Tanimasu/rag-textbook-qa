import unittest

from rag_textbook_qa.providers import ComputeSettings, ProviderError


class ComputeSettingsTests(unittest.TestCase):
    def test_default_is_local_and_does_not_require_remote_configuration(self):
        settings = ComputeSettings.from_env({})

        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.device, "auto")
        self.assertIsNone(settings.remote_url)
        self.assertFalse(settings.query_fallback_to_local)

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
