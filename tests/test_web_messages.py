import unittest

from rag_textbook_qa.web.messages import answer_message, compute_trace_items


class WebMessageTests(unittest.TestCase):
    def test_answer_is_preferred_when_generation_succeeds(self):
        self.assertEqual(
            answer_message({"answer": "教材答案", "error": None}),
            "教材答案",
        )

    def test_generation_error_is_shown_instead_of_generic_fallback(self):
        message = answer_message(
            {
                "answer": None,
                "error": "LLM 不可用：必须设置 LLM_API_KEY",
                "success": False,
            }
        )

        self.assertIn("未能生成答案", message)
        self.assertIn("LLM_API_KEY", message)
        self.assertNotEqual(message, "抱歉，未能生成答案。")

    def test_very_long_provider_errors_are_bounded(self):
        message = answer_message(
            {
                "answer": "第三方客户端生成的冗长错误",
                "error": "x" * 1000,
                "success": False,
            }
        )

        self.assertLessEqual(len(message), 510)
        self.assertTrue(message.endswith("..."))

    def test_compute_trace_labels_remote_cuda_and_local_mps_fallback(self):
        remote = compute_trace_items(
            {
                "embedding": {
                    "backend": "remote",
                    "device": "cuda",
                    "platform": "Windows",
                    "elapsed_seconds": 0.14,
                    "calls": 1,
                    "fallback_used": False,
                },
                "reranker": {
                    "backend": "local",
                    "device": "mps",
                    "platform": "Darwin",
                    "elapsed_seconds": 1.23,
                    "calls": 1,
                    "fallback_used": True,
                },
                "retrieval_seconds": 1.5,
                "generation_seconds": 2,
                "total_seconds": 3.5,
            }
        )

        self.assertIn("远程 Worker（Windows） · CUDA", remote[0]["text"])
        self.assertIn("已回退到本地（macOS） · MPS", remote[1]["text"])
        self.assertEqual(remote[1]["kind"], "fallback")
        self.assertIn("总计 3.500 秒", remote[2]["text"])

    def test_compute_trace_is_empty_for_legacy_messages(self):
        self.assertEqual(compute_trace_items(None), [])


if __name__ == "__main__":
    unittest.main()
