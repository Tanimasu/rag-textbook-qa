import unittest

from rag_textbook_qa.web.messages import answer_message


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


if __name__ == "__main__":
    unittest.main()
