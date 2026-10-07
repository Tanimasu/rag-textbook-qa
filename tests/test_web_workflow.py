"""Exercise the ordinary UI flow without contacting a model or the Worker."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit UI extra is not installed")
class OrdinaryWebWorkflowTests(unittest.TestCase):
    def app(self, engine=None, error=None, results=None):
        from streamlit.testing.v1 import AppTest

        from rag_textbook_qa.web import services

        self.enterContext(
            patch.object(
                services, "load_available_books", return_value=[("操作系统", "os"), ("全部", None)]
            )
        )
        self.enterContext(patch.object(services, "load_ragas_results", return_value=results))
        loader = self.enterContext(
            patch.object(services, "load_engine", return_value=engine, side_effect=error)
        )
        return AppTest.from_file(str(ROOT / "src/rag_textbook_qa/web/app.py")).run(
            timeout=20
        ), loader

    def test_evaluation_view_preserves_original_question_ids_and_numeric_order(self):
        import pandas as pd

        for indices in (None, [2, 10]):
            with self.subTest(indices=indices):
                results = pd.DataFrame({"question": ["问题甲", "问题乙"], "faithfulness": [0.8, 0.6]})
                if indices is not None:
                    results["question_index"] = indices
                app, _ = self.app(results=results)
                self.assertFalse(app.exception)
                expected = ["Q1", "Q2"] if indices is None else ["Q2", "Q10"]
                self.assertEqual(list(app.dataframe[0].value["题号"]), expected)
                self.assertEqual(list(app.dataframe[1].value["题号"]), list(reversed(expected)))
                self.assertNotIn("question_index", app.dataframe[1].value.columns)

    def test_streaming_answer_sources_history_clear_and_defaults(self):
        engine = MagicMock()

        def ask(**kwargs):
            kwargs["on_answer_chunk"]("普通回答")
            return {
                "success": True,
                "answer": "普通回答【参考资料 1】",
                "context_sources": [
                    {
                        "book_name": "os",
                        "content": "实际输入片段",
                        "chapter": "第2章",
                        "citation_id": 1,
                    }
                ],
                "results": [{"book_name": "os", "content": "未发送的候选"}],
            }

        engine.ask.side_effect = ask
        app, loader = self.app(engine)
        self.assertFalse(app.exception)
        loader.assert_not_called()
        self.assertTrue(all(not toggle.value for toggle in app.toggle))
        app.chat_input[0].set_value("什么是进程？").run(timeout=20)
        self.assertFalse(app.exception)
        kwargs = engine.ask.call_args.kwargs
        self.assertFalse(kwargs["use_decomposition"])
        self.assertFalse(kwargs["verify_citations"])
        self.assertFalse(kwargs["use_hyde"])
        self.assertFalse(kwargs["use_adjacent_context"])
        self.assertEqual(kwargs["book_name"], "os")
        self.assertEqual(len(app.session_state["messages"]), 2)
        self.assertEqual(app.session_state["messages"][1]["sources"][0]["content"], "实际输入片段")
        app.run(timeout=20)
        self.assertFalse(app.exception)
        engine.ask.assert_called_once()
        next(b for b in app.button if b.label == "清空对话").click().run(timeout=20)
        self.assertEqual(app.session_state["messages"], [])

    def test_initialization_failure_is_readable_and_does_not_leak_details(self):
        app, _ = self.app(error=RuntimeError("secret endpoint"))
        app.chat_input[0].set_value("问题").run(timeout=20)
        self.assertFalse(app.exception)
        text = app.session_state["messages"][-1]["content"]
        self.assertIn("暂时无法", text)
        self.assertNotIn("secret", text)

    def test_retrieval_failure_and_empty_results_render_without_crashing(self):
        from rag_textbook_qa.providers.base import AuthenticationError

        engine = MagicMock()
        engine.ask.side_effect = AuthenticationError("secret token")
        app, _ = self.app(engine)
        app.chat_input[0].set_value("问题").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertNotIn("secret", app.session_state["messages"][-1]["content"])
        engine.ask.side_effect = None
        engine.ask.return_value = {
            "success": False,
            "answer": "没有找到相关内容",
            "context_sources": [],
        }
        app.chat_input[0].set_value("另一个问题").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["messages"][-1]["content"], "没有找到相关内容")
