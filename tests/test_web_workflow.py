"""Exercise the ordinary UI flow without contacting a model or the Worker."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit UI extra is not installed")
class OrdinaryWebWorkflowTests(unittest.TestCase):
    def test_web_evaluations_keep_previous_runs_and_load_latest_completed_csv(self):
        import pandas as pd

        from rag_textbook_qa.web import services

        with tempfile.TemporaryDirectory() as directory:
            paths = MagicMock()
            paths.evaluations = Path(directory) / "evaluations"
            paths.evaluation_data = Path(directory) / "data"
            paths.evaluations.mkdir()
            legacy = paths.evaluations / "ragas_evaluation_results.csv"
            original = b"question,faithfulness\nlegacy,0.1\n"
            legacy.write_bytes(original)
            os.utime(legacy, (1, 1))
            destinations = []

            def evaluate(engine, questions, output):
                destination = Path(output)
                self.assertFalse(list(destination.iterdir()))
                destinations.append(destination)
                frame = pd.DataFrame({"question": [f"run-{len(destinations)}"],
                                      "faithfulness": [0.8]})
                frame.to_csv(destination / "ragas_evaluation_results.csv", index=False, mode="x")
                os.utime(destination / "ragas_evaluation_results.csv",
                         (len(destinations) + 1, len(destinations) + 1))
                return frame

            with (
                patch.object(services, "_settings", return_value=MagicMock(paths=paths)),
                patch.object(services, "load_engine", return_value=MagicMock()),
                patch("rag_textbook_qa.evaluation.load_test_questions", return_value=[{"question": "问题"}]),
                patch("rag_textbook_qa.evaluation.run_evaluation", side_effect=evaluate),
            ):
                self.assertEqual(services.load_ragas_results()["question"].tolist(), ["legacy"])
                for count in (1, 2):
                    self.assertEqual(services.run_ragas_evaluation()["question"].tolist(), [f"run-{count}"])
                # A newer unfinished run must not hide the latest saved results.
                (paths.evaluations / "ragas-runs" / "ragas-unfinished").mkdir()
                self.assertEqual(services.load_ragas_results()["question"].tolist(), ["run-2"])
            self.assertNotEqual(destinations[0], destinations[1])
            self.assertEqual(legacy.read_bytes(), original)
            self.assertTrue(all((path / "ragas_evaluation_results.csv").exists() for path in destinations))

    def app(self, engine=None, error=None, results=None, result_reader=None):
        from streamlit.testing.v1 import AppTest

        from rag_textbook_qa.web import services

        self.enterContext(
            patch.object(
                services, "load_available_books", return_value=[("操作系统", "os"), ("全部", None)]
            )
        )
        self.enterContext(patch.object(services, "load_ragas_results",
                                      new=result_reader or MagicMock(return_value=results)))
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

    def test_unscored_questions_remain_visible_and_search_uses_literal_text(self):
        import pandas as pd

        results = pd.DataFrame({"question": ["数组 a[0]", "未评分的问题"],
                                "faithfulness": [0.8, None]})
        app, _ = self.app(results=results)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[1].value), 2)
        self.assertEqual(set(app.dataframe[1].value["有效指标"]), {"1/1", "0/1"})
        self.assertTrue(any("1/2 题有得分" in item.value for item in app.caption))
        search = next(w for w in app.text_input if w.label == "搜索问题")
        search.set_value("[").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[1].value["问题"].tolist(), ["数组 a[0]"])
        next(w for w in app.text_input if w.label == "搜索问题").set_value("").run(timeout=20)
        next(w for w in app.slider if w.label == "最高平均分").set_value(0.5).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[1].value["问题"].tolist(), ["未评分的问题"])
        self.assertEqual(app.dataframe[1].value["平均分"].tolist(), ["未评分"])
        self.assertEqual(results["faithfulness"].tolist()[0], 0.8)
        self.assertTrue(pd.isna(results["faithfulness"].iloc[1]))

    def test_unreadable_results_can_be_reloaded_without_running_paid_evaluation(self):
        import pandas as pd

        from rag_textbook_qa.web import services

        read = MagicMock(side_effect=pd.errors.EmptyDataError("fixture private path"))
        with patch.object(services, "run_ragas_evaluation") as evaluate:
            app, loader = self.app(result_reader=read)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_input), 1)
            self.assertTrue(any("暂时无法读取" in item.value for item in app.warning))
            read.side_effect = None
            read.return_value = pd.DataFrame({"question": ["恢复的结果"], "faithfulness": [0.8]})
            next(w for w in app.button if w.label == "重新读取结果").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[1].value["问题"].tolist(), ["恢复的结果"])
            evaluate.assert_not_called()
            loader.assert_not_called()

    def test_failed_evaluation_keeps_previous_report_without_automatic_retry(self):
        import pandas as pd

        from rag_textbook_qa.web import services

        results = pd.DataFrame({"question": ["已有结果"], "faithfulness": [0.8]})
        with patch.object(services, "run_ragas_evaluation", side_effect=RuntimeError("fixture secret")) as evaluate:
            app, _ = self.app(results=results)
            next(w for w in app.button if w.label == "运行评估").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any("未完成" in item.value for item in app.error))
            self.assertFalse(any("secret" in item.value for item in app.error))
            self.assertEqual(app.dataframe[1].value["问题"].tolist(), ["已有结果"])
            next(w for w in app.text_input if w.label == "搜索问题").set_value("已有").run(timeout=20)
            evaluate.assert_called_once()

    def test_invalid_metric_text_is_shown_as_unknown_without_mutating_original(self):
        import pandas as pd

        results = pd.DataFrame({"question": ["已评分", "格式异常"],
                                "faithfulness": ["0.8", "invalid"]})
        app, _ = self.app(results=results)
        self.assertFalse(app.exception)
        self.assertTrue(any("得分格式异常" in item.value for item in app.warning))
        detail = app.dataframe[1].value
        self.assertEqual(len(detail), 2)
        self.assertEqual(detail.loc[detail["问题"] == "格式异常", "平均分"].tolist(), ["未评分"])
        self.assertEqual(results["faithfulness"].tolist(), ["0.8", "invalid"])

    def test_nonfinite_and_boolean_scores_are_unknown_and_rows_remain_visible(self):
        import pandas as pd

        results = pd.DataFrame({"question": ["正常", "正无穷", "负无穷", "布尔值"],
                                "faithfulness": [0.8, float("inf"), float("-inf"), True]})
        original = results.copy(deep=True)
        app, _ = self.app(results=results)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "0.800")
        self.assertTrue(any("1/4 题有得分" in item.value for item in app.caption))
        self.assertTrue(any("得分格式异常" in item.value for item in app.warning))
        detail = app.dataframe[1].value.set_index("问题")
        self.assertEqual(len(detail), 4)
        for question in ("正无穷", "负无穷", "布尔值"):
            self.assertEqual(detail.loc[question, "平均分"], "未评分")
            self.assertEqual(detail.loc[question, "有效指标"], "0/1")
        pd.testing.assert_frame_equal(results, original)

    def test_boolean_only_metric_has_no_valid_summary_score(self):
        import pandas as pd

        results = pd.DataFrame({"question": ["布尔真", "布尔假"], "faithfulness": [True, False]})
        app, _ = self.app(results=results)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "—")
        self.assertTrue(any("0/2 题有得分" in item.value for item in app.caption))
        self.assertEqual(app.dataframe[1].value["平均分"].tolist(), ["未评分", "未评分"])

    def test_invalid_question_ids_use_result_rows_without_rewriting_identity(self):
        import pandas as pd

        for indices, displayed in (
            ([2, None], ["未提供", "2.0"]),
            ([2, 2], ["2", "2"]),
            ([True, False], ["False", "True"]),
            ([2.5, 3], ["3.0", "2.5"]),
        ):
            with self.subTest(indices=indices):
                results = pd.DataFrame({"question": ["第一行", "第二行"],
                                        "faithfulness": [0.8, 0.6], "question_index": indices})
                original = results.copy(deep=True)
                app, _ = self.app(results=results)
                self.assertFalse(app.exception)
                self.assertTrue(any("不能据此配对题目" in w.value for w in app.warning))
                self.assertEqual(app.dataframe[0].value["题号"].tolist(), ["结果行1", "结果行2"])
                detail = app.dataframe[1].value
                self.assertEqual(detail["题号"].tolist(), ["结果行2", "结果行1"])
                self.assertEqual(detail["原题号"].tolist(), displayed)
                pd.testing.assert_frame_equal(results, original)

    def test_streaming_answer_sources_history_clear_and_defaults(self):
        engine = MagicMock()

        def ask(**kwargs):
            kwargs["on_generation_start"]()
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
        self.assertFalse(any('<div class="empty-state">' in item.value for item in app.markdown))
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

    def test_download_uses_original_question_and_actual_sources_without_model_reruns(self):
        engine = MagicMock()
        engine.ask.return_value = {"success": True, "answer": "答案【参考资料 7】",
            "context_sources": [{"citation_id": 7, "book_name": "os", "content": "实际输入末尾"}],
            "results": [{"content": "未发送候选"}]}
        with patch("rag_textbook_qa.web.chat_page.st.download_button") as download:
            app, _ = self.app(engine)
            app.chat_input[0].set_value("原来的问题").run(timeout=20)
            app.radio[0].set_value("全部").run(timeout=20)
            app.run(timeout=20)
        engine.ask.assert_called_once()
        self.assertFalse(app.exception)
        options = download.call_args.kwargs
        self.assertIn("原来的问题", options["data"])
        self.assertIn("实际输入末尾", options["data"])
        self.assertNotIn("未发送候选", options["data"])
        self.assertEqual(options["on_click"], "ignore")

    def test_retry_preserves_original_book_and_settings_without_automatic_calls(self):
        engine = MagicMock()
        engine.ask.return_value = {"success": False, "answer": "服务暂时不可用", "context_sources": []}
        app, _ = self.app(engine)
        app.chat_input[0].set_value("什么是进程？").run(timeout=20)
        self.assertTrue(any(b.label == "重试本题" for b in app.button))
        app.run(timeout=20)
        original = engine.ask.call_args.kwargs.copy()
        self.assertEqual(engine.ask.call_count, 1)
        # Changing the controls must not silently turn a retry into a different request.
        app.radio[0].set_value("全部").run(timeout=20)
        app.slider[0].set_value(8).run(timeout=20)
        engine.ask.return_value = {"success": True, "answer": "重试后的回答", "context_sources": []}
        next(b for b in app.button if b.label == "重试本题").click().run(timeout=20)
        retried = engine.ask.call_args.kwargs
        callbacks = {"on_answer_chunk", "on_generation_start"}
        self.assertEqual({k: v for k, v in retried.items() if k not in callbacks},
                         {k: v for k, v in original.items() if k not in callbacks})
        self.assertEqual(engine.ask.call_count, 2)
        self.assertEqual(app.session_state["messages"][-1]["content"], "重试后的回答")
        self.assertFalse(app.exception)

    def test_source_view_keeps_full_text_and_declared_citation_number(self):
        engine = MagicMock()
        content = "教材原文\n" + "前文" * 150 + "末尾的关键定义 <原样显示>"
        engine.ask.return_value = {"success": True, "answer": "答案【参考资料 7】",
            "context_sources": [{"citation_id": 7, "book_name": "computer_organization",
                "content": content, "chapter": "第 4 章 存储系统", "section_h2": "4.5",
                "section_h3": "4.5.8 写入策略", "section_h4": "1.写回法（Write-Back，WB）",
                "truncated": True, "table_compacted": True}]}
        app, _ = self.app(engine)
        app.chat_input[0].set_value("问题").run(timeout=20)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("参考资料 7", rendered)
        self.assertIn("末尾的关键定义 &lt;原样显示&gt;", rendered)
        self.assertIn("4.5.8 写入策略 &gt; 1.写回法（Write-Back，WB）", rendered)
        self.assertIn("片段已截断", rendered)
        self.assertIn("表格按行整理", rendered)
        self.assertNotIn("分数 0.000", rendered)
        self.assertFalse(app.exception)

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
