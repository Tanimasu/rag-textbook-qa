import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.evaluation import RAGASEvaluator, load_test_questions
from rag_textbook_qa.evaluation.ragas import _ragas_embedding_model


class EvaluationTests(unittest.TestCase):
    def test_ragas_embedding_defaults_to_the_retrieval_model(self):
        with patch.dict(
            os.environ,
            {"RAG_QA_EMBEDDING_MODEL": "example/embedding-model"},
            clear=True,
        ):
            self.assertEqual(_ragas_embedding_model(), "example/embedding-model")

    def test_judge_thinking_is_only_disabled_when_asked(self):
        from rag_textbook_qa.evaluation.ragas import judge_model_kwargs

        self.assertEqual(judge_model_kwargs({}), {})
        self.assertEqual(judge_model_kwargs({"RAGAS_DISABLE_THINKING": "no"}), {})
        self.assertEqual(
            judge_model_kwargs({"RAGAS_DISABLE_THINKING": "true"}),
            {"extra_body": {"enable_thinking": False}},
        )

    def test_legacy_script_is_a_thin_compatibility_entrypoint(self):
        repository_root = Path(__file__).resolve().parents[1]
        source = (repository_root / "project" / "ragas_evaluation.py").read_text(encoding="utf-8")

        self.assertIn("rag_textbook_qa.evaluation", source)
        self.assertNotIn("from ragas", source)
        self.assertNotIn("from datasets", source)
        self.assertLessEqual(len(source.splitlines()), 20)

    def test_question_file_validation_is_lightweight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            questions_path = Path(temporary_directory) / "questions.json"
            questions_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "什么是进程？",
                            "book_name": "os",
                            "ground_truth": "标准答案",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            questions = load_test_questions(questions_path)

        self.assertEqual(questions[0]["book_name"], "os")

    def test_invalid_question_file_is_rejected_before_models_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            questions_path = Path(temporary_directory) / "questions.json"
            questions_path.write_text('[{"book_name": "os"}]', encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "缺少 question"):
                load_test_questions(questions_path)

    def test_prepare_evaluation_data_preserves_existing_schema_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "evaluations"
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = output_dir
            engine = MagicMock()
            engine.ask.return_value = {
                "success": True,
                "answer": "进程是程序的一次执行过程。",
                "context": "[os - 第1章 - 进程]\n实际提供的上下文",
                "context_sources": [
                    {"context_text": "[os - 第1章 - 进程]\n"},
                    {"context_text": "实际提供的上下文"},
                ],
                "results": [
                    {
                        "book_name": "os",
                        "chapter": "第1章",
                        "section_h2": "进程",
                        "content": "教材上下文",
                    }
                ],
            }
            questions = [
                {
                    "question": "什么是进程？",
                    "book_name": "os",
                    "ground_truth": "标准答案",
                }
            ]

            with patch(
                "rag_textbook_qa.evaluation.ragas._dataset_from_dict",
                side_effect=lambda data: data,
            ):
                dataset = evaluator.prepare_evaluation_data(engine, questions)

            comparison = json.loads(
                (output_dir / "ragas_qa_comparison.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            set(dataset),
            {"question", "answer", "contexts", "ground_truth"},
        )
        self.assertEqual("".join(dataset["contexts"][0]), engine.ask.return_value["context"])
        self.assertEqual(len(dataset["contexts"][0]), 2)
        self.assertNotIn("教材上下文", dataset["contexts"][0][0])
        self.assertEqual(comparison[0]["ground_truth"], "标准答案")
        engine.ask.assert_called_once_with(
            query="什么是进程？",
            book_name="os",
            top_k=8,
            use_llm=True,
        )

    def test_failure_summary_includes_failed_and_exception_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = Path(directory)
            engine = MagicMock()
            engine.ask.side_effect = [
                {"success": False, "error": "private error detail"},
                RuntimeError("private runtime detail"),
                {"success": True, "answer": "答案", "context": "实际证据"},
            ]
            with patch(
                "rag_textbook_qa.evaluation.ragas._dataset_from_dict", side_effect=lambda data: data
            ):
                dataset = evaluator.prepare_evaluation_data(
                    engine, [{"question": str(i)} for i in range(3)]
                )
            summary_text = (Path(directory) / "ragas_run_summary.json").read_text()
            summary = json.loads(summary_text)
        self.assertEqual(summary["total_questions"], 3)
        self.assertEqual(summary["failed_questions"], 2)
        self.assertEqual(summary["success_rate"], 1 / 3)
        self.assertEqual(dataset["contexts"], [["实际证据"]])
        self.assertNotIn("private", summary_text)

    def test_all_failed_run_still_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = Path(directory)
            engine = MagicMock()
            engine.ask.return_value = {"success": False}
            with self.assertRaisesRegex(ValueError, "失败摘要已保存"):
                evaluator.prepare_evaluation_data(engine, [{"question": "问题"}])
            summary = json.loads((Path(directory) / "ragas_run_summary.json").read_text())
            self.assertEqual(summary["success_rate"], 0)
            self.assertEqual(summary["failed_questions"], 1)

    def test_print_results_handles_an_all_nan_evaluation(self):
        import pandas as pd

        evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
        result = MagicMock()
        result.to_pandas.return_value = pd.DataFrame(
            {
                "question": ["什么是进程？"],
                "faithfulness": [float("nan")],
            }
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dataframe = evaluator.print_results(result)

        self.assertIsNotNone(dataframe)
        self.assertIn("没有可汇总的有效指标分数", output.getvalue())


if __name__ == "__main__":
    unittest.main()
