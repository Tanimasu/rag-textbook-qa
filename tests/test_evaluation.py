import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.evaluation import RAGASEvaluator, load_test_questions


class EvaluationTests(unittest.TestCase):
    def test_legacy_script_is_a_thin_compatibility_entrypoint(self):
        repository_root = Path(__file__).resolve().parents[1]
        source = (repository_root / "project" / "ragas_evaluation.py").read_text(
            encoding="utf-8"
        )

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
        self.assertIn("[os - 第1章 - 进程]", dataset["contexts"][0][0])
        self.assertEqual(comparison[0]["ground_truth"], "标准答案")
        engine.ask.assert_called_once_with(
            query="什么是进程？",
            book_name="os",
            top_k=8,
            use_llm=True,
        )


if __name__ == "__main__":
    unittest.main()
