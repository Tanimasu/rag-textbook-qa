import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.cli import main


class EvaluateCliTests(unittest.TestCase):
    def test_evaluate_command_delegates_without_loading_real_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src" / "rag_textbook_qa").mkdir(parents=True)
            (root / "project").mkdir()
            (root / "data" / "evaluation").mkdir(parents=True)
            (root / "pyproject.toml").write_text(
                "[project]\nname='test'\n",
                encoding="utf-8",
            )
            (root / "project" / ".env").write_text("", encoding="utf-8")
            questions_path = root / "questions.json"
            questions_path.write_text("[]", encoding="utf-8")
            database_path = root / "custom-db"
            output_path = root / "smoke-results"
            questions = [{"question": "什么是进程？", "book_name": "os"}]
            engine = MagicMock()
            engine.__enter__.return_value = engine

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "rag_textbook_qa.evaluation.load_test_questions",
                    return_value=questions,
                ) as load_questions,
                patch("rag_textbook_qa.evaluation.run_evaluation") as run_evaluation,
                patch("rag_textbook_qa.rag.RAGEngine", return_value=engine) as engine_type,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "evaluate",
                        "--questions",
                        str(questions_path),
                        "--db-path",
                        str(database_path),
                        "--output-dir",
                        str(output_path),
                        "--baseline",
                    ]
                )

            self.assertEqual(exit_code, 0)
            load_questions.assert_called_once_with(questions_path)
            engine_type.assert_called_once_with(
                db_path=database_path,
                enable_llm=True,
                verbose=False,
                enable_hyde=True,
            )
            run_evaluation.assert_called_once_with(
                engine,
                questions,
                output_path,
                include_baseline=True,
            )
            engine.__exit__.assert_called_once()

    def test_retrieval_evaluate_compares_real_strategies_without_llm_or_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src" / "rag_textbook_qa").mkdir(parents=True)
            (root / "project").mkdir()
            (root / "data" / "evaluation").mkdir(parents=True)
            (root / "pyproject.toml").write_text(
                "[project]\nname='test'\n",
                encoding="utf-8",
            )
            (root / "project" / ".env").write_text("", encoding="utf-8")
            questions_path = root / "questions.json"
            questions_path.write_text("[]", encoding="utf-8")
            database_path = root / "custom-db"
            output_path = root / "retrieval-results"
            saved_path = output_path / "retrieval_test.json"
            questions = [object()]
            report = {
                "question_count": 1,
                "top_k": 7,
                "strategies": {
                    strategy: {
                        "mean_recall_at_k": 1.0,
                        "hit_rate_at_k": 1.0,
                        "mrr": 1.0,
                        "mean_latency_seconds": 0.01,
                    }
                    for strategy in ("bm25", "embedding", "hybrid", "hybrid-rerank")
                },
            }
            engine = MagicMock()
            engine.__enter__.return_value = engine
            output = io.StringIO()

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "rag_textbook_qa.evaluation.load_retrieval_questions",
                    return_value=questions,
                ) as load_questions,
                patch(
                    "rag_textbook_qa.evaluation.run_retrieval_strategies",
                    return_value=report,
                ) as run_strategies,
                patch(
                    "rag_textbook_qa.evaluation.save_retrieval_report",
                    return_value=saved_path,
                ) as save_report,
                patch("rag_textbook_qa.rag.RAGEngine", return_value=engine) as engine_type,
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "--workspace",
                        str(root),
                        "evaluate-retrieval",
                        "--questions",
                        str(questions_path),
                        "--db-path",
                        str(database_path),
                        "--output-dir",
                        str(output_path),
                        "--strategy",
                        "all",
                        "--top-k",
                        "7",
                    ]
                )

            self.assertEqual(exit_code, 0)
            load_questions.assert_called_once_with(questions_path)
            engine_kwargs = engine_type.call_args.kwargs
            self.assertEqual(engine_kwargs["db_path"], database_path)
            self.assertFalse(engine_kwargs["enable_llm"])
            self.assertTrue(engine_kwargs["enable_reranker"])
            self.assertFalse(engine_kwargs["enable_hyde"])
            self.assertFalse(engine_kwargs["compute_settings"].query_fallback_to_local)
            run_strategies.assert_called_once_with(
                engine,
                questions,
                ("bm25", "embedding", "hybrid", "hybrid-rerank"),
                top_k=7,
            )
            save_report.assert_called_once_with(report, output_path)
            self.assertIn("检索评测完成", output.getvalue())


if __name__ == "__main__":
    unittest.main()
