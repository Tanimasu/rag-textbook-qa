import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_textbook_qa.evaluation import (
    RAGASEvaluator,
    build_evaluation_plan,
    load_test_questions,
    render_evaluation_plan,
)
from rag_textbook_qa.evaluation.ragas import (
    _attach_question_indices,
    _paired_baseline_report,
    _ragas_embedding_model,
    run_evaluation,
)


class EvaluationTests(unittest.TestCase):
    def test_reference_metrics_require_a_complete_nonblank_reference_column(self):
        for references, expected in (
            (None, ["faithfulness", "answer_relevancy"]),
            (["", ""], ["faithfulness", "answer_relevancy"]),
            (["标准答案", " \n "], ["faithfulness", "answer_relevancy"]),
            (["标准一", "标准二"],
             ["faithfulness", "context_precision", "answer_relevancy", "context_recall"]),
        ):
            with self.subTest(references=references):
                evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
                for attribute in ("_faithfulness", "_answer_relevancy",
                                  "_context_precision", "_context_recall"):
                    setattr(evaluator, attribute, MagicMock(name=attribute))
                    getattr(evaluator, attribute).name = attribute.removeprefix("_")
                evaluator.embeddings = object()
                evaluator.llm = object()
                evaluator._evaluate = MagicMock()
                evaluator._run_config_type = MagicMock()
                evaluator._stabilize_relevancy = MagicMock()
                dataset = MagicMock()
                dataset.column_names = [] if references is None else ["ground_truth"]
                dataset.__getitem__.return_value = references
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluator.evaluate(dataset)
                selected = evaluator._evaluate.call_args.kwargs["metrics"]
                self.assertEqual([metric.name for metric in selected], expected)

    def test_used_output_is_rejected_before_generation_or_evaluator_creation(self):
        questions = [{"question": "问题"}]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "previous-run"
            destination.mkdir()
            previous = destination / "ragas_qa_comparison.json"
            original = b'[{"answer":"original"}]'
            previous.write_bytes(original)
            engine = MagicMock()
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = destination
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "输出目录"):
                    evaluator.prepare_evaluation_data(engine, questions)
                with patch("rag_textbook_qa.evaluation.ragas.RAGASEvaluator") as create:
                    with self.assertRaisesRegex(ValueError, "输出目录"):
                        run_evaluation(engine, questions, destination)
                    create.assert_not_called()
            engine.ask.assert_not_called()
            self.assertEqual(previous.read_bytes(), original)
            self.assertEqual(list(destination.iterdir()), [previous])

    def test_output_created_during_generation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = destination
            collision = destination / "ragas_run_summary.json"
            original = b'{"owner":"another-run"}'

            def generate(**kwargs):
                collision.write_bytes(original)
                return {"success": True, "answer": "答案", "context": "实际证据"}

            engine = MagicMock()
            engine.ask.side_effect = generate
            with (
                patch("rag_textbook_qa.evaluation.ragas._dataset_from_dict", side_effect=lambda data: data),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(FileExistsError),
            ):
                evaluator.prepare_evaluation_data(engine, [{"question": "问题"}])
            self.assertEqual(collision.read_bytes(), original)
            self.assertFalse((destination / "ragas_qa_comparison.json").exists())

    def test_invalid_question_batch_is_rejected_before_any_generation(self):
        invalid_rows = (
            {"question": " \n "},
            {"question": "问题", "book_name": ["os"]},
            {"question": "问题", "ground_truth": 42},
        )
        for invalid in invalid_rows:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
                evaluator.output_dir = Path(directory)
                engine = MagicMock()
                engine.ask.return_value = {"success": True, "answer": "固定答案", "context": "固定证据"}
                engine.llm.generate_answer.return_value = {"success": True, "answer": "固定基线"}
                questions = [{"question": "有效的第一题"}, invalid]
                for method in (evaluator.prepare_evaluation_data, evaluator.prepare_baseline_data):
                    with (
                        self.subTest(method=method.__name__),
                        patch("rag_textbook_qa.evaluation.ragas._dataset_from_dict", side_effect=lambda data: data),
                        contextlib.redirect_stdout(io.StringIO()),
                        self.assertRaises((TypeError, ValueError)),
                    ):
                        method(engine, questions)
                engine.ask.assert_not_called()
                engine.llm.generate_answer.assert_not_called()

    def test_invalid_question_file_and_run_fail_before_creating_evaluator(self):
        questions = [{"question": "有效题"}, {"question": "问题", "ground_truth": ["错误类型"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaises((TypeError, ValueError)):
                load_test_questions(path)
            with patch("rag_textbook_qa.evaluation.ragas.RAGASEvaluator") as create:
                with self.assertRaises((TypeError, ValueError)):
                    run_evaluation(MagicMock(), questions, directory)
                with self.assertRaisesRegex(ValueError, "top_k"):
                    run_evaluation(MagicMock(), [{"question": "有效题"}], directory, top_k=0)
                create.assert_not_called()
            with patch("rag_textbook_qa.evaluation.ragas.create_llm_client") as create:
                with self.assertRaises((TypeError, ValueError)):
                    build_evaluation_plan(questions, output_dir=directory, environ={})
                create.assert_not_called()

    def test_paired_comparison_excludes_failed_scores_and_has_no_empty_delta(self):
        import pandas as pd

        rag = pd.DataFrame({"question_index": [1, 2, 3],
                            "answer_relevancy": [float("nan"), 0.5, 0.4]})
        baseline = pd.DataFrame({"question_index": [1, 2, 3],
                                 "answer_relevancy": [0.8, float("inf"), 0.2]})
        report = _paired_baseline_report(rag, baseline, [1, 2, 3], [1, 2, 3], 3)
        self.assertEqual(report["paired_question_indices"], [3])
        self.assertEqual(report["invalid_score_question_indices"], [1, 2])
        self.assertAlmostEqual(report["delta"], 0.2)
        json.dumps(report, allow_nan=False)

        report = _paired_baseline_report(rag.iloc[:1], baseline.iloc[1:2], [1], [2], 3)
        self.assertEqual(report["paired_questions"], 0)
        self.assertIsNone(report["delta"])
        self.assertIsNone(report["rag_mean"])
        self.assertIsNone(report["baseline_mean"])
        json.dumps(report, allow_nan=False)

    def test_result_identity_is_rejected_when_rows_are_missing_or_reordered(self):
        import pandas as pd

        questions = [{"question": "问题一"}, {"question": "问题二"}]
        with self.assertRaisesRegex(ValueError, "行数"):
            _attach_question_indices(pd.DataFrame({"answer_relevancy": [0.5]}), [1, 2], questions)
        for column in ("question", "user_input"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "顺序"):
                _attach_question_indices(pd.DataFrame({column: ["问题二", "问题一"]}), [1, 2], questions)

    def test_baseline_comparison_pairs_original_question_positions(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
            evaluator.output_dir = Path(directory)
            evaluator._answer_relevancy = MagicMock(name="answer_relevancy")
            frames = [
                pd.DataFrame({"answer_relevancy": [0.9, 0.8]}),
                pd.DataFrame({"answer_relevancy": [0.1, 0.6]}),
            ]
            evaluator.evaluate = MagicMock(side_effect=[
                MagicMock(to_pandas=MagicMock(return_value=frame)) for frame in frames
            ])
            engine = MagicMock()
            engine.enable_hyde = engine.enable_adjacent_context = False
            engine.ask.side_effect = [
                {"success": False},
                {"success": True, "answer": "回答二", "context": "证据二"},
                {"success": True, "answer": "回答三", "context": "证据三"},
            ]
            engine.llm.generate_answer.side_effect = [
                {"success": True, "answer": "基线一"},
                {"success": False},
                {"success": True, "answer": "基线三"},
            ]
            # Identical wording must not merge questions from different books.
            questions = [{"question": "相同问题", "book_name": book}
                         for book in ("os", "database", "computer_networks")]
            output = io.StringIO()
            with (
                patch("rag_textbook_qa.evaluation.ragas.RAGASEvaluator", return_value=evaluator),
                patch("rag_textbook_qa.evaluation.ragas._dataset_from_dict", side_effect=lambda data: data),
                contextlib.redirect_stdout(output),
            ):
                run_evaluation(engine, questions, directory, include_baseline=True)
            self.assertIn("delta=+0.2000", output.getvalue())
            report = json.loads((Path(directory) / "ragas_baseline_comparison.json").read_text())
            self.assertEqual(report["paired_question_indices"], [3])
            self.assertEqual(report["paired_questions"], 1)
            self.assertEqual(report["total_questions"], 3)
            self.assertAlmostEqual(report["delta"], 0.2)
            self.assertEqual(report["rag_successful_questions"], 2)
            self.assertEqual(report["baseline_successful_questions"], 2)
            for name, expected in (("ragas_evaluation_results.csv", [2, 3]),
                                   ("ragas_baseline_results.csv", [1, 3])):
                self.assertEqual(list(pd.read_csv(Path(directory) / name)["question_index"]), expected)

    def test_evaluation_plan_is_secret_free_and_exposes_cost_factors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = build_evaluation_plan(
                [{"question": "问题一"}, {"question": "问题二"}],
                output_dir=Path(temporary_directory) / "fresh-output",
                top_k=7,
                enable_hyde=True,
                enable_adjacent_context=True,
                include_baseline=True,
                environ={
                    "LLM_API_KEY": "shared-secret",
                    "RAG_API_BASE": "https://generator.example/v1",
                    "RAG_MODEL": "generator-model",
                    "RAGAS_API_KEY": "judge-secret",
                    "RAGAS_API_BASE": "https://judge.example/v1",
                    "RAGAS_MODEL": "judge-model",
                    "RAGAS_EMBEDDING_MODEL": "judge-embedding",
                    "RAGAS_RELEVANCY_SAMPLES": "3",
                    "RAG_QA_COMPUTE_BACKEND": "local",
                    "RAG_QA_DEVICE": "mps",
                },
            )

        serialized = json.dumps(plan)
        self.assertNotIn("shared-secret", serialized)
        self.assertNotIn("judge-secret", serialized)
        self.assertEqual(plan["question_count"], 2)
        self.assertEqual(plan["minimum_generation_calls"], 6)
        self.assertEqual(plan["ragas_scoring_passes"], 6)
        self.assertEqual(plan["product_path"]["top_k"], 7)
        self.assertTrue(plan["product_path"]["hyde"])
        self.assertTrue(plan["product_path"]["adjacent_context"])
        self.assertEqual(plan["compute"]["device"], "mps")
        self.assertEqual(plan["warnings"], [])
        rendered = render_evaluation_plan(plan)
        self.assertIn("未调用 API", rendered)
        self.assertIn("至少 6 次回答生成", rendered)
        self.assertIn("计算后端: local / 设备 mps", rendered)
        self.assertIn("相邻片段 开", rendered)

    def test_evaluation_plan_warns_before_mixing_results_or_backends(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "old.csv").write_text("old", encoding="utf-8")
            plan = build_evaluation_plan(
                [{"question": "问题"}],
                output_dir=output_dir,
                environ={
                    "LLM_API_KEY": "secret",
                    "LLM_API_BASE": "https://same.example/v1",
                    "LLM_MODEL": "same-model",
                    "RAG_QA_COMPUTE_BACKEND": "remote",
                    "RAG_QA_REMOTE_URL": "http://100.64.0.1:8765",
                    "RAG_QA_WORKER_TOKEN": "worker-secret",
                    "RAG_QA_QUERY_FALLBACK_TO_LOCAL": "true",
                },
            )

        warnings = "\n".join(plan["warnings"])
        self.assertIn("同一 API 服务", warnings)
        self.assertIn("同一模型", warnings)
        self.assertIn("已有 1 项内容", warnings)
        self.assertIn("回退本地", warnings)
        self.assertNotIn("worker-secret", json.dumps(plan))
        self.assertIn("本地回退 开（auto）", render_evaluation_plan(plan))

    def test_ragas_embedding_defaults_to_the_retrieval_model(self):
        with patch.dict(
            os.environ,
            {"RAG_QA_EMBEDDING_MODEL": "example/embedding-model"},
            clear=True,
        ):
            self.assertEqual(_ragas_embedding_model(), "example/embedding-model")

    def test_relevancy_sample_count_is_configurable_and_validated(self):
        from rag_textbook_qa.evaluation.ragas import (
            DEFAULT_RELEVANCY_SAMPLES,
            relevancy_samples,
        )

        self.assertEqual(relevancy_samples({}), DEFAULT_RELEVANCY_SAMPLES)
        self.assertEqual(relevancy_samples({"RAGAS_RELEVANCY_SAMPLES": "5"}), 5)
        with self.assertRaisesRegex(ValueError, "大于等于 1"):
            relevancy_samples({"RAGAS_RELEVANCY_SAMPLES": "0"})
        with self.assertRaisesRegex(ValueError, "必须是整数"):
            relevancy_samples({"RAGAS_RELEVANCY_SAMPLES": "三"})

    def test_averaging_ignores_rows_a_run_failed_to_score(self):
        from rag_textbook_qa.evaluation.ragas import average_samples

        averaged = average_samples([[0.8, 0.4, float("nan")], [0.6, None, float("nan")]])
        self.assertAlmostEqual(averaged[0], 0.7)
        self.assertAlmostEqual(averaged[1], 0.4)
        self.assertIsNone(averaged[2])
        self.assertEqual(average_samples([]), [])
        finite = average_samples([[float("inf"), float("-inf")], [0.6, None]])
        self.assertEqual(finite, [0.6, None])
        with self.assertRaisesRegex(ValueError, "行数不一致"):
            average_samples([[0.5], [0.5, 0.5]])

    def test_averaged_result_replaces_only_its_own_column(self):
        import pandas as pd

        from rag_textbook_qa.evaluation.ragas import _AveragedResult

        class Stub:
            marker = "kept"

            def to_pandas(self):
                return pd.DataFrame({"answer_relevancy": [0.1, 0.2], "faithfulness": [1.0, 1.0]})

        wrapped = _AveragedResult(Stub(), "answer_relevancy", [0.5, 0.6])
        frame = wrapped.to_pandas()

        self.assertEqual(list(frame["answer_relevancy"]), [0.5, 0.6])
        self.assertEqual(list(frame["faithfulness"]), [1.0, 1.0])
        self.assertEqual(wrapped.marker, "kept")

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
            engine.enable_hyde = False
            engine.enable_adjacent_context = False
            engine.ask.return_value = {
                "success": True,
                "answer": "进程是程序的一次执行过程。",
                "llm_response": {"finish_reason": "stop", "model": "saved-generator"},
                "prompt": "实际完整提示词",
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
                    "id": "process-01",
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
            summary = json.loads((output_dir / "ragas_run_summary.json").read_text())

        self.assertEqual(
            set(dataset),
            {"question", "answer", "contexts", "ground_truth"},
        )
        self.assertEqual("".join(dataset["contexts"][0]), engine.ask.return_value["context"])
        self.assertEqual(len(dataset["contexts"][0]), 2)
        self.assertNotIn("教材上下文", dataset["contexts"][0][0])
        self.assertEqual(comparison[0]["ground_truth"], "标准答案")
        self.assertEqual(comparison[0]["question_id"], "process-01")
        self.assertEqual(comparison[0]["context"], engine.ask.return_value["context"])
        self.assertEqual(comparison[0]["contexts"], dataset["contexts"][0])
        self.assertEqual(comparison[0]["context_sources"], engine.ask.return_value["context_sources"])
        self.assertEqual(comparison[0]["finish_reason"], "stop")
        self.assertEqual(comparison[0]["generation_model"], "saved-generator")
        self.assertEqual(comparison[0]["prompt"], "实际完整提示词")
        engine.ask.assert_called_once_with(
            query="什么是进程？",
            book_name="os",
            top_k=5,
            use_llm=True,
            use_hyde=False,
            use_adjacent_context=False,
            use_decomposition=False,
            verify_citations=False,
        )

        self.assertEqual(
            summary["product_path"],
            {
                "top_k": 5,
                "hyde": False,
                "adjacent_context": False,
                "query_decomposition": False,
                "citation_verification": False,
            },
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

    def test_print_results_treats_infinite_scores_as_missing_without_losing_valid_scores(self):
        import pandas as pd

        evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
        result = MagicMock()
        original = pd.DataFrame({"faithfulness": [float("inf"), float("-inf"), 0.6]})
        result.to_pandas.return_value = original
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dataframe = evaluator.print_results(result)
        self.assertEqual(list(dataframe["faithfulness"].isna()), [True, True, False])
        self.assertAlmostEqual(dataframe["faithfulness"].mean(), 0.6)
        self.assertIn("0.6000", output.getvalue())
        self.assertEqual(list(original["faithfulness"].isna()), [False, False, False])


if __name__ == "__main__":
    unittest.main()
