"""RAGAS evaluation for the packaged textbook QA engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _dataset_from_dict(data: dict[str, list[Any]]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "运行评估需要安装 eval 依赖：uv sync --extra eval"
        ) from exc
    return Dataset.from_dict(data)


def _default_output_dir() -> Path:
    from rag_textbook_qa.config import Settings

    return Settings.load().paths.evaluations


class RAGASEvaluator:
    """Evaluate RAG answers through an OpenAI-compatible RAGAS judge."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import RunConfig, evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise RuntimeError(
                "运行评估需要安装 eval 依赖：uv sync --extra eval"
            ) from exc

        resolved_api_key = api_key or os.getenv("RAGAS_API_KEY") or os.getenv(
            "LLM_API_KEY", ""
        )
        resolved_base_url = (
            base_url
            or os.getenv("RAGAS_API_BASE")
            or os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")
        )
        resolved_model = (
            model
            or os.getenv("RAGAS_MODEL")
            or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")
        )

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._evaluate = evaluate
        self._run_config_type = RunConfig
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy
        self._context_precision = context_precision
        self._context_recall = context_recall

        print("初始化 RAGAS 评估器...")
        print(f"  API: {resolved_base_url}")
        print(f"  模型: {resolved_model}")

        self.llm = ChatOpenAI(
            model=resolved_model,
            openai_api_key=resolved_api_key,
            openai_api_base=resolved_base_url,
            temperature=0.0,
            request_timeout=60,
            max_retries=3,
        )
        print("  LLM 初始化成功")

        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=resolved_api_key,
                openai_api_base=resolved_base_url,
                request_timeout=60,
            )
            print("  Embeddings 初始化成功（API）")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"  API Embeddings 不可用 ({exc})，降级到本地 HuggingFace 模型")
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("  Embeddings 初始化成功（本地）")

        print("初始化完成\n")

    def prepare_evaluation_data(
        self,
        rag_engine: Any,
        test_questions: list[dict[str, Any]],
    ) -> Any:
        """Run the RAG engine over questions and return a RAGAS dataset."""

        print("=" * 60)
        print("准备评估数据")
        print("=" * 60)

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []
        qa_records: list[dict[str, str]] = []

        for index, item in enumerate(test_questions, 1):
            question = item["question"]
            print(f"[{index}/{len(test_questions)}] {question}")
            try:
                result = rag_engine.ask(
                    query=question,
                    book_name=item.get("book_name"),
                    top_k=8,
                    use_llm=True,
                )
                if not result["success"]:
                    print(f"  失败: {result.get('error', 'unknown')}")
                    continue

                ground_truth = item.get("ground_truth", "")
                answer = result["answer"]
                questions.append(question)
                answers.append(answer)
                contexts.append(
                    [
                        f"[{source['book_name']} - {source['chapter']} - "
                        f"{source['section_h2']}]\n{source['content']}"
                        for source in result["results"]
                    ]
                )
                ground_truths.append(ground_truth)
                qa_records.append(
                    {
                        "question": question,
                        "answer": answer,
                        "ground_truth": ground_truth,
                    }
                )
                preview = f"{answer[:300]}{'...' if len(answer) > 300 else ''}"
                print(f"  [回答] {preview}")
                if ground_truth:
                    ground_truth_preview = (
                        f"{ground_truth[:300]}"
                        f"{'...' if len(ground_truth) > 300 else ''}"
                    )
                    print(f"  [标准] {ground_truth_preview}")
                print()
            except (
                AttributeError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                print(f"  异常: {exc}")

        if not questions:
            raise ValueError("没有成功处理任何问题，无法评估。")

        output_dir = self.output_dir or _default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "ragas_qa_comparison.json"
        comparison_path.write_text(
            json.dumps(qa_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"问答对比已保存到: {comparison_path}")
        print(f"\n数据准备完成：{len(questions)} 条\n")

        data: dict[str, list[Any]] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if any(ground_truths):
            data["ground_truth"] = ground_truths
        return _dataset_from_dict(data)

    def evaluate(self, dataset: Any, metrics: list[Any] | None = None) -> Any:
        """Execute RAGAS metrics and return its evaluation result."""

        if metrics is None:
            metrics = [self._faithfulness, self._context_precision]
            if self.embeddings:
                metrics.append(self._answer_relevancy)
            if "ground_truth" in dataset.column_names and any(dataset["ground_truth"]):
                metrics.append(self._context_recall)

        print("=" * 60)
        print(
            f"开始评估  数据集: {len(dataset)} 条  "
            f"指标: {[metric.name for metric in metrics]}"
        )
        print("=" * 60)
        result = self._evaluate(
            dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
            run_config=self._run_config_type(
                max_retries=5,
                timeout=180,
                max_workers=2,
            ),
        )
        print("评估完成\n")
        return result

    def print_results(self, result: Any) -> Any | None:
        """Print metric summaries and return a reduced DataFrame when available."""

        print("=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)

        dataframe = result.to_pandas() if hasattr(result, "to_pandas") else None
        if dataframe is not None:
            excluded = {
                "response",
                "answer",
                "retrieved_contexts",
                "contexts",
                "reference",
                "ground_truth",
            }
            dataframe = dataframe.drop(
                columns=[column for column in dataframe.columns if column in excluded],
                errors="ignore",
            )

            numeric_columns = dataframe.select_dtypes(include="number").columns
            nan_mask = dataframe[numeric_columns].isna()
            if nan_mask.any().any():
                print("⚠️  以下问题的部分指标评估失败（NaN），已从均值计算中排除：")
                question_column = next(
                    (
                        column
                        for column in dataframe.columns
                        if column in {"user_input", "question"}
                    ),
                    None,
                )
                for index, row in nan_mask.iterrows():
                    failed = [column for column in numeric_columns if row[column]]
                    if failed:
                        question = (
                            f"{dataframe.loc[index, question_column][:40]}..."
                            if question_column
                            else f"第{index + 1}条"
                        )
                        print(f"  [{index + 1}] {question}  →  {', '.join(failed)}")
                print()

            scores = {
                column: float(dataframe[column].mean()) for column in numeric_columns
            }
        else:
            try:
                scores = {
                    key: value
                    for key, value in result.items()
                    if isinstance(value, (int, float))
                }
            except AttributeError:
                scores = {}

        for metric, score in scores.items():
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            if score >= 0.85:
                grade = "优秀"
            elif score >= 0.70:
                grade = "良好"
            elif score >= 0.50:
                grade = "及格"
            else:
                grade = "需改进"
            print(f"{metric:20s} | {bar} | {score:.4f} | {grade}")
        print()

        if dataframe is not None:
            print("逐条详情:")
            print(dataframe.to_string())
            print()
        return dataframe

    def prepare_baseline_data(
        self,
        rag_engine: Any,
        test_questions: list[dict[str, Any]],
    ) -> Any:
        """Build a no-retrieval baseline dataset with the RAG engine's LLM."""

        print("=" * 60)
        print("准备 Baseline 数据（无 RAG，直接 LLM）")
        print("=" * 60)
        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        for index, item in enumerate(test_questions, 1):
            question = item["question"]
            print(f"[{index}/{len(test_questions)}] {question}")
            try:
                prompt = (
                    "你是一个计算机课程的专业 AI 助教，请回答以下问题。"
                    "先给出简明答案（2-3句话），再给出详细解释。\n\n"
                    f"问题：{question}\n\n请开始你的回答："
                )
                result = rag_engine.llm.generate_answer(
                    prompt,
                    temperature=0.7,
                    max_tokens=2000,
                )
                if not result["success"]:
                    print(f"  失败: {result.get('error', 'unknown')}")
                    continue
                questions.append(question)
                answers.append(result["answer"])
                contexts.append([""])
                ground_truths.append(item.get("ground_truth", ""))
                print(f"  [回答] {result['answer'][:200]}...")
            except (
                AttributeError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                print(f"  异常: {exc}")

        if not questions:
            raise ValueError("没有成功处理任何问题，无法评估。")
        print(f"\nBaseline 数据准备完成：{len(questions)} 条\n")
        data: dict[str, list[Any]] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if any(ground_truths):
            data["ground_truth"] = ground_truths
        return _dataset_from_dict(data)


def load_test_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate an evaluation question JSON file."""

    questions_path = Path(path)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"评估问题必须是非空 JSON 数组: {questions_path}")
    for index, item in enumerate(questions, 1):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise TypeError(f"第 {index} 条评估问题缺少 question 字符串")
    return questions


def run_evaluation(
    rag_engine: Any,
    test_questions: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    include_baseline: bool = False,
) -> Any | None:
    """Run the existing RAGAS workflow and persist its result CSV files."""

    destination = Path(output_dir)
    evaluator = RAGASEvaluator(output_dir=destination)
    rag_dataset = evaluator.prepare_evaluation_data(rag_engine, test_questions)
    rag_result = evaluator.evaluate(rag_dataset)
    print("\n【RAG 系统评估结果】")
    rag_dataframe = evaluator.print_results(rag_result)
    if rag_dataframe is not None:
        destination.mkdir(parents=True, exist_ok=True)
        rag_output = destination / "ragas_evaluation_results.csv"
        rag_dataframe.to_csv(rag_output, index=False, encoding="utf-8-sig")
        print(f"结果已保存到: {rag_output}")

    if not include_baseline:
        return rag_dataframe

    baseline_dataset = evaluator.prepare_baseline_data(rag_engine, test_questions)
    baseline_result = evaluator.evaluate(
        baseline_dataset,
        metrics=[evaluator._answer_relevancy],
    )
    print("\n【Baseline 评估结果（无 RAG）】")
    baseline_dataframe = evaluator.print_results(baseline_result)
    if baseline_dataframe is not None:
        destination.mkdir(parents=True, exist_ok=True)
        baseline_output = destination / "ragas_baseline_results.csv"
        baseline_dataframe.to_csv(
            baseline_output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"结果已保存到: {baseline_output}")

    print("\n" + "=" * 60)
    print("RAG vs Baseline 对比摘要")
    print("=" * 60)
    metric = "answer_relevancy"
    rag_score = (
        float(rag_dataframe[metric].mean())
        if rag_dataframe is not None and metric in rag_dataframe.columns
        else float("nan")
    )
    baseline_score = (
        float(baseline_dataframe[metric].mean())
        if baseline_dataframe is not None and metric in baseline_dataframe.columns
        else float("nan")
    )
    delta = rag_score - baseline_score
    sign = "+" if delta >= 0 else ""
    print(
        f"  {metric:20s}  RAG={rag_score:.4f}  "
        f"Baseline={baseline_score:.4f}  delta={sign}{delta:.4f}"
    )
    print("=" * 60)
    return rag_dataframe


def create_test_dataset() -> list[dict[str, str]]:
    """Return the original five-question fallback dataset."""

    return [
        {
            "question": "什么是进程？",
            "book_name": "os",
            "ground_truth": (
                "进程是正在运行的程序及其所需资源的动态实体，"
                "是操作系统进行资源分配和调度的基本单位。"
            ),
        },
        {
            "question": "进程和线程有什么区别？",
            "book_name": "os",
            "ground_truth": (
                "进程有独立的内存空间，是资源分配的基本单位；"
                "线程共享进程的内存空间，是CPU调度的基本单位。"
                "线程的创建和切换开销比进程小。"
            ),
        },
        {
            "question": "死锁的四个必要条件是什么？",
            "book_name": "os",
            "ground_truth": (
                "死锁的四个必要条件是：1)互斥条件 2)请求与保持条件 "
                "3)不可剥夺条件 4)循环等待条件。"
            ),
        },
        {
            "question": "常见的进程调度算法有哪些？",
            "book_name": "os",
            "ground_truth": (
                "常见的进程调度算法包括：先来先服务(FCFS)、短作业优先(SJF)、"
                "时间片轮转(RR)、优先级调度、多级反馈队列等。"
            ),
        },
        {
            "question": "CPU 的主要功能是什么？",
            "book_name": "computer_organization",
            "ground_truth": (
                "CPU的主要功能包括：指令控制、操作控制、时间控制、数据加工等，"
                "是计算机的运算和控制核心。"
            ),
        },
    ]
