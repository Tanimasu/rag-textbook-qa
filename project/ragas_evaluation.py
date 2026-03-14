"""
ragas_evaluation.py
使用 RAGAS 框架评估 RAG 系统质量，计算忠实度、相关性、上下文精确度等指标。
运行方式：python ragas_evaluation.py
"""
import json
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_engine import RAGEngine

# 评估使用的 API 配置（与 RAG 引擎的 API 可以不同）
API_KEY  = os.getenv("RAGAS_API_KEY")  or os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("RAGAS_API_BASE") or os.getenv("LLM_API_BASE", "https://api.ohmygpt.com/v1")
MODEL    = os.getenv("RAGAS_MODEL")    or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")

DB_PATH      = "./vector_db"
RUN_BASELINE = False   # 设为 True 才运行无 RAG 的 baseline 对比（费 token）


class RAGASEvaluator:
    """RAGAS 评估器，通过 LangChain 调用 LLM 计算各项评估指标。"""

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, model: str = MODEL):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = base_url

        print(f"初始化 RAGAS 评估器...")
        print(f"  API: {base_url}")
        print(f"  模型: {model}")

        # LLM
        try:
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.0,
                request_timeout=60,
                max_retries=3,
            )
            print("  LLM 初始化成功")
        except Exception as e:
            print(f"  LLM 初始化失败: {e}")
            raise

        # Embeddings（优先用 API，失败则降级到本地模型）
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key,
                openai_api_base=base_url,
                request_timeout=60,
            )
            print("  Embeddings 初始化成功（API）")
        except Exception as e:
            print(f"  API Embeddings 不可用 ({e})，降级到本地 HuggingFace 模型")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("  Embeddings 初始化成功（本地）")

        print("初始化完成\n")

    def prepare_evaluation_data(self, rag_engine: RAGEngine, test_questions: List[Dict]) -> Dataset:
        """
        用 RAG 引擎逐条回答测试问题，收集 (question, answer, contexts, ground_truth)，
        返回 RAGAS Dataset。
        """
        print("=" * 60)
        print("准备评估数据")
        print("=" * 60)

        questions, answers, contexts, ground_truths = [], [], [], []
        qa_records = []  # 用于写入对比文件

        for i, item in enumerate(test_questions, 1):
            question = item["question"]
            print(f"[{i}/{len(test_questions)}] {question}")

            try:
                result = rag_engine.ask(
                    query=question,
                    book_name=item.get("book_name"),
                    top_k=8,
                    use_llm=True,
                )

                if result["success"]:
                    gt = item.get("ground_truth", "")
                    questions.append(question)
                    answers.append(result["answer"])
                    contexts.append([
                        f"[{r['book_name']} - {r['chapter']} - {r['section_h2']}]\n{r['content']}"
                        for r in result["results"]
                    ])
                    ground_truths.append(gt)
                    qa_records.append({
                        "question": question,
                        "answer": result["answer"],
                        "ground_truth": gt,
                    })
                    print(f"  [回答] {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")
                    if gt:
                        print(f"  [标准] {gt[:300]}{'...' if len(gt) > 300 else ''}")
                    print()
                else:
                    print(f"  失败: {result.get('error', 'unknown')}")

            except Exception as e:
                print(f"  异常: {e}")

        if not questions:
            raise ValueError("没有成功处理任何问题，无法评估。")

        # 将问题、回答、标准答案写入对比文件
        qa_out = "ragas_qa_comparison.json"
        with open(qa_out, "w", encoding="utf-8") as f:
            json.dump(qa_records, f, ensure_ascii=False, indent=2)
        print(f"问答对比已保存到: {qa_out}")

        print(f"\n数据准备完成：{len(questions)} 条\n")

        data = {"question": questions, "answer": answers, "contexts": contexts}
        if any(ground_truths):
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    def evaluate(self, dataset: Dataset, metrics: List = None) -> Dict:
        """执行 RAGAS 评估，返回结果字典。"""
        if metrics is None:
            metrics = [faithfulness, context_precision]
            if self.embeddings:
                metrics.append(answer_relevancy)
            if "ground_truth" in dataset.column_names and any(dataset["ground_truth"]):
                metrics.append(context_recall)

        print("=" * 60)
        print(f"开始评估  数据集: {len(dataset)} 条  指标: {[m.name for m in metrics]}")
        print("=" * 60)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,  # 单条失败不中断整体评估
            run_config=RunConfig(max_retries=5, timeout=180, max_workers=2),
        )

        print("评估完成\n")
        return result

    def print_results(self, result):
        """打印评估结果，含进度条和评级，并返回 DataFrame（如可用）。"""
        print("=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)

        df = None
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            # 只保留问题和各指标列，去掉回答正文、标准答案和检索上下文
            drop_cols = [c for c in df.columns if c in {"response", "answer", "retrieved_contexts", "contexts", "reference", "ground_truth"}]
            df = df.drop(columns=drop_cols, errors="ignore")

        # 检测并报告 NaN（评估 LLM 输出格式异常时产生）
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns
            nan_mask = df[numeric_cols].isna()
            if nan_mask.any().any():
                print("⚠️  以下问题的部分指标评估失败（NaN），已从均值计算中排除：")
                question_col = next((c for c in df.columns if c in {"user_input", "question"}), None)
                for idx, row in nan_mask.iterrows():
                    failed = [col for col in numeric_cols if row[col]]
                    if failed:
                        q = df.loc[idx, question_col][:40] + "..." if question_col else f"第{idx+1}条"
                        print(f"  [{idx+1}] {q}  →  {', '.join(failed)}")
                print()

        # 获取各指标的汇总分数
        # 新版 RAGAS EvaluationResult 不再是 dict，用 to_pandas() 取列均值
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns
            scores = {col: float(df[col].mean()) for col in numeric_cols}
        else:
            # 兼容旧版（dict-like）
            try:
                scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}
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

        if df is not None:
            print("逐条详情:")
            print(df.to_string())
            print()
            return df

        return None


    def prepare_baseline_data(self, rag_engine: RAGEngine, test_questions: List[Dict]) -> Dataset:
        """
        不使用 RAG，直接让 LLM 回答问题，作为 baseline 对比。
        上下文传空字符串，仅评估 answer_relevancy。
        """
        print("=" * 60)
        print("准备 Baseline 数据（无 RAG，直接 LLM）")
        print("=" * 60)

        questions, answers, contexts, ground_truths = [], [], [], []

        for i, item in enumerate(test_questions, 1):
            question = item["question"]
            print(f"[{i}/{len(test_questions)}] {question}")
            try:
                prompt = (
                    "你是一个计算机课程的专业 AI 助教，请回答以下问题。"
                    "先给出简明答案（2-3句话），再给出详细解释。\n\n"
                    f"问题：{question}\n\n请开始你的回答："
                )
                result = rag_engine.llm.generate_answer(prompt, temperature=0.7, max_tokens=2000)
                if result["success"]:
                    questions.append(question)
                    answers.append(result["answer"])
                    contexts.append([""])          # 无检索上下文
                    ground_truths.append(item.get("ground_truth", ""))
                    print(f"  [回答] {result['answer'][:200]}...")
                else:
                    print(f"  失败: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"  异常: {e}")

        if not questions:
            raise ValueError("没有成功处理任何问题，无法评估。")

        print(f"\nBaseline 数据准备完成：{len(questions)} 条\n")
        data = {"question": questions, "answer": answers, "contexts": contexts}
        if any(ground_truths):
            data["ground_truth"] = ground_truths
        return Dataset.from_dict(data)


def create_test_dataset() -> List[Dict]:
    """内置测试问题集（也可以直接使用 eval_dataset.json）。"""
    return [
        {"question": "什么是进程？",           "book_name": "os",                   "ground_truth": "进程是正在运行的程序及其所需资源的动态实体，是操作系统进行资源分配和调度的基本单位。"},
        {"question": "进程和线程有什么区别？",   "book_name": "os",                   "ground_truth": "进程有独立的内存空间，是资源分配的基本单位；线程共享进程的内存空间，是CPU调度的基本单位。线程的创建和切换开销比进程小。"},
        {"question": "死锁的四个必要条件是什么？","book_name": "os",                   "ground_truth": "死锁的四个必要条件是：1)互斥条件 2)请求与保持条件 3)不可剥夺条件 4)循环等待条件。"},
        {"question": "常见的进程调度算法有哪些？","book_name": "os",                   "ground_truth": "常见的进程调度算法包括：先来先服务(FCFS)、短作业优先(SJF)、时间片轮转(RR)、优先级调度、多级反馈队列等。"},
        {"question": "CPU 的主要功能是什么？",   "book_name": "computer_organization", "ground_truth": "CPU的主要功能包括：指令控制、操作控制、时间控制、数据加工等，是计算机的运算和控制核心。"},
    ]


if __name__ == "__main__":
    # 初始化 RAG 引擎
    rag = RAGEngine(db_path=DB_PATH, enable_llm=True, verbose=False, enable_hyde=True)

    # 初始化评估器
    evaluator = RAGASEvaluator()

    # 加载测试题（优先读文件，否则用内置集）
    try:
        with open("test_questions.json", encoding="utf-8") as f:
            test_questions = json.load(f)
    except FileNotFoundError:
        print("test_questions.json 不存在，使用内置测试集")
        test_questions = create_test_dataset()

    # ── 1. RAG 评估 ──────────────────────────────────────────
    rag_dataset = evaluator.prepare_evaluation_data(rag, test_questions)
    rag_result  = evaluator.evaluate(rag_dataset)
    print("\n【RAG 系统评估结果】")
    rag_df = evaluator.print_results(rag_result)
    if rag_df is not None:
        rag_df.to_csv("ragas_evaluation_results.csv", index=False, encoding="utf-8-sig")
        print("结果已保存到: ragas_evaluation_results.csv")

    # ── 2. Baseline 评估（无 RAG，直接 LLM）──────────────────
    if RUN_BASELINE:
        baseline_dataset = evaluator.prepare_baseline_data(rag, test_questions)
        baseline_result  = evaluator.evaluate(baseline_dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        print("\n【Baseline 评估结果（无 RAG）】")
        base_df = evaluator.print_results(baseline_result)
        if base_df is not None:
            base_df.to_csv("ragas_baseline_results.csv", index=False, encoding="utf-8-sig")
            print("结果已保存到: ragas_baseline_results.csv")

        # ── 3. 对比摘要 ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("RAG vs Baseline 对比摘要")
        print("=" * 60)
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            rag_score  = float(rag_df[metric].mean())  if rag_df  is not None and metric in rag_df.columns  else float("nan")
            base_score = float(base_df[metric].mean()) if base_df is not None and metric in base_df.columns else float("nan")
            delta = rag_score - base_score
            sign  = "+" if delta >= 0 else ""
            print(f"  {metric:20s}  RAG={rag_score:.4f}  Baseline={base_score:.4f}  delta={sign}{delta:.4f}")
        print("=" * 60)
    else:
        print("\n（Baseline 对比已跳过，设 RUN_BASELINE=True 可开启）")
