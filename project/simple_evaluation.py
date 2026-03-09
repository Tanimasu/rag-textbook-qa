"""
simple_evaluation.py
不依赖 RAGAS 的轻量评估方案，通过关键词重叠计算答案质量。
指标：整体相似度（Jaccard）、关键词覆盖率、长度合适度、综合得分。
结果同时保存为 JSON 和 Excel。
运行方式：python simple_evaluation.py
"""
import json
import re
import time
from typing import List, Dict

import pandas as pd
from rag_engine import RAGEngine

DB_PATH       = "./vector_db"
EVAL_DATASET  = "./eval_dataset.json"
OUTPUT_JSON   = "./simple_eval_results.json"
OUTPUT_EXCEL  = "./simple_eval_results.xlsx"


class SimpleEvaluator:
    """
    轻量 RAG 评估器。
    无需外部评估服务，仅用中文关键词匹配衡量答案质量。
    综合得分 = 相似度×0.4 + 关键词覆盖率×0.4 + 长度合适度×0.2
    """

    def __init__(self, db_path: str = DB_PATH, eval_dataset_path: str = EVAL_DATASET):
        self.eval_dataset_path = eval_dataset_path
        self.rag = RAGEngine(db_path=db_path, enable_llm=True, verbose=False)
        print(f"评估器初始化完成  DB: {db_path}  数据集: {eval_dataset_path}\n")

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def load_eval_dataset(self) -> List[Dict]:
        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data["questions"]
        print(f"加载评估数据集：{len(questions)} 条问题")
        return questions

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    def _extract_chinese_words(self, text: str, min_len: int = 2) -> set:
        """提取长度 >= min_len 的中文词。"""
        return {w for w in re.findall(r"[\u4e00-\u9fa5]+", text) if len(w) >= min_len}

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """两段文本的中文关键词 Jaccard 相似度。"""
        w1, w2 = self._extract_chinese_words(text1), self._extract_chinese_words(text2)
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _keyword_coverage(self, answer: str, ground_truth: str) -> Dict:
        """
        计算答案对标准答案关键词的覆盖情况。
        返回 coverage（0-1）、matched、missed 三个字段。
        """
        gt_words  = self._extract_chinese_words(ground_truth)
        ans_words = self._extract_chinese_words(answer)
        matched   = gt_words & ans_words
        missed    = gt_words - ans_words
        coverage  = len(matched) / len(gt_words) if gt_words else 0.0
        return {"coverage": coverage, "matched": list(matched), "missed": list(missed)}

    def _length_score(self, answer: str, ground_truth: str) -> float:
        """
        答案长度合适度（0-1）。
        长度在标准答案的 0.8~2.0 倍范围内得满分，超出则线性衰减。
        """
        gt_len = len(ground_truth)
        if gt_len == 0:
            return 0.0
        ratio = len(answer) / gt_len
        if 0.8 <= ratio <= 2.0:
            return 1.0
        return (ratio / 0.8) if ratio < 0.8 else (2.0 / ratio)

    def _score_answer(self, answer: str, ground_truth: str, contexts: List[str]) -> Dict:
        """计算单条答案的所有指标，返回指标字典。"""
        similarity = self._jaccard_similarity(answer, ground_truth)
        coverage   = self._keyword_coverage(answer, ground_truth)
        length     = self._length_score(answer, ground_truth)
        overall    = similarity * 0.4 + coverage["coverage"] * 0.4 + length * 0.2

        return {
            "similarity":        similarity,
            "keyword_coverage":  coverage["coverage"],
            "length_score":      length,
            "overall_score":     overall,
            "context_count":     len(contexts),
            "matched_keywords":  coverage["matched"],
            "missed_keywords":   coverage["missed"],
            "answer_length":     len(answer),
            "ground_truth_length": len(ground_truth),
        }

    # ------------------------------------------------------------------
    # 主评估流程
    # ------------------------------------------------------------------
    def run_evaluation(self) -> List[Dict]:
        """逐条运行评估，打印进度，返回结果列表。"""
        questions = self.load_eval_dataset()
        results   = []

        print("=" * 60)
        for i, item in enumerate(questions, 1):
            question    = item["question"]
            ground_truth = item["ground_truth"]
            book_name   = item.get("book_name")

            print(f"[{i}/{len(questions)}] {question}")

            try:
                rag_result = self.rag.ask(
                    query=question, book_name=book_name, top_k=5, use_llm=True
                )

                if rag_result["success"]:
                    answer   = rag_result["answer"]
                    contexts = [r.get("content", "") for r in rag_result["results"]]
                    metrics  = self._score_answer(answer, ground_truth, contexts)

                    print(f"  相似度 {metrics['similarity']:.2f}  "
                          f"关键词覆盖 {metrics['keyword_coverage']:.2f}  "
                          f"长度 {metrics['length_score']:.2f}  "
                          f"综合 {metrics['overall_score']:.2f}")
                    if metrics["missed_keywords"]:
                        print(f"  遗漏关键词: {', '.join(metrics['missed_keywords'][:5])}")

                    results.append({
                        "question": question, "answer": answer,
                        "ground_truth": ground_truth, "book_name": book_name,
                        "metrics": metrics, "success": True,
                    })
                else:
                    print(f"  RAG 失败: {rag_result.get('error', 'unknown')}")
                    results.append({"question": question, "success": False})

            except Exception as e:
                print(f"  异常: {e}")
                results.append({"question": question, "success": False})

            time.sleep(0.5)

        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # 汇总与保存
    # ------------------------------------------------------------------
    def _print_summary(self, results: List[Dict]):
        """打印汇总统计。"""
        successful = [r for r in results if r.get("success")]
        if not successful:
            print("没有成功的评估结果。")
            return

        n = len(successful)
        metrics_keys = ["similarity", "keyword_coverage", "length_score", "overall_score"]
        avgs = {k: sum(r["metrics"][k] for r in successful) / n for k in metrics_keys}

        print("\n" + "=" * 60)
        print(f"评估总结  成功 {n}/{len(results)} 条")
        print("=" * 60)
        labels = {"similarity": "整体相似度", "keyword_coverage": "关键词覆盖率",
                  "length_score": "长度合适度", "overall_score": "综合得分"}
        for k, label in labels.items():
            bar = "█" * int(avgs[k] * 20) + "░" * (20 - int(avgs[k] * 20))
            print(f"  {label:12s} | {bar} | {avgs[k]:.4f}")

        best  = max(successful, key=lambda r: r["metrics"]["overall_score"])
        worst = min(successful, key=lambda r: r["metrics"]["overall_score"])
        print(f"\n  最高分: {best['question']}  ({best['metrics']['overall_score']:.4f})")
        print(f"  最低分: {worst['question']}  ({worst['metrics']['overall_score']:.4f})")
        print("=" * 60)

    def save_results(self, results: List[Dict],
                     json_path: str = OUTPUT_JSON, excel_path: str = OUTPUT_EXCEL):
        """将评估结果保存为 JSON 和 Excel。"""
        successful = [r for r in results if r.get("success")]

        # JSON
        summary = {}
        if successful:
            n = len(successful)
            summary = {
                "total": len(results), "successful": n,
                "success_rate": n / len(results),
                "avg_similarity":       sum(r["metrics"]["similarity"]       for r in successful) / n,
                "avg_keyword_coverage": sum(r["metrics"]["keyword_coverage"] for r in successful) / n,
                "avg_length_score":     sum(r["metrics"]["length_score"]     for r in successful) / n,
                "avg_overall_score":    sum(r["metrics"]["overall_score"]    for r in successful) / n,
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=2)
        print(f"JSON 结果已保存: {json_path}")

        # Excel
        rows = [{
            "问题":     r["question"],
            "教材":     r.get("book_name", ""),
            "综合得分": f"{r['metrics']['overall_score']:.2f}",
            "相似度":   f"{r['metrics']['similarity']:.2f}",
            "关键词覆盖": f"{r['metrics']['keyword_coverage']:.2f}",
            "长度得分": f"{r['metrics']['length_score']:.2f}",
            "答案长度": r["metrics"]["answer_length"],
            "上下文数": r["metrics"]["context_count"],
            "遗漏关键词": ", ".join(r["metrics"]["missed_keywords"][:3]),
        } for r in successful]

        pd.DataFrame(rows).to_excel(excel_path, index=False, engine="openpyxl")
        print(f"Excel 报告已保存: {excel_path}")


if __name__ == "__main__":
    evaluator = SimpleEvaluator()
    results   = evaluator.run_evaluation()
    evaluator.save_results(results)
