import json
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.evaluation.generation import (
    Arm,
    GenerationCase,
    extraction_prompt,
    load_generation_cases,
    mean_pairwise_overlap,
    paired_permutation_p,
    parse_arm,
    parse_json_object,
    quote_found,
    score_answer,
    summarize,
    validate_extraction,
    validate_verification,
    verification_prompt,
)

SOURCE = {
    "citation_id": 1,
    "content": "进程是程序在一个数据集合上运行的过程，是系统进行资源分配和调度的一个独立单位。",
}


def write_cases(directory: str, cases: list[dict]) -> Path:
    path = Path(directory) / "cases.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


class ArmAndCaseTests(unittest.TestCase):
    def test_parses_generated_and_stored_arms(self):
        self.assertEqual(parse_arm("t07=baseline@0.7"), Arm("t07", "baseline", 0.7))
        self.assertEqual(parse_arm("reviewed=baseline@stored"), Arm("reviewed", "baseline", None))
        with self.assertRaises(ValueError):
            parse_arm("baseline@0.7")

    def test_rejects_sources_the_generator_never_saw(self):
        case = {
            "id": "01",
            "question": "什么是进程？",
            "variants": {"baseline": {"context": "【参考资料 1】\n别的内容", "sources": [SOURCE]}},
        }
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            load_generation_cases(write_cases(directory, [case]))

    def test_loads_requirements_and_stored_answers(self):
        variant = {"context": "前缀" + SOURCE["content"], "sources": [SOURCE], "answers": ["答"]}
        case = {
            "id": "01",
            "question": "什么是进程？",
            "requirements": ["进程的含义"],
            "variants": {"baseline": variant},
        }
        with tempfile.TemporaryDirectory() as directory:
            [loaded] = load_generation_cases(write_cases(directory, [case]))
        self.assertEqual(loaded.requirements, ("进程的含义",))
        self.assertEqual(loaded.variants["baseline"].answers, ("答",))


class QuoteTests(unittest.TestCase):
    def test_tolerates_width_whitespace_and_punctuation_style(self):
        self.assertTrue(quote_found("是系统进行资源分配和调度的 一个独立单位.", SOURCE["content"]))
        self.assertTrue(quote_found("运行的过程,是系统进行", SOURCE["content"]))

    def test_ellipsis_parts_must_appear_in_order(self):
        self.assertTrue(quote_found("进程是程序……资源分配和调度", SOURCE["content"]))
        self.assertFalse(quote_found("资源分配和调度……进程是程序", SOURCE["content"]))

    def test_short_or_invented_quotes_do_not_count(self):
        self.assertFalse(quote_found("进程", SOURCE["content"]))
        self.assertFalse(quote_found("进程是资源分配的最小单位", SOURCE["content"]))


class JudgeOutputTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        self.assertEqual(parse_json_object('```json\n{"claims": []}\n```'), {"claims": []})
        with self.assertRaises(ValueError):
            parse_json_object("没有 JSON")

    def test_extraction_requires_every_requirement(self):
        payload = {
            "claims": [{"id": 1, "type": "fact", "text": "进程是独立单位"}],
            "coverage": [{"id": 1, "addressed": True}],
        }
        claims, coverage = validate_extraction(payload, 1)
        self.assertEqual((len(claims), coverage), (1, [True]))
        with self.assertRaises(ValueError):
            validate_extraction(payload, 2)
        with self.assertRaises(ValueError):
            validate_extraction({**payload, "claims": [{"id": 1, "type": "opinion", "text": "x"}]}, 1)

    def test_verification_accepts_the_four_labels_once_per_claim(self):
        verdict = {"id": 1, "label": "supported", "evidence": [{"source_id": 1, "quote": "独立单位"}]}
        self.assertEqual(validate_verification({"verdicts": [verdict]}, [1])[1]["label"], "supported")
        minor = validate_verification({"verdicts": [{**verdict, "label": "minor"}]}, [1])
        self.assertEqual(minor[1]["label"], "minor")
        for broken in ([verdict, verdict], [], [{**verdict, "label": "partial"}]):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_verification({"verdicts": broken}, [1])

    def test_prompts_embed_the_data_as_json(self):
        self.assertIn(
            '"requirements": [{"id": 1, "text": "要求"}]',
            extraction_prompt("问题", "回答", ["要求"]),
        )
        prompt = verification_prompt([SOURCE], [{"id": 3, "text": "陈述"}])
        self.assertIn(json.dumps(SOURCE["content"], ensure_ascii=False), prompt)
        self.assertIn('"claims": [{"id": 3, "text": "陈述"}]', prompt)


class ScoreTests(unittest.TestCase):
    def test_support_without_a_findable_quote_counts_as_a_problem(self):
        claims = [
            {"id": 1, "type": "fact", "text": "进程是资源分配和调度的独立单位"},
            {"id": 2, "type": "fact", "text": "进程是资源分配的最小单位"},
            {"id": 3, "type": "fact", "text": "线程共享地址空间"},
            {"id": 4, "type": "meta", "text": "资料没有提到线程"},
        ]
        supported = {"label": "supported", "reason": ""}
        verdicts = {
            1: {**supported, "evidence": [{"source_id": 1, "quote": "资源分配和调度的一个独立单位"}]},
            2: {**supported, "evidence": [{"source_id": 1, "quote": "资源分配的最小单位"}]},
            3: {"label": "unsupported", "evidence": [], "reason": "片段没有线程"},
        }
        score = score_answer(claims, [True, False], verdicts, [SOURCE])
        self.assertEqual((score["fact_claims"], score["meta_claims"]), (3, 1))
        self.assertEqual((score["supported"], score["unverified"], score["unsupported"]), (1, 1, 1))
        self.assertEqual(score["problem_claims"], 2)
        self.assertAlmostEqual(score["problem_rate"], 2 / 3)
        self.assertEqual([row["id"] for row in score["claims"]], [1, 2, 3])
        self.assertEqual(score["coverage"], 0.5)

    def test_minor_drift_is_not_a_problem_but_counts_in_the_strict_view(self):
        claims = [
            {"id": 1, "type": "fact", "text": "进程是资源分配的重要独立单位"},
            {"id": 2, "type": "fact", "text": "进程由操作系统创建"},
        ]
        verdicts = {
            1: {"label": "minor", "reason": "加强语气",
                "evidence": [{"source_id": 1, "quote": "资源分配和调度的一个独立单位"}]},
            2: {"label": "minor", "reason": "",
                "evidence": [{"source_id": 1, "quote": "进程由操作系统创建"}]},
        }
        score = score_answer(claims, [True], verdicts, [SOURCE])
        self.assertEqual((score["minor"], score["unverified"], score["problem_claims"]), (1, 1, 1))
        self.assertAlmostEqual(score["problem_rate"], 0.5)
        self.assertAlmostEqual(score["strict_rate"], 1.0)

    def test_an_abstention_has_no_rates(self):
        score = score_answer([{"id": 1, "type": "meta", "text": "证据不足"}], [False], {}, [SOURCE])
        self.assertIsNone(score["problem_rate"])
        self.assertIsNone(score["strict_rate"])
        self.assertEqual((score["problem_claims"], score["coverage"]), (0, 0.0))


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_p_values(self):
        self.assertEqual(paired_permutation_p([1.0] * 5), 2 / 32)
        self.assertEqual(paired_permutation_p([0.0, 0.0, 0.0]), 1.0)
        self.assertEqual(paired_permutation_p([1.0, -1.0]), 1.0)

    def test_overlap_separates_identical_from_different_answers(self):
        self.assertEqual(mean_pairwise_overlap(["进程是独立单位", "进程是独立单位"]), 1.0)
        self.assertLess(mean_pairwise_overlap(["进程是独立单位", "线程共享地址空间"]), 0.2)
        self.assertIsNone(mean_pairwise_overlap(["只有一个样本"]))

    def test_summary_averages_within_cases_before_pairing(self):
        def record(case_id: str, arm: str, problems: int) -> dict:
            return {
                "case_id": case_id,
                "arm": arm,
                "answer": f"{case_id}-{arm}-{problems}-{len(case_id + arm)}",
                "finish_reason": "stop",
                "score": {
                    "fact_claims": 4,
                    "problem_claims": problems,
                    "problem_rate": problems / 4,
                    "strict_rate": problems / 4,
                    "coverage": 1.0,
                },
            }

        cases = [GenerationCase("01", "问", (), {}), GenerationCase("02", "问", (), {})]
        arms = [Arm("hot", "baseline", 0.7), Arm("cool", "baseline", 0.2)]
        records = [
            record("01", "hot", 2), record("01", "hot", 1), record("01", "cool", 1),
            record("02", "hot", 0), record("02", "cool", 0), record("02", "cool", 1),
        ]
        report = summarize(records, cases, arms)

        self.assertAlmostEqual(report["arms"]["hot"]["problem_claims"], 0.75)
        comparison = report["comparisons"]["cool"]["problem_claims"]
        self.assertAlmostEqual(comparison["mean_difference"], 0.0)
        self.assertEqual((comparison["lower"], comparison["higher"]), (1, 1))
        self.assertAlmostEqual(report["cases"]["01"]["hot"]["problem_share"], 1.0)
        self.assertTrue(report["cases"]["02"]["cool"]["mixed_verdicts"])
        self.assertEqual(report["arms"]["cool"]["mixed_verdicts"], 1)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
