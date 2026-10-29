import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.evaluation.paired_preparation import (
    CASE_IDS,
    GENERATOR_MODEL,
    SCREENING_MODEL,
    SCREENING_PROMPT,
    cost_scenario,
    prepare_plan,
    screening_request,
    validate_screening,
)
from rag_textbook_qa.rag.context import build_prompt
from scripts.prepare_paired_generation import main

COPY = "Cache把常访问的数据副本保存在快速SRAM中。"
BOTTLENECK = "主存速度已经成为瓶颈，可以通过并行工作提高存取速度。"
ANSWER = "主存是瓶颈，副本保存在SRAM中。【参考资料 2】"


def fixtures():
    sources = [{"citation_id": 1, "content": COPY}, {"citation_id": 2, "content": BOTTLENECK}]
    context = "【参考资料 1】\n内容:\n" + COPY + "\n---\n【参考资料 2】\n内容:\n" + BOTTLENECK
    extra = {"citation_id": 3, "content": "这是候选补充的完整解释，只作为合成测试资料，不是验收金标准。"}
    cases = [{"id": cid, "question": f"{cid} 的工作原理是什么？", "arms": {
        "baseline": {"context": context, "sources": sources},
        "focused_parent": {"context": context + "\n【参考资料 3】\n" + extra["content"],
                           "sources": [*sources, extra]},
    }} for cid in CASE_IDS]
    report = {"split": "dev", "reuses_saved_rankings": True, "cases": cases,
              "index_revision": "fixture-index", "dataset_sha256": "fixture-dataset"}
    evidence = [{"id": cid, "point": "数据副本机制"} for cid in CASE_IDS]
    return report, evidence


def generation(job):
    return {"label": f"gen-{job['ordinal']:02d}-{job['case_id']}-{job['arm']}-{job['repeat']}",
            "role": "generator", "model": GENERATOR_MODEL, "request_sha256": job["request_sha256"],
            "content": ANSWER, "finish_reason": "stop", "complete": True}


class PairedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.report, self.evidence = fixtures()
        self.plan = prepare_plan(self.report, self.evidence)
        self.job = self.plan["generation_requests"][0]
        self.generation = generation(self.job)

    def screening(self):
        request = screening_request(self.plan, self.job, self.generation)
        payload = {"claims": [{
            "id": "C1", "answer_quote": "副本保存在SRAM中", "statement": "数据副本保存在SRAM中",
            "qualifiers": ["SRAM"], "citation_markers": ["【参考资料 2】"],
            "context_support": "supported", "citation_support": "unsupported",
            "context_evidence": [{"source_id": 1, "quote": COPY}],
            "citation_evidence": [{"source_id": 2, "quote": BOTTLENECK}],
            "reason": "原引用2只支持瓶颈，副本说明来自来源1。",
        }], "coverage": [{"id": 1, "status": "covered", "reason": "回应副本机制"}], "omissions": []}
        response = {"role": "screening", "model": SCREENING_MODEL,
                    "generator_label": request["generator_label"],
                    "request_sha256": request["request_sha256"], "finish_reason": "stop", "complete": True,
                    "content": json.dumps(payload, ensure_ascii=False),
                    "recorded_at_utc": "2026-10-29T00:00:00Z"}
        return payload, response

    def test_thirty_requests_are_blocked_paired_and_only_context_differs(self):
        jobs = self.plan["generation_requests"]
        self.assertEqual(len(jobs), 30)
        self.assertEqual(jobs, prepare_plan(self.report, self.evidence)["generation_requests"])
        self.assertNotEqual(jobs, prepare_plan(self.report, self.evidence, seed=5)["generation_requests"])
        by_id = {c["id"]: c for c in self.plan["cases"]["cases"]}
        for i in range(0, 30, 2):
            a, b = jobs[i:i + 2]
            self.assertEqual((a["case_id"], a["repeat"]), (b["case_id"], b["repeat"]))
            self.assertEqual({a["arm"], b["arm"]}, {"baseline", "focused"})
            self.assertEqual({k: v for k, v in a["request"].items() if k != "messages"},
                             {k: v for k, v in b["request"].items() if k != "messages"})
        for job in jobs:
            case = by_id[job["case_id"]]
            self.assertEqual(job["request"]["messages"][0]["content"],
                             build_prompt(case["question"], case["variants"][job["arm"]]["context"]))

    def test_provider_parameters_and_execution_remain_explicit_and_disabled(self):
        protocol = self.plan["protocol"]
        self.assertFalse(protocol["execution_enabled"])
        self.assertFalse(protocol["budget_authorized"])
        self.assertFalse(protocol["legacy_llm_config_fallback"])
        self.assertFalse(protocol["siliconflow_allowed"])
        self.assertEqual(self.job["request"]["thinking"], {"type": "disabled"})
        request = screening_request(self.plan, self.job, self.generation)["request"]
        self.assertEqual(request["max_completion_tokens"], 8192)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertFalse({"temperature", "top_p", "max_tokens", "thinking"} & request.keys())
        data = json.loads(request["messages"][0]["content"][len(SCREENING_PROMPT):])
        self.assertIn("【参考资料 1】", data["sources"][0]["content"])
        self.assertEqual(data["answer_body"], ANSWER)

    def test_incomplete_wrong_provider_or_stale_answers_cannot_trigger_screening(self):
        for key, value in (("finish_reason", "length"), ("complete", False), ("content", ""),
                           ("model", "deepseek-ai/DeepSeek-V4-Pro"), ("request_sha256", "wrong"),
                           ("label", "another-paid-sample")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                screening_request(self.plan, self.job, {**self.generation, key: value})

    def test_large_answer_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, "不截断"):
            screening_request(self.plan, self.job, {**self.generation, "content": "中文" * 20000})

    def test_changed_materials_or_screening_config_cannot_reuse_a_generation(self):
        for change in ("context", "options", "checklist"):
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan = copy.deepcopy(self.plan)
                if change == "context":
                    case = next(c for c in plan["cases"]["cases"] if c["id"] == self.job["case_id"])
                    case["variants"][self.job["arm"]]["context"] += "另加的未送入生成模型的解释。"
                elif change == "options":
                    plan["screening_spec"]["request_options"]["prompt_cache_options"]["ttl"] = "1h"
                else:
                    plan["cases"]["cases"][0]["requirements"].append("后来改写的要点")
                screening_request(plan, self.job, self.generation)
        self.assertEqual(prepare_plan(self.report, self.evidence)["screening_spec"]["request_options"]
                         ["prompt_cache_options"]["ttl"], "5m")

    def test_returned_requests_do_not_mutate_frozen_template_options(self):
        first = screening_request(self.plan, self.job, self.generation)
        first["request"]["prompt_cache_options"]["ttl"] = "1h"
        self.assertEqual(screening_request(self.plan, self.job, self.generation)["request"]
                         ["prompt_cache_options"]["ttl"], "5m")

    def test_original_wrong_citation_is_kept_even_when_other_source_supports(self):
        _, response = self.screening()
        result = validate_screening(self.plan, self.job, self.generation, response)
        self.assertTrue(result["literal_bindings_valid"])
        self.assertFalse(result["semantic_validity_verified"])
        row = result["claim_review"]["claims"][0]
        self.assertEqual(row["original_citation_ids"], [2])
        self.assertEqual(row["context_support"], "supported")
        self.assertEqual(row["citation_support"], "unsupported")

    def test_truncated_ambiguous_or_misbinding_screening_does_not_validate(self):
        payload, response = self.screening()
        broken = []
        item = copy.deepcopy(payload)
        item["claims"][0]["citation_evidence"] = [{"source_id": 1, "quote": COPY}]
        broken.append(json.dumps(item))
        item = copy.deepcopy(payload)
        item["coverage"] = []
        broken.append(json.dumps(item))
        broken.append('{"claims":[],"claims":[],"coverage":[],"omissions":[]}')
        for raw in broken:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_screening(self.plan, self.job, self.generation, {**response, "content": raw})
        with self.assertRaises(ValueError):
            validate_screening(self.plan, self.job, self.generation, {**response, "finish_reason": "length"})

    def test_ambiguous_repeated_quote_is_not_assigned_a_guessed_span(self):
        answer = {**self.generation, "content": ANSWER + "\n\n" + ANSWER}
        request = screening_request(self.plan, self.job, answer)
        _, response = self.screening()
        response["request_sha256"] = request["request_sha256"]
        with self.assertRaisesRegex(ValueError, "唯一绑定"):
            validate_screening(self.plan, self.job, answer, response)

    def test_changed_arms_holdout_missing_points_and_duplicate_ids_are_rejected(self):
        for change in ("split", "duplicate", "lost_baseline", "missing_points"):
            report, evidence = copy.deepcopy((self.report, self.evidence))
            if change == "split":
                report["split"] = "holdout"
            elif change == "duplicate":
                report["cases"].append(report["cases"][0])
            elif change == "lost_baseline":
                report["cases"][0]["arms"]["focused_parent"]["sources"] = []
            else:
                evidence = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_plan(report, evidence)

    def test_cost_is_a_disclosed_scenario_not_authorization_or_a_bill(self):
        result = cost_scenario(self.plan)
        self.assertFalse(result["budget_authorized"])
        self.assertTrue(result["not_a_hard_billing_upper_bound"])
        self.assertGreater(result["total_scenario_usd"], result["generation_scenario_usd"])
        self.assertEqual(result["kimi_5m_cache_write_per_million"], 2)
        self.assertTrue(result["kimi_cache_breakdown_partitions_prompt_tokens"])
        self.assertAlmostEqual(result["screening_scenario_usd"], 4.43136)

    def test_cli_checks_frozen_inputs_and_refuses_old_output_without_any_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            source.write_text("合成原文")
            evidence = [{**e, "source_path": str(source),
                         "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                        for e in self.evidence]
            report = root / "report.json"
            report.write_text(json.dumps(self.report))
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({"replay_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                                         "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
                                         "index_revision": "fixture-index", "dataset_sha256": "fixture-dataset"}))
            ev = root / "evidence.json"
            ev.write_text(json.dumps(evidence))
            output = root / "out"
            argv = ["prepare", "prepare", "--replay-report", str(report), "--candidate-freeze", str(freeze),
                    "--evidence", str(ev), "--output-dir", str(output)]
            with patch("socket.socket.connect", side_effect=AssertionError("No networking")), patch("sys.argv", argv):
                main()
            before = (output / "plan.json").read_bytes()
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()
            self.assertEqual((output / "plan.json").read_bytes(), before)
            source.write_text("原文已修改")
            with patch("sys.argv", [*argv[:-1], str(root / "other")]), self.assertRaises(SystemExit):
                main()
            self.assertFalse((root / "other").exists())


if __name__ == "__main__":
    unittest.main()
