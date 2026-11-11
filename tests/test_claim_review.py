import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_textbook_qa.evaluation.claim_review import (
    citation_markers,
    digest,
    encoded,
    prepare_packet,
    review_template,
    validate_review,
)
from scripts.prepare_claim_review import main

MECHANISM = "Cache将主存中经常访问的数据副本放在快速的SRAM中。"
BOTTLENECK = "主存速度是瓶颈，并行工作可以提高存取速度。"
CONTEXT = "【参考资料 1】\n" + MECHANISM + "\n【参考资料 2】\n" + BOTTLENECK
ANSWER = ("副本机制的概述。【参考资料 1】\n\n"
          "主存速度是瓶颈，Cache将数据副本保存在SRAM中。【参考资料 2】\n\n"
          "## 参考章节\n章节信息【参考资料 1】")


class ClaimReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.batch = Path(temporary.name) / "batch"
        self.batch.mkdir()
        (self.batch / "responses").mkdir()
        self.case = {"cases": [{"id": "test", "question": "Cache如何工作？", "requirements": [],
                              "variants": {"base": {"context": CONTEXT, "sources": [
                                  {"citation_id": 1, "content": MECHANISM},
                                  {"citation_id": 2, "content": BOTTLENECK},
                              ]}}}]}
        request = {"model": "fixture-no-calls", "messages": [{
            "role": "user", "content": "Cache如何工作？\n" + CONTEXT}], "temperature": 0.7}
        self.job = {"ordinal": 0, "case_id": "test", "arm": "base", "repeat": 0,
                    "request": request, "request_sha256": digest(encoded(request))}
        self.response = {"label": "gen-00-test-base-0", "role": "generator", "model": "fixture-no-calls",
                         "request_sha256": self.job["request_sha256"], "content": ANSWER,
                         "complete": True, "finish_reason": "stop"}
        self.write()

    def write(self):
        (self.batch / "generation-cases.json").write_text(json.dumps(self.case), encoding="utf-8")
        (self.batch / "generation-requests.json").write_text(json.dumps([self.job]), encoding="utf-8")
        (self.batch / "responses/gen-00-test-base-0.json").write_text(json.dumps(self.response), encoding="utf-8")

    def packet(self):
        return prepare_packet(self.batch, run_label="合成资料离线检查")

    def review(self, packet):
        reviewed = review_template(packet)
        reviewed.update(review_status="partial", reviewer="fixture reviewer", reviewer_kind="model",
                        reviewed_at_utc="2026-10-29T00:00:00+00:00")
        body = packet["records"][0]["answer_body"]
        quote = "Cache将数据副本保存在SRAM中"
        start = body.index(quote)
        marker = next(m for m in citation_markers(body) if m["citation_ids"] == [2])
        reviewed["records"][0].update(status="targeted", claims=[{
            "id": "C1", "start": start, "end": start + len(quote), "statement": quote,
            "answer_qualifiers": ["SRAM"], "citation_marker_offsets": [marker["start"]],
            "context_support": "supported", "citation_support": "unsupported",
            "context_evidence": [{"source_id": 1, "quote": MECHANISM}],
            "citation_evidence": [{"source_id": 2, "quote": BOTTLENECK}],
            "reason": "实际引用2只说明瓶颈，副本机制在来源1。",
        }])
        return reviewed

    def test_packet_preserves_actual_answer_prompt_sources_and_pending_labels(self):
        before = {p: p.read_bytes() for p in self.batch.rglob("*.json")}
        packet = self.packet()
        row = packet["records"][0]
        self.assertEqual(row["answer"], ANSWER)
        self.assertEqual(row["context"], CONTEXT)
        self.assertEqual(row["prompt"], self.job["request"]["messages"][0]["content"])
        self.assertNotIn("参考章节", row["answer_body"])
        self.assertEqual([m["citation_ids"] for m in row["citation_markers"]], [[1], [2]])
        self.assertEqual(review_template(packet)["records"][0]["claims"], [])
        self.assertEqual(before, {p: p.read_bytes() for p in self.batch.rglob("*.json")})

    def test_context_support_cannot_repair_an_original_wrong_citation(self):
        packet = self.packet()
        result = validate_review(packet, self.review(packet))
        self.assertEqual(result["context_support_counts"], {"supported": 1})
        self.assertEqual(result["citation_support_counts"], {"unsupported": 1})
        self.assertEqual(result["claims"][0]["original_citation_ids"], [2])
        self.assertFalse(result["identity_atomicity_semantics_independence_verified"])

    def test_rejects_borrowed_support_and_qualifiers_or_replaced_raw_quotes(self):
        packet = self.packet()
        reviewed = self.review(packet)
        first_marker = packet["records"][0]["citation_markers"][0]["start"]
        mutations = {
            "citation_evidence": [{"source_id": 1, "quote": MECHANISM}],
            "context_evidence": [{"source_id": 2, "quote": MECHANISM}],
            "answer_qualifiers": ["只限非负整数"],
            "citation_marker_offsets": [first_marker],
            "answer_quote": "编造替换原文",
            "start": True,
        }
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                altered = copy.deepcopy(reviewed)
                altered["records"][0]["claims"][0][key] = value
                validate_review(packet, altered)

    def test_empty_pending_missing_and_truncated_are_not_complete_reviews(self):
        packet = self.packet()
        with self.assertRaises(ValueError):
            validate_review(packet, review_template(packet))
        reviewed = self.review(packet)
        reviewed["review_status"] = "complete"
        with self.assertRaises(ValueError):
            validate_review(packet, reviewed)
        for missing in (True, False):
            with self.subTest(missing=missing):
                if missing:
                    (self.batch / "responses/gen-00-test-base-0.json").unlink()
                else:
                    self.response["finish_reason"] = "length"
                    self.response["complete"] = False
                    self.write()
                row = self.packet()["records"][0]
                self.assertFalse(row["eligible_complete_answer"])
                self.assertEqual(row["recorded_finish_reason"], None if missing else "length")

    def test_repeated_identical_requests_bind_by_sample_not_only_hash(self):
        second = {**self.job, "ordinal": 1, "repeat": 1}
        (self.batch / "generation-requests.json").write_text(json.dumps([self.job, second]))
        other = {**self.response, "label": "gen-01-test-base-1", "content": "另一份完整回答。"}
        (self.batch / "responses/gen-01-test-base-1.json").write_text(json.dumps(other))
        rows = self.packet()["records"]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["answer"], rows[1]["answer"])
        self.assertEqual(rows[0]["request_sha256"], rows[1]["request_sha256"])

    def test_rejects_stale_context_hash_model_and_duplicate_sample_keys(self):
        mutations = [("request_sha256", "wrong"), ("model", "different-model")]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(ValueError):
                original = self.response[key]
                self.response[key] = value
                self.write()
                try:
                    self.packet()
                finally:
                    self.response[key] = original
        self.write()
        (self.batch / "generation-requests.json").write_text(json.dumps([self.job, self.job]))
        with self.assertRaises(ValueError):
            self.packet()

    def test_modified_packet_cannot_use_previous_review(self):
        packet = self.packet()
        reviewed = self.review(packet)
        packet["records"][0]["context"] += "改动"
        with self.assertRaises(ValueError):
            validate_review(packet, reviewed)

    def test_cli_publishes_a_packet_and_refuses_existing_outputs(self):
        output = self.batch.parent / "review-packet"
        argv = ["review", "prepare", "--batch-dir", str(self.batch), "--run-label", "fixture",
                "--output-dir", str(output)]
        with patch("sys.argv", argv):
            main()
        self.assertTrue((output / "review.md").is_file())
        before = (output / "packet.json").read_bytes()
        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            main()
        self.assertEqual((output / "packet.json").read_bytes(), before)

    def test_cli_rejects_ambiguous_inputs_before_publishing_a_directory(self):
        path = self.batch / "generation-cases.json"
        path.write_text('{"cases":[],"cases":[]}')
        output = self.batch.parent / "review-packet"
        with patch("sys.argv", ["review", "prepare", "--batch-dir", str(self.batch),
                                "--run-label", "fixture", "--output-dir", str(output)]), self.assertRaises(SystemExit):
            main()
        self.assertFalse(output.exists())

    def test_missing_review_labels_and_absent_citations_cannot_be_supported(self):
        packet = self.packet()
        for field, value in (("context_support", None), ("citation_marker_offsets", [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                reviewed = self.review(packet)
                claim = reviewed["records"][0]["claims"][0]
                claim[field] = value
                claim["citation_support"] = "supported"
                validate_review(packet, reviewed)

    def test_grouped_ranged_unknown_and_code_markers_keep_their_original_text(self):
        text = "说明【参考资料 1、2】【参考资料 3-5】【参考资料 N】\n\n```\n【参考资料 9】\n```"
        rows = citation_markers(text)
        self.assertEqual(rows[0]["citation_ids"], [1, 2])
        self.assertEqual(rows[1]["citation_ids"], [3, 4, 5])
        self.assertFalse(rows[2]["syntax_valid"])
        self.assertTrue(rows[3]["in_code"])
        self.assertTrue(all(text[r["start"]:r["end"]] == r["raw"] for r in rows))


if __name__ == "__main__":
    unittest.main()
