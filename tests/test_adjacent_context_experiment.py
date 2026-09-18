import unittest

from scripts.experiment_adjacent_context import (
    adjacent_candidates,
    section_body_shingles,
)


def _chunk(chunk_id: str, section: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "book_name": "computer_network",
        "chapter": "第3章",
        "section_h2": "3.1 数据链路层",
        "section_h3": "3.1.2 三个基本问题",
        "section_h4": section,
        "content": content,
        "has_code": False,
        "has_image": False,
        "char_count": len(content),
        "level": 4,
    }


class AdjacentContextExperimentTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            _chunk("p1", "1.封装成帧", "封装成帧正文"),
            _chunk("p2", "2.透明传输", "透明传输先说明问题"),
            _chunk("p3", "2.透明传输", "透明传输再说明转义方法"),
            _chunk("p4", "3.差错检测", "差错检测正文"),
        ]

    def test_interleaved_adds_only_same_section_neighbour(self):
        results = [{**self.corpus[1], "rank": 1, "method": "hybrid-rerank"}]

        expanded = adjacent_candidates(results, self.corpus, placement="interleaved")

        self.assertEqual([row["chunk_id"] for row in expanded], ["p2", "p3"])
        self.assertEqual(expanded[1]["adjacent_of"], "p2")
        self.assertEqual(expanded[1]["adjacent_direction"], "next")

    def test_base_first_preserves_result_order_and_deduplicates(self):
        results = [
            {**self.corpus[1], "rank": 1},
            {**self.corpus[2], "rank": 2},
        ]

        expanded = adjacent_candidates(results, self.corpus, placement="base_first")

        self.assertEqual([row["chunk_id"] for row in expanded], ["p2", "p3"])

    def test_section_body_proxy_uses_all_exact_heading_chunks(self):
        expected = section_body_shingles(self.corpus, ["2.透明传输"])
        combined = section_body_shingles(self.corpus[1:3], ["2.透明传输"])

        self.assertEqual(expected, combined)
        self.assertTrue(expected)

    def test_missing_annotated_section_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "找不到标注章节"):
            section_body_shingles(self.corpus, ["9.9 不存在"])


if __name__ == "__main__":
    unittest.main()
