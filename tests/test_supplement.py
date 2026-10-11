import unittest

from rag_textbook_qa.rag.context import select_context
from rag_textbook_qa.rag.supplement import pack_selected_excerpts, paragraph_spans


def source(identifier, content, book="os"):
    return {"chunk_id": identifier, "book_name": book, "chapter": "第1章",
            "section_h2": "1.1 原理", "content": content}


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.ranked = [source("a", "原有完整证据及编号必须保留。" * 3)]
        self.candidate = source("b", "定义说明必须来自教材中的原文，不允许模型改写。\n\n"
                                "流程说明也必须来自同一来源的连续段落，并保留换行。\n\n"
                                "后面的其他内容与本次问题没有直接联系，可以不加入。")
        self.selected = {"selected": [{"chunk_id": "b", "start_paragraph": 0,
                                       "end_paragraph": 1}]}

    def test_preserves_exact_original_substring_and_baseline_citations(self):
        baseline, sources = select_context(self.ranked)
        context, packed = pack_selected_excerpts(self.ranked, [self.candidate], self.selected)
        self.assertTrue(context.startswith(baseline))
        self.assertEqual(packed[0], sources[0])
        self.assertEqual(packed[1]["citation_id"], 2)
        excerpt = packed[1]
        self.assertEqual(excerpt["content"], self.candidate["content"][
            excerpt["excerpt_start"]:excerpt["excerpt_end"]])
        self.assertNotIn("后面的", excerpt["content"])
        self.assertFalse(excerpt["truncated"])

    def test_rejects_invented_duplicate_cross_book_or_replacement_sources(self):
        candidates = [self.candidate, source("c", self.candidate["content"], "database"),
                      self.ranked[0]]
        for identifiers in (["invented"], ["b", "b"], ["c"], ["a"]):
            with self.subTest(ids=identifiers), self.assertRaises(ValueError):
                pack_selected_excerpts(self.ranked, candidates, {"selected": [
                    {"chunk_id": identifier, "start_paragraph": 0, "end_paragraph": 0}
                    for identifier in identifiers
                ]})

    def test_rejects_ambiguous_ranges_and_model_supplied_prose(self):
        for changes in ({"start_paragraph": True}, {"end_paragraph": 90},
                        {"start_paragraph": 2}, {"content": "模型新写的文字"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pack_selected_excerpts(self.ranked, [self.candidate], {"selected": [
                    {**self.selected["selected"][0], **changes},
                ]})

    def test_excludes_code_including_long_fences_tables_and_image_paragraphs(self):
        text = ("这是可供选择的完整解释段落，需要足够长度才能参与补充。\n\n"
                "````python\nx = '```'\n\n"
                "代码里的这段话不能因为空行被错误视为教材解释段落。\n\n"
                "````\n\n<table>\n\n表格里的这段文字也不能被拆成普通解释段落。\n\n</table>\n\n"
                "![](images/example.png)\n\n"
                "这里是代码和表格之后的完整正文，可以安全按原文选取。")
        spans = paragraph_spans(text)
        selectable = [p["text"] for p in spans if p["selectable"]]
        self.assertEqual(selectable, [text.split("\n\n")[0], text.split("\n\n")[-1]])
        bad = next(p["index"] for p in spans if "代码里的" in p["text"])
        with self.assertRaises(ValueError):
            pack_selected_excerpts(self.ranked, [source("b", text)], {"selected": [
                {"chunk_id": "b", "start_paragraph": bad, "end_paragraph": bad},
            ]})

    def test_over_budget_or_partial_excerpts_fail_instead_of_silent_clipping(self):
        for options in ({"max_excerpt_chars": 10}, {"context_budget": 120}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pack_selected_excerpts(self.ranked, [self.candidate], self.selected, **options)
        self.assertEqual(pack_selected_excerpts(self.ranked, [self.candidate], {"selected": []}),
                         select_context(self.ranked))

    def test_unfenced_c_function_is_excluded_even_with_a_damaged_closing_brace(self):
        for ending in ("}", "1"):
            code = "int sumvec()\n{\nint total = 0;\nreturn total;\n" + ending
            text = ("这里是代码之前的完整教材解释段落，需要单独保留其原文。\n\n"
                    + code + "\n\n这里是代码之后的完整教材解释段落，也可以安全选取。")
            spans = paragraph_spans(text)
            with self.subTest(ending=ending):
                self.assertFalse(spans[1]["selectable"])
                self.assertTrue(spans[0]["selectable"])
                self.assertTrue(spans[2]["selectable"])
                with self.assertRaises(ValueError):
                    pack_selected_excerpts(self.ranked, [source("b", text)],
                                           {"selected": [{"chunk_id": "b",
                                                          "start_paragraph": 1,
                                                          "end_paragraph": 1}]})

    def test_image_alt_text_and_markdown_tables_are_not_prose_evidence(self):
        paragraphs = [
            "![这是一张仅有图片，没有正文依据的教材示意图](images/cache-layout.png)",
            '<img src="images/cache-layout.png" alt="这里的替代文字不能充当教材正文依据">',
            "|存储层次及含义|数据访问时间和示例|\n|:---|---:|\n|高速缓存|示例内容仅供查看表格|",
            "存储层次及含义 | 数据访问时间和示例\n--- | ---\n高速缓存 | 示例内容仅供查看表格",
        ]
        for paragraph in paragraphs:
            with self.subTest(paragraph=paragraph):
                self.assertFalse(paragraph_spans(paragraph)[0]["selectable"])
                with self.assertRaises(ValueError):
                    pack_selected_excerpts(self.ranked, [source("b", paragraph)],
                                           {"selected": [{"chunk_id": "b",
                                                          "start_paragraph": 0,
                                                          "end_paragraph": 0}]})

    def test_math_only_and_damaged_image_paths_do_not_become_explanatory_prose(self):
        for text in (
            "$$\n\\mathrm { a + b ^ { * } c - d - e / f }\n$$",
            r"\[ a + b + c + d + e + f + g \]",
            "4c918d762842be839e1f701997085d7ddce32f580579.jpg)\n图5.15 节点形式\n图5.16 存储结构",
        ):
            with self.subTest(text=text):
                self.assertFalse(paragraph_spans(text)[0]["selectable"])
        # Inline mathematics accompanied by an explanation is still prose.
        prose = "其中 $N = q \\times d + r$ 表示商和余数的关系，这段教材说明有完整含义。"
        self.assertTrue(paragraph_spans(prose)[0]["selectable"])

    def test_invalid_limits_fail_even_for_empty_selection(self):
        for name in ("context_budget", "max_sources", "max_excerpt_chars"):
            for value in (True, 0, -1, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    pack_selected_excerpts(self.ranked, [self.candidate], {"selected": []},
                                           **{name: value})


if __name__ == "__main__":
    unittest.main()
