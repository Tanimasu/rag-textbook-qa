import unittest

import jieba

from rag_textbook_qa.rag.tokenizer import BM25Tokenizer, heading_terms


class HeadingTermTests(unittest.TestCase):
    def test_numbering_is_stripped_and_fragments_are_mined(self):
        terms = heading_terms(["5.9.1 TCP 的连接建立", "3.5 死锁概述"])

        self.assertIn("TCP", terms)
        self.assertIn("连接建立", terms)
        self.assertIn("死锁概述", terms)

    def test_fragments_bounded_by_connectives_are_dropped(self):
        # "的连接建立" as a dictionary entry would mis-segment ordinary prose.
        self.assertNotIn("的连接建立", heading_terms(["5.9.1 TCP 的连接建立"]))

    def test_single_characters_and_pure_punctuation_are_ignored(self):
        self.assertEqual(heading_terms(["1.1 的", "2.2 （）", "   "]), set())


class TokenizerTests(unittest.TestCase):
    def test_dictionary_does_not_mutate_other_instances_or_global_jieba(self):
        text = "阿尔法贝塔伽马"
        plain = BM25Tokenizer("jieba")
        hybrid = BM25Tokenizer("hybrid")
        before = plain(text), hybrid(text), list(jieba.cut(text))
        dictionary = BM25Tokenizer("dictionary", terms=[text])
        self.assertEqual(dictionary(text), [text])
        self.assertEqual((plain(text), hybrid(text), list(jieba.cut(text))), before)
        self.assertEqual(BM25Tokenizer("jieba")(text), before[0])

    def test_dictionary_instances_keep_their_own_terms(self):
        first = BM25Tokenizer("dictionary", terms=["阿尔法贝塔伽马"])
        before = first("贝塔伽马德尔塔")
        second = BM25Tokenizer("dictionary", terms=["贝塔伽马德尔塔"])
        self.assertEqual(first("贝塔伽马德尔塔"), before)
        self.assertEqual(second("贝塔伽马德尔塔"), ["贝塔伽马德尔塔"])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知的 BM25 分词模式"):
            BM25Tokenizer("word2vec")

    def test_bigrams_recover_a_term_the_dictionary_splits(self):
        # jieba cuts 散列表 into 散/列表, so the term itself never matches.
        words = BM25Tokenizer("jieba")("散列表")
        bigrams = BM25Tokenizer("bigram")("散列表")

        self.assertNotIn("散列表", words)
        self.assertIn("散列", bigrams)

    def test_hybrid_carries_both_word_and_bigram_evidence(self):
        tokens = BM25Tokenizer("hybrid")("散列表")

        self.assertIn("列表", tokens)
        self.assertIn("散列", tokens)

    def test_latin_runs_stay_whole_and_lowercase(self):
        self.assertIn("tcp", BM25Tokenizer("bigram")("TCP 三次握手"))

    def test_punctuation_and_whitespace_never_reach_the_index(self):
        for mode in ("jieba", "bigram", "hybrid"):
            with self.subTest(mode=mode):
                tokens = BM25Tokenizer(mode)("死锁，是指：多个进程 —— 互相等待。")
                self.assertTrue(all(token.strip() for token in tokens))
                self.assertFalse({"，", "：", "。", "——", " "} & set(tokens))


if __name__ == "__main__":
    unittest.main()
