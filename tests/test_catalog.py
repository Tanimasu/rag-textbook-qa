import hashlib
import unittest

from rag_textbook_qa.catalog import book_id_from_chunk_stem


class CatalogTests(unittest.TestCase):
    def test_known_chunk_stems_use_stable_book_ids(self):
        self.assertEqual(book_id_from_chunk_stem("操作系统_chunks"), "os")
        self.assertEqual(
            book_id_from_chunk_stem("数据库原理及应用教程_mineru_chunks"),
            "database_mineru",
        )

    def test_unknown_non_ascii_stem_uses_deterministic_digest(self):
        stem = "新教材"
        expected = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]

        self.assertEqual(book_id_from_chunk_stem(stem), f"book_{expected}")
        self.assertEqual(book_id_from_chunk_stem("custom-book_chunks"), "custom-book")

        mixed_stem = "教材-v2"
        mixed_digest = hashlib.sha256(mixed_stem.encode("utf-8")).hexdigest()[:8]
        self.assertEqual(
            book_id_from_chunk_stem(mixed_stem),
            f"v2_{mixed_digest}",
        )


if __name__ == "__main__":
    unittest.main()
