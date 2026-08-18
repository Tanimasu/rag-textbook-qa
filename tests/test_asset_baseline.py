import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "legacy_assets.sha256"
SOURCE_COMMIT = "13b84805ae16971ced68c444b820b55f020f6c58"

EXPECTED_CHUNK_COUNTS = {
    "data/chunks/操作系统_chunks.json": 1183,
    "data/chunks/操作系统_mineru_chunks.json": 1164,
    "data/chunks/数据库原理及应用教程_chunks.json": 999,
    "data/chunks/数据库原理及应用教程_mineru_chunks.json": 903,
    "data/chunks/数据结构_chunks.json": 697,
    "data/chunks/数据结构_mineru_chunks.json": 729,
    "data/chunks/计算机组成原理_chunks.json": 1082,
    "data/chunks/计算机组成原理_mineru_chunks.json": 1103,
    "data/chunks/计算机网络_chunks.json": 624,
    "data/chunks/计算机网络_mineru_chunks.json": 937,
}

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "chapter",
    "section_h2",
    "section_h3",
    "section_h4",
    "content",
    "level",
    "char_count",
    "has_code",
    "has_image",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected_hash, relative_path = line.split("  ", 1)
        entries[relative_path] = expected_hash
    return entries


class LegacyAssetBaselineTests(unittest.TestCase):
    def test_manifest_records_original_commit(self):
        first_line = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first_line, f"# source_commit={SOURCE_COMMIT}")

    def test_all_legacy_asset_hashes_are_unchanged(self):
        manifest = load_manifest()
        self.assertEqual(len(manifest), 44)
        for relative_path, expected_hash in manifest.items():
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                self.assertTrue(path.is_file())
                self.assertEqual(sha256(path), expected_hash)

    def test_legacy_assets_are_grouped_by_lifecycle_stage(self):
        parsed = list((REPOSITORY_ROOT / "data" / "parsed").glob("*.md"))
        cleaned = list((REPOSITORY_ROOT / "data" / "cleaned").glob("*.md"))
        chunks = list((REPOSITORY_ROOT / "data" / "chunks").glob("*.json"))
        self.assertEqual(len(parsed), 10)
        self.assertEqual(len(cleaned), 10)
        self.assertEqual(len(chunks), 10)
        self.assertEqual(
            len(list((REPOSITORY_ROOT / "data" / "chunks" / "previews").glob("*.txt"))),
            10,
        )
        self.assertFalse((REPOSITORY_ROOT / "project" / "output").exists())

    def test_chunk_counts_and_schema_are_unchanged(self):
        total = 0
        for relative_path, expected_count in EXPECTED_CHUNK_COUNTS.items():
            with self.subTest(path=relative_path):
                chunks = json.loads((REPOSITORY_ROOT / relative_path).read_text("utf-8"))
                self.assertEqual(len(chunks), expected_count)
                self.assertTrue(chunks)
                for chunk in chunks:
                    self.assertTrue(REQUIRED_CHUNK_FIELDS.issubset(chunk))
                total += len(chunks)
        self.assertEqual(total, 9421)

    def test_evaluation_question_distribution_is_unchanged(self):
        questions = json.loads(
            (REPOSITORY_ROOT / "data" / "evaluation" / "test_questions.json").read_text(
                "utf-8"
            )
        )
        self.assertEqual(len(questions), 50)
        self.assertEqual(
            Counter(question["book_name"] for question in questions),
            {
                "os": 10,
                "computer_organization": 10,
                "computer_network": 10,
                "data_structure": 10,
                "database": 10,
            },
        )

    def test_evaluation_output_row_counts_are_unchanged(self):
        qa = json.loads(
            (
                REPOSITORY_ROOT
                / "artifacts"
                / "evaluations"
                / "ragas_qa_comparison.json"
            ).read_text("utf-8")
        )
        self.assertEqual(len(qa), 50)

        for name in ("ragas_evaluation_results.csv", "ragas_baseline_results.csv"):
            with self.subTest(path=name), (
                REPOSITORY_ROOT / "artifacts" / "evaluations" / name
            ).open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 50)


if __name__ == "__main__":
    unittest.main()
