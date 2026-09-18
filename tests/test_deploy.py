import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path

from rag_textbook_qa.cli import build_parser
from rag_textbook_qa.providers.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from scripts.fetch_models import MODELS
from scripts.prepare_hf_space import (
    MARKER,
    SPACE_CARD,
    SPACE_FILES,
    SPACE_TREES,
    assemble_space,
    index_problems,
)

ROOT = Path(__file__).resolve().parents[1]


def dockerfile_instructions():
    """Yield (instruction, argument) pairs with line continuations joined."""

    logical = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if logical and logical[-1].endswith("\\"):
            logical[-1] = logical[-1][:-1] + " " + stripped
        else:
            logical.append(stripped)
    for line in logical:
        instruction, _, argument = line.partition(" ")
        yield instruction.upper(), argument.strip()


def copy_sources():
    sources = []
    for instruction, argument in dockerfile_instructions():
        if instruction != "COPY":
            continue
        tokens = shlex.split(argument)
        if any(token.startswith("--from=") for token in tokens):
            continue
        paths = [token for token in tokens if not token.startswith("--")]
        sources.extend(paths[:-1])
    return sources


def allowed_by_dockerignore(path):
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "*", "the build context must be an allow-list"
    allowed = [rule[1:].rstrip("/") for rule in rules if rule.startswith("!")]
    return any(path == entry or path.startswith(entry + "/") for entry in allowed)


class DockerImageContractTests(unittest.TestCase):
    def test_the_container_command_is_a_valid_public_serve_on_the_space_port(self):
        commands = [argument for name, argument in dockerfile_instructions() if name == "CMD"]
        self.assertEqual(len(commands), 1)
        command = json.loads(commands[0])
        self.assertEqual(command[0], "rag-qa")

        args = build_parser().parse_args(command[1:])

        self.assertEqual(args.command, "serve")
        self.assertEqual((args.host, args.port), ("0.0.0.0", 7860))
        self.assertEqual(args.db_path.as_posix(), "/app/artifacts/vector_db")
        exposed = [argument for name, argument in dockerfile_instructions() if name == "EXPOSE"]
        self.assertEqual(exposed, [str(args.port)])
        card = (ROOT / SPACE_CARD).read_text(encoding="utf-8")
        self.assertRegex(card, r"(?m)^sdk: docker$")
        self.assertRegex(card, rf"(?m)^app_port: {args.port}$")

    def test_the_build_context_admits_only_what_the_image_copies(self):
        sources = copy_sources()

        self.assertIn("artifacts/vector_db", sources)
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(allowed_by_dockerignore(source))
        for secret in ("project/.env", ".env", "data/chunks", "artifacts/product"):
            with self.subTest(secret=secret):
                self.assertFalse(allowed_by_dockerignore(secret))

    def test_the_space_folder_carries_every_file_the_image_copies(self):
        shipped = set(SPACE_FILES) | set(SPACE_TREES) | {"README.md"}
        for source in copy_sources():
            with self.subTest(source=source):
                self.assertIn(source, shipped)

    def test_baked_models_are_the_service_defaults_at_pinned_commits(self):
        # verify_offline unpacks the table as (embedding, reranker).
        self.assertEqual(list(MODELS), [DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL])
        for revision, weights in MODELS.values():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
            self.assertIn(weights, {"pytorch_model.bin", "model.safetensors"})

    def test_the_runtime_never_reaches_a_model_hub(self):
        environment = " ".join(
            argument for name, argument in dockerfile_instructions() if name == "ENV"
        )
        for setting in (
            "HF_HUB_OFFLINE=1",
            "RAG_QA_COMPUTE_BACKEND=local",
            "RAG_QA_DEVICE=cpu",
        ):
            self.assertIn(setting, environment)
        self.assertNotIn("project/.env", " ".join(copy_sources()))


def healthy_report(**changes):
    report = {
        "books": [
            {"book_name": "os", "count": 12, "embedding_model": DEFAULT_EMBEDDING_MODEL},
        ],
        "empty_books": [],
        "conflict_problems": [],
    }
    report.update(changes)
    return report


class PrepareSpaceTests(unittest.TestCase):
    def workspace(self, root):
        for relative in SPACE_FILES:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(relative, encoding="utf-8")
        package = root / "src" / "rag_textbook_qa"
        (package / "__pycache__").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__pycache__" / "cli.cpython-311.pyc").write_bytes(b"\0")
        index = root / "artifacts" / "vector_db"
        index.mkdir(parents=True)
        (index / "chroma.sqlite3").write_bytes(b"index")
        (root / SPACE_CARD).parent.mkdir(parents=True)
        (root / SPACE_CARD).write_text("---\nsdk: docker\n---\n", encoding="utf-8")
        (root / "project").mkdir()
        (root / "project" / ".env").write_text("LLM_API_KEY=sk-live-secret\n", encoding="utf-8")

    def test_the_space_folder_holds_the_card_and_index_but_no_secrets_or_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            self.workspace(root)
            output = Path(directory) / "space"

            manifest = assemble_space(root, output)

            files = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual((output / "README.md").read_text(encoding="utf-8"), "---\nsdk: docker\n---\n")
            self.assertIn("artifacts/vector_db/chroma.sqlite3", files)
            self.assertIn("src/rag_textbook_qa/__init__.py", files)
            self.assertFalse(any("__pycache__" in name or name.endswith(".env") for name in files))
            self.assertNotIn("sk-live-secret", "".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in output.rglob("*") if path.is_file()
            ))
            self.assertEqual(set(manifest["files"]) | {MARKER}, files)

    def test_regeneration_needs_force_and_never_deletes_a_foreign_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            self.workspace(root)
            output = Path(directory) / "space"
            assemble_space(root, output)

            with self.assertRaisesRegex(FileExistsError, "--force"):
                assemble_space(root, output)
            assemble_space(root, output, force=True)

            foreign = Path(directory) / "notes"
            foreign.mkdir()
            (foreign / "keep.txt").write_text("mine", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "不是本脚本生成"):
                assemble_space(root, foreign, force=True)
            self.assertTrue((foreign / "keep.txt").is_file())

    def test_an_index_that_would_answer_wrongly_is_refused(self):
        self.assertEqual(index_problems(healthy_report(), DEFAULT_EMBEDDING_MODEL), [])
        cases = {
            "教材集合": healthy_report(books=[], empty_books=[]),
            "集合为空": healthy_report(empty_books=["os"]),
            "镜像内置": healthy_report(
                books=[{"book_name": "os", "count": 12, "embedding_model": "other/model"}]
            ),
            "冲突锚点": healthy_report(
                conflict_problems=[{"rule": "r", "chunk_id": "c", "status": "missing"}]
            ),
        }
        for expected, report in cases.items():
            with self.subTest(expected=expected):
                problems = index_problems(report, DEFAULT_EMBEDDING_MODEL)
                self.assertTrue(any(re.search(expected, problem) for problem in problems))


if __name__ == "__main__":
    unittest.main()
