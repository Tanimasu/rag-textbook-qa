"""Assemble a Hugging Face Docker Space from this checkout, ready to upload.

The Space repository needs its own README (Hugging Face reads its YAML front matter)
and the exact local index, which is gitignored. Rather than keep a second copy of
either in Git, this script checks the index and then builds the Space folder under
the ignored artifacts/ directory. It publishes nothing: uploading needs the owner's
Hugging Face account, and the printed commands leave that step to them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SPACE_FILES = (
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "scripts/fetch_models.py",
)
SPACE_TREES = ("src", "artifacts/vector_db")
SPACE_CARD = Path("deploy/huggingface/README.md")
# Marks a folder this script created, so --force never deletes anything else.
MARKER = ".rag-qa-space.json"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def index_problems(report: dict[str, Any], embedding_model: str) -> list[str]:
    """Reasons this index must not ship; an empty list means it may."""

    problems = []
    if not report["books"]:
        problems.append("向量库里没有任何教材集合")
    problems.extend(f"{book} 集合为空" for book in report["empty_books"])
    for book in report["books"]:
        if book["embedding_model"] != embedding_model:
            problems.append(
                f"{book['book_name']} 由 {book['embedding_model'] or '未知模型'} 建立，"
                f"镜像内置的是 {embedding_model}"
            )
    problems.extend(
        f"冲突锚点失效: {item['rule']} / {item['chunk_id']}（{item['status']}）"
        for item in report["conflict_problems"]
    )
    return problems


def assemble_space(root: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    """Copy exactly the files the Dockerfile needs into ``output``."""

    root = root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        if not (output / MARKER).is_file():
            raise FileExistsError(f"{output} 已存在且不是本脚本生成的目录，拒绝覆盖")
        if not force:
            raise FileExistsError(f"{output} 已存在；确认重新生成请加 --force")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for relative in SPACE_FILES:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    for relative in SPACE_TREES:
        shutil.copytree(root / relative, output / relative, ignore=ignore)
    shutil.copy2(root / SPACE_CARD, output / "README.md")

    manifest = {
        "source_commit": _git(root, "rev-parse", "HEAD") or None,
        "source_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        "files": sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        ),
    }
    (output / MARKER).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成可上传的 Hugging Face Docker Space 目录")
    parser.add_argument("--output", type=Path, help="默认 artifacts/hf-space（已被 Git 忽略）")
    parser.add_argument("--force", action="store_true", help="重新生成本脚本之前创建的目录")
    args = parser.parse_args(argv)

    from rag_textbook_qa.cli import _index_health, _render_index_health
    from rag_textbook_qa.config import Settings
    from rag_textbook_qa.providers.config import DEFAULT_EMBEDDING_MODEL

    settings = Settings.load()
    root = settings.paths.root
    report = _index_health(settings.paths.vector_db)
    print(_render_index_health(report))
    # The image runs with the default models, baked by scripts/fetch_models.py.
    problems = index_problems(report, DEFAULT_EMBEDDING_MODEL)
    if problems:
        print("\n索引不能用于部署：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    output = args.output or settings.paths.artifacts / "hf-space"
    try:
        manifest = assemble_space(root, output, force=args.force)
    except FileExistsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    commit = manifest["source_commit"] or "未知"
    print(f"\nSpace 目录: {output.resolve()}（{len(manifest['files'])} 个文件，源提交 {commit[:12]}）")
    if manifest["source_dirty"]:
        print("警告: 工作区有未提交的改动，Space 内容与这个提交并不完全一致", file=sys.stderr)
    print(
        "\n上传需要你自己的 Hugging Face 账号（token 需要写权限）：\n"
        "  hf auth login\n"
        "  hf repos create <用户名>/<space 名> --repo-type space --space-sdk docker --public\n"
        f"  hf upload <用户名>/<space 名> {output.resolve()} . --repo-type space\n"
        "然后在 Space 的 Settings → Variables and secrets 里配置 LLM_API_KEY（Secret）、"
        "LLM_API_BASE、LLM_MODEL。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
