"""Verify release archives contain runnable assets and exclude local or secret data."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

WHEEL_REQUIRED = {
    "rag_textbook_qa/api/app.py",
    "rag_textbook_qa/api/static/index.html",
    "rag_textbook_qa/cli.py",
    "rag_textbook_qa/web/app.py",
    "rag_textbook_qa/worker.py",
}
SDIST_REQUIRED = {
    "environment.yml",
    "project/.env.example",
    "scripts/check_distribution.py",
    "scripts/windows/start-worker.ps1",
}


def _archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def _contains(names: list[str], relative_path: str) -> bool:
    return any(name == relative_path or name.endswith(f"/{relative_path}") for name in names)


def _forbidden(names: list[str]) -> list[str]:
    blocked = []
    for name in names:
        parts = Path(name).parts
        if (
            name.endswith(("/.env", "/CLAUDE.md"))
            or "artifacts" in parts
            or "vector_db" in parts
            or "feedback.sqlite3" in parts
        ):
            blocked.append(name)
    return blocked


def check_distribution(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    source_archives = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        raise ValueError("发布目录必须恰好包含一个 wheel 和一个源码包")

    for archive, required in (
        (wheels[0], WHEEL_REQUIRED),
        (source_archives[0], SDIST_REQUIRED),
    ):
        names = _archive_names(archive)
        missing = sorted(path for path in required if not _contains(names, path))
        forbidden = _forbidden(names)
        if missing:
            raise ValueError(f"{archive.name} 缺少发布文件: {missing}")
        if forbidden:
            raise ValueError(f"{archive.name} 包含本地或敏感文件: {forbidden}")
        print(f"{archive.name}: {len(names)} files, manifest OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="包含 wheel 和源码包的目录")
    args = parser.parse_args()
    check_distribution(args.directory)


if __name__ == "__main__":
    main()
