"""Verify release archives contain runnable assets and exclude local or secret data."""

from __future__ import annotations

import argparse
import ast
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

WHEEL_REQUIRED = {
    "rag_textbook_qa/__init__.py",
    "rag_textbook_qa/api/app.py",
    "rag_textbook_qa/api/static/index.html",
    "rag_textbook_qa/api/static/retrieval.html",
    "rag_textbook_qa/api/static/imports.html",
    "rag_textbook_qa/api/static/history-backup.js",
    "rag_textbook_qa/api/static/markdown.js",
    "rag_textbook_qa/cli.py",
    "rag_textbook_qa/indexing/exact.py",
    "rag_textbook_qa/web/app.py",
    "rag_textbook_qa/worker.py",
}
SDIST_REQUIRED = {
    "pyproject.toml",
    "uv.lock",
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


def _source_text(archive: tarfile.TarFile, filename: str) -> str:
    candidates = [entry for entry in archive.getmembers() if entry.isfile()
                  and len(Path(entry.name).parts) <= 2 and Path(entry.name).name == filename]
    if len(candidates) != 1:
        raise ValueError(f"源码包必须包含唯一的根目录 {filename}")
    source = archive.extractfile(candidates[0])
    if source is None:
        raise ValueError(f"源码包无法读取 {filename}")
    return source.read().decode("utf-8")


def _check_versions(wheel: Path, source_archive: Path) -> None:
    with tarfile.open(source_archive) as archive:
        project = tomllib.loads(_source_text(archive, "pyproject.toml"))["project"]
        lock = tomllib.loads(_source_text(archive, "uv.lock"))
    name, version = project["name"], project["version"]
    locked = [entry for entry in lock.get("package", []) if entry.get("name") == name]
    if len(locked) != 1 or locked[0].get("version") != version:
        raise ValueError("源码包的项目版本与依赖锁中的项目版本不一致")
    with zipfile.ZipFile(wheel) as archive:
        entries = [entry for entry in archive.namelist() if entry.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError("wheel 必须包含唯一的版本元数据")
        metadata = Parser().parsestr(archive.read(entries[0]).decode("utf-8"))
        tree = ast.parse(archive.read("rag_textbook_qa/__init__.py").decode("utf-8"))
    if metadata.get_all("Name") != [name] or metadata.get_all("Version") != [version]:
        raise ValueError("wheel 名称或版本与源码包不一致")
    runtime_versions = [entry.value.value for entry in tree.body if isinstance(entry, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in entry.targets)
                        and isinstance(entry.value, ast.Constant) and isinstance(entry.value.value, str)]
    if runtime_versions != [version]:
        raise ValueError("wheel 运行时版本与发布元数据不一致")
    print(f"project version {version}: metadata, runtime and lock OK")


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
    _check_versions(wheels[0], source_archives[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="包含 wheel 和源码包的目录")
    args = parser.parse_args()
    check_distribution(args.directory)


if __name__ == "__main__":
    main()
