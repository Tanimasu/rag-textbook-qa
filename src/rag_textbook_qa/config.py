"""Workspace discovery and path configuration without heavyweight imports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ENV_VAR = "RAG_QA_HOME"


class WorkspaceNotFoundError(RuntimeError):
    """Raised when the project workspace cannot be located safely."""


def _is_workspace(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (
        path / "src" / "rag_textbook_qa"
    ).is_dir()


def _find_workspace_upwards(start: Path) -> Path | None:
    candidate = start.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for path in (candidate, *candidate.parents):
        if _is_workspace(path):
            return path
    return None


def resolve_workspace(explicit: str | Path | None = None) -> Path:
    """Resolve the repository workspace independently of the current directory.

    Resolution order is an explicit argument, ``RAG_QA_HOME``, the current
    directory and its parents, then this package's editable-source checkout.
    """

    if explicit is not None:
        workspace = Path(explicit).expanduser().resolve()
        if not _is_workspace(workspace):
            raise WorkspaceNotFoundError(f"不是有效的 rag-textbook-qa 工作区: {workspace}")
        return workspace

    configured = os.getenv(WORKSPACE_ENV_VAR)
    if configured:
        workspace = Path(configured).expanduser().resolve()
        if not _is_workspace(workspace):
            raise WorkspaceNotFoundError(
                f"{WORKSPACE_ENV_VAR} 指向的目录不是有效工作区: {workspace}"
            )
        return workspace

    from_cwd = _find_workspace_upwards(Path.cwd())
    if from_cwd is not None:
        return from_cwd

    source_checkout = Path(__file__).resolve().parents[2]
    if _is_workspace(source_checkout):
        return source_checkout

    raise WorkspaceNotFoundError(
        f"无法定位项目工作区；请设置 {WORKSPACE_ENV_VAR} 或使用 --workspace"
    )


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    data: Path
    raw_data: Path
    processed_data: Path
    chunks: Path
    evaluation_data: Path
    artifacts: Path
    vector_db: Path
    evaluations: Path

    @classmethod
    def from_root(cls, root: Path) -> "WorkspacePaths":
        data = root / "data"
        artifacts = root / "artifacts"
        return cls(
            root=root,
            data=data,
            raw_data=data / "raw",
            processed_data=data / "processed",
            chunks=data / "chunks",
            evaluation_data=data / "evaluation",
            artifacts=artifacts,
            vector_db=artifacts / "vector_db",
            evaluations=artifacts / "evaluations",
        )

    def as_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in vars(self).items()}


@dataclass(frozen=True)
class Settings:
    paths: WorkspacePaths

    @classmethod
    def load(cls, workspace: str | Path | None = None) -> "Settings":
        return cls(paths=WorkspacePaths.from_root(resolve_workspace(workspace)))
