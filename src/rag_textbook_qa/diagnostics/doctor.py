"""Diagnostics that never import or download model runtimes."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass

from rag_textbook_qa.config import Settings


@dataclass(frozen=True)
class Diagnostic:
    name: str
    status: str
    detail: str


def _module_status(import_name: str, extra: str) -> Diagnostic:
    installed = importlib.util.find_spec(import_name) is not None
    return Diagnostic(
        name=f"module:{import_name}",
        status="ok" if installed else "optional",
        detail="已安装" if installed else f"未安装；需要 uv sync --extra {extra}",
    )


def collect_diagnostics(settings: Settings) -> list[Diagnostic]:
    """Collect cheap diagnostics without importing any optional dependency."""

    paths = settings.paths
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    diagnostics = [
        Diagnostic("platform", "ok", platform.platform()),
        Diagnostic(
            "python",
            "ok" if supported_python else "warning",
            f"{platform.python_version()}（项目候选范围: >=3.11,<3.13）",
        ),
        Diagnostic("workspace", "ok", str(paths.root)),
        Diagnostic(
            "data:raw-pdfs",
            "ok" if any(paths.raw_data.glob("*.pdf")) else "optional",
            str(paths.raw_data),
        ),
        Diagnostic(
            "data:parsed",
            "ok" if paths.parsed_data.is_dir() else "pending",
            str(paths.parsed_data),
        ),
        Diagnostic(
            "data:cleaned",
            "ok" if paths.cleaned_data.is_dir() else "pending",
            str(paths.cleaned_data),
        ),
        Diagnostic(
            "data:chunks",
            "ok" if paths.chunks.is_dir() else "pending",
            str(paths.chunks),
        ),
        Diagnostic(
            "data:evaluation",
            "ok" if paths.evaluation_data.is_dir() else "pending",
            str(paths.evaluation_data),
        ),
        Diagnostic(
            "artifact:vector-db",
            "ok" if paths.vector_db.is_dir() else "optional",
            str(paths.vector_db),
        ),
        Diagnostic(
            "artifact:evaluations",
            "ok" if paths.evaluations.is_dir() else "pending",
            str(paths.evaluations),
        ),
    ]

    for executable in ("uv", "tailscale"):
        location = shutil.which(executable)
        diagnostics.append(
            Diagnostic(
                f"command:{executable}",
                "ok" if location else "optional",
                location or "未找到",
            )
        )

    diagnostics.extend(
        [
            _module_status("sentence_transformers", "local-models"),
            _module_status("streamlit", "ui"),
            _module_status("ragas", "eval"),
            _module_status("docling", "docling"),
            _module_status("mineru", "mineru"),
        ]
    )
    return diagnostics


def diagnostics_as_dict(settings: Settings) -> dict[str, object]:
    checks = collect_diagnostics(settings)
    return {
        "workspace": settings.paths.as_dict(),
        "checks": [asdict(check) for check in checks],
    }


def render_diagnostics(settings: Settings) -> str:
    lines = ["rag-textbook-qa 环境检查", "=" * 40]
    for check in collect_diagnostics(settings):
        lines.append(f"[{check.status.upper():8}] {check.name}: {check.detail}")
    lines.append("=" * 40)
    lines.append("doctor 只检查配置，不会加载或下载模型。")
    return "\n".join(lines)
