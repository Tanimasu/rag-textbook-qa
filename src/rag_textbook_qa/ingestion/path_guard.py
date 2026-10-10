"""Reject output aliases before a textbook transformation writes files."""

from collections.abc import Sequence
from pathlib import Path


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def validate_output_paths(sources: Sequence[Path], outputs: Sequence[Path]) -> None:
    """Protect inputs and keep distinct output formats in distinct files."""
    for index, output in enumerate(outputs):
        if any(_same_file(source, output) for source in sources):
            raise ValueError("输入和输出不能是同一个文件（包括链接）")
        if any(_same_file(other, output) for other in outputs[:index]):
            raise ValueError("多个输出不能指向同一个文件（包括链接）")
