"""Copy a quiet published index for local inspection and experiments."""

from __future__ import annotations

import shutil
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_textbook_qa.indexing.revision import index_revision


class TemporaryIndexDirectory(TemporaryDirectory):
    """Private index copies whose Windows native handles may close asynchronously.

    Only retry Windows sharing violations during removal of this owned temporary
    directory. Persistent leaks still fail after one second; index operations and
    other permission errors are never retried.
    """

    def cleanup(self) -> None:
        deadline = time.monotonic() + 1.0
        while True:
            try:
                super().cleanup()
                return
            except PermissionError as error:
                if (sys.platform != "win32" or getattr(error, "winerror", None) != 32
                        or time.monotonic() >= deadline):
                    raise
            time.sleep(0.01)


def copy_index(source: Path, target: Path) -> str:
    """Back up SQLite and reject a catalog publication during the copy.

    Pause index writes first: SQLite backup is transactional, while native vector
    files are copied separately. Revision checks do not freeze those files.
    """

    source, target = source.expanduser().resolve(), target.expanduser().resolve()
    if target.is_relative_to(source):
        raise ValueError("副本不能写入原索引目录")
    if not (source / "chroma.sqlite3").is_file():
        raise FileNotFoundError("原索引缺少 chroma.sqlite3")
    revision = index_revision(source)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("chroma.sqlite3*", "*.lock"))
    with (
        closing(sqlite3.connect(f"{(source / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True)) as connection,
        closing(sqlite3.connect(target / "chroma.sqlite3")) as copied,
    ):
        connection.backup(copied)
    if index_revision(source) != revision or index_revision(target) != revision:
        raise RuntimeError("索引复制期间发生更新，请重新运行")
    return revision
