"""Small local-only Markdown/chunks import jobs with durable progress metadata."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.catalog import BOOK_LABELS
from rag_textbook_qa.indexing.vectorizer import _load_chunks
from rag_textbook_qa.ingestion.chunker import SmartTextbookChunker
from rag_textbook_qa.json_utils import loads_strict

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
BuildImport = Callable[[Path, str, Callable[[int, int, str], None]], None]


class ImportBusyError(RuntimeError):
    pass


class ImportManager:
    def __init__(self, root: Path, build: BuildImport) -> None:
        self.root = root.resolve()
        self._build = build
        self._lock = threading.RLock()
        self._active: set[str] = set()
        self._threads: list[threading.Thread] = []

    def _folder(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise LookupError("没有这个导入任务")
        folder = self.root / job_id
        if folder.resolve().parent != self.root or folder.is_symlink():
            raise LookupError("没有这个导入任务")
        return folder

    def _write(self, job: dict[str, Any]) -> None:
        folder = self._folder(job["job_id"])
        temporary = folder / "job.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(folder / "job.json")

    def _load(self, job_id: str) -> dict[str, Any]:
        try:
            job = loads_strict((self._folder(job_id) / "job.json").read_text(encoding="utf-8"))
            if not isinstance(job, dict) or job.get("job_id") != job_id:
                raise ValueError("导入任务记录不完整")
            return job
        except FileNotFoundError:
            raise LookupError("没有这个导入任务") from None

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._load(job_id)
            if job["state"] == "building" and job_id not in self._active:
                job.update(state="interrupted", message="任务执行已中断，未确认完成；请检查教材列表后重试。")
                self._write(job)
            return job

    def list_jobs(self, limit: int = 20) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("任务列表数量必须在 1–100 之间")
        with self._lock:
            if not self.root.exists():
                return {"jobs": [], "unreadable_count": 0}
            jobs, unreadable = [], 0
            for folder in sorted(self.root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
                if not folder.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                    continue
                try:
                    job = self.status(folder.name)
                    fields = ("job_id", "book_id", "book", "filename", "created_at", "state",
                              "completed_chunks", "total_chunks", "message")
                    jobs.append({key: job[key] for key in fields})
                except (OSError, ValueError, LookupError, TypeError):
                    unreadable += 1
                if len(jobs) >= limit:
                    break
            return {"jobs": jobs, "unreadable_count": unreadable}

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._load(job_id)
            if job_id in self._active:
                raise ImportBusyError("任务仍在建立索引，不能清理导入副本")
            shutil.rmtree(self._folder(job_id))

    def prepare(self, book_id: str, filename: str, content: bytes) -> dict[str, Any]:
        if book_id not in BOOK_LABELS:
            raise ValueError("请选择支持的教材")
        name = filename.replace("\\", "/").rsplit("/", 1)[-1][:160]
        extension = Path(name).suffix.lower()
        if extension not in {".md", ".json"}:
            raise ValueError("本版支持 UTF-8 Markdown（.md）或已有分块 JSON（.json）")
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("教材文件不能为空，且不能超过 8 MB")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("请上传 UTF-8 编码的教材文件") from None
        if extension == ".json":
            chunks = loads_strict(text)
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("分块 JSON 必须包含非空片段列表")
        else:
            chunks = None
        job_id = uuid.uuid4().hex
        folder = self._folder(job_id)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / ("source" + extension)).write_bytes(content)
        if chunks is None:
            markdown = folder / "parsed.md"
            markdown.write_text(text, encoding="utf-8")
            chunks = [asdict(chunk) for chunk in SmartTextbookChunker().chunk_document(markdown)]
        chunks_file = folder / "chunks.json"
        chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        validated = _load_chunks(chunks_file)
        if len(validated) > 20000:
            raise ValueError("单次最多导入 20000 个片段")
        job = {
            "job_id": job_id, "book_id": book_id, "book": BOOK_LABELS[book_id], "filename": name,
            "created_at": datetime.now(UTC).isoformat(), "state": "prepared", "stage": "parsed",
            "completed_chunks": 0, "total_chunks": len(validated), "message": "解析完成，确认预览后可建立索引。",
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "chunks_sha256": hashlib.sha256(chunks_file.read_bytes()).hexdigest(),
            "preview": [
                {"section": " > ".join(str(row.get(key) or "") for key in
                                         ("chapter", "section_h2", "section_h3", "section_h4")
                                         if row.get(key)), "excerpt": row["content"][:1000]}
                for row in validated[:3]
            ],
        }
        with self._lock:
            self._write(job)
        return job

    def start(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._load(job_id)
            if job["state"] in {"complete", "building"} and (job["state"] == "complete" or job_id in self._active):
                return job
            if self._active:
                raise ImportBusyError("另一个教材正在建立索引，请等待它完成")
            chunks_file = self._folder(job_id) / "chunks.json"
            if hashlib.sha256(chunks_file.read_bytes()).hexdigest() != job["chunks_sha256"]:
                raise ValueError("预览后的分块文件发生变化，请重新上传教材")
            if job["book_id"] not in BOOK_LABELS:
                raise ValueError("不支持的教材")
            job.update(state="building", stage="embedding", completed_chunks=0, message="开始建立索引…")
            self._write(job)
            self._active.add(job_id)
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True,
                                      name=f"textbook-import-{job_id[:8]}")
            self._threads = [previous for previous in self._threads if previous.is_alive()]
            self._threads.append(thread)
            thread.start()
            return job

    def _run(self, job_id: str) -> None:
        def progress(done: int, total: int, stage: str) -> None:
            with self._lock:
                job = self._load(job_id)
                job.update(completed_chunks=done, total_chunks=total, stage=stage,
                           message="正在发布索引…" if stage == "publishing" else "正在向量化教材片段…")
                self._write(job)
        try:
            job = self.status(job_id)
            self._build(self._folder(job_id) / "chunks.json", job["book_id"], progress)
            with self._lock:
                job = self._load(job_id)
                job.update(state="complete", stage="complete", completed_chunks=job["total_chunks"],
                           message="索引已建立，可以返回聊天选择这本教材。")
                self._write(job)
        except Exception as exc:  # noqa: BLE001 - do not expose model or filesystem details
            print(f"教材导入失败: {type(exc).__name__}", flush=True)
            with self._lock:
                job = self._load(job_id)
                job.update(state="failed", message="建索引失败，原始上传和分块已保留。可检查服务配置后重试。")
                self._write(job)
        finally:
            with self._lock:
                self._active.discard(job_id)

    def close(self) -> None:
        for thread in self._threads:
            thread.join()
