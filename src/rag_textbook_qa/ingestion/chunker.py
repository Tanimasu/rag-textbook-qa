"""Structure-aware chunking for cleaned textbook Markdown files.

This module preserves the chunking behaviour of the graduation-project script,
while exposing path-based APIs that work on Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class TextChunk:
    """A chunk of textbook text together with its heading context."""

    chunk_id: str
    chapter: str
    section_h2: str
    section_h3: str
    section_h4: str
    content: str
    level: int
    char_count: int
    has_code: bool
    has_image: bool


@dataclass(frozen=True)
class BatchChunkResult:
    """Files created or skipped by :func:`batch_chunk_markdown`."""

    created: tuple[Path, ...]
    skipped_existing: tuple[Path, ...]


class SmartTextbookChunker:
    """Split textbook Markdown while retaining the current heading hierarchy."""

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size 必须大于 0")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size 必须大于 0")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size 不能大于 max_chunk_size")
        if overlap_size < 0:
            raise ValueError("overlap_size 不能小于 0")

        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

        self.current_chapter = ""
        self.current_h2 = ""
        self.current_h3 = ""
        self.current_h4 = ""

        self.chapter_num = 0
        self.chunk_counter = 0

    def parse_markdown(self, content: str) -> list[dict[str, object]]:
        """Parse Markdown into the same section representation as the legacy script."""

        print("📖 解析文档结构...")
        sections: list[dict[str, object]] = []
        current_section: dict[str, object] = {
            "level": 0,
            "title": "",
            "content": [],
        }

        for line in content.split("\n"):
            title_match = re.match(r"^(#{1,4})\s+(.+)$", line)

            if title_match:
                current_content = current_section["content"]
                if current_content:
                    current_section["content"] = "\n".join(  # type: ignore[arg-type]
                        current_content
                    )
                    sections.append(current_section.copy())

                current_section = {
                    "level": len(title_match.group(1)),
                    "title": title_match.group(2).strip(),
                    "content": [],
                }
            elif line.strip():
                current_content = current_section["content"]
                assert isinstance(current_content, list)
                current_content.append(line)

        current_content = current_section["content"]
        if current_content:
            current_section["content"] = "\n".join(current_content)  # type: ignore[arg-type]
            sections.append(current_section)

        print(f"   ✅ 解析完成，共 {len(sections)} 个段落")
        return sections

    def update_context(self, level: int, title: str) -> None:
        """Update the active heading hierarchy."""

        if level == 1:
            self.current_chapter = title
            self.current_h2 = ""
            self.current_h3 = ""
            self.current_h4 = ""
            self.chapter_num += 1
        elif level == 2:
            self.current_h2 = title
            self.current_h3 = ""
            self.current_h4 = ""
        elif level == 3:
            self.current_h3 = title
            self.current_h4 = ""
        elif level == 4:
            self.current_h4 = title

    def generate_chunk_id(self) -> str:
        """Generate a stable chunk ID using the current chapter and section."""

        self.chunk_counter += 1

        chapter_match = re.search(r"第\s*(\d+)\s*章", self.current_chapter)
        chapter_number = chapter_match.group(1) if chapter_match else str(self.chapter_num)

        h2_match = re.match(r"(\d+\.\d+)", self.current_h2)
        h2_number = h2_match.group(1).replace(".", "_") if h2_match else ""

        if h2_number:
            return f"ch{chapter_number}_s{h2_number}_p{self.chunk_counter:03d}"
        return f"ch{chapter_number}_p{self.chunk_counter:03d}"

    def create_chunk(self, content: str, level: int) -> TextChunk:
        """Create one chunk using the current heading context."""

        return TextChunk(
            chunk_id=self.generate_chunk_id(),
            chapter=self.current_chapter,
            section_h2=self.current_h2,
            section_h3=self.current_h3,
            section_h4=self.current_h4,
            content=content.strip(),
            level=level,
            char_count=len(content),
            has_code="```" in content,
            has_image="📷" in content or "[图片]" in content,
        )

    def split_long_content(self, content: str, level: int) -> list[TextChunk]:
        """Split oversized content by paragraph, then by Chinese sentence marks."""

        chunks: list[TextChunk] = []
        paragraphs = re.split(r"\n\n+", content)
        current_content = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chunk_size:
                if current_content:
                    chunks.append(self.create_chunk(current_content, level))
                    current_content = ""

                sentences = re.split(r"([。！？\n])", paragraph)
                temporary_content = ""

                for index in range(0, len(sentences), 2):
                    sentence = sentences[index]
                    separator = sentences[index + 1] if index + 1 < len(sentences) else ""

                    if (
                        len(temporary_content) + len(sentence) + len(separator)
                        > self.max_chunk_size
                    ):
                        if temporary_content:
                            chunks.append(self.create_chunk(temporary_content, level))
                        temporary_content = sentence + separator
                    else:
                        temporary_content += sentence + separator

                if temporary_content:
                    current_content = temporary_content
            elif len(current_content) + len(paragraph) > self.max_chunk_size:
                if current_content:
                    chunks.append(self.create_chunk(current_content, level))
                current_content = paragraph
            else:
                current_content += ("\n\n" if current_content else "") + paragraph

        if current_content:
            chunks.append(self.create_chunk(current_content, level))

        return chunks

    def chunk_document(self, markdown_path: str | Path) -> list[TextChunk]:
        """Chunk a complete UTF-8 Markdown document."""

        print("=" * 70)
        print("🚀 开始智能分块")
        print("=" * 70)
        print("配置:")
        print(f"  最大块大小: {self.max_chunk_size} 字符")
        print(f"  最小块大小: {self.min_chunk_size} 字符")
        print(f"  重叠大小: {self.overlap_size} 字符")
        print("=" * 70)

        content = Path(markdown_path).read_text(encoding="utf-8")
        sections = self.parse_markdown(content)

        print("\n📦 开始分块...")
        all_chunks: list[TextChunk] = []

        for section in sections:
            level = section["level"]
            title = section["title"]
            section_content = section["content"]
            assert isinstance(level, int)
            assert isinstance(title, str)
            assert isinstance(section_content, str)

            self.update_context(level, title)

            if not section_content or len(section_content.strip()) < 10:
                continue

            if len(section_content) > self.max_chunk_size:
                if section_content.lstrip().startswith("<table"):
                    all_chunks.append(self.create_chunk(section_content, level))
                else:
                    all_chunks.extend(self.split_long_content(section_content, level))
            else:
                all_chunks.append(self.create_chunk(section_content, level))

        print(f"   ✅ 分块完成，共 {len(all_chunks)} 个块")

        processed: list[TextChunk] = []
        for chunk in all_chunks:
            if chunk.char_count >= self.min_chunk_size:
                processed.append(chunk)
            elif (
                processed
                and processed[-1].char_count + chunk.char_count + 2 <= self.max_chunk_size
            ):
                previous = processed[-1]
                previous.content = previous.content + "\n\n" + chunk.content
                previous.char_count = len(previous.content)
                previous.has_code = previous.has_code or chunk.has_code
                previous.has_image = previous.has_image or chunk.has_image

        print(f"   ✅ 合并过小块后，共 {len(processed)} 个块")
        self.print_statistics(processed)
        return processed

    def print_statistics(self, chunks: list[TextChunk]) -> None:
        """Print the legacy chunk statistics report."""

        print("\n" + "=" * 70)
        print("📊 分块统计")
        print("=" * 70)

        total_chars = sum(chunk.char_count for chunk in chunks)
        average_chars = total_chars / len(chunks) if chunks else 0
        level_counts: dict[int, int] = {}
        for chunk in chunks:
            level_counts[chunk.level] = level_counts.get(chunk.level, 0) + 1

        code_chunks = sum(1 for chunk in chunks if chunk.has_code)
        image_chunks = sum(1 for chunk in chunks if chunk.has_image)

        print(f"总块数:        {len(chunks)}")
        print(f"总字符数:      {total_chars:,}")
        print(f"平均块大小:    {average_chars:.0f} 字符")
        print("\n按层级分布:")
        for level in sorted(level_counts):
            print(f"  Level {level}: {level_counts[level]} 个块")
        print("\n特殊内容:")
        print(f"  包含代码: {code_chunks} 个块")
        print(f"  包含图片: {image_chunks} 个块")
        print("=" * 70)

    @staticmethod
    def preview_path(output_path: str | Path) -> Path:
        """Return the preview path paired with a JSON output path."""

        output = Path(output_path)
        return output.with_name(f"{output.stem}_preview.txt")

    def save_chunks(
        self,
        chunks: list[TextChunk],
        output_path: str | Path,
        *,
        overwrite: bool = False,
        write_preview: bool = True,
    ) -> None:
        """Save chunk JSON and an optional readable preview.

        Existing files are rejected by default so invoking the compatibility
        script cannot silently replace preserved graduation-project assets.
        """

        output = Path(output_path)
        preview = self.preview_path(output)
        destinations = [output]
        if write_preview:
            destinations.append(preview)
        existing = [path for path in destinations if path.exists()]
        if existing and not overwrite:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"输出文件已存在，未覆盖: {paths}")

        output.parent.mkdir(parents=True, exist_ok=True)
        print("\n💾 保存分块结果...")
        output.write_text(
            json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"   ✅ 已保存至: {output}")

        if not write_preview:
            return

        preview_lines: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            preview_lines.extend(
                [
                    f"\n{'=' * 70}",
                    f"Chunk {index}/{len(chunks)}",
                    f"ID: {chunk.chunk_id}",
                    "=" * 70,
                ]
            )
            if chunk.chapter:
                preview_lines.append(f"📖 {chunk.chapter}")
            if chunk.section_h2:
                preview_lines.append(f"  └─ {chunk.section_h2}")
            if chunk.section_h3:
                preview_lines.append(f"      └─ {chunk.section_h3}")
            if chunk.section_h4:
                preview_lines.append(f"          └─ {chunk.section_h4}")

            details = f"\n[Level {chunk.level}] ({chunk.char_count} 字符)"
            if chunk.has_code:
                details += " 📝代码"
            if chunk.has_image:
                details += " 🖼️图片"
            preview_lines.extend([details, "", chunk.content[:500]])
            if len(chunk.content) > 500:
                preview_lines.append("\n... (内容已截断) ...")

        preview.write_text("\n".join(preview_lines) + "\n", encoding="utf-8")
        print(f"   ✅ 预览文件: {preview}")


def chunk_markdown(
    input_path: str | Path,
    output_json: str | Path,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
    overlap_size: int = 50,
    *,
    overwrite: bool = False,
    write_preview: bool = True,
) -> list[TextChunk]:
    """Chunk one Markdown file and save its JSON representation."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 Markdown 文件: {source}")
    output = Path(output_json)
    if source.resolve() == output.resolve():
        raise ValueError("输入 Markdown 和输出 JSON 不能是同一个文件")

    chunker = SmartTextbookChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
    )
    chunks = chunker.chunk_document(source)
    chunker.save_chunks(
        chunks,
        output,
        overwrite=overwrite,
        write_preview=write_preview,
    )
    return chunks


def batch_chunk_markdown(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
    overlap_size: int = 50,
    *,
    pattern: str = "*_cleaned.md",
    overwrite: bool = False,
    write_preview: bool = True,
) -> BatchChunkResult:
    """Chunk all matching cleaned Markdown files in deterministic name order."""

    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"找不到输入目录: {source_dir}")
    destination_dir = Path(output_dir) if output_dir is not None else source_dir

    sources = sorted(source_dir.glob(pattern), key=lambda path: path.name)
    if not sources:
        raise FileNotFoundError(
            f"输入目录中没有匹配 {pattern!r} 的 Markdown 文件: {source_dir}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    skipped: list[Path] = []
    for source in sources:
        output_name = source.stem.replace("_cleaned", "_chunks") + ".json"
        output = destination_dir / output_name
        preview = SmartTextbookChunker.preview_path(output)
        if not overwrite and (output.exists() or (write_preview and preview.exists())):
            skipped.append(output)
            continue

        chunk_markdown(
            source,
            output,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size,
            overwrite=overwrite,
            write_preview=write_preview,
        )
        created.append(output)

    return BatchChunkResult(tuple(created), tuple(skipped))


def chunk_single_file(input_file: Path, output_json: Path, *, overwrite: bool = False) -> bool:
    """Backward-compatible single-file helper used by the legacy entry point."""

    try:
        chunk_markdown(input_file, output_json, overwrite=overwrite)
    except (OSError, ValueError) as error:
        print(f"错误: {error}")
        return False
    return True


def batch_chunk_cleaned(
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> BatchChunkResult:
    """Backward-compatible alias for batching files in one directory."""

    directory = output_dir if output_dir is not None else Path.cwd()
    return batch_chunk_markdown(directory, overwrite=overwrite)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按教材标题结构分块 Markdown")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--single", type=Path, metavar="MARKDOWN", help="处理单个 Markdown 文件"
    )
    mode.add_argument("--batch", type=Path, metavar="DIRECTORY", help="批量处理 *_cleaned.md")
    parser.add_argument("--output", type=Path, help="单文件输出 JSON 路径")
    parser.add_argument(
        "--output-dir", type=Path, help="批量输出目录（默认为输入目录）"
    )
    parser.add_argument("--max-chunk-size", type=int, default=800)
    parser.add_argument("--min-chunk-size", type=int, default=100)
    parser.add_argument("--overlap-size", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    parser.add_argument("--no-preview", action="store_true", help="不生成文本预览")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone/legacy-compatible chunking command."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.single is not None:
            output = args.output or args.single.with_name(
                args.single.stem.replace("_cleaned", "_chunks") + ".json"
            )
            chunk_markdown(
                args.single,
                output,
                max_chunk_size=args.max_chunk_size,
                min_chunk_size=args.min_chunk_size,
                overlap_size=args.overlap_size,
                overwrite=args.force,
                write_preview=not args.no_preview,
            )
            return 0

        result = batch_chunk_markdown(
            args.batch,
            args.output_dir,
            max_chunk_size=args.max_chunk_size,
            min_chunk_size=args.min_chunk_size,
            overlap_size=args.overlap_size,
            overwrite=args.force,
            write_preview=not args.no_preview,
        )
        print(
            f"批量分块完成：新建 {len(result.created)}，"
            f"跳过 {len(result.skipped_existing)}"
        )
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"错误: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
