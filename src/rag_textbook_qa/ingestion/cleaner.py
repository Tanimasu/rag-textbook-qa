"""Markdown cleaning logic migrated from the original graduation project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class SmartMarkdownCleaner:
    """Clean parsed textbook Markdown while preserving the legacy V4 rules."""

    def __init__(self) -> None:
        self.invalid_title_patterns = [
            r"^(int|void|char|float|double|boolean|typedef|struct|enum|class|const|static)",
            r"^[a-zA-Z_]\w*\s*[\[\(=;]",
            r"^[\d\s]+[）\)]$",
            r"^[}{\[\];,\(\)]+$",
            r"^\d+\s*(退出区|剩余区|临界区)",
            r"^while|^for|^if|^return",
        ]

        # 标题层级规则的顺序很重要。
        self.title_patterns = [
            (r"^(第\s*[0-9０-９]+\s*章)", 1),
            (r"^(\d+\.\d+\.\d+\.\d+)", 4),
            (r"^(\d+\.\d+\.\d+)", 3),
            (r"^(\d+\.\d+)", 2),
            (r"^(\d+)[．\.][\u4e00-\u9fa5]", 4),
            (r"^(\d+)[．\.\s]", 2),
        ]

        self.header_footer_keywords = [
            "存储器管理",
            "存储器管",
            "操作系统",
            "计算机系统",
            "计算机操作系统",
            "慕课版",
            "上册",
            "下册",
        ]

    def is_valid_title(self, text: str) -> bool:
        """判断是否是有效的标题。"""
        text = text.strip()

        for pattern in self.invalid_title_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False

        if len(text) < 2:
            return False

        if re.match(r"^[\W\d\s]+$", text):
            return False

        return True

    def detect_title_level(self, text: str) -> tuple[int, str]:
        """智能检测标题层级，返回层级和清理后的文本。"""
        text = text.strip()
        text = re.sub(r"^#+\s*", "", text)

        if not self.is_valid_title(text):
            return (0, text)

        for pattern, level in self.title_patterns:
            if re.match(pattern, text):
                return (level, text)

        return (3, text)

    def remove_page_headers_footers(self, content: str) -> tuple[str, int]:
        """移除页眉页脚残留。"""
        print("\n🧹 移除页眉页脚...")
        removed_count = 0

        pattern = r"(> 📷 \*\*\[图片\]\*\*.*?\n\n)([^\n#]{1,15}\n\n)"
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, r"\1", content)

        pattern = r"\n\n([\u4e00-\u9fa5]{1,10})\n\n"
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, "\n\n", content)

        for keyword in self.header_footer_keywords:
            pattern = f"\n\n{re.escape(keyword)}\n\n"
            count = content.count(pattern)
            if count > 0:
                removed_count += count
                content = content.replace(pattern, "\n\n")

        print(f"   ✅ 移除了 {removed_count} 处页眉页脚残留")
        return content, removed_count

    def remove_isolated_fragments(self, content: str) -> tuple[str, int]:
        """移除孤立的文本碎片。"""
        print("\n🧹 移除孤立碎片...")
        removed_count = 0

        pattern = r"\n\n(\d{1,4})\n\n"
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, "\n\n", content)

        pattern = r"\n\n(第\s*\d+\s*章)\n\n"
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, "\n\n", content)

        pattern = r"(#{1,6}\s+.+?\n\n)([^\n#]{1,15}\n\n)(?=[^\n])"
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, r"\1", content)

        print(f"   ✅ 移除了 {removed_count} 处孤立碎片")
        return content, removed_count

    def fix_broken_paragraphs(self, content: str) -> tuple[str, int]:
        """修复被错误分割的段落。"""
        print("\n🔧 修复段落...")

        pattern = (
            r"\n\n([^\n#]{1,10})\n\n"
            r"(计算机操作系统|操作系统|上述|下面|因此|所以|"
            r"但是|然而|同时|此外)"
        )
        fixed_count = len(re.findall(pattern, content))
        content = re.sub(pattern, r"\n\n\2", content)

        print(f"   ✅ 修复了 {fixed_count} 处段落分割")
        return content, fixed_count

    def normalize_titles(self, content: str) -> str:
        """规范化标题层级。"""
        lines = content.split("\n")
        result = []

        for line in lines:
            if line.strip().startswith("#"):
                title_text = re.sub(r"^#+\s*", "", line.strip())
                level, clean_text = self.detect_title_level(title_text)

                if level == 0:
                    result.append(clean_text)
                else:
                    result.append("#" * level + " " + clean_text)
            else:
                result.append(line)

        return "\n".join(result)

    def validate_content(self, content: str) -> dict[str, Any]:
        """验证清理后的内容质量。"""
        print("\n📊 内容质量检查...")

        isolated_paragraphs = re.findall(r"\n\n([^\n#]{1,15})\n\n", content)
        h1_count = len(re.findall(r"^\# ", content, re.MULTILINE))
        h2_count = len(re.findall(r"^\## ", content, re.MULTILINE))
        h3_count = len(re.findall(r"^\### ", content, re.MULTILINE))
        h4_count = len(re.findall(r"^\#### ", content, re.MULTILINE))
        paragraph_count = len(re.findall(r"\n\n[^\n#].{20,}", content))

        stats = {
            "isolated_paragraphs": len(isolated_paragraphs),
            "isolated_examples": isolated_paragraphs[:5],
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "h4_count": h4_count,
            "paragraph_count": paragraph_count,
            "total_length": len(content),
        }

        print(f"   # 章节标题: {h1_count}")
        print(f"   ## 二级标题: {h2_count}")
        print(f"   ### 三级标题: {h3_count}")
        print(f"   #### 四级标题: {h4_count}")
        print(f"   段落数: {paragraph_count}")
        print(f"   总字符数: {len(content):,}")

        if isolated_paragraphs:
            print(f"\n   ⚠️  仍有 {len(isolated_paragraphs)} 处疑似孤立文本:")
            for example in isolated_paragraphs[:5]:
                print(f"      - '{example}'")
        else:
            print("   ✅ 未发现明显的孤立文本")

        return stats

    def clean(self, input_path: str | Path, output_path: str | Path) -> str:
        """执行完整清洗流程并返回清洗后的 Markdown。"""
        input_path = Path(input_path)
        output_path = Path(output_path)

        print("=" * 70)
        print("🚀 智能 Markdown 清洗 V4")
        print("=" * 70)

        content = input_path.read_text(encoding="utf-8")
        original_length = len(content)

        print("\n📝 步骤 1/15: 移除乱码...")
        garbage_patterns = [
            r"订ካ఻୲ͻጇፒ",
            r"啖ᚂ䄫❵啗",
            r"订ካ఻୲ͻ紊ፒ",
            r"ኄ",
            r"୲ͻጇፒ",
            r"ᝠካ఻",
        ]
        for pattern in garbage_patterns:
            content = re.sub(pattern, "", content)
        print("   ✅ 已移除乱码")

        print("\n📝 步骤 2/15: 合并分离的标题...")
        content = re.sub(r"## (第\d+章)\s*\n+## ([^\n]+)", r"## \1 \2", content)
        print("   ✅ 已合并标题")

        print("\n📝 步骤 3/15: 处理图片占位符...")
        image_count = content.count("<!-- image -->")
        content = re.sub(r"<!-- image -->", "\n\n> 📷 **[图片]**\n\n", content)
        print(f"   ✅ 已处理 {image_count} 个图片")

        content, removed_headers = self.remove_page_headers_footers(content)
        content, removed_fragments = self.remove_isolated_fragments(content)
        content, fixed_paragraphs = self.fix_broken_paragraphs(content)

        print("\n📝 步骤 7/15: 🔥 智能规范化标题层级...")
        before_titles = len(re.findall(r"^#", content, re.MULTILINE))
        content = self.normalize_titles(content)
        after_titles = len(re.findall(r"^#{1,4} ", content, re.MULTILINE))
        print(f"   ✅ 标题数量: {before_titles} → {after_titles}")

        print("\n📝 步骤 8/15: 规范列表符号...")
        circle_numbers = {
            "①": "(1)",
            "②": "(2)",
            "③": "(3)",
            "④": "(4)",
            "⑤": "(5)",
            "⑥": "(6)",
            "⑦": "(7)",
            "⑧": "(8)",
            "⑨": "(9)",
            "⑩": "(10)",
        }
        for old, new in circle_numbers.items():
            content = content.replace(old, new)
        print("   ✅ 已规范列表")

        print("\n📝 步骤 9/15: 规范标点...")
        content = re.sub(r"(\d+)．", r"\1.", content)
        print("   ✅ 已规范标点")

        print("\n📝 步骤 10/15: 转换HTML实体...")
        html_entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&nbsp;": " ",
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)
        print("   ✅ 已转换HTML实体")

        print("\n📝 步骤 11/15: 修复转义...")
        content = re.sub(r"(\w)\\_(\w)", r"\1_\2", content)
        content = content.replace(r"\_", "_")
        print("   ✅ 已修复转义")

        print("\n📝 步骤 12/15: 修复代码块...")

        def fix_code_punctuation(match: re.Match[str]) -> str:
            code = match.group(0)
            code = code.replace("；", ";").replace("，", ",")
            return code.replace("（", "(").replace("）", ")")

        content = re.sub(r"```[\s\S]*?```", fix_code_punctuation, content)
        print("   ✅ 已修复代码块")

        print("\n📝 步骤 13/15: 清理空行...")
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)
        print("   ✅ 已清理空行")

        print("\n📝 步骤 14/15: 优化格式...")
        content = re.sub(r"([^\n])\n(#{1,4} )", r"\1\n\n\2", content)
        content = re.sub(r"(#{1,4} [^\n]+)\n([^\n#])", r"\1\n\n\2", content)
        content = re.sub(r"([^\n])\n(```)", r"\1\n\n\2", content)
        content = re.sub(r"(```)\n([^\n`])", r"\1\n\n\2", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        print("   ✅ 已优化格式")

        stats = self.validate_content(content)

        print("\n📝 步骤 15/15: 最终整理...")
        content = content.lstrip("\n").rstrip("\n") + "\n"
        print("   ✅ 最终整理完成")

        print("\n📝 保存文件...")
        output_path.write_text(content, encoding="utf-8")
        print(f"   ✅ 已保存至: {output_path}")

        self.print_statistics(
            content,
            original_length,
            image_count,
            removed_headers,
            removed_fragments,
            fixed_paragraphs,
            stats,
            output_path,
        )

        return content

    def print_statistics(
        self,
        content: str,
        original_length: int,
        image_count: int,
        removed_headers: int,
        removed_fragments: int,
        fixed_paragraphs: int,
        stats: dict[str, Any],
        output_path: Path,
    ) -> None:
        """打印统计信息。"""
        cleaned_length = len(content)

        print("\n" + "=" * 70)
        print("📊 清洗统计")
        print("=" * 70)
        print(f"原始字符数:    {original_length:,}")
        print(f"清洗后字符数:  {cleaned_length:,}")
        print(f"变化:          {cleaned_length - original_length:+,}")

        print("\n🧹 清理统计:")
        print(f"  移除页眉页脚:  {removed_headers} 处")
        print(f"  移除孤立碎片:  {removed_fragments} 处")
        print(f"  修复断裂段落:  {fixed_paragraphs} 处")

        chapters = stats["h1_count"]
        level2 = stats["h2_count"]
        level3 = stats["h3_count"]
        level4 = stats["h4_count"]
        code_blocks = len(re.findall(r"^```", content, re.MULTILINE)) // 2

        print("\n📚 结构统计:")
        print(f"  # 章节:        {chapters} 个")
        print(f"  ## 二级标题:   {level2} 个")
        print(f"  ### 三级标题:  {level3} 个")
        print(f"  #### 四级标题: {level4} 个")
        print(f"  代码块:        {code_blocks} 个")
        print(f"  图片:          {image_count} 个")

        print("\n✅ 清洗完成！")
        print("=" * 70)


def clean_markdown(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> str:
    """Clean one Markdown file using the legacy V4 algorithm.

    Both paths are explicit so importing this module or invoking its API never
    selects or overwrites a historical project asset implicitly.
    """
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 Markdown 文件: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("输入和输出不能是同一个文件")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，未覆盖: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    return SmartMarkdownCleaner().clean(source, destination)
