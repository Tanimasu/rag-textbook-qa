# chunk_textbook.py
"""
📚 教材智能分块工具
支持多层级标题结构，保持上下文完整性
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TextChunk:
    """文本块数据结构"""
    chunk_id: str  # 块ID: "ch1_s1.1_p001"
    chapter: str  # 章节: "第1章 操作系统引论"
    section_h2: str  # 二级标题: "1.1 操作系统的目标"
    section_h3: str  # 三级标题: "1.1.1 操作系统的目标"
    section_h4: str  # 四级标题: "1.方便性"
    content: str  # 实际内容
    level: int  # 标题层级 (1-4)
    char_count: int  # 字符数
    has_code: bool  # 是否包含代码
    has_image: bool  # 是否包含图片


class SmartTextbookChunker:
    """智能教材分块器"""

    def __init__(self,
                 max_chunk_size: int = 800,
                 min_chunk_size: int = 100,
                 overlap_size: int = 50):
        """
        初始化分块器

        Args:
            max_chunk_size: 最大块大小（字符数）
            min_chunk_size: 最小块大小（字符数）
            overlap_size: 重叠大小（字符数）
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

        # 当前上下文
        self.current_chapter = ""  # 当前章
        self.current_h2 = ""  # 当前二级标题
        self.current_h3 = ""  # 当前三级标题
        self.current_h4 = ""  # 当前四级标题

        # 章节计数器
        self.chapter_num = 0
        self.chunk_counter = 0

    def parse_markdown(self, content: str) -> List[Dict]:
        """解析 Markdown 文档结构"""
        print("📖 解析文档结构...")

        sections = []
        lines = content.split('\n')

        current_section = {
            'level': 0,
            'title': '',
            'content': []
        }

        for line in lines:
            # 检测标题
            title_match = re.match(r'^(#{1,4})\s+(.+)$', line)

            if title_match:
                # 保存上一个section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section.copy())

                # 开始新section
                level = len(title_match.group(1))
                title = title_match.group(2).strip()

                current_section = {
                    'level': level,
                    'title': title,
                    'content': []
                }
            else:
                # 累积内容
                if line.strip():  # 跳过纯空行
                    current_section['content'].append(line)

        # 保存最后一个section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)

        print(f"   ✅ 解析完成，共 {len(sections)} 个段落")
        return sections

    def update_context(self, level: int, title: str):
        """更新当前标题上下文"""
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
        """生成块ID"""
        self.chunk_counter += 1

        # 提取章节编号
        ch_match = re.search(r'第\s*(\d+)\s*章', self.current_chapter)
        ch_num = ch_match.group(1) if ch_match else str(self.chapter_num)

        # 提取二级标题编号
        h2_match = re.match(r'(\d+\.\d+)', self.current_h2)
        h2_num = h2_match.group(1).replace('.', '_') if h2_match else ''

        if h2_num:
            return f"ch{ch_num}_s{h2_num}_p{self.chunk_counter:03d}"
        else:
            return f"ch{ch_num}_p{self.chunk_counter:03d}"

    def create_chunk(self, content: str, level: int) -> TextChunk:
        """创建文本块"""

        # 检测特殊内容
        has_code = '```' in content
        has_image = '📷' in content or '[图片]' in content

        chunk = TextChunk(
            chunk_id=self.generate_chunk_id(),
            chapter=self.current_chapter,
            section_h2=self.current_h2,
            section_h3=self.current_h3,
            section_h4=self.current_h4,
            content=content.strip(),
            level=level,
            char_count=len(content),
            has_code=has_code,
            has_image=has_image
        )

        return chunk

    def split_long_content(self, content: str, level: int) -> List[TextChunk]:
        """分割过长的内容"""
        chunks = []

        # 按段落分割
        paragraphs = re.split(r'\n\n+', content)

        current_content = ""

        for para in paragraphs:
            # 如果当前段落本身就很长
            if len(para) > self.max_chunk_size:
                # 先保存之前的内容
                if current_content:
                    chunks.append(self.create_chunk(current_content, level))
                    current_content = ""

                # 按句子分割长段落
                sentences = re.split(r'([。！？\n])', para)
                temp_content = ""

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    separator = sentences[i + 1] if i + 1 < len(sentences) else ''

                    if len(temp_content) + len(sentence) + len(separator) > self.max_chunk_size:
                        if temp_content:
                            chunks.append(self.create_chunk(temp_content, level))
                        temp_content = sentence + separator
                    else:
                        temp_content += sentence + separator

                if temp_content:
                    current_content = temp_content

            # 如果加上这个段落会超长
            elif len(current_content) + len(para) > self.max_chunk_size:
                if current_content:
                    chunks.append(self.create_chunk(current_content, level))
                current_content = para

            # 否则累积
            else:
                current_content += ("\n\n" if current_content else "") + para

        # 保存剩余内容
        if current_content:
            chunks.append(self.create_chunk(current_content, level))

        return chunks

    def chunk_document(self, markdown_path: str) -> List[TextChunk]:
        """分块整个文档"""

        print("=" * 70)
        print("🚀 开始智能分块")
        print("=" * 70)
        print(f"配置:")
        print(f"  最大块大小: {self.max_chunk_size} 字符")
        print(f"  最小块大小: {self.min_chunk_size} 字符")
        print(f"  重叠大小: {self.overlap_size} 字符")
        print("=" * 70)

        # 读取文件
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析结构
        sections = self.parse_markdown(content)

        # 分块
        print("\n📦 开始分块...")
        all_chunks = []

        for section in sections:
            level = section['level']
            title = section['title']
            content = section['content']

            # 更新上下文
            self.update_context(level, title)

            # 跳过纯标题（无内容）
            if not content or len(content.strip()) < 10:
                continue

            # 如果内容过长，分割
            if len(content) > self.max_chunk_size:
                chunks = self.split_long_content(content, level)
                all_chunks.extend(chunks)
            else:
                # 直接创建块
                chunk = self.create_chunk(content, level)
                all_chunks.append(chunk)

        print(f"   ✅ 分块完成，共 {len(all_chunks)} 个块")

        # 后处理：合并过小的块
        processed = []
        for chunk in all_chunks:
            if chunk.char_count >= self.min_chunk_size:
                processed.append(chunk)
            else:
                # 尝试合并到前一个块
                if processed and (processed[-1].char_count + chunk.char_count + 2) <= self.max_chunk_size:
                    prev = processed[-1]
                    prev.content = prev.content + "\n\n" + chunk.content
                    prev.char_count = len(prev.content)
                    prev.has_code = prev.has_code or chunk.has_code
                    prev.has_image = prev.has_image or chunk.has_image
                # 否则丢弃（纯图片占位符、孤立碎片等）
        all_chunks = processed
        print(f"   ✅ 合并过小块后，共 {len(all_chunks)} 个块")

        # 统计信息
        self.print_statistics(all_chunks)

        return all_chunks

    def print_statistics(self, chunks: List[TextChunk]):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 分块统计")
        print("=" * 70)

        total_chars = sum(c.char_count for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0

        level_counts = {}
        for chunk in chunks:
            level_counts[chunk.level] = level_counts.get(chunk.level, 0) + 1

        code_chunks = sum(1 for c in chunks if c.has_code)
        image_chunks = sum(1 for c in chunks if c.has_image)

        print(f"总块数:        {len(chunks)}")
        print(f"总字符数:      {total_chars:,}")
        print(f"平均块大小:    {avg_chars:.0f} 字符")

        print(f"\n按层级分布:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} 个块")

        print(f"\n特殊内容:")
        print(f"  包含代码: {code_chunks} 个块")
        print(f"  包含图片: {image_chunks} 个块")

        print("=" * 70)

    def save_chunks(self, chunks: List[TextChunk], output_path: str):
        """保存分块结果"""
        print(f"\n💾 保存分块结果...")

        # 转换为字典列表
        chunks_data = [asdict(chunk) for chunk in chunks]

        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 已保存至: {output_path}")

        # 同时保存一个可读的文本版本
        txt_path = output_path.replace('.json', '_preview.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n{'=' * 70}\n")
                f.write(f"Chunk {i}/{len(chunks)}\n")
                f.write(f"ID: {chunk.chunk_id}\n")
                f.write(f"{'=' * 70}\n")

                if chunk.chapter:
                    f.write(f"📖 {chunk.chapter}\n")
                if chunk.section_h2:
                    f.write(f"  └─ {chunk.section_h2}\n")
                if chunk.section_h3:
                    f.write(f"      └─ {chunk.section_h3}\n")
                if chunk.section_h4:
                    f.write(f"          └─ {chunk.section_h4}\n")

                f.write(f"\n[Level {chunk.level}] ({chunk.char_count} 字符)")
                if chunk.has_code:
                    f.write(" 📝代码")
                if chunk.has_image:
                    f.write(" 🖼️图片")
                f.write("\n\n")

                f.write(chunk.content[:500])  # 只显示前500字符
                if len(chunk.content) > 500:
                    f.write("\n\n... (内容已截断) ...")
                f.write("\n")

        print(f"   ✅ 预览文件: {txt_path}")


def main():
    """主函数"""

    # 配置路径
    input_file = Path(r"D:\CodeField\Graduation_project\project\output\计算机网络_cleaned.md")
    output_json = Path(r"D:\CodeField\Graduation_project\project\output\计算机网络_chunks.json")

    # 检查输入文件
    if not input_file.exists():
        print(f"❌ 错误：找不到文件 {input_file}")
        exit(1)

    print(f"📂 输入文件: {input_file}")
    print(f"📂 输出文件: {output_json}")
    print(f"📂 文件大小: {input_file.stat().st_size:,} 字节\n")

    try:
        # 创建分块器
        chunker = SmartTextbookChunker(
            max_chunk_size=800,  # 最大800字符
            min_chunk_size=100,  # 最小100字符
            overlap_size=50  # 重叠50字符
        )

        # 执行分块
        chunks = chunker.chunk_document(str(input_file))

        # 保存结果
        chunker.save_chunks(chunks, str(output_json))

        # 显示示例
        print("\n" + "=" * 70)
        print("📄 分块示例（前3个块）")
        print("=" * 70)

        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n【Chunk {i}】")
            print(f"ID: {chunk.chunk_id}")
            print(f"章节: {chunk.chapter}")
            if chunk.section_h2:
                print(f"二级: {chunk.section_h2}")
            if chunk.section_h3:
                print(f"三级: {chunk.section_h3}")
            if chunk.section_h4:
                print(f"四级: {chunk.section_h4}")
            print(f"长度: {chunk.char_count} 字符")
            print(f"内容预览:\n{chunk.content[:200]}...")

        print("\n" + "=" * 70)
        print("🎉 分块完成！")
        print("\n💡 下一步:")
        print(f"   1. 检查 JSON 文件: {output_json}")
        print(f"   2. 检查预览文件: {output_json.parent / (output_json.stem + '_preview.txt')}")
        print(f"   3. 确认分块质量后，可进行向量化")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
