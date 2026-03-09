# ultimate_clean_markdown_v4.py
"""
📝 Markdown 智能清洗脚本 V4
修复：正确识别列表项标题为四级标题
"""

import re
from pathlib import Path
from typing import List, Tuple


class SmartMarkdownCleaner:
    def __init__(self):
        """初始化清洗器"""

        # 无效标题模式（代码、变量等）
        self.invalid_title_patterns = [
            r'^(int|void|char|float|double|boolean|typedef|struct|enum|class|const|static)',
            r'^[a-zA-Z_]\w*\s*[\[\(=;]',
            r'^[\d\s]+[）\)]$',
            r'^[}{\[\];,\(\)]+$',
            r'^\d+\s*(退出区|剩余区|临界区)',
            r'^while|^for|^if|^return',
        ]

        # ✅ 修正后的标题层级规则（顺序很重要！）
        self.title_patterns = [
            # 1. 章节标题（最高优先级）
            (r'^(第\s*[0-9０-９]+\s*章)', 1),  # 第X章 → #

            # 2. 多级编号（从长到短匹配，避免被截断）
            (r'^(\d+\.\d+\.\d+\.\d+)', 4),  # X.X.X.X → ####
            (r'^(\d+\.\d+\.\d+)', 3),  # X.X.X → ###
            (r'^(\d+\.\d+)', 2),  # X.X → ##

            # 3. ✅ 列表项编号（单个数字+句号+中文）
            (r'^(\d+)[．\.][\u4e00-\u9fa5]', 4),  # 1.方便性 → ####

            # 4. 纯数字编号（后面是空格或英文）
            (r'^(\d+)[．\.\s]', 2),  # 其他 X. → ##（默认）
        ]

        # 常见的页眉页脚关键词
        self.header_footer_keywords = [
            '存储器管理', '存储器管', '操作系统', '计算机系统',
            '计算机操作系统', '慕课版', '上册', '下册'
        ]

    def is_valid_title(self, text: str) -> bool:
        """判断是否是有效的标题"""
        text = text.strip()

        # 检查无效模式
        for pattern in self.invalid_title_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False

        # 标题不能太短
        if len(text) < 2:
            return False

        # 标题不能全是符号
        if re.match(r'^[\W\d\s]+$', text):
            return False

        return True

    def detect_title_level(self, text: str) -> Tuple[int, str]:
        """
        智能检测标题层级
        返回: (层级, 清理后的文本)
        """
        text = text.strip()

        # 去掉开头的 #
        text = re.sub(r'^#+\s*', '', text)

        # 检查是否是有效标题
        if not self.is_valid_title(text):
            return (0, text)  # 0表示不是标题

        # ✅ 按规则匹配层级（顺序很重要！）
        for pattern, level in self.title_patterns:
            match = re.match(pattern, text)
            if match:
                # 调试输出（可选）
                # print(f"   匹配: '{text[:30]}' → 层级{level}")
                return (level, text)

        # 默认返回三级标题
        return (3, text)

    def remove_page_headers_footers(self, content: str) -> Tuple[str, int]:
        """移除页眉页脚残留"""
        print("\n🧹 移除页眉页脚...")

        removed_count = 0

        # 1. 移除图片后的孤立短文本（很可能是页眉）
        pattern = r'(> 📷 \*\*\[图片\]\*\*.*?\n\n)([^\n#]{1,15}\n\n)'
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, r'\1', content)

        # 2. 移除段落间的孤立短文本（1-10个汉字单独成段）
        pattern = r'\n\n([\u4e00-\u9fa5]{1,10})\n\n'
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, '\n\n', content)

        # 3. 移除特定的页眉页脚关键词（单独成段的）
        for keyword in self.header_footer_keywords:
            pattern = f'\n\n{re.escape(keyword)}\n\n'
            count = content.count(pattern)
            if count > 0:
                removed_count += count
                content = content.replace(pattern, '\n\n')

        print(f"   ✅ 移除了 {removed_count} 处页眉页脚残留")
        return content, removed_count

    def remove_isolated_fragments(self, content: str) -> Tuple[str, int]:
        """移除孤立的文本碎片"""
        print("\n🧹 移除孤立碎片...")

        removed_count = 0

        # 1. 移除孤立的页码（单独成段的纯数字）
        pattern = r'\n\n(\d{1,4})\n\n'
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, '\n\n', content)

        # 2. 移除孤立的章节引用（如 "第5章"）
        pattern = r'\n\n(第\s*\d+\s*章)\n\n'
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, '\n\n', content)

        # 3. 移除标题后的孤立短文本（标题和下一段之间的碎片）
        pattern = r'(#{1,6}\s+.+?\n\n)([^\n#]{1,15}\n\n)(?=[^\n])'
        matches = re.findall(pattern, content)
        removed_count += len(matches)
        content = re.sub(pattern, r'\1', content)

        print(f"   ✅ 移除了 {removed_count} 处孤立碎片")
        return content, removed_count

    def fix_broken_paragraphs(self, content: str) -> Tuple[str, int]:
        """修复被错误分割的段落"""
        print("\n🔧 修复段落...")

        # 修复 "存储器管" + "计算机操作系统" 这类情况
        pattern = r'\n\n([^\n#]{1,10})\n\n(计算机操作系统|操作系统|上述|下面|因此|所以|但是|然而|同时|此外)'
        fixed_count = len(re.findall(pattern, content))
        content = re.sub(pattern, r'\n\n\2', content)

        print(f"   ✅ 修复了 {fixed_count} 处段落分割")
        return content, fixed_count

    def normalize_titles(self, content: str) -> str:
        """规范化标题层级"""
        lines = content.split('\n')
        result = []

        for line in lines:
            # 检测是否是标题行
            if line.strip().startswith('#'):
                # 提取标题文本
                title_text = re.sub(r'^#+\s*', '', line.strip())

                # 智能判断层级
                level, clean_text = self.detect_title_level(title_text)

                if level == 0:
                    # 不是有效标题，作为普通文本
                    result.append(clean_text)
                else:
                    # 生成正确层级的标题
                    result.append('#' * level + ' ' + clean_text)
            else:
                result.append(line)

        return '\n'.join(result)

    def validate_content(self, content: str) -> dict:
        """验证清理后的内容质量"""
        print("\n📊 内容质量检查...")

        # 检查是否还有孤立的短段落
        isolated_paragraphs = re.findall(r'\n\n([^\n#]{1,15})\n\n', content)

        # 检查标题数量
        h1_count = len(re.findall(r'^\# ', content, re.MULTILINE))
        h2_count = len(re.findall(r'^\## ', content, re.MULTILINE))
        h3_count = len(re.findall(r'^\### ', content, re.MULTILINE))
        h4_count = len(re.findall(r'^\#### ', content, re.MULTILINE))

        # 检查段落数量
        paragraph_count = len(re.findall(r'\n\n[^\n#].{20,}', content))

        stats = {
            'isolated_paragraphs': len(isolated_paragraphs),
            'isolated_examples': isolated_paragraphs[:5],
            'h1_count': h1_count,
            'h2_count': h2_count,
            'h3_count': h3_count,
            'h4_count': h4_count,
            'paragraph_count': paragraph_count,
            'total_length': len(content)
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

    def clean(self, input_path: str, output_path: str):
        """执行完整清洗流程"""

        print("=" * 70)
        print("🚀 智能 Markdown 清洗 V4")
        print("=" * 70)

        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_length = len(content)

        # ========== 1. 移除乱码 ==========
        print("\n📝 步骤 1/15: 移除乱码...")
        garbage_patterns = [
            r'订ካ఻୲ͻጇፒ',
            r'啖ᚂ䄫❵啗',
            r'订ካ఻୲ͻ紊ፒ',
            r'ኄ', r'୲ͻጇፒ', r'ᝠካ఻',
        ]
        for pattern in garbage_patterns:
            content = re.sub(pattern, '', content)
        print("   ✅ 已移除乱码")

        # ========== 2. 合并分离的章节标题 ==========
        print("\n📝 步骤 2/15: 合并分离的标题...")
        content = re.sub(r'## (第\d+章)\s*\n+## ([^\n]+)', r'## \1 \2', content)
        print("   ✅ 已合并标题")

        # ========== 3. 处理图片占位符 ==========
        print("\n📝 步骤 3/15: 处理图片占位符...")
        image_count = content.count('<!-- image -->')
        content = re.sub(r'<!-- image -->', '\n\n> 📷 **[图片]**\n\n', content)
        print(f"   ✅ 已处理 {image_count} 个图片")

        # ========== 4. 移除页眉页脚 ==========
        content, removed_headers = self.remove_page_headers_footers(content)

        # ========== 5. 移除孤立碎片 ==========
        content, removed_fragments = self.remove_isolated_fragments(content)

        # ========== 6. 修复断裂段落 ==========
        content, fixed_paragraphs = self.fix_broken_paragraphs(content)

        # ========== 7. 🔥 智能规范化标题层级 🔥 ==========
        print("\n📝 步骤 7/15: 🔥 智能规范化标题层级...")
        before_titles = len(re.findall(r'^#', content, re.MULTILINE))
        content = self.normalize_titles(content)
        after_titles = len(re.findall(r'^#{1,4} ', content, re.MULTILINE))
        print(f"   ✅ 标题数量: {before_titles} → {after_titles}")

        # ========== 8. 规范列表符号 ==========
        print("\n📝 步骤 8/15: 规范列表符号...")
        circle_numbers = {
            '①': '(1)', '②': '(2)', '③': '(3)', '④': '(4)', '⑤': '(5)',
            '⑥': '(6)', '⑦': '(7)', '⑧': '(8)', '⑨': '(9)', '⑩': '(10)'
        }
        for old, new in circle_numbers.items():
            content = content.replace(old, new)
        print("   ✅ 已规范列表")

        # ========== 9. 规范标点符号 ==========
        print("\n📝 步骤 9/15: 规范标点...")
        content = re.sub(r'(\d+)．', r'\1.', content)
        print("   ✅ 已规范标点")

        # ========== 10. 转换HTML实体 ==========
        print("\n📝 步骤 10/15: 转换HTML实体...")
        html_entities = {
            '&lt;': '<', '&gt;': '>', '&amp;': '&',
            '&quot;': '"', '&apos;': "'", '&nbsp;': ' ',
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)
        print("   ✅ 已转换HTML实体")

        # ========== 11. 修复转义字符 ==========
        print("\n📝 步骤 11/15: 修复转义...")
        content = re.sub(r'(\w)\\\_(\w)', r'\1_\2', content)
        content = content.replace(r'\_', '_')
        print("   ✅ 已修复转义")

        # ========== 12. 修复代码块中的中文标点 ==========
        print("\n📝 步骤 12/15: 修复代码块...")

        def fix_code_punctuation(match):
            code = match.group(0)
            code = code.replace('；', ';').replace('，', ',')
            code = code.replace('（', '(').replace('）', ')')
            return code

        content = re.sub(r'```[\s\S]*?```', fix_code_punctuation, content)
        print("   ✅ 已修复代码块")

        # ========== 13. 清理多余空行 ==========
        print("\n📝 步骤 13/15: 清理空行...")
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        print("   ✅ 已清理空行")

        # ========== 14. 优化格式 ==========
        print("\n📝 步骤 14/15: 优化格式...")
        # 标题前后空行
        content = re.sub(r'([^\n])\n(#{1,4} )', r'\1\n\n\2', content)
        content = re.sub(r'(#{1,4} [^\n]+)\n([^\n#])', r'\1\n\n\2', content)
        # 代码块前后空行
        content = re.sub(r'([^\n])\n(```)', r'\1\n\n\2', content)
        content = re.sub(r'(```)\n([^\n`])', r'\1\n\n\2', content)
        # 再次清理多余空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        print("   ✅ 已优化格式")

        # ========== 15. 内容质量验证 ==========
        stats = self.validate_content(content)

        # ========== 16. 最终整理 ==========
        print("\n📝 步骤 15/15: 最终整理...")
        content = content.lstrip('\n').rstrip('\n') + '\n'
        print("   ✅ 最终整理完成")

        # ========== 17. 保存文件 ==========
        print("\n📝 保存文件...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ 已保存至: {output_path}")

        # ========== 统计信息 ==========
        self.print_statistics(
            content, original_length, image_count,
            removed_headers, removed_fragments, fixed_paragraphs,
            stats, output_path
        )

        return content

    def print_statistics(self, content, original_length, image_count,
                         removed_headers, removed_fragments, fixed_paragraphs,
                         stats, output_path):
        """打印统计信息"""
        cleaned_length = len(content)

        print("\n" + "=" * 70)
        print("📊 清洗统计")
        print("=" * 70)
        print(f"原始字符数:    {original_length:,}")
        print(f"清洗后字符数:  {cleaned_length:,}")
        print(f"变化:          {cleaned_length - original_length:+,}")

        print(f"\n🧹 清理统计:")
        print(f"  移除页眉页脚:  {removed_headers} 处")
        print(f"  移除孤立碎片:  {removed_fragments} 处")
        print(f"  修复断裂段落:  {fixed_paragraphs} 处")

        # 章节统计
        chapters = stats['h1_count']
        level2 = stats['h2_count']
        level3 = stats['h3_count']
        level4 = stats['h4_count']
        code_blocks = len(re.findall(r'^```', content, re.MULTILINE)) // 2

        print(f"\n📚 结构统计:")
        print(f"  # 章节:        {chapters} 个")
        print(f"  ## 二级标题:   {level2} 个")
        print(f"  ### 三级标题:  {level3} 个")
        print(f"  #### 四级标题: {level4} 个")
        print(f"  代码块:        {code_blocks} 个")
        print(f"  图片:          {image_count} 个")

        print("\n✅ 清洗完成！")
        print("=" * 70)


def main():
    """主函数"""
    # 配置路径
    input_file = Path(r"D:\CodeField\Graduation_project\project\output\计算机网络.md")
    output_file = Path(r"D:\CodeField\Graduation_project\project\output\计算机网络_cleaned.md")

    # 检查输入文件
    if not input_file.exists():
        print(f"❌ 错误：找不到文件 {input_file}")
        exit(1)

    print(f"📂 输入: {input_file}")
    print(f"📂 输出: {output_file}")
    print(f"📂 大小: {input_file.stat().st_size:,} 字节\n")

    # 执行清洗
    try:
        cleaner = SmartMarkdownCleaner()
        cleaned_content = cleaner.clean(str(input_file), str(output_file))

        # 预览
        print("\n" + "=" * 70)
        print("📄 清洗后预览（前2000字符）")
        print("=" * 70)
        preview = cleaned_content[:2000]

        # 显示标题结构
        titles = re.findall(r'^(#{1,4} .+)$', preview, re.MULTILINE)
        if titles:
            print("\n📑 标题结构预览:")
            for title in titles[:20]:
                level = len(re.match(r'^#+', title).group())
                indent = '  ' * (level - 1)
                print(f"{indent}{title}")

        print("\n" + "=" * 70)

        print("\n🎉 全部完成！")
        print(f"\n💡 预期结果示例：")
        print(f"   # 第1章 操作系统引论        ← 一级")
        print(f"   ## 1.1 操作系统的目标        ← 二级")
        print(f"   ### 1.1.1 操作系统的目标     ← 三级")
        print(f"   #### 1.方便性                 ← 四级 ✅")
        print(f"   #### 2.有效性                 ← 四级 ✅")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
