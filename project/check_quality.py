# check_quality.py
"""
文件名: check_quality.py
主要功能:
    - 加载 JSON 格式的文本分块结果，快速评估分块质量
    - 检测常见问题：截断的代码块、残留行号、超大块、空内容块等
    - 统计问题发生率，给出 优秀 / 良好 / 较差 三级质量评级
    - 展示最多 3 个问题示例，辅助人工排查
所属模块: 数据预处理 / 质量检查
"""
import json
from pathlib import Path


def check_chunks_quality(json_path: str):
    """检查分块质量"""

    print("=" * 70)
    print("🔍 分块质量检查")
    print("=" * 70)

    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    total = len(chunks)

    # 检查各种问题
    issues = {
        'code_truncated': [],  # 代码被截断
        'has_line_numbers': [],  # 有行号
        'too_small': [],  # 过小
        'too_large': [],  # 过大
        'empty_code': [],  # 空代码块
    }

    for chunk in chunks:
        cid = chunk['chunk_id']
        content = chunk['content']
        char_count = chunk['char_count']

        # 检查代码截断（代码块以}或)开头）
        if chunk['has_code']:
            if content.strip().startswith('```\n\n}') or \
                    content.strip().startswith('```\n\n)'):
                issues['code_truncated'].append(cid)

        # 检查行号问题
        import re
        if re.search(r'^\s*\d+\s+[a-zA-Z_]\w*\s*\(', content, re.MULTILINE):  # "13 if(" 行首代码行号
            issues['has_line_numbers'].append(cid)

        # 检查大小
        if char_count < 100:
            issues['too_small'].append(cid)
        elif char_count > 2000:
            issues['too_large'].append(cid)

        # 检查空代码块
        if chunk['has_code']:
            code_content = re.search(r'```.*?```', content, re.DOTALL)
            if code_content and len(code_content.group(0)) < 50:
                issues['empty_code'].append(cid)

    # 打印结果
    print(f"\n总块数: {total}\n")

    print("问题统计:")
    print("-" * 70)

    for issue_type, chunk_ids in issues.items():
        count = len(chunk_ids)
        percentage = (count / total * 100) if total > 0 else 0

        status = "✅" if percentage < 5 else "⚠️" if percentage < 15 else "❌"

        print(f"{status} {issue_type:20s}: {count:4d} 个 ({percentage:5.1f}%)")

        # 显示前3个有问题的
        if chunk_ids and count <= 3:
            for cid in chunk_ids:
                print(f"      - {cid}")

    # 总体评估
    print("\n" + "=" * 70)
    print("📋 总体评估:")
    print("-" * 70)

    total_issues = sum(len(ids) for ids in issues.values())
    issue_rate = (total_issues / total * 100) if total > 0 else 0

    if issue_rate < 5:
        print("✅ 质量优秀！问题率 < 5%，可以直接使用")
        return "excellent"
    elif issue_rate < 15:
        print("⚠️  质量良好。问题率 < 15%，可以使用，建议关注特定问题块")
        return "good"
    else:
        print("❌ 质量较差。问题率 >= 15%，建议修复后再使用")
        return "poor"


if __name__ == "__main__":
    # 修改这里指定要检查的 chunks JSON 文件
    json_path = "./output/计算机组成原理_chunks.json"

    if not Path(json_path).exists():
        print(f"❌ 文件不存在: {json_path}")
        exit(1)

    result = check_chunks_quality(json_path)

    print("\n💡 建议:")
    if result == "excellent":
        print("   → 直接使用，无需修复")
    elif result == "good":
        print("   → 可以使用，后续再优化少数问题块")
        print("   → 对于 RAG 应用，这个质量已经足够")
    else:
        print("   → 建议修复代码块问题后再使用")
