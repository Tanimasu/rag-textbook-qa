"""
check_parsing_quality.py
检查 PDF 解析后 Markdown 文件的质量：统计字符数、标题数、图片占位符等，
并标记常见问题。通常在 parsingPDF.py 和 clean_markdown.py 之后运行。
运行方式：python check_parsing_quality.py
"""
from pathlib import Path


def check_markdown_quality(md_path: str) -> dict:
    """
    分析 Markdown 文件的结构质量，打印报告，返回统计字典。

    Args:
        md_path: Markdown 文件路径
    """
    content = Path(md_path).read_text(encoding="utf-8")

    stats = {
        "总字符数":       len(content),
        "总行数":         len(content.split("\n")),
        "二级标题（##）": content.count("\n## "),
        "三级标题（###）":content.count("\n### "),
        "图片占位符":     content.count("<!-- image -->"),
        "代码块":         content.count("```"),
        "空行数":         content.count("\n\n"),
    }

    print("=" * 50)
    print(f"Markdown 质量报告: {Path(md_path).name}")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    issues = []
    if stats["图片占位符"] > 0:
        issues.append(f"有 {stats['图片占位符']} 个图片占位符（<!-- image -->），需人工处理或过滤")
    if stats["总字符数"] < 10000:
        issues.append("总字符数不足 10000，文档可能解析不完整")

    print()
    if issues:
        print("发现问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("未发现明显问题，文档质量良好。")

    print("=" * 50)
    return stats


if __name__ == "__main__":
    # 修改这里指定要检查的 Markdown 文件
    MD_PATH = "./output/操作系统_cleaned.md"

    if not Path(MD_PATH).exists():
        print(f"文件不存在: {MD_PATH}")
    else:
        check_markdown_quality(MD_PATH)
