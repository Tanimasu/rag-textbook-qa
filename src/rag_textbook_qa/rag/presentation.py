"""Terminal rendering for retrieved chunks, kept out of the retrieval engine."""

from __future__ import annotations

from typing import Any

_RULE = "─" * 70


def render_search_results(results: list[dict[str, Any]]) -> str:
    """Render retrieved chunks for a terminal.

    Returns text instead of printing it: the engine owns retrieval, not screen
    layout, and a pure function can be asserted on without capturing stdout.
    """

    if not results:
        return "没有找到相关内容"

    lines = [f"找到 {len(results)} 条相关内容：", ""]
    for index, result in enumerate(results, 1):
        content = result["content"]
        lines.extend(
            [
                _RULE,
                f"【结果 {index}】",
                f"相似度: {result['similarity']:.4f} | 方法: {result['method']}",
                f"教材: {result['book_name']}",
                f"章节: {result['chapter']} | {result['section_h2']}",
                f"内容: {content[:150]}{'...' if len(content) > 150 else ''}",
            ]
        )
        extra = []
        if result.get("has_code"):
            extra.append("含代码")
        if result.get("has_image"):
            extra.append("含图片")
        if extra:
            lines.append(f"标签: {' | '.join(extra)}")
    lines.append(_RULE)
    return "\n".join(lines)
