"""Prepare/validate local claim review; never generate answers or semantic labels."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from rag_textbook_qa.evaluation.claim_review import prepare_packet, review_template, validate_review
from rag_textbook_qa.json_utils import loads_strict


def render(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def packet_markdown(packet):
    def literal(value):
        text = value if isinstance(value, str) else "[缺失，未补造]"
        fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
        return fence + "\n" + text + "\n" + fence + "\n"

    sections = ["# 已存回答逐主张复核材料\n\n所有语义标签待填；引用标记存在不等于支持陈述。\n"]
    for row in packet["records"]:
        sections.extend([
            f"## {row['id']}\n\n问题：{row['question']}\n\n",
            f"结束状态：{row['recorded_finish_reason']}；完整回答：{row['eligible_complete_answer']}。\n\n",
            "### 回答正文（start/end 偏移以此为准）\n\n", literal(row["answer_body"]),
            "### 原始引用标记及偏移\n\n", literal(render(row["citation_markers"])),
            "### 实际冻结上下文\n\n", literal(row["context"]),
        ])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--batch-dir", type=Path, action="append", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--run-label", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--packet", type=Path, required=True)
    validate.add_argument("--review", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.output_dir.exists():
                raise ValueError("输出目录已存在，请保留旧材料并使用新目录")
            packets = [prepare_packet(p, run_label=args.run_label) for p in args.batch_dir]
            packet = {**packets[0], "records": [r for p in packets for r in p["records"]],
                      "input_sha256": {k: v for p in packets for k, v in p["input_sha256"].items()},
                      "unmatched_generator_responses": [r for p in packets for r in p["unmatched_generator_responses"]]}
            if len({r["id"] for r in packet["records"]}) != len(packet["records"]):
                raise ValueError("批次/样本身份重复")
            template = review_template(packet)
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
                staging = Path(temporary) / "packet"
                staging.mkdir()
                (staging / "packet.json").write_text(render(packet), encoding="utf-8")
                (staging / "review-template.json").write_text(render(template), encoding="utf-8")
                (staging / "review.md").write_text(packet_markdown(packet), encoding="utf-8")
                staging.rename(args.output_dir)
            print(f"Prepared {len(packet['records'])} saved/planned answers; all semantic labels pending.")
        else:
            if args.output.exists():
                raise ValueError("输出文件已存在")
            packet = loads_strict(args.packet.read_text(encoding="utf-8"))
            reviewed = loads_strict(args.review.read_text(encoding="utf-8"))
            result = validate_review(packet, reviewed)
            content = render(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(render({k: v for k, v in result.items() if k not in {"claims", "per_answer"}}))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
