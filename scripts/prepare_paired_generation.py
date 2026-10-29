"""Freeze development requests or inspect saved responses offline. Never sends HTTP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from rag_textbook_qa.evaluation.paired_preparation import (
    cost_scenario,
    prepare_plan,
    screening_request,
    validate_screening,
)
from rag_textbook_qa.json_utils import loads_strict


def read(path):
    return loads_strict(path.read_text(encoding="utf-8"))


def render(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--replay-report", type=Path, required=True)
    prepare.add_argument("--candidate-freeze", type=Path, required=True)
    prepare.add_argument("--evidence", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20261029)
    for name in ("screen-request", "validate-screening"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--ordinal", type=int, required=True)
        command.add_argument("--generation-response", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "validate-screening":
            command.add_argument("--screening-response", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.output_dir.exists():
                raise ValueError("输出目录已存在，请保留旧协议")
            freeze = read(args.candidate_freeze)
            if hashlib.sha256(args.replay_report.read_bytes()).hexdigest() != freeze["replay_report_sha256"]:
                raise ValueError("回放报告不属于冻结候选")
            for name, expected in freeze["source_sha256"].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"冻结实现改变：{name}")
            report = read(args.replay_report)
            if any(report[k] != freeze[k] for k in ("index_revision", "dataset_sha256")):
                raise ValueError("冻结数据集/索引身份不符")
            evidence = read(args.evidence)
            for item in evidence:
                if hashlib.sha256(Path(item["source_path"]).read_bytes()).hexdigest() != item["source_file_sha256"]:
                    raise ValueError("开发答案要点所据原文改变")
            plan = prepare_plan(report, evidence, seed=args.seed)
            material = [args.replay_report, args.candidate_freeze, args.evidence,
                        Path(__file__), Path("src/rag_textbook_qa/evaluation/paired_preparation.py")]
            plan["protocol"].update({
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "input_and_implementation_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in material},
            })
            files = {"plan.json": plan, "generation-cases.json": plan["cases"],
                     "generation-requests.json": plan["generation_requests"],
                     "screening-spec.json": plan["screening_spec"],
                     "protocol.json": plan["protocol"], "cost-scenario.json": cost_scenario(plan)}
            rendered = {name: render(value) for name, value in files.items()}
            manifest = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in rendered.items()}
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as directory:
                staging = Path(directory) / "prepared"
                staging.mkdir()
                for name, text in rendered.items():
                    (staging / name).write_text(text, encoding="utf-8")
                (staging / "manifest.json").write_text(render(manifest), encoding="utf-8")
                staging.rename(args.output_dir)
            print(render({"generation_requests": 30, "screening_max": 30, "model_calls": 0,
                          "execution_enabled": False, "cost_scenario": files["cost-scenario.json"]}))
        else:
            if args.output.exists():
                raise ValueError("输出已存在")
            plan = read(args.plan)
            jobs = [j for j in plan["generation_requests"] if j["ordinal"] == args.ordinal]
            if len(jobs) != 1:
                raise ValueError("生成样本序号无效")
            response = read(args.generation_response)
            result = (screening_request(plan, jobs[0], response) if args.command == "screen-request" else
                      validate_screening(plan, jobs[0], response, read(args.screening_response)))
            text = render(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
            print("Saved offline result; no model request was sent.")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
