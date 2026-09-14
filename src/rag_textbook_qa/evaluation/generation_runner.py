"""Run a frozen-context generation experiment: sample, judge, resume, report.

Each finished generation and judgment is appended to JSONL the moment it exists,
so a dropped connection costs only the requests in flight and rerunning the same
command resumes. The first run freezes the protocol; a rerun with other settings
is refused, so one output directory can never mix two experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook_qa.evaluation.generation import (
    EXTRACTION_PROMPT,
    VERIFICATION_PROMPT,
    Arm,
    GenerationCase,
    context_blocks,
    extraction_prompt,
    parse_json_object,
    score_answer,
    summarize,
    validate_extraction,
    validate_verification,
    verification_prompt,
)

Generator = Callable[[str, float], Mapping[str, Any]]
Judge = Callable[[str], str]
Key = tuple[str, str, int]

JUDGE_ATTEMPTS = 3
# Bump whenever what the judge sees or how its output is scored changes, so a
# directory judged under older rules refuses to resume under newer ones.
JUDGE_VERSION = 3
TRANSIENT_ERRORS = frozenset(
    {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}
)


def with_retries(
    call: Callable[[], Any],
    *,
    attempts: int = 4,
    delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    """Retry transient provider failures only; anything else is a real error."""

    attempt = 1
    while True:
        try:
            return call(), attempt
        except Exception as exc:
            transient = any(cls.__name__ in TRANSIENT_ERRORS for cls in type(exc).__mro__)
            if not transient or attempt >= attempts:
                raise
        sleep(delay * attempt)
        attempt += 1


def generation_request(
    model: str, prompt: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """The request LLMClient.generate_answer sends, so the experiment measures the product."""

    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


def openai_generator(
    sdk_client: Any, model: str, *, max_tokens: int, timeout: float = 180.0
) -> Generator:
    client = sdk_client.with_options(timeout=timeout, max_retries=0)

    def generate(prompt: str, temperature: float) -> dict[str, Any]:
        started = time.monotonic()
        response, attempts = with_retries(
            lambda: client.chat.completions.create(
                **generation_request(model, prompt, temperature, max_tokens)
            )
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return {
            "answer": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            # Only the length is kept; reasoning text is never stored.
            "reasoning_chars": len(getattr(choice.message, "reasoning_content", None) or ""),
            "tokens": {
                "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion": int(getattr(usage, "completion_tokens", 0) or 0),
            },
            "seconds": round(time.monotonic() - started, 3),
            "attempts": attempts,
        }

    return generate


def openai_judge(
    sdk_client: Any,
    model: str,
    *,
    extra: Mapping[str, Any],
    max_tokens: int = 4096,
    timeout: float = 180.0,
) -> Judge:
    client = sdk_client.with_options(timeout=timeout, max_retries=0)

    def judge(prompt: str) -> str:
        response, _ = with_retries(
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                stream=False,
                **extra,
            )
        )
        return response.choices[0].message.content or ""

    return judge


def llm_pair_from_env() -> tuple[Any, Any, dict[str, Any]]:
    """Generator resolved like the engine (RAG_*), judge like RAGAS (RAGAS_*), both over LLM_*.

    A model judging its own answers reintroduces self-preference, so a matching pair
    is refused outright.
    """

    from rag_textbook_qa.evaluation.ragas import judge_model_kwargs
    from rag_textbook_qa.llm.client import create_llm_client

    generator = create_llm_client(
        api_key=os.getenv("RAG_API_KEY") or None,
        base_url=os.getenv("RAG_API_BASE") or None,
        model=os.getenv("RAG_MODEL") or None,
        verbose=False,
    )
    judge = create_llm_client(
        api_key=os.getenv("RAGAS_API_KEY") or None,
        base_url=os.getenv("RAGAS_API_BASE") or None,
        model=os.getenv("RAGAS_MODEL") or None,
        verbose=False,
    )
    if judge.default_model == generator.default_model:
        raise ValueError("评判模型与生成模型相同，会带来自我偏好偏差；请设置 RAGAS_MODEL")
    return generator, judge, judge_model_kwargs()


def _key(record: Mapping[str, Any]) -> Key:
    return record["case_id"], record["arm"], record["index"]


class JsonlLog:
    """Append-only JSONL keyed by (case, arm, sample index), shared by worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self.lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash is simply redone
        self.records = {_key(line): line for line in self.lines}

    def append(self, record: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.lines.append(record)
            self.records[_key(record)] = record


def sample_keys(
    cases: Sequence[GenerationCase], arms: Sequence[Arm], samples: int, seed: int
) -> list[Key]:
    """Shuffle every planned sample together so provider drift touches all arms alike."""

    if samples < 1:
        raise ValueError("每个方案至少采样 1 次")
    keys: list[Key] = []
    for case in cases:
        for arm in arms:
            if arm.variant not in case.variants:
                raise ValueError(f"题目 {case.case_id} 没有上下文 {arm.variant}")
            stored = case.variants[arm.variant].answers
            if arm.temperature is None and not stored:
                raise ValueError(f"题目 {case.case_id} 的 {arm.variant} 没有保存的回答")
            count = len(stored) if arm.temperature is None else samples
            keys.extend((case.case_id, arm.name, index) for index in range(count))
    random.Random(seed).shuffle(keys)
    return keys


def freeze_protocol(path: Path, settings: Mapping[str, Any]) -> None:
    normalized = json.loads(json.dumps(settings, ensure_ascii=False))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["settings"] != normalized:
            raise ValueError("该输出目录的实验协议已冻结；参数不同请换一个输出目录")
        return
    frozen = {"frozen_at_utc": datetime.now(UTC).isoformat(), "settings": normalized}
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ask(judge: Judge, prompt: str, validate: Callable[[dict[str, Any]], Any]) -> Any:
    """Re-ask on malformed output, naming the error; persistent failure fails one sample.

    At temperature 0 the identical prompt tends to return the identical broken JSON,
    so every retry says what could not be parsed or validated.
    """

    error: Exception | None = None
    for attempt in range(JUDGE_ATTEMPTS):
        request = prompt if attempt == 0 else (
            f"{prompt}\n\n注意：上一次输出不是合法且完整的 JSON（{error}）。"
            "请只输出合法 JSON，字符串里的英文双引号必须转义。"
        )
        try:
            return validate(parse_json_object(judge(request)))
        except (TypeError, ValueError) as exc:
            error = exc
    raise ValueError(f"judge_output_invalid:{error}")


def extract_claims(
    case: GenerationCase, answer: str, judge: Judge
) -> tuple[list[dict[str, Any]], list[bool]]:
    """Claims and requirement coverage from the answer alone; the evidence is never shown."""

    return _ask(
        judge,
        extraction_prompt(case.question, answer, case.requirements),
        lambda payload: validate_extraction(payload, len(case.requirements)),
    )


def verify_claims(
    case: GenerationCase,
    arm: Arm,
    claims: Sequence[Mapping[str, Any]],
    coverage: Sequence[bool],
    judge: Judge,
) -> dict[str, Any]:
    """Check fact claims against each citation's whole block, headings included."""

    variant = case.variants[arm.variant]
    blocks = context_blocks(variant.context, variant.sources)
    facts = [claim for claim in claims if claim["type"] == "fact"]
    verdicts = (
        _ask(
            judge,
            verification_prompt(blocks, facts),
            lambda payload: validate_verification(payload, [claim["id"] for claim in facts]),
        )
        if facts
        else {}
    )
    return score_answer(claims, coverage, verdicts, blocks)


def judge_answer(case: GenerationCase, arm: Arm, answer: str, judge: Judge) -> dict[str, Any]:
    """Extract, then verify; two stages so a blind audit can sit between them."""

    started = time.monotonic()
    claims, coverage = extract_claims(case, answer, judge)
    return {
        "score": verify_claims(case, arm, claims, coverage, judge),
        "judge_seconds": round(time.monotonic() - started, 3),
    }


def _run_stage(
    stage: str,
    pending: Sequence[Key],
    work: Callable[[Key], None],
    *,
    failures: JsonlLog,
    concurrency: int,
    log: Callable[[str], None],
) -> None:
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, key): key for key in pending}
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                future.result()
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - one failed sample must not end the run
                message = str(exc)
                failures.append(
                    {
                        "case_id": key[0],
                        "arm": key[1],
                        "index": key[2],
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        # Provider messages may echo request details; keep only our codes.
                        "detail": message if message.startswith("judge_output_invalid") else "",
                        "at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                status = type(exc).__name__
            log(f"[{stage}] {done}/{len(pending)} {key[0]} {key[1]} #{key[2]} {status}")


def _usage(records: Sequence[Mapping[str, Any]], arms: Sequence[Arm]) -> dict[str, Any]:
    usage = {}
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm.name and arm.temperature is not None]
        usage[arm.name] = {
            "prompt_tokens": sum(r["tokens"]["prompt"] for r in rows),
            "completion_tokens": sum(r["tokens"]["completion"] for r in rows),
            "mean_seconds": statistics.fmean(r["seconds"] for r in rows) if rows else None,
            "mean_reasoning_chars": (
                statistics.fmean(r["reasoning_chars"] for r in rows) if rows else None
            ),
            "retried": sum(r["attempts"] > 1 for r in rows),
        }
    return usage


def run_generation_experiment(
    cases: Sequence[GenerationCase],
    arms: Sequence[Arm],
    *,
    output_dir: Path,
    generator: Generator,
    judge: Judge,
    samples: int,
    seed: int,
    concurrency: int,
    protocol: Mapping[str, Any],
    prompt_builder: Callable[[str, str], str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("方案名称重复")
    if concurrency < 1:
        raise ValueError("并发数至少为 1")
    if prompt_builder is None:
        # The packing module is light; importing the engine would pull in Chroma.
        from rag_textbook_qa.rag.context import build_prompt

        prompt_builder = build_prompt
    keys = sample_keys(cases, arms, samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_prompts = (EXTRACTION_PROMPT + VERIFICATION_PROMPT).encode("utf-8")
    settings = {
        **protocol,
        "arms": [asdict(arm) for arm in arms],
        "samples": samples,
        "seed": seed,
        "judge_prompts_sha256": hashlib.sha256(judge_prompts).hexdigest(),
        "judge_version": JUDGE_VERSION,
    }
    freeze_protocol(output_dir / "protocol.json", settings)
    generations = JsonlLog(output_dir / "generations.jsonl")
    judgments = JsonlLog(output_dir / "judgments.jsonl")
    failures = JsonlLog(output_dir / "failures.jsonl")
    cases_by_id = {case.case_id: case for case in cases}
    arms_by_name = {arm.name: arm for arm in arms}

    def generate(key: Key) -> None:
        case, arm = cases_by_id[key[0]], arms_by_name[key[1]]
        variant = case.variants[arm.variant]
        record: dict[str, Any] = {"case_id": key[0], "arm": key[1], "index": key[2]}
        if arm.temperature is None:
            record.update(answer=variant.answers[key[2]], finish_reason="stored")
        else:
            prompt = prompt_builder(case.question, variant.context)
            record.update(generator(prompt, arm.temperature))
        generations.append(record)

    def grade(key: Key) -> None:
        case, arm = cases_by_id[key[0]], arms_by_name[key[1]]
        answer = generations.records[key]["answer"]
        judgments.append(
            {"case_id": key[0], "arm": key[1], "index": key[2],
             **judge_answer(case, arm, answer, judge)}
        )

    options = {"failures": failures, "concurrency": concurrency, "log": log}
    _run_stage("generate", [k for k in keys if k not in generations.records], generate, **options)
    ungraded = [k for k in keys if k in generations.records and k not in judgments.records]
    _run_stage("judge", ungraded, grade, **options)

    records = [
        {**generations.records[k], "score": judgments.records.get(k, {}).get("score")}
        for k in keys
        if k in generations.records
    ]
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "settings": settings,
        "planned": len(keys),
        "generated": len(records),
        "judged": sum(record["score"] is not None for record in records),
        "failures_logged": len(failures.lines),
        "usage": _usage(records, arms),
        **summarize(records, cases, arms, seed=seed),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
