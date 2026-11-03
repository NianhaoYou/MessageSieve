from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from .base import DatasetConfig, TaskSample


def load_task_samples(config: DatasetConfig) -> List[TaskSample]:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    records = _read_records(path)
    split = (config.split or "").strip()
    if split:
        filtered_records = []
        for record in records:
            record_split = str(record.get("split", "") or "")
            if record_split == split:
                filtered_records.append(record)
        records = filtered_records

    start_index = max(int(config.start_index or 0), 0)
    end_index = None if config.limit is None else start_index + max(int(config.limit), 0)
    sliced = records[start_index:end_index]
    return [build_task_sample(config.dataset_id, index=start_index + offset, record=record) for offset, record in enumerate(sliced)]


def build_task_sample(dataset_id: str, *, index: int, record: Mapping[str, Any]) -> TaskSample:
    task_id = str(
        record.get("task_id")
        or record.get("id")
        or record.get("question_id")
        or f"{dataset_id}_{index}"
    )
    target = record.get("target")
    if target is None:
        target = record.get("answer")
    if target is None:
        target = record.get("canonical_solution")
    if target is None:
        target = record.get("solution")
    prompt = _build_prompt(dataset_id, record)
    metadata = dict(record)
    return TaskSample(
        task_id=task_id,
        dataset_id=dataset_id,
        prompt=prompt,
        target=target,
        metadata=metadata,
    )


def _read_records(path: Path) -> List[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "data", "examples", "records"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate
        raise ValueError(f"Unsupported JSON dataset payload structure: {path}")
    raise ValueError(f"Unsupported dataset file format: {path.suffix}")


def _build_prompt(dataset_id: str, record: Mapping[str, Any]) -> str:
    normalized = dataset_id.strip().lower()
    if normalized == "mbppplus":
        return _build_mbppplus_prompt(record)

    if normalized == "livecodebench_codegen_lite":
        prompt = record.get("prompt")
        if prompt:
            return str(prompt)
        title = str(record.get("question_title") or "").strip()
        content = str(record.get("question_content") or "").strip()
        starter_code = str(record.get("starter_code") or "").strip()
        lines = []
        if title:
            lines.append(f"Title: {title}")
        if content:
            lines.append(content)
        if starter_code:
            lines.append("Starter code:\n```python\n" + starter_code + "\n```")
        return "\n\n".join(lines).strip()

    if normalized in {"gpqa_diamond", "aqua"}:
        if record.get("prompt"):
            return str(record["prompt"])
        question = str(record.get("question", "")).strip()
        raw_choices = record.get("choices") or record.get("options") or []
        if isinstance(raw_choices, Mapping):
            labels = ["A", "B", "C", "D", "E", "F"]
            choices = [raw_choices[label] for label in labels if label in raw_choices]
        else:
            choices = list(raw_choices)
        lines = [question]
        labels = ["A", "B", "C", "D", "E", "F"]
        for idx, choice in enumerate(choices):
            label = labels[idx] if idx < len(labels) else f"Option{idx + 1}"
            lines.append(f"Option {label}: {choice}")
        return "\n".join(lines).strip()

    if normalized == "math":
        if record.get("prompt"):
            return str(record["prompt"])
        question = record.get("problem") or record.get("question") or record.get("text") or ""
        return str(question).strip()

    raise ValueError(f"Unsupported dataset: {dataset_id}")


def _build_mbppplus_prompt(record: Mapping[str, Any]) -> str:
    task_text = str(record.get("prompt") or record.get("text") or record.get("question") or "").strip()
    lines: list[str] = []
    lines.extend(
        [
            ">>> Specification Priority:",
            "Follow the visible tests exactly. If the task wording and the visible tests conflict, the visible tests win.",
        ]
    )
    if task_text:
        lines.extend(["", ">>> Problem:", task_text])

    public_tests = _load_mbppplus_public_tests(record.get("test_list"))
    if public_tests:
        lines.extend(["", ">>> Test Cases:", "```python"])
        lines.extend(public_tests)
        lines.append("```")
        observations = _infer_mbppplus_behavior_observations(public_tests)
        if observations:
            lines.extend(["", ">>> Visible Test Observations:"])
            lines.extend(f"- {item}" for item in observations)
        lines.extend(
            [
                "",
                ">>> Behavior Contract:",
                "Follow the visible problem statement and the visible test cases exactly.",
                "If the task wording is broader or ambiguous, do not expand beyond what the visible tests demonstrate.",
            ]
        )

    return "\n".join(lines).strip()


def _load_mbppplus_public_tests(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return [line.strip() for line in value.splitlines() if line.strip().startswith("assert ")]
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    return []


def _infer_mbppplus_behavior_observations(tests: Sequence[str]) -> list[str]:
    has_int_literals = False
    has_float_literals = False
    string_literals: list[str] = []
    for test in tests:
        try:
            expr = ast.parse(str(test)).body[0].test
        except Exception:
            continue
        for node in ast.walk(expr):
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    has_int_literals = True
                elif isinstance(value, float):
                    has_float_literals = True
                elif isinstance(value, str):
                    string_literals.append(value)

    observations: list[str] = []
    if has_int_literals and not has_float_literals:
        observations.append("The visible tests only show integer literals, so use integers only in the implementation instead of adding float support.")

    has_tab_or_newline = any("\t" in item or "\n" in item or "\r" in item for item in string_literals)
    has_space = any(" " in item for item in string_literals)
    if has_space and not has_tab_or_newline:
        observations.append("The visible tests show space characters but do not show tabs or newlines, so remove literal spaces only instead of all whitespace characters.")

    return observations
