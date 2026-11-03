"""Convert user-selected benchmark records to evaluation JSONL in input order."""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping


FORMATS = {".csv", ".json", ".jsonl", ".parquet"}


@dataclass(frozen=True)
class SourceRow:
    """A payload and its original zero-based row/line and file provenance."""

    record: dict[str, Any]
    index: int
    path: str


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("items", "data", "examples", "records", "test", "train"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("Expected JSON records to be objects or an array of objects")
    return payload


def _read_file(path: Path) -> list[SourceRow]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            indexed = list(enumerate(csv.DictReader(handle)))
    elif suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8-sig")
        if text.startswith("version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Source is a Git LFS pointer, not data: {path}")
        if suffix == ".json":
            try:
                indexed = list(enumerate(_payload_records(json.loads(text))))
            except json.JSONDecodeError:
                # Official AQuA calls its newline-delimited files *.json.
                indexed = [(i, json.loads(line)) for i, line in enumerate(text.split("\n")) if line.strip()]
        else:
            # Only LF delimits JSONL. Unicode separators inside JSON strings
            # (present in GPQA's raw annotations) are ordinary string content.
            indexed = [(i, json.loads(line)) for i, line in enumerate(text.split("\n")) if line.strip()]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Install pyarrow to read parquet sources") from exc
        indexed = list(enumerate(pq.read_table(path).to_pylist()))
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}")
    if not all(isinstance(row, dict) for _, row in indexed):
        raise ValueError(f"Expected object records: {path}")
    source_path = str(path.resolve())
    return [SourceRow(dict(row), i, source_path) for i, row in indexed]


def read_source(dataset: str, path: Path) -> list[SourceRow]:
    """Read a file or one unambiguous upstream split from a directory.

    Known filenames take precedence over other files. Parquet shards are read
    in filename order with continuous split indices. Arbitrary directories
    containing competing exports are rejected rather than silently mixed.
    """
    def read_file(file: Path) -> list[SourceRow]:
        rows = _read_file(file)
        return [SourceRow(row.record, i, row.path) for i, row in enumerate(rows)]

    if not path.is_dir():
        return read_file(path)
    stems = {"gpqa_diamond": "gpqa_diamond", "livecodebench": "test6"}
    stem = stems.get(dataset, "test")
    for suffix in (".csv", ".jsonl", ".json", ".parquet"):
        matches = sorted(path.rglob(stem + suffix))
        if len(matches) > 1:
            raise ValueError(f"Multiple {stem + suffix} sources; pass the intended file directly")
        if matches:
            return read_file(matches[0])
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in FORMATS)
    shards = [p for p in files if p.suffix == ".parquet" and
              (p.name.startswith(stem + "-") or p.parent.name == stem)]
    if shards:
        rows: list[SourceRow] = []
        for shard in shards:
            offset = len(rows)
            rows.extend(SourceRow(r.record, offset + r.index, r.path) for r in read_file(shard))
        return rows
    if len(files) == 1:
        return read_file(files[0])
    if dataset == "math500" and files and all(p.suffix == ".json" for p in files):
        # Read subject files in filename order and retain their upstream IDs.
        rows = []
        for file in files:
            offset = len(rows)
            for row in _read_file(file):
                payload = dict(row.record)
                parts = file.parts
                if "test" in parts:
                    payload.setdefault("unique_id", "/".join(parts[parts.index("test"):]))
                rows.append(SourceRow(payload, offset + row.index, row.path))
        return rows
    raise ValueError(f"No unambiguous {stem} split in {path}; pass a file or a split/shard directory")


def read_huggingface(name: str, *, config: str | None = None, split: str = "test",
                     revision: str | None = None) -> list[SourceRow]:
    """Load records from a Hugging Face dataset split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to use --download with Hugging Face") from exc
    options = {"split": split}
    if revision:
        options["revision"] = revision
    records = load_dataset(name, config, **options)
    location = f"hf://datasets/{name}@{revision or 'main'}/{split}"
    return [SourceRow(dict(row), i, location) for i, row in enumerate(records)]


def _text(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def task_id(record: Mapping[str, Any]) -> str:
    """Resolve IDs without discarding a numeric zero."""
    for key in ("task_id", "id", "question_id", "Record ID", "unique_id"):
        if record.get(key) is not None and str(record[key]).strip():
            return str(record[key])
    return ""


def _structured(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Expected a JSON/list literal in a structured source field") from exc
    return value


def _index(value: Any, field: str) -> int:
    try:
        index = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if index < 0 or str(value) != str(index):
        raise ValueError(f"Invalid {field}: {value!r}")
    return index


def _gpqa(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    question = _text(record, "Question", "question", "prompt", "text")
    correct = _text(record, "Correct Answer", "correct_answer", "correct")
    options = [correct] + [_text(record, f"Incorrect Answer {i}", f"incorrect_answer_{i}") for i in range(1, 4)]
    if not question or not all(options):
        raise ValueError("Raw GPQA requires Question, Correct Answer and three Incorrect Answer fields")
    random.Random(42 + index).shuffle(options)
    choices = dict(zip("ABCD", options))
    answer = next(label for label, value in choices.items() if value == correct)
    identifier = task_id(record) or f"gpqa_{index}"
    return {"id": identifier, "task_id": identifier,
            "subject": _text(record, "High-level domain", "Subdomain", "subject"),
            "question": question, "choices": choices, "answer": answer, "split": "test",
            "prompt": "\n".join([question] + [f"Option {k}: {v}" for k, v in choices.items()]).strip(),
            "raw_row_index": index, "shuffle_seed": 42 + index}


def _option(value: Any, fallback: str) -> tuple[str, str]:
    text = str(value).strip()
    match = re.match(r"^\s*([A-Z])\s*[\)\].:\-]\s*(.+?)\s*$", text, flags=re.IGNORECASE)
    return (match.group(1).upper(), match.group(2).strip()) if match else (fallback, text)


def _aqua(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    question = _text(record, "question", "Question")
    raw = _structured(record.get("options", record.get("choices")), [])
    choices: dict[str, str] = {}
    if isinstance(raw, dict):
        labels, texts = raw.get("label", raw.get("labels")), raw.get("text", raw.get("texts"))
        pairs = zip(labels, texts) if isinstance(labels, list) and isinstance(texts, list) else raw.items()
    elif isinstance(raw, list):
        pairs = zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", raw)
    else:
        raise ValueError("AQuA options must be an array or choice mapping")
    for label, value in pairs:
        label, text = _option(value, str(label).strip().upper())
        choices[label] = text
    choices = {label: choices[label] for label in sorted(choices) if choices[label]}
    raw_answer = next((record[k] for k in ("correct", "answer", "target", "answerKey", "answer_key", "label")
                       if record.get(k) is not None), "")
    answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[raw_answer] if isinstance(raw_answer, int) and 0 <= raw_answer < 26 else str(raw_answer).strip().upper()
    if answer not in choices:
        match = re.search(r"\b([A-Z])\b", answer)
        answer = match.group(1) if match and match.group(1) in choices else next(
            (label for label, text in choices.items() if str(raw_answer).strip().lower() == text.lower()), "")
    if not question or len(choices) != 5 or answer not in choices:
        raise ValueError("AQuA requires a question, five choices and a valid correct option")
    source_id = _text(record, "id", "question_id") or f"test_{index}"
    identifier = f"aqua_{source_id}"
    prompt = "\n".join([f"Question: {question}", "", "Answer choices:"] +
                       [f"Option {k}: {v}" for k, v in choices.items()] +
                       ["", "Choose the single best option letter."])
    return {"id": identifier, "task_id": identifier, "dataset_id": "aqua", "split": "test",
            "raw_split": "test", "question": question, "choices": choices, "answer": answer,
            "target": answer, "rationale": _text(record, "rationale", "solution", "explanation"),
            "prompt": prompt.strip(), "raw_row_index": index}


def _math(record: Mapping[str, Any], index: int, path: str) -> dict[str, Any]:
    problem, solution = _text(record, "problem", "prompt"), _text(record, "solution")
    if not problem or record.get("answer") is None:
        raise ValueError("MATH500 requires problem and answer fields; solution text is not an answer substitute")
    answer = str(record["answer"]).strip()
    subject = _text(record, "subject", "type")
    level = _text(record, "level")
    if level.isdigit():
        level = f"Level {level}"
    identifier = task_id(record) or f"test-{index + 1}"
    return {"id": identifier, "task_id": identifier, "subject": subject, "level": level,
            "problem": problem, "solution": solution, "answer": answer, "target": answer,
            "raw_split": "test", "split": "test", "prompt": problem, "source_path": path,
            "unique_id": _text(record, "unique_id"), "source_row_index": index,
            "selection_level": level, "selection_subject": subject}


def _mbpp(record: Mapping[str, Any]) -> dict[str, Any]:
    identifier = task_id(record)
    prompt, code, test = _text(record, "prompt", "text"), _text(record, "code"), _text(record, "test")
    if not identifier or not prompt or not code or not test:
        raise ValueError("Raw MBPP+ requires task_id, prompt, code and executable test (plain MBPP is insufficient)")
    source_file = _text(record, "source_file")
    result = {"id": identifier, "task_id": identifier, "dataset_id": "mbppplus", "split": "test",
              "prompt": prompt, "target": code, "answer": code, "canonical_solution": code,
              "test_list": _structured(record.get("test_list"), []),
              "test_setup_code": _text(record, "test_setup_code"),
              "challenge_test_list": _structured(record.get("challenge_test_list"), []),
              "source_file": source_file, "test_imports": _structured(record.get("test_imports"), []),
              "test": test, "metadata": {"source_dataset": "evalplus/mbppplus", "source_file": source_file}}
    for key in ("test_list", "challenge_test_list", "test_imports"):
        if not isinstance(result[key], list):
            raise ValueError(f"MBPP+ {key} must be a list")
    return result


def _lcb(record: Mapping[str, Any], index: int, source_file: str) -> dict[str, Any]:
    identifier = _text(record, "question_id")
    title, content, starter = (_text(record, k) for k in ("question_title", "question_content", "starter_code"))
    if not identifier or not content or not record.get("public_test_cases") or not record.get("private_test_cases"):
        raise ValueError("Raw LCB requires question_id, question_content and public/private_test_cases")
    def serialized(key: str) -> str:
        value = record.get(key)
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "").strip()
    metadata = serialized("metadata")
    try:
        parsed = json.loads(metadata or "{}")
    except json.JSONDecodeError:
        parsed = {}
    func = _text(parsed, "func_name", "function_name", "entry_point") if isinstance(parsed, dict) else ""
    mode = "functional" if starter or func else "stdin"
    parts = ([f"Title: {title}"] if title else []) + [content]
    if starter:
        parts.append("Starter code:\n```python\n" + starter + "\n```")
    if mode == "functional":
        note = ("This is a functional-input task. Complete the required function/class implementation; "
                "preserve the starter signature exactly and do not read from standard input or print to standard output.")
        if func:
            note += f" The required entry function is `{func}`."
    else:
        note = ("This is a stdin-stdout task. Write a complete Python program that reads from standard input "
                "and writes the answer to standard output.")
    parts.append(note)
    return {"id": identifier, "task_id": f"lcb_{identifier}", "dataset_id": "livecodebench_codegen_lite",
            "question_title": title, "question_content": content,
            "platform": str(record.get("platform") or ""), "question_id": identifier,
            "contest_id": str(record.get("contest_id") or ""), "contest_date": str(record.get("contest_date") or ""),
            "starter_code": starter, "difficulty": str(record.get("difficulty") or ""),
            "public_test_cases": serialized("public_test_cases"), "private_test_cases": serialized("private_test_cases"),
            "metadata": metadata, "io_mode": mode, "func_name": func, "answer": "", "target": "",
            "prompt": "\n\n".join(parts).strip(), "source_file": source_file,
            "raw_row_index": index, "split": "test"}


def normalize(dataset: str, rows: list[SourceRow], *, source_format: str = "auto") -> list[dict[str, Any]]:
    """Normalize records without selecting or reordering samples."""
    index_key = "source_row_index" if dataset == "math500" else "raw_row_index"
    result = []
    for row in rows:
        record = row.record
        # A prompt alone is upstream data for MBPP+; targets distinguish the normalized form.
        is_normalized = "prompt" in record and any(
            k in record for k in ("target", "answer", "canonical_solution"))
        if source_format == "normalized" or (source_format == "auto" and is_normalized):
            converted = dict(record)
            converted["task_id"] = task_id(record) or f"{dataset}_{row.index}"
            converted["split"] = "test"
            result.append(converted)
            continue
        coordinate = record.get(index_key)
        index = row.index if coordinate is None else _index(coordinate, index_key)
        if dataset == "gpqa_diamond":
            converted = _gpqa(record, index)
        elif dataset == "aqua_rat":
            converted = _aqua(record, index)
        elif dataset == "math500":
            converted = _math(record, index, row.path)
        elif dataset == "mbppplus":
            converted = _mbpp(record)
        elif dataset == "livecodebench":
            source_file = _text(record, "source_file") or Path(row.path).name
            converted = _lcb(record, index, source_file)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        result.append(converted)
    return result
