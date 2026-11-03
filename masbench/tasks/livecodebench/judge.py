from __future__ import annotations

import json
import math
from time import monotonic

from ...datasets.base import TaskSample
from ..sandbox import run_python_program_in_sandbox


def _remaining_timeout(deadline: float) -> int:
    return max(0, math.ceil(deadline - monotonic()))


def _sandbox_stdout(code: str, test_input: str, deadline: float) -> str | None:
    remaining = _remaining_timeout(deadline)
    if remaining <= 0:
        return None
    result = run_python_program_in_sandbox(
        code,
        remaining,
        stdin_text=test_input,
    )
    return result.stdout if result.passed else None


def run_livecodebench_stdin_tests(
    code: str,
    *,
    tests: list[dict],
    timeout_sec: int,
    sample: TaskSample | None = None,
) -> float:
    task_metadata = {}
    if sample is not None and isinstance(sample.metadata, dict):
        task_metadata = sample.metadata
    task_id = str(task_metadata.get("task_id") or getattr(sample, "task_id", "") or "").strip().lower()
    if task_id:
        special_judge = _get_livecodebench_special_judge(task_id)
        if special_judge is not None:
            return special_judge(code=code, tests=tests, timeout_sec=timeout_sec)
    deadline = monotonic() + timeout_sec
    for test in tests:
        test_input = str(test.get("input") or "")
        expected = str(test.get("output") or "").strip()
        stdout = _sandbox_stdout(code, test_input, deadline)
        if stdout is None or stdout.strip() != expected:
            return 0.0
    return 1.0


def run_livecodebench_call_tests(
    code: str,
    *,
    starter_code: str,
    func_name: str,
    tests: list[dict],
    timeout_sec: int,
) -> float:
    tests_payload = json.dumps(tests, ensure_ascii=False)
    # Starter code is an interface template and may have unfinished bodies.
    script_parts = ["from typing import *"]
    script_parts.append(code)
    script_parts.append(
        "import json as __masbench_json\n"
        f"__fn = globals().get({func_name!r})\n"
        "if __fn is None and 'Solution' in globals():\n"
        f"    __fn = getattr(globals()['Solution'](), {func_name!r})\n"
        "if __fn is None:\n"
        "    raise NameError('LiveCodeBench function not found')\n"
        "import inspect as __masbench_inspect\n"
        "try:\n"
        "    __param_count = len(__masbench_inspect.signature(__fn).parameters)\n"
        "except Exception:\n"
        "    __param_count = None\n"
        f"__tests = __masbench_json.loads({tests_payload!r})\n"
        "if not __tests:\n"
        "    raise ValueError('LiveCodeBench test list is empty')\n"
        "def __parse_scalar(__value):\n"
        "    if not isinstance(__value, str):\n"
        "        return __value\n"
        "    __text = __value.strip()\n"
        "    try:\n"
        "        return __masbench_json.loads(__text)\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        import ast as __masbench_ast\n"
        "        return __masbench_ast.literal_eval(__text)\n"
        "    except Exception:\n"
        "        return __value\n"
        "def __parse_input(__value):\n"
        "    if isinstance(__value, str) and __param_count is not None and __param_count > 1:\n"
        "        __lines = [__line.strip() for __line in __value.strip().splitlines() if __line.strip()]\n"
        "        if len(__lines) == __param_count:\n"
        "            return [__parse_scalar(__line) for __line in __lines]\n"
        "    return __parse_scalar(__value)\n"
        "for __case in __tests:\n"
        "    __inp = __parse_input(__case.get('input'))\n"
        "    __expected = __parse_scalar(__case.get('output'))\n"
        "    if isinstance(__inp, list) and __param_count is not None and __param_count > 1 and __param_count == len(__inp):\n"
        "        __got = __fn(*__inp)\n"
        "    elif isinstance(__inp, dict):\n"
        "        __got = __fn(**__inp)\n"
        "    else:\n"
        "        __got = __fn(__inp)\n"
        "    assert __got == __expected, (__got, __expected)\n"
    )
    result = run_python_program_in_sandbox(
        "\n\n".join(script_parts),
        timeout_sec,
        require_completion=True,
    )
    return 1.0 if result.passed else 0.0


def _get_livecodebench_special_judge(task_id: str):
    return {
        "lcb_arc191_c": _judge_arc191_c,
        "lcb_arc195_c": _judge_arc195_c,
    }.get(task_id)


def _judge_arc191_c(*, code: str, tests: list[dict], timeout_sec: int) -> float:
    deadline = monotonic() + timeout_sec
    for test in tests:
        test_input = str(test.get("input") or "")
        stdout = _sandbox_stdout(code, test_input, deadline)
        if stdout is None or not _validate_arc191_c_case(test_input, stdout):
            return 0.0
    return 1.0


def _validate_arc191_c_case(test_input: str, prediction_output: str) -> bool:
    tokens = [part for part in prediction_output.strip().split() if part]
    input_tokens = test_input.strip().split()
    if not input_tokens:
        return False
    try:
        t = int(input_tokens[0])
        ns = [int(value) for value in input_tokens[1:]]
    except ValueError:
        return False
    if t < 1 or len(ns) != t or len(tokens) != 2 * t:
        return False
    cursor = 0
    for n in ns:
        try:
            a = int(tokens[cursor])
            m = int(tokens[cursor + 1])
        except ValueError:
            return False
        cursor += 2
        if not (1 <= n <= 10**9 and 1 <= a <= 10**18 and 1 <= m <= 10**18):
            return False
        identity = 1 % m
        if pow(a, n, m) != identity:
            return False
        # If the order divides n, checking n / p for each prime factor p proves it is exactly n.
        remaining = n
        factor = 2
        while factor * factor <= remaining:
            if remaining % factor == 0:
                if pow(a, n // factor, m) == identity:
                    return False
                while remaining % factor == 0:
                    remaining //= factor
            factor += 1 if factor == 2 else 2
        if remaining > 1 and pow(a, n // remaining, m) == identity:
            return False
    return True


def _judge_arc195_c(*, code: str, tests: list[dict], timeout_sec: int) -> float:
    deadline = monotonic() + timeout_sec
    for test in tests:
        test_input = str(test.get("input") or "")
        stdout = _sandbox_stdout(code, test_input, deadline)
        if stdout is None or not _validate_arc195_c_case(test_input, stdout):
            return 0.0
    return 1.0


def _validate_arc195_c_case(test_input: str, prediction_output: str) -> bool:
    input_tokens = test_input.strip().split()
    if not input_tokens:
        return False
    try:
        t = int(input_tokens[0])
        counts = [int(value) for value in input_tokens[1:]]
    except ValueError:
        return False
    if t < 1 or len(counts) != 2 * t:
        return False
    cases = list(zip(counts[::2], counts[1::2]))

    lines = [line.rstrip() for line in prediction_output.strip().splitlines() if line.strip()]
    line_index = 0
    for red_count, blue_count in cases:
        if red_count < 0 or blue_count < 0 or red_count + blue_count < 2:
            return False
        if line_index >= len(lines):
            return False
        header = lines[line_index].strip()
        line_index += 1
        if header == "No":
            # A placement exists iff R is even and either R > 0 or B is even.
            if red_count % 2 == 0 and (red_count > 0 or blue_count % 2 == 0):
                return False
            continue
        if header != "Yes":
            return False
        total = red_count + blue_count
        placement_lines = lines[line_index: line_index + total]
        if len(placement_lines) != total:
            return False
        line_index += total
        parsed: list[tuple[str, int, int]] = []
        red_seen = 0
        blue_seen = 0
        used_cells = set()
        for raw in placement_lines:
            parts = raw.split()
            if len(parts) != 3:
                return False
            piece_type = parts[0]
            if piece_type not in {"R", "B"}:
                return False
            try:
                row = int(parts[1])
                col = int(parts[2])
            except ValueError:
                return False
            if not (1 <= row <= 10**9 and 1 <= col <= 10**9):
                return False
            if (row, col) in used_cells:
                return False
            used_cells.add((row, col))
            parsed.append((piece_type, row, col))
            if piece_type == "R":
                red_seen += 1
            else:
                blue_seen += 1
        if red_seen != red_count or blue_seen != blue_count:
            return False
        for idx, (piece_type, row, col) in enumerate(parsed):
            _, next_row, next_col = parsed[(idx + 1) % total]
            dr = next_row - row
            dc = next_col - col
            if piece_type == "R":
                if abs(dr) + abs(dc) != 1:
                    return False
            else:
                if abs(dr) != 1 or abs(dc) != 1:
                    return False
    return line_index == len(lines)
