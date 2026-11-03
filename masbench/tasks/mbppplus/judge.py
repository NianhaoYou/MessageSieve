from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ...datasets.base import TaskSample
from ..sandbox import run_python_program_in_sandbox


def _run_script_in_sandbox(script: str, timeout_sec: int) -> tuple[bool, str]:
    result = run_python_program_in_sandbox(
        script,
        timeout_sec,
        require_completion=True,
    )
    if result.passed:
        return True, ""
    return False, (result.stderr or result.stdout or "runtime error")[:4096].strip()


def _looks_like_python_solution_block(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if re.search(r"(?m)^\s*(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*\s*[\(:]", stripped):
        return True
    if re.search(r"(?m)^\s*(?:from|import)\s+\S+", stripped):
        return True
    return False


def extract_python_code(text: str) -> str | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    python_blocks = re.findall(
        r"```python\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL
    )
    if python_blocks:
        for block in reversed(python_blocks):
            candidate = block.strip()
            if _looks_like_python_solution_block(candidate):
                return candidate
        return python_blocks[-1].strip()
    fenced_blocks = re.findall(r"```\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced_blocks:
        for block in reversed(fenced_blocks):
            candidate = block.strip()
            if _looks_like_python_solution_block(candidate):
                return candidate
        return fenced_blocks[-1].strip()
    if (
        stripped.startswith("def ")
        or stripped.startswith("from ")
        or stripped.startswith("import ")
        or "class " in stripped
    ):
        return stripped
    return None


def _run_assert_in_subprocess(
    code: str,
    assertion: str,
    setup_code: str,
    timeout_sec: int,
    *,
    imports: Sequence[str] | None = None,
) -> tuple[bool, str]:
    script = ["from typing import *"]
    if imports:
        script.extend(str(item).strip() for item in imports if str(item).strip())
    if setup_code.strip():
        script.append(setup_code.strip())
    script.append(code)
    script.append(assertion)
    full_script = "\n\n".join(script)
    return _run_script_in_sandbox(full_script, timeout_sec)


def run_mbppplus_harness(
    code: str,
    *,
    test: str,
    test_imports: Sequence[str] | None = None,
    timeout_sec: int,
) -> float:
    """Run the official MBPP+ generated test harness against candidate code."""
    if not code.strip() or not str(test or "").strip():
        return 0.0
    script_parts = ["from typing import *"]
    if test_imports:
        script_parts.extend(
            str(item).strip() for item in test_imports if str(item).strip()
        )
    script_parts.append(code)
    script_parts.append(str(test).strip())
    full_script = "\n\n".join(script_parts)
    passed, _ = _run_script_in_sandbox(full_script, timeout_sec)
    return 1.0 if passed else 0.0


def run_mbppplus_tests(
    code: str,
    *,
    tests: Sequence[str],
    setup_code: str,
    imports: Sequence[str] | None = None,
    timeout_sec: int,
) -> float:
    if not code.strip() or not tests:
        return 0.0
    passed = 0
    total = len(tests)
    for test in tests:
        ok, _ = _run_assert_in_subprocess(
            code, test, setup_code, timeout_sec, imports=imports
        )
        if ok:
            passed += 1
    return 1.0 if passed == total else 0.0


def run_mbppplus_sample(
    code: str,
    *,
    sample: TaskSample,
    timeout_sec: int,
) -> float:
    test = str(sample.metadata.get("test") or "").strip()
    if test:
        return run_mbppplus_harness(
            code,
            test=test,
            test_imports=_load_test_imports(sample.metadata.get("test_imports")),
            timeout_sec=timeout_sec,
        )
    tests = build_mbppplus_public_tests(sample)
    setup_code = str(sample.metadata.get("test_setup_code") or "")
    return run_mbppplus_tests(
        code,
        tests=tests,
        setup_code=setup_code,
        imports=_load_test_imports(sample.metadata.get("test_imports")),
        timeout_sec=timeout_sec,
    )


def build_mbppplus_public_tests(sample: TaskSample) -> list[str]:
    test_list = sample.metadata.get("test_list")
    if isinstance(test_list, list):
        return [str(item) for item in test_list if str(item).strip()]
    if isinstance(test_list, str) and test_list.strip():
        try:
            payload = json.loads(test_list)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item).strip()]
    return []


def _load_test_imports(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return [
                line
                for line in value.splitlines()
                if line.strip() and re.match(r"\s*(?:import|from)\s+", line)
            ]
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item).strip()]
    return []
