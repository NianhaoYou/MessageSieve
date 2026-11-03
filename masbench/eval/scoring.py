from __future__ import annotations

import base64
import binascii
from io import BytesIO
import pickle
import re
import json
import zlib
import math
from typing import Callable

from ..datasets.base import TaskSample

try:
    import regex as _regex
except ImportError:  # pragma: no cover - optional dependency
    _regex = None

N = None
pi = None
simplify = None
sqrt = None
real_root = None
parse_latex = None
parse_expr = None
implicit_multiplication_application = None
standard_transformations = ()
_SYMPY_LOAD_ATTEMPTED = False


def _ensure_sympy_loaded() -> bool:
    global N, pi, simplify, sqrt, real_root, parse_latex, parse_expr
    global implicit_multiplication_application, standard_transformations
    global _SYMPY_LOAD_ATTEMPTED
    if _SYMPY_LOAD_ATTEMPTED:
        return simplify is not None and N is not None and parse_expr is not None
    _SYMPY_LOAD_ATTEMPTED = True
    try:
        from sympy import N as sympy_N
        from sympy import pi as sympy_pi
        from sympy import real_root as sympy_real_root
        from sympy import simplify as sympy_simplify
        from sympy import sqrt as sympy_sqrt
        from sympy.parsing.latex import parse_latex as sympy_parse_latex
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application as sympy_implicit_multiplication_application,
        )
        from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr
        from sympy.parsing.sympy_parser import standard_transformations as sympy_standard_transformations
    except ImportError:  # pragma: no cover - optional dependency
        return False
    N = sympy_N
    pi = sympy_pi
    simplify = sympy_simplify
    sqrt = sympy_sqrt
    real_root = sympy_real_root
    parse_latex = sympy_parse_latex
    parse_expr = sympy_parse_expr
    implicit_multiplication_application = sympy_implicit_multiplication_application
    standard_transformations = sympy_standard_transformations
    return True

UtilityFn = Callable[[str, TaskSample], float]


def build_utility_fn(dataset_id: str) -> UtilityFn:
    key = dataset_id.strip().lower()
    if key == "livecodebench_codegen_lite":
        return livecodebench_codegen_pass_utility
    if key == "mbppplus":
        return mbppplus_pass_utility
    if key in {"gpqa_diamond", "aqua"}:
        return choice_exact_match_utility
    if key == "math":
        if not _ensure_sympy_loaded():
            raise RuntimeError(
                "MATH scoring requires sympy in the active Python environment. "
                "Without it, mathematically equivalent expressions can be mis-scored."
            )
        return math_exact_match_utility
    raise ValueError(f"Unsupported dataset: {dataset_id}")


def choice_exact_match_utility(prediction: str, sample: TaskSample) -> float:
    target = "" if sample.target is None else str(sample.target).strip().upper()
    pred = _extract_choice_letter(prediction)
    return 1.0 if pred == target else 0.0


def math_exact_match_utility(prediction: str, sample: TaskSample) -> float:
    target = "" if sample.target is None else str(sample.target)
    normalized_pred = _extract_math_answer(prediction)
    # Targets are answer expressions, not model explanations to search for numbers.
    normalized_target = _clean_math_answer_candidate(target)
    if _math_answers_equivalent(normalized_pred, normalized_target):
        return 1.0
    # Do not fall back to math_verify here. In this benchmark setting it can
    # parse only numeric fragments from LaTeX expressions such as 2\sqrt{41},
    # causing false positives against unrelated expressions.
    return 0.0


def livecodebench_codegen_pass_utility(prediction: str, sample: TaskSample, *, timeout_sec: int = 100) -> float:
    tests = _load_livecodebench_tests(sample)
    if not tests:
        return 0.0
    code = _extract_python_code(prediction) or prediction.strip()
    if not code:
        return 0.0
    task_metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    metadata = _loads_jsonish(task_metadata.get("metadata"))
    func_name = _livecodebench_func_name(metadata)
    starter_code = str(task_metadata.get("starter_code") or "").strip()
    if func_name:
        from ..tasks.livecodebench.judge import run_livecodebench_call_tests

        return run_livecodebench_call_tests(
            code,
            starter_code=starter_code,
            func_name=func_name,
            tests=tests,
            timeout_sec=timeout_sec,
        )
    from ..tasks.livecodebench.judge import run_livecodebench_stdin_tests

    return run_livecodebench_stdin_tests(
        code,
        tests=tests,
        timeout_sec=timeout_sec,
        sample=sample,
    )


def mbppplus_pass_utility(prediction: str, sample: TaskSample, *, timeout_sec: int = 100) -> float:
    from ..tasks.mbppplus.judge import extract_python_code as _extract_mbppplus_python_code
    from ..tasks.mbppplus.judge import run_mbppplus_sample

    code = _extract_mbppplus_python_code(prediction) or prediction.strip()
    if not code:
        return 0.0
    return run_mbppplus_sample(code, sample=sample, timeout_sec=timeout_sec)


def _extract_choice_letter(text: str) -> str:
    final_answer_matches = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*[\(\[\{]?\s*([A-H])\s*[\)\]\}]?\s*(?:$|[.\s,:;-])",
        str(text or ""),
    )
    if final_answer_matches:
        return final_answer_matches[-1].upper()
    for line in text.splitlines():
        candidate = line.strip().upper()
        if candidate in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            return candidate
    option_matches = re.findall(r"\b([A-H])\b", text.upper())
    return option_matches[-1] if option_matches else ""


def validate_code_scorer_sample(dataset_id: str, sample: TaskSample) -> None:
    dataset_key = str(dataset_id or "").strip().lower()
    if dataset_key == "livecodebench_codegen_lite":
        raw_private = sample.metadata.get("private_test_cases")
        tests = _load_livecodebench_test_blob(raw_private)
        if not isinstance(tests, list) or not tests:
            raise ValueError(
                f"LiveCodeBench sample {sample.task_id!r} has no decodable private test cases"
            )
        malformed = [
            index
            for index, test in enumerate(tests)
            if not isinstance(test, dict) or "input" not in test or "output" not in test
        ]
        if malformed:
            raise ValueError(
                f"LiveCodeBench sample {sample.task_id!r} has malformed private test cases "
                f"at indices {malformed[:5]}"
            )
        return
    if dataset_key == "mbppplus":
        private_test = str(sample.metadata.get("test") or "").strip()
        if not private_test:
            raise ValueError(
                f"MBPP+ sample {sample.task_id!r} has no executable test metadata "
                "(private test missing)"
            )
        try:
            compile(private_test, f"<mbppplus-private-test:{sample.task_id}>", "exec")
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(
                f"MBPP+ sample {sample.task_id!r} has malformed private test metadata"
            ) from exc


def _load_livecodebench_tests(sample: TaskSample) -> list[dict]:
    raw_private = sample.metadata.get("private_test_cases")
    raw_public = sample.metadata.get("public_test_cases")
    tests = _load_livecodebench_test_blob(raw_private)
    if not tests:
        tests = _load_livecodebench_test_blob(raw_public)
    if isinstance(tests, list):
        return [test for test in tests if isinstance(test, dict)]
    return []


def _load_livecodebench_test_blob(value):
    direct = _loads_jsonish(value)
    if isinstance(direct, list):
        return direct

    text = str(value or "").strip()
    if not text:
        return None

    decoded = _decode_livecodebench_private_tests(text)
    if decoded is not None:
        return decoded
    return direct


class _RestrictedDataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"global object loading is disabled ({module}.{name})"
        )


def _restricted_pickle_loads(payload: bytes):
    return _RestrictedDataUnpickler(BytesIO(payload)).load()


def _decode_livecodebench_private_tests(text: str):
    try:
        compressed = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        return None

    decompressed = None
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, 15 | 32):
        try:
            decompressed = zlib.decompress(compressed, wbits)
            break
        except zlib.error:
            continue
    if decompressed is None:
        return None

    candidates = []
    try:
        candidates.append(_restricted_pickle_loads(decompressed))
    except (EOFError, pickle.UnpicklingError, ValueError, TypeError):
        pass
    candidates.append(decompressed)

    for candidate in candidates:
        if isinstance(candidate, (list, dict)):
            return candidate
        if isinstance(candidate, bytes):
            try:
                candidate = candidate.decode("utf-8")
            except UnicodeDecodeError:
                continue
        if isinstance(candidate, str):
            loaded = _loads_jsonish(candidate)
            if loaded is not None:
                return loaded
    return None


def _loads_jsonish(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _livecodebench_func_name(metadata) -> str:
    if not isinstance(metadata, dict):
        return ""
    for key in ("func_name", "function_name", "entry_point"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_math_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    explicit_answers = list(
        re.finditer(
            r"(?im)^[ \t]*(?:#+[ \t]*)?"
            r"(?:final[ \t]+answer[ \t]*:|(?:the[ \t]+)?answer[ \t]+is)"
            r"[ \t]*([^\r\n]+)$",
            stripped,
        )
    )
    # A later boxed conclusion takes precedence over an earlier answer preamble.
    if explicit_answers and stripped.rfind(r"\boxed{") > explicit_answers[-1].end():
        boxed = _extract_boxed_answer(stripped)
        if boxed:
            return boxed
    for match in reversed(explicit_answers):
        candidate = _clean_math_answer_candidate(match.group(1))
        if candidate and not candidate.endswith(":"):
            return candidate

    boxed = _extract_boxed_answer(stripped)
    if boxed:
        return boxed

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.search(r"(?:answer|ans)\s*[:=]\s*(.+)", line, flags=re.IGNORECASE)
        if match:
            candidate = _clean_math_answer_candidate(match.group(1).strip())
            if candidate:
                return candidate

    math_like = _extract_last_math_like_answer(lines, stripped)
    if math_like:
        return math_like

    if _looks_like_math_expression(stripped):
        return _strip_math_wrappers(stripped)

    numbers = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?%?", stripped.replace(",", ""))
    if numbers:
        value = numbers[-1]
        return value[:-1] if value.endswith(".") else value

    return _strip_math_wrappers(lines[-1] if lines else stripped)


def _extract_last_math_like_answer(lines: list[str], stripped: str) -> str:
    """Keep exact LaTeX/interval answers before falling back to a last-number heuristic."""
    candidates: list[str] = []
    for line in reversed(lines):
        cleaned = _clean_math_answer_candidate(line)
        if not cleaned:
            continue
        lowered = cleaned.lower().strip()
        if lowered in {"final answer", "answer", "final", "$$", "\\[", "\\]"}:
            continue
        if set(lowered) <= {"-", "_", "=", " "}:
            continue
        candidates.append(cleaned)

    for candidate in candidates:
        if _looks_like_math_expression(candidate):
            return candidate

    cleaned_full = _strip_math_wrappers(stripped)
    if _looks_like_math_expression(cleaned_full):
        return cleaned_full
    return ""


def _looks_like_math_expression(text: str) -> bool:
    value = _strip_math_wrappers(text)
    if not value:
        return False
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return True
    word_tokens = re.findall(r"[A-Za-z]+", value)
    if re.search(r"\\(?:frac|dfrac|tfrac|sqrt|infty|cup|cap|leq|geq|cdot|times|pi|theta|circ|sin|cos|tan|cot|sec|csc|log|ln|exp)\b", value):
        return True
    if re.search(r"\\(?:frac|dfrac|tfrac)\s*[+-]?\d\s*[+-]?\d\b", value):
        return True
    if re.search(r"\\sqrt\s*(?:\d+|[A-Za-z]|\()", value):
        return True
    if re.search(r"√\s*(?:\d+|[A-Za-zπ]|\(|\{)", value):
        return True
    if "π" in value and re.fullmatch(r"[\d\sπ+\-*/^().{}]+", value):
        return True
    if re.search(r"[\^_{}]", value):
        return True
    if "=" in value and len(word_tokens) <= 8:
        return True
    if re.search(r"[\[\]\(\)]", value) and re.search(r"\\infty|,|\\cup|\\cap", value):
        return True
    if re.search(r"[A-Za-z]", value) and re.search(r"[+\-*/^]", value) and len(word_tokens) <= 6:
        return True
    if re.search(r"\d+\s*[+\-*/]\s*\d+", value):
        return True
    return False


def _clean_math_answer_candidate(text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    boxed = _extract_boxed_answer(candidate)
    if boxed:
        return boxed
    lowered = candidate.lower()
    for prefix in ("final answer:", "answer:", "ans:", "the answer is"):
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            lowered = candidate.lower()
    return _strip_math_wrappers(candidate)


def _extract_boxed_answer(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    if _regex is not None:
        matches = _regex.findall(pattern, text, _regex.DOTALL)
        if matches:
            return _strip_math_wrappers(matches[-1].strip())
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""
    index = start + len(marker)
    depth = 1
    collected: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _strip_math_wrappers("".join(collected).strip())
        if depth > 0:
            collected.append(char)
        index += 1
    return ""


def _strip_math_wrappers(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace(r"\$", "").replace("$", "").strip()
    for marker in ("**", "__"):
        if cleaned.startswith(marker) and cleaned.endswith(marker) and len(cleaned) > 2 * len(marker):
            cleaned = cleaned[len(marker) : -len(marker)].strip()
            break
    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
        cleaned = cleaned[2:-2].strip()
    return cleaned


def _normalize_latex_expr(text: str) -> str:
    normalized = _strip_math_wrappers(text)
    normalized = normalized.replace("π", "pi").replace("√", r"\sqrt")
    normalized = normalized.replace("\\%", "%")
    normalized = re.sub(r"\s*\^?\s*\\circ\b", "", normalized)
    normalized = normalized.replace("°", "")
    normalized = normalized.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    normalized = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", normalized)
    normalized = re.sub(r"\\frac\s*([+-]?\d)\s*([+-]?\d)\b", r"\\frac{\1}{\2}", normalized)
    normalized = _replace_latex_fraction_and_sqrt(normalized)
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("\\pi", "pi")
    normalized = re.sub(r"\\frac\s*\(([^()]*)\)\s*\(([^()]*)\)", r"(\1)/(\2)", normalized)
    normalized = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"sqrt(\1)", normalized)
    normalized = normalized.replace("\\", "")
    normalized = re.sub(r"(?<=[\d)])(?=[A-Za-z(])", "*", normalized)
    normalized = re.sub(r"(?<=\d)(?=pi|sqrt|\()", "*", normalized)
    normalized = re.sub(r"(?<=\))(?=pi|sqrt|\d)", "*", normalized)
    normalized = re.sub(r"(?<=pi)(?=sqrt|\d|\()", "*", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _replace_latex_fraction_and_sqrt(text: str) -> str:
    normalized = text
    while True:
        replaced = _replace_first_latex_sqrt(_replace_first_latex_frac(normalized))
        if replaced == normalized:
            return normalized
        normalized = replaced


def _replace_first_latex_sqrt(text: str) -> str:
    marker = "\\sqrt"
    start = text.find(marker)
    if start < 0:
        return text
    prefix = re.match(r"\\sqrt\s*(?:\[\s*(\d+)\s*\]\s*)?", text[start:])
    degree = int(prefix.group(1)) if prefix.group(1) is not None else 2
    if degree == 0:
        return text
    brace_start = start + prefix.end()
    group = _extract_balanced_brace_group(text, brace_start)
    if group is None:
        return text
    content, end = group
    root = f"sqrt({content})" if degree == 2 else f"real_root(({content}), {degree})"
    return f"{text[:start]}{root}{text[end:]}"


def _replace_first_latex_frac(text: str) -> str:
    marker = "\\frac"
    start = text.find(marker)
    if start < 0:
        return text
    numerator_start = text.find("{", start + len(marker))
    if numerator_start < 0:
        return text
    numerator = _extract_balanced_brace_group(text, numerator_start)
    if numerator is None:
        return text
    numerator_text, numerator_end = numerator
    denominator_start = text.find("{", numerator_end)
    if denominator_start < 0:
        return text
    denominator = _extract_balanced_brace_group(text, denominator_start)
    if denominator is None:
        return text
    denominator_text, denominator_end = denominator
    return f"{text[:start]}(({numerator_text})/({denominator_text})){text[denominator_end:]}"


def _extract_balanced_brace_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    collected: list[str] = []
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth > 1:
                collected.append(char)
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return "".join(collected), index + 1
            collected.append(char)
            continue
        collected.append(char)
    return None


def _parse_number(value: str) -> float | None:
    cleaned = value.strip()
    if "," in cleaned and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", cleaned):
        return None
    cleaned = cleaned.replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1]) / 100.0
            except ValueError:
                return None
    return None


def _parse_math_expr(value: str):
    normalized = _normalize_latex_expr(value)
    if not normalized:
        return None
    _ensure_sympy_loaded()
    if parse_expr is not None:
        try:
            transformations = standard_transformations
            if implicit_multiplication_application is not None:
                transformations = transformations + (implicit_multiplication_application,)
            local_dict = {}
            if pi is not None:
                local_dict["pi"] = pi
            if sqrt is not None:
                local_dict["sqrt"] = sqrt
            if real_root is not None:
                local_dict["real_root"] = real_root
            return parse_expr(normalized, transformations=transformations, local_dict=local_dict)
        except Exception:
            pass
    if parse_latex is not None:
        try:
            return parse_latex(_strip_math_wrappers(value))
        except Exception:
            pass
    return None


def _canonical_math_surface(text: str) -> str:
    value = _strip_math_wrappers(text)
    if not value:
        return ""
    value = value.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("~", " ")
    for command in ("text", "mathrm", "operatorname"):
        pattern = re.compile(r"\\" + command + r"\{([^{}]*)\}")
        while True:
            replaced = pattern.sub(r"\1", value)
            if replaced == value:
                break
            value = replaced
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _strip_membership_prefix(text: str) -> str:
    value = _canonical_math_surface(text)
    match = re.match(r"^[A-Za-z]\s*(?:\\in|in)\s*(.+)$", value)
    return match.group(1).strip() if match else value


def _choice_letter_answer(text: str) -> str:
    value = _canonical_math_surface(text).upper()
    if re.fullmatch(r"\(?[A-H]\)?", value):
        return value.strip("()")
    match = re.search(r"\(([A-H])\)", value)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:OPTION|CHOICE|ANSWER)\s*[:\-]?\s*\(?([A-H])\)?\b", value)
    return match.group(1) if match else ""


def _compact_digit_answer(text: str) -> str:
    value = _canonical_math_surface(text)
    compact = re.sub(r"\s+", "", value)
    if re.search(r"[,']", compact) and not re.fullmatch(
        r"-?\d{1,3}(?:[,']\d{3})+(?:\.\d+)?", compact
    ):
        return ""
    compact = compact.replace(",", "").replace("'", "")
    return compact if re.fullmatch(r"-?\d+(?:\.\d+)?", compact) else ""


def _plain_text_answer(text: str) -> str:
    value = _canonical_math_surface(text)
    if not re.search(r"[A-Za-z]", value):
        return ""
    if re.search(r"\\(?:frac|sqrt|infty|cup|cap|leq|geq|cdot|times|pi|theta|circ)\b", value):
        return ""
    if re.search(r"[\^_{}=+\-*/\[\](),]", value):
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", value).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _strip_simple_assignment(text: str) -> str:
    value = _strip_math_wrappers(text)
    match = re.fullmatch(r"\s*[A-Za-z]\s*=\s*(.+?)\s*", value, flags=re.DOTALL)
    return match.group(1).strip() if match else value


def _numeric_sequence(text: str) -> tuple[float, ...]:
    value = _strip_math_wrappers(text)
    value = re.sub(r"\\(?:begin|end)\{[pbvBV]?matrix\}", "", value)
    parts = [part.strip() for part in re.split(r"\\\\|,", value) if part.strip()]
    if len(parts) < 2:
        return ()
    parsed = tuple(_parse_number(part) for part in parts)
    if any(item is None for item in parsed):
        return ()
    return tuple(float(item) for item in parsed if item is not None)


def _numbers_close(left: float, right: float) -> bool:
    # Absolute tolerance alone would accept zero for small nonzero probabilities.
    return abs(left - right) <= 1e-3 and math.isclose(
        left, right, rel_tol=1e-3, abs_tol=1e-9
    )


def _math_answers_equivalent(prediction: str, target: str) -> bool:
    pred = _strip_math_wrappers(prediction)
    gold = _strip_math_wrappers(target)
    if not pred or not gold:
        return False
    if pred == gold:
        return True

    pred = _strip_simple_assignment(pred)
    gold = _strip_simple_assignment(gold)
    if pred == gold:
        return True
    pred_sequence = _numeric_sequence(pred)
    gold_sequence = _numeric_sequence(gold)
    if pred_sequence and gold_sequence and len(pred_sequence) == len(gold_sequence):
        return all(_numbers_close(left, right) for left, right in zip(pred_sequence, gold_sequence))

    normalized_pred = _normalize_latex_expr(pred)
    normalized_gold = _normalize_latex_expr(gold)
    if normalized_pred == normalized_gold:
        return True
    if normalized_pred.replace(" ", "") == normalized_gold.replace(" ", ""):
        return True

    surface_pred = _canonical_math_surface(pred)
    surface_gold = _canonical_math_surface(gold)
    if surface_pred and surface_gold:
        if surface_pred == surface_gold:
            return True
        if surface_pred.replace(" ", "") == surface_gold.replace(" ", ""):
            return True
        pred_member = _strip_membership_prefix(surface_pred)
        gold_member = _strip_membership_prefix(surface_gold)
        if pred_member and gold_member and pred_member.replace(" ", "") == gold_member.replace(" ", ""):
            return True
        pred_choice = _choice_letter_answer(surface_pred)
        gold_choice = _choice_letter_answer(surface_gold)
        if pred_choice and gold_choice and pred_choice == gold_choice:
            return True
        pred_digits = _compact_digit_answer(surface_pred)
        gold_digits = _compact_digit_answer(surface_gold)
        if pred_digits and gold_digits and pred_digits == gold_digits:
            return True
        pred_text = _plain_text_answer(surface_pred)
        gold_text = _plain_text_answer(surface_gold)
        if pred_text and gold_text and pred_text == gold_text:
            return True

    pred_num = _parse_number(pred)
    gold_num = _parse_number(gold)
    if pred_num is not None and gold_num is not None:
        return _numbers_close(pred_num, gold_num)

    if not _ensure_sympy_loaded() or simplify is None or N is None:
        return False

    pred_expr = _parse_math_expr(pred)
    gold_expr = _parse_math_expr(gold)
    if pred_expr is None or gold_expr is None:
        return False

    try:
        if simplify(pred_expr - gold_expr) == 0:
            return True
    except Exception:
        pass

    try:
        return _numbers_close(float(N(pred_expr)), float(N(gold_expr)))
    except Exception:
        return False

def _extract_python_code(text: str) -> str:
    fenced_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        return fenced_blocks[-1].strip()
    generic_blocks = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()
    content = text.strip()
    if not content:
        return ""
    start_patterns = [
        r"^\s*def\s+\w+\s*\(",
        r"^\s*async\s+def\s+\w+\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*from\s+\w+",
        r"^\s*import\s+\w+",
        r"^\s*@",
    ]
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if any(re.match(pattern, line) for pattern in start_patterns):
            return "\n".join(lines[index:]).strip()
    if "def " in content or "class " in content:
        return content
    return ""
