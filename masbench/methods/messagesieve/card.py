from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ...core.agent import AgentProfile
from ...core.message import Message

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


_FENCED_CODE_RE = re.compile(r"```([A-Za-z0-9_-]*)\s*\n(.*?)```", re.DOTALL)


_PYTHON_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", re.MULTILINE
)


_PYTHON_CLASS_RE = re.compile(
    r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(|:)", re.MULTILINE
)


_PYTHON_EXECUTION_RE = re.compile(
    r"^\s*(?:import\s+\w+|from\s+\w+\s+import\s+|print\s*\(|[A-Za-z_][A-Za-z0-9_]*\s*=|if\s+__name__\s*==|for\s+.+\s+in\s+|while\s+.+:)",
    re.MULTILINE,
)


_SECTION_HEADERS = ("Summary", "Key Evidence", "Deliverable", "Final Answer")


_SECTION_HEADER_ALIASES = {
    "answer copy": "Final Answer",
    "answer": "Final Answer",
    "canonical answer": "Final Answer",
    "check": "Deliverable",
    "claim": "Summary",
    "decisive evidence": "Key Evidence",
    "critical check": "Deliverable",
    "reasoning": "Deliverable",
    "rationale": "Deliverable",
    "evidence": "Key Evidence",
    "verification target": "Deliverable",
    "risk": "Deliverable",
    "risk / alternative": "Deliverable",
    "risk/alternative": "Deliverable",
    "verification": "Deliverable",
}


_LCB_NATIVE_SECTION_MAP = {
    "insight": "Summary",
    "algorithm": "Key Evidence",
    "complexity": "Key Evidence",
    "edge_cases": "Key Evidence",
    "code_hint": "Deliverable",
    "final_code_candidate": "Deliverable",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(float(value), upper))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _rough_token_count(text: str) -> int:
    stripped = _safe_text(text)
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def _tokens(text: str) -> list[str]:
    return [
        item.lower() for item in _WORD_RE.findall(_safe_text(text)) if len(item) > 1
    ]


def _vectorize(text: str) -> dict[str, float]:
    counter: Counter[str] = Counter(_tokens(text))
    if not counter:
        return {}
    norm = math.sqrt(sum(value * value for value in counter.values()))
    if norm <= 1e-12:
        return {}
    return {key: value / norm for key, value in counter.items()}


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _extract_sections(text: str) -> dict[str, str]:
    lines = _safe_text(text).splitlines()
    sections: dict[str, str] = {}
    current = ""
    buffer: list[str] = []
    in_fence = False
    header_names = "|".join(
        re.escape(header)
        for header in (
            *_SECTION_HEADERS,
            "INSIGHT",
            "ALGORITHM",
            "COMPLEXITY",
            "EDGE_CASES",
            "CODE_HINT",
            "FINAL_CODE_CANDIDATE",
            "Answer Copy",
            "Canonical Answer",
            "Check",
            "Claim",
            "Answer",
            "Decisive Evidence",
            "Critical Check",
            "Verification Target",
            "Risk / Alternative",
            "Reasoning",
            "Rationale",
            "Evidence",
            "Risk",
            "Verification",
            "Confidence",
        )
    )
    header_pattern = re.compile(
        rf"^\s*(?:#{{1,6}}\s*)?({header_names})\s*:?\s*(.*)$",
        re.IGNORECASE,
    )
    canonical = {header.lower(): header for header in _SECTION_HEADERS}
    canonical.update(_SECTION_HEADER_ALIASES)
    canonical.update(_LCB_NATIVE_SECTION_MAP)

    def flush_section() -> None:
        if not current:
            return
        content = "\n".join(buffer).strip()
        if not content:
            return
        if current in sections:
            sections[current] = f"{sections[current]}\n\n{content}".strip()
        else:
            sections[current] = content

    for raw_line in lines:
        line = raw_line.rstrip()
        if re.match(r"^\s*DEBATE_HANDOFF\s*:\s*$", line, re.IGNORECASE):
            continue
        if re.match(r"^\s*ANSWER_AUDIT_CARD\s*:\s*$", line, re.IGNORECASE):
            continue
        match = None if in_fence else header_pattern.match(line)
        if match:
            flush_section()
            header_key = match.group(1).lower()
            if header_key == "confidence":
                current = ""
                buffer = []
                continue
            current = canonical[header_key]
            inline = match.group(2).strip()
            buffer = [inline] if inline else []
        elif current:
            buffer.append(line)
        if "```" in line:
            in_fence = not in_fence
    flush_section()
    return sections


def _coerce_metadata_sections(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    sections: dict[str, str] = {}
    for header in _SECTION_HEADERS:
        text = _safe_text(value.get(header))
        if text:
            sections[header] = text
    return sections


def _extract_answer(
    text: str,
    sections: Mapping[str, str],
    *,
    normalize_wrapped_option: bool = False,
    allow_option_prefix: bool = True,
) -> str:
    final = _safe_text(sections.get("Final Answer"))
    search_space = final or _safe_text(text)
    math_final = re.search(
        r"(?im)^\s*(?:the\s+answer\s+is|answer\s+is)\s+(.+?)\s*$",
        search_space,
    )
    if math_final:
        return _clean_extracted_answer(
            math_final.group(1),
            normalize_wrapped_option=normalize_wrapped_option,
        )[:120]
    if allow_option_prefix:
        first_line_option = re.search(
            r"(?im)^\s*([A-H])(?:\s*[:.)-].*)?\s*$",
            search_space,
        )
        if first_line_option:
            return first_line_option.group(1).upper()
        option = re.search(
            r"(?:final\s+answer|answer|option|choice)\s*[:\-]?\s*\(?\s*([A-H])\s*\)?",
            search_space,
            re.IGNORECASE,
        )
        if option:
            return option.group(1).upper()
    else:
        option = _extract_wrapped_option_letter(_clean_extracted_answer(search_space))
        if option:
            return option
    line_answer = re.search(
        r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)$",
        search_space,
        re.IGNORECASE | re.MULTILINE,
    )
    if line_answer:
        return _clean_extracted_answer(
            line_answer.group(1),
            normalize_wrapped_option=normalize_wrapped_option,
        )[:120]
    boxed = _extract_balanced_boxed(search_space)
    if boxed:
        return boxed
    if final:
        return _clean_extracted_answer(
            final.splitlines()[0],
            normalize_wrapped_option=normalize_wrapped_option,
        )[:120]
    return ""


def _clean_extracted_answer(
    text: str, *, normalize_wrapped_option: bool = False
) -> str:
    candidate = _safe_text(text).strip().rstrip(".")
    boxed = _extract_balanced_boxed(candidate)
    if boxed:
        return boxed.strip()
    if normalize_wrapped_option:
        option = _extract_wrapped_option_letter(candidate)
        if option:
            return option
    candidate = candidate.replace("\\left", "").replace("\\right", "")
    candidate = candidate.replace("$", "").strip()
    if candidate.startswith("\\(") and candidate.endswith("\\)"):
        candidate = candidate[2:-2].strip()
    if candidate.startswith("\\[") and candidate.endswith("\\]"):
        candidate = candidate[2:-2].strip()
    return candidate


def _extract_wrapped_option_letter(text: str) -> str:
    candidate = _safe_text(text).strip()
    match = re.fullmatch(
        r"[\s*_`~]*[\(\[\{]?[\s*_`~]*([A-H])[\s*_`~]*[\)\]\}]?[\s*_`~]*",
        candidate,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def _extract_balanced_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = _safe_text(text).rfind(marker)
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
                return "".join(collected).strip()
        if depth > 0:
            collected.append(char)
        index += 1
    return ""


def _has_code(text: str) -> bool:
    stripped = _safe_text(text)
    if _looks_like_python_code(stripped):
        return True
    for match in _FENCED_CODE_RE.finditer(stripped):
        lang = _safe_text(match.group(1)).lower()
        body = _safe_text(match.group(2))
        if lang in {"python", "py"}:
            return _looks_like_python_code(body)
        if not lang and _looks_like_python_code(body):
            return True
    return False


def _looks_like_python_code(text: str) -> bool:
    stripped = _safe_text(text)
    if not stripped:
        return False
    if _PYTHON_DEF_RE.search(stripped) or _PYTHON_CLASS_RE.search(stripped):
        return True
    return bool(
        _PYTHON_EXECUTION_RE.search(stripped)
        and ("input(" in stripped or "sys.stdin" in stripped or "print(" in stripped)
    )


def _trim_lines(text: str, *, max_lines: int) -> str:
    lines = [line.rstrip() for line in _safe_text(text).splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]).strip()


_NO_UPDATE_RE = re.compile(
    r"\b(no\s+(?:change|update)|same\s+as\s+before|unchanged|previous\s+(?:answer|plan)\s+still\s+stands)\b",
    re.IGNORECASE,
)


_ANSWER_MARKER_RE = re.compile(
    r"\b(answer|final|option|choice|therefore|hence|boxed)\b", re.IGNORECASE
)


_EVIDENCE_MARKER_RE = re.compile(
    r"\b(evidence|because|since|therefore|derive|derivation|check|verify|test|feedback|edge case|counterexample)\b",
    re.IGNORECASE,
)


_MATH_EVIDENCE_MARKER_RE = re.compile(
    r"(?:\\boxed|\\frac|\\sqrt|\\sum|\\prod|=|≤|>=|<=|\b(?:step|case|solve|substitut|"
    r"simplif|factor|equation|identity|therefore|hence|verify|check|probability|count|area|angle)\b)",
    re.IGNORECASE,
)


_FORMAT_RISK_RE = re.compile(
    r"\b(ignore|disregard|override|system prompt|developer message|not enough information)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MessageSieveCard:
    raw_text: str
    raw_tokens: int
    evidence_vector: Mapping[str, float]
    relevance: float
    freshness: float
    topology: Mapping[str, Any]
    cues: Mapping[str, Any]
    bucket: str
    answer: str
    sections: Mapping[str, str]


class CardBuilder:
    """Build receiver-conditioned message cards."""

    def prepare_message(
        self,
        *,
        preset,
        sender: AgentProfile | None,
        receiver: AgentProfile,
        round_index: int,
        task_input: str,
        message: Message,
    ) -> Message:
        card = self._build_card(
            sender=sender,
            receiver=receiver,
            task_input=task_input,
            message=message,
        )
        self.cards[message.message_id] = card
        return replace(
            message,
            metadata={
                **message.metadata,
                "message_sieve_structured_sections": dict(card.sections),
            },
        )

    def _build_card(
        self,
        *,
        sender: AgentProfile | None,
        receiver: AgentProfile,
        task_input: str,
        message: Message,
    ) -> MessageSieveCard:
        raw_text = _safe_text(message.content)
        evidence_vector = _vectorize(raw_text)
        receiver_context = " ".join(
            [
                task_input,
                receiver.role_name,
                _safe_text(receiver.system_prompt),
            ]
        )
        demand_vector = _vectorize(receiver_context)
        relevance = _cosine(evidence_vector, demand_vector)
        prior_vector = self.previous_pair_vectors.get(
            (message.source, message.target), {}
        )
        if message.channel == "temporal" and prior_vector and evidence_vector:
            freshness = _clamp(1.0 - _cosine(evidence_vector, prior_vector))
        else:
            freshness = 1.0
        self.previous_pair_vectors[(message.source, message.target)] = evidence_vector

        sections = _coerce_metadata_sections(
            message.metadata.get("message_sieve_structured_sections")
        )
        sections.update(_extract_sections(raw_text))
        bucket = self._bucket()
        answer = _extract_answer(
            raw_text, sections, allow_option_prefix=bucket != "math"
        )
        sender_role = _safe_text(message.metadata.get("sender_role"))
        if not sender_role and sender is not None:
            sender_role = sender.role_name
        sender_role = sender_role or message.source
        receiver_role = receiver.role_name
        has_code = _has_code(raw_text) or _looks_like_python_code(raw_text)
        has_evidence = bool(_EVIDENCE_MARKER_RE.search(raw_text))
        if bucket == "math":
            has_evidence = has_evidence or bool(
                _MATH_EVIDENCE_MARKER_RE.search(raw_text)
            )
        cues = {
            "has_answer": bool(answer) or bool(_ANSWER_MARKER_RE.search(raw_text)),
            "has_evidence": has_evidence,
            "has_code": has_code,
            "has_sections": bool(sections),
            "has_feedback": bool(
                _safe_text(message.metadata.get("internal_test_feedback"))
                or _safe_text(message.metadata.get("public_test_feedback"))
                or re.search(
                    r"\b(test|assert|fail|error|traceback|feedback)\b", raw_text, re.I
                )
            ),
            "verification_pass": self._coerce_optional_bool(
                message.metadata.get("internal_test_pass")
            ),
            "no_update": bool(_NO_UPDATE_RE.search(raw_text)),
            "low_content": len(raw_text.strip()) < 24 or len(evidence_vector) <= 3,
            "format_risk": bool(_FORMAT_RISK_RE.search(raw_text)),
            "fragile": has_code
            or bool(
                re.search(
                    r"\\boxed|```|def\s+\w+\(|class\s+\w+|required\s+format|exact\s+output",
                    raw_text,
                    re.I,
                )
            ),
        }
        topology = {
            "channel": message.channel,
            "sender_role": sender_role,
            "receiver_role": receiver_role,
            "receiver_is_judge": "judge" in receiver_role.lower()
            or "judge" in receiver.agent_id.lower(),
        }
        if bucket == "code":
            structured_sections = message.metadata.get(
                "message_sieve_structured_sections"
            )
            structured_feedback = ""
            if isinstance(structured_sections, Mapping):
                structured_feedback = _safe_text(
                    structured_sections.get("Feedback")
                    or structured_sections.get("Test")
                    or structured_sections.get("Tests")
                )
            visible_feedback = structured_feedback or self._explicit_feedback_span(
                raw_text
            )
            metadata_feedback = _safe_text(
                message.metadata.get("internal_test_feedback")
            )
            if not metadata_feedback:
                metadata_feedback = _safe_text(
                    message.metadata.get("public_test_feedback")
                )
            feedback = visible_feedback or metadata_feedback
            if feedback and not any(
                _safe_text(sections.get(key)) for key in ("Feedback", "Test", "Tests")
            ):
                sections["Feedback"] = feedback

            verification_pass = cues["verification_pass"]
            if verification_pass is None:
                verification_pass = self._coerce_optional_bool(
                    message.metadata.get("public_test_pass")
                )
            cues["has_feedback"] = bool(feedback) or bool(cues.get("has_feedback"))
            cues["verification_pass"] = verification_pass
            cues["has_sections"] = bool(sections)

        return MessageSieveCard(
            raw_text=raw_text,
            raw_tokens=_rough_token_count(raw_text),
            evidence_vector=evidence_vector,
            relevance=relevance,
            freshness=freshness,
            topology=topology,
            cues=cues,
            bucket=bucket,
            answer=answer,
            sections=sections,
        )

    @staticmethod
    def _explicit_feedback_span(text: str) -> str:
        match = re.search(
            r"(?ims)^\s*(?:public[- ]test\s+|test\s+)?feedback\s*:\s*(.*?)"
            r"(?=^\s*(?:summary|key evidence|deliverable|final answer|answer|"
            r"algorithm|complexity|edge[_ -]?cases?|fix|action)\s*:|\Z)",
            _safe_text(text),
        )
        return _trim_lines(match.group(1), max_lines=2) if match else ""

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = _safe_text(value).lower()
        if text in {"1", "true", "yes", "pass", "passed"}:
            return True
        if text in {"0", "false", "no", "fail", "failed"}:
            return False
        return None
