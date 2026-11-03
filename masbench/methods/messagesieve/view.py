from __future__ import annotations

import math
import re

from .card import (
    _ANSWER_MARKER_RE,
    _EVIDENCE_MARKER_RE,
    _MATH_EVIDENCE_MARKER_RE,
    MessageSieveCard,
    _cosine,
    _rough_token_count,
    _safe_text,
    _trim_lines,
    _vectorize,
)
from .gate import AdmissionDecision
from .relation import MessageSieveRelation

LITE_BUDGET_RATIO = 0.8
CODE_LITE_BUDGET_RATIO = 0.9


_CODE_EVIDENCE_RE = re.compile(
    r"\b(algorithm|invariant|complexity|constraint|counterexample|edge[ -]?case|"
    r"boundary|interface|signature|test|feedback|fix|risk|off-by-one|time|memory)\b|"
    r"\bO\s*\(",
    re.IGNORECASE,
)
_FENCED_BLOCK_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_CODE_CORRECTION_RE = re.compile(
    r"\b(counterexample|incorrect|wrong|bug|fix|backtrack|off-by-one)\b",
    re.IGNORECASE,
)
_CODE_EQUATION_RE = re.compile(r"==|!=|<=|>=|(?<![<>=!])=(?!=)|≤|≥")
_MATH_CHECK_RE = re.compile(
    r"\b(check|verif\w*|substitut\w*|counterexample|contradiction|invalid|valid)\b",
    re.IGNORECASE,
)
_MATH_FORMULA_RE = re.compile(r"[=<>≤≥]|\\(?:frac|sqrt|sum|prod|int|equiv|neq|leq|geq|in)\b")


def _sigmoid(value: float) -> float:
    if value >= 40:
        return 1.0
    if value <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _first_sentences(text: str, *, max_sentences: int = 1) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", _safe_text(text))
        if sentence.strip()
    ]
    return " ".join(sentences[: max(1, max_sentences)])


class ViewProjector:
    """Realize admitted messages as full text or compact task-family views."""

    def _project_view(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
        admission: AdmissionDecision,
    ) -> str:
        if relation.relation_reason:
            family_view = self._project_family_relation_view(
                card=card, relation=relation
            )
            if family_view is not None:
                return family_view
        fidelity = self._fidelity_score(
            card=card, relation=relation, admission=admission
        )
        if fidelity >= 0.5:
            return card.raw_text
        lite = self._lite_view(card=card)
        lite_tokens = _rough_token_count(lite)
        max_lite_tokens = max(1, int(card.raw_tokens * LITE_BUDGET_RATIO))
        if (
            not lite
            or lite == card.raw_text
            or lite_tokens >= card.raw_tokens
            or lite_tokens > max_lite_tokens
        ):
            return card.raw_text
        return lite

    def _project_family_relation_view(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
    ) -> str | None:
        if card.bucket == "code":
            return self._project_code_relation_view(card=card, relation=relation)
        if card.bucket == "math":
            return self._project_math_relation_view(card=card, relation=relation)
        if card.bucket == "qa":
            return self._project_qa_relation_view(card=card, relation=relation)
        return None

    def _project_qa_relation_view(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
    ) -> str:
        reason = relation.relation_reason
        if (
            card.bucket == "qa"
            and reason in {"disagreement_relation", "novelty_relation"}
            and (
                relation.relation_score >= 4
                or relation.conflict > 0.0
                or relation.novelty >= 0.72
            )
        ):
            return card.raw_text
        lite = self._lite_view(card=card)
        lite_tokens = _rough_token_count(lite)
        max_lite_tokens = max(1, int(card.raw_tokens * LITE_BUDGET_RATIO))
        if (
            not lite
            or lite == card.raw_text
            or lite_tokens >= card.raw_tokens
            or lite_tokens > max_lite_tokens
        ):
            lite = self._minimal_qa_lite_view(card=card)
        if not lite:
            lite = "[LITE]\nSummary: message retained by receiver-local relation gate."
            if card.answer:
                lite += f"\nFinal Answer: {card.answer}"
        return lite

    def _project_math_relation_view(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
    ) -> str:
        reason = relation.relation_reason
        if reason in {
            "math_bounded_full_anchor",
            "math_self_temporal_anchor",
            "math_disagreement_representative",
        }:
            return card.raw_text
        if reason == "math_bounded_secondary_anchor":
            return self._math_lite_view(card=card) or (
                "[LITE]\nEvidence: retained secondary derivation anchor."
            )
        lite = self._lite_view(card=card)
        lite_tokens = _rough_token_count(lite)
        max_lite_tokens = max(1, int(card.raw_tokens * LITE_BUDGET_RATIO))
        if (
            not lite
            or lite == card.raw_text
            or lite_tokens >= card.raw_tokens
            or lite_tokens > max_lite_tokens
        ):
            return card.raw_text
        return lite

    def _project_code_relation_view(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
    ) -> str:
        reason = relation.relation_reason
        if reason == "code_failed_feedback_relation":
            return self._compact_code_lite_view(card=card) or (
                "[LITE]\nFeedback: Executable candidate failed available tests; "
                "code omitted from this view."
            )
        if reason == "code_cross_source_artifact_relation":
            return card.raw_text
        if (
            reason in {"code_artifact_relation", "code_anchor_full_relation"}
            and card.cues.get("has_code")
            and card.cues.get("verification_pass") is not False
        ):
            return card.raw_text
        if self._should_project_code_representative_full(card=card, reason=reason):
            return card.raw_text
        lite = self._compact_code_lite_view(card=card)
        max_lite_tokens = max(1, int(card.raw_tokens * CODE_LITE_BUDGET_RATIO))
        lite = self._fit_code_lite_budget(lite, max_tokens=max_lite_tokens)
        if not lite or _rough_token_count(lite) >= card.raw_tokens:
            return card.raw_text
        return lite

    @staticmethod
    def _fit_code_lite_budget(text: str, *, max_tokens: int) -> str:
        compact = _safe_text(text)
        if compact and _rough_token_count(compact) <= max_tokens:
            return compact
        # The caller restores FULL rather than cutting a formula or repair span.
        return ""

    def _should_project_code_representative_full(
        self,
        *,
        card: MessageSieveCard,
        reason: str,
    ) -> bool:
        if self.task_family != "code":
            return False
        return (
            reason == "spatial_representative_relation"
            and _safe_text(card.topology.get("channel")) == "spatial"
            and not card.topology.get("receiver_is_judge")
            and card.raw_tokens >= 70
        )

    def _fidelity_score(
        self,
        *,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
        admission: AdmissionDecision,
    ) -> float:
        fragile = 1.0 if card.cues.get("fragile") else 0.0
        protected = 1.0 if "protected_artifact" in admission.support_predicates else 0.0
        decisive = 1.0 if "useful_disagreement" in admission.support_predicates else 0.0
        score = (
            1.35 * fragile
            + 1.1 * protected
            + 0.85 * decisive
            + 0.45 * relation.conflict
            + 0.2 * max(admission.margin, 0.0)
            - 0.55 * relation.coverage
        )
        return _sigmoid(score)

    def _lite_view(self, *, card: MessageSieveCard) -> str:
        if card.bucket == "code":
            return self._compact_code_lite_view(card=card)
        if card.bucket == "math":
            return self._math_lite_view(card=card)
        return self._qa_lite_view(card=card)

    def _math_lite_view(self, *, card: MessageSieveCard) -> str:
        fields: list[tuple[str, str]] = []
        if card.answer:
            fields.append(("Final", card.answer))
        evidence = self._math_evidence_span(card.raw_text, final_answer=card.answer)
        if evidence:
            fields.append(("Evidence", evidence))
        if not fields:
            first = self._first_evidence_span(card.raw_text) or _trim_lines(
                card.raw_text, max_lines=1
            )
            if first:
                fields.append(("Evidence", first))
        if not fields:
            return ""
        return "[LITE]\n" + "\n".join(
            f"{label}: {value}" for label, value in fields[:3] if value
        )

    def _compact_code_lite_view(self, *, card: MessageSieveCard) -> str:
        fields: list[tuple[str, str]] = []
        incomplete_evidence = False

        def add(
            label: str,
            value: str,
            *,
            max_lines: int = 1,
            force: bool = False,
        ) -> None:
            nonlocal incomplete_evidence
            compact = _trim_lines(
                _FENCED_BLOCK_RE.sub("", _safe_text(value)),
                max_lines=max_lines,
            )
            if len(compact) > 320:
                if label == "Key Evidence":
                    incomplete_evidence = True
                    return
                # Keep complete sentences; never cut an expression mid-token.
                sentences = re.split(r"(?<=[.!?])\s+|\n", compact)
                compact = ""
                for sentence in sentences:
                    candidate = (compact + " " + sentence).strip()
                    if len(candidate) > 320:
                        break
                    compact = candidate
            if not compact or compact == "<code omitted>":
                return
            if not force:
                compact_vector = _vectorize(compact)
                for _, existing in fields:
                    if _cosine(compact_vector, _vectorize(existing)) >= 0.9:
                        return
            fields.append((label, compact))

        feedback = _safe_text(
            card.sections.get("Feedback")
            or card.sections.get("Test")
            or card.sections.get("Tests")
        )
        if feedback:
            add("Feedback", feedback, max_lines=2, force=True)
        if not fields and card.cues.get("verification_pass") is False:
            add(
                "Feedback",
                "Executable candidate failed available tests; code omitted from this LITE view.",
                force=True,
            )

        signature = self._first_matching_line(
            card.raw_text,
            r"^\s*(?:(?:async\s+)?def\s+\w+\s*\(|class\s+\w+)",
        )
        add("Interface", signature, force=True)

        summary = _safe_text(card.sections.get("Summary") or card.sections.get("Plan"))
        if not summary:
            prose = _FENCED_BLOCK_RE.sub("", card.raw_text)
            summary = _first_sentences(
                prose,
                max_sentences=2,
            )
        add("Summary", summary, max_lines=2, force=True)

        evidence = _safe_text(
            card.sections.get("Key Evidence") or card.sections.get("Evidence")
        )
        if not evidence:
            evidence = self._code_evidence_span(card.raw_text)
        add(
            "Key Evidence",
            evidence,
            max_lines=2,
        )

        deliverable = _safe_text(
            card.sections.get("Deliverable")
            or card.sections.get("Action")
            or card.sections.get("Fix")
        )
        add("Deliverable", deliverable, force=True)
        final_answer = _safe_text(card.sections.get("Final Answer") or card.answer)
        add("Answer", final_answer, force=True)
        # Failed candidates retain feedback and prose, never their executable.
        if incomplete_evidence and card.cues.get("verification_pass") is not False:
            return ""
        if not fields:
            return ""
        return "[LITE]\n" + "\n".join(
            f"{label}: {value}" for label, value in fields[:4]
        )

    def _code_evidence_span(self, text: str) -> str:
        prose = _FENCED_BLOCK_RE.sub("", _safe_text(text))
        candidates: list[tuple[int, int, str]] = []
        heading = ""
        for line in prose.splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            if re.match(r"^#{1,6}\s", clean) or re.fullmatch(
                r"\*\*[^*]+\*\*:?", clean
            ):
                heading = clean
                continue
            context = heading + " " + clean
            # Concrete equations and corrections outrank generic complexity notes.
            priority = (
                3 if _CODE_EQUATION_RE.search(clean)
                else 2 if _CODE_CORRECTION_RE.search(context)
                else 1 if _CODE_EVIDENCE_RE.search(context)
                else 0
            )
            if priority:
                candidates.append((-priority, len(candidates), clean))
            heading = ""
        selected = sorted(candidates)[:2]
        return "\n".join(item[2] for item in sorted(selected, key=lambda item: item[1]))

    def _qa_lite_view(self, *, card: MessageSieveCard) -> str:
        cleaned_lines: list[str] = []
        for line in card.raw_text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if re.search(r"^final\s+answer\s*:", clean, re.IGNORECASE):
                continue
            if re.search(
                r"give\s+a\s+brief|end\s+with\s+a\s+final\s+line|exactly\s+in\s+this\s+format",
                clean,
                re.IGNORECASE,
            ):
                continue
            cleaned_lines.append(clean)
        cleaned_text = "\n".join(cleaned_lines)
        summary = _safe_text(card.sections.get("Summary"))
        if not summary:
            summary = self._first_evidence_span(cleaned_text) or _trim_lines(
                cleaned_text, max_lines=2
            )
        summary = _trim_lines(summary, max_lines=2)
        fields: list[str] = []
        if summary:
            fields.append(f"Summary: {summary}")
        if card.answer:
            fields.append(f"Final Answer: {card.answer}")
        if not fields:
            return ""
        return "[LITE]\n" + "\n".join(fields)

    def _minimal_qa_lite_view(self, *, card: MessageSieveCard) -> str:
        fields: list[str] = []
        evidence = self._first_evidence_span(card.raw_text)
        if evidence:
            fields.append(f"Summary: {_trim_lines(evidence, max_lines=2)}")
        if card.answer:
            fields.append(f"Final Answer: {card.answer}")
        if not fields:
            first = _trim_lines(card.raw_text, max_lines=1)
            if first:
                fields.append(f"Summary: {first}")
        if not fields:
            return ""
        return "[LITE]\n" + "\n".join(fields[:2])

    @staticmethod
    def _first_matching_line(text: str, pattern: str) -> str:
        regex = re.compile(pattern, re.IGNORECASE)
        for line in text.splitlines():
            clean = line.strip()
            if clean and regex.search(clean):
                return _trim_lines(clean, max_lines=1)
        return ""

    @staticmethod
    def _first_evidence_span(text: str) -> str:
        for line in text.splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            if _EVIDENCE_MARKER_RE.search(clean) or _ANSWER_MARKER_RE.search(clean):
                return _trim_lines(clean, max_lines=1)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            clean = sentence.strip()
            if clean:
                return _trim_lines(clean, max_lines=1)
        return ""

    def _math_evidence_span(self, text: str, *, final_answer: str) -> str:
        candidates: list[tuple[bool, bool, int, list[str]]] = []
        block: list[str] = []
        closing = ""
        context = ""

        def add(lines: list[str]) -> None:
            value = "\n".join(lines)
            if not lines or len(lines) > 6 or value.count("{") != value.count("}"):
                return
            if final_answer and value.strip("$ ") in {
                final_answer, r"\boxed{" + final_answer + "}"
            }:
                return
            mathematical = bool(_MATH_FORMULA_RE.search(value))
            checked = bool(_MATH_CHECK_RE.search(context + " " + value))
            if not mathematical and value.endswith(":"):
                return
            if (
                mathematical or _MATH_EVIDENCE_MARKER_RE.search(value)
                or checked or _EVIDENCE_MARKER_RE.search(value)
            ):
                candidates.append((checked and mathematical, mathematical, len(candidates), lines))

        for line in _safe_text(text).splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            if closing:
                if clean == closing:
                    add(block + [clean] if closing.startswith(r"\end{") else block)
                    block, closing = [], ""
                    context = ""
                else:
                    block.append(clean)
                continue
            if clean in {"$$", r"\["}:
                closing = "$$" if clean == "$$" else r"\]"
                continue
            environment = re.match(r"\\begin\{([^}]+)\}", clean)
            if environment:
                end = r"\end{" + environment.group(1) + "}"
                if end not in clean:
                    block, closing = [clean], end
                    continue
            if re.match(r"^#{1,6}\s", clean) or re.fullmatch(r"\*\*[^*]+\*\*:?", clean):
                context = clean
                continue
            if re.match(r"^(?:the answer is|final(?: answer)?\s*:)", clean, re.I):
                continue
            add([clean])
            context = clean

        selected: list[tuple[int, list[str]]] = []
        used = 0
        seen: set[str] = set()
        # Prefer complete verification equations, then later derivation steps.
        for _, _, index, lines in sorted(candidates, reverse=True):
            key = re.sub(r"\s+", " ", "\n".join(lines))
            if key not in seen and used + len(lines) <= 6:
                selected.append((index, lines))
                used += len(lines)
                seen.add(key)
        return "\n".join(line for _, lines in sorted(selected) for line in lines)
