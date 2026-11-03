from __future__ import annotations

from dataclasses import dataclass, replace

from ...core.message import Message
from .card import MessageSieveCard
from .relation import MessageSieveRelation

GATE_THRESHOLD = 0.0
COVERAGE_THRESHOLD = 0.7
RESIDUAL_SUPPORT_THRESHOLD = 0.3


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    support_score: float
    risk_score: float
    margin: float
    support_predicates: tuple[str, ...]
    risk_predicates: tuple[str, ...]


class AdmissionGate:
    """Admit worker messages using support, risk, and route eligibility."""

    @staticmethod
    def _family_relation_support(*, bucket: str, reason: str) -> list[str]:
        support = [reason]
        if bucket == "qa":
            if reason in {"disagreement_relation", "novelty_relation"}:
                support.append("useful_disagreement_or_novelty")
            if reason == "temporal_self_relation":
                support.append("stable_self_temporal_memory")
        elif bucket == "math":
            if reason in {
                "math_self_temporal_anchor",
                "math_disagreement_representative",
            }:
                support.append("math_high_fidelity_anchor")
        if bucket == "math" and reason in {
            "math_bounded_full_anchor",
            "math_bounded_secondary_anchor",
        }:
            support.append("math_high_fidelity_anchor")
        if bucket == "code" and reason == "code_failed_feedback_relation":
            support.append("receiver_relevant_feedback")
        return support

    def _admit(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation: MessageSieveRelation,
        bucket: str,
    ) -> AdmissionDecision:
        support: list[str] = []
        risk: list[str] = []
        cues = card.cues
        if relation.relation_reason:
            support.extend(
                self._family_relation_support(
                    bucket=bucket, reason=relation.relation_reason
                )
            )
        else:
            risk.append(f"not_in_{bucket}_relation_plan")
        local_signal = bool(
            cues.get("has_evidence") or cues.get("has_feedback") or cues.get("has_code")
        )
        if cues.get("has_code") or (bucket == "code" and cues.get("has_sections")):
            support.append("protected_artifact")
        answer_has_local_value = (
            cues.get("has_evidence")
            or relation.conflict > 0.0
            or relation.receiver_value >= 0.45
        )
        answer_coverage_limit = 0.45
        if (
            cues.get("has_answer")
            and relation.coverage < answer_coverage_limit
            and answer_has_local_value
        ):
            support.append("unique_or_minority_answer_evidence")
        temporal_has_value = (
            local_signal
            or relation.conflict > 0.0
            or (
                bucket == "code"
                and cues.get("has_sections")
                and relation.receiver_value >= 0.3
            )
        )
        if (
            message.channel == "temporal"
            and card.freshness >= 0.4
            and not cues.get("no_update")
            and temporal_has_value
        ):
            support.append("fresh_temporal_memory")
        cross_support_threshold = 0.6
        cross_support_has_value = (
            local_signal
            or relation.conflict > 0.0
            or relation.receiver_value >= 0.45
            or (bucket == "code" and cues.get("has_sections"))
        )
        if relation.support >= cross_support_threshold and cross_support_has_value:
            support.append("cross_sender_support")
        if relation.receiver_value >= 0.35 and (
            cues.get("has_feedback") or cues.get("has_evidence")
        ):
            support.append("receiver_relevant_feedback")
        if relation.conflict > 0.0 and (
            cues.get("has_evidence") or cues.get("has_answer")
        ):
            support.append("useful_disagreement")

        if cues.get("low_content"):
            risk.append("low_content_text")
        if message.channel == "temporal" and (
            card.freshness < 0.2 or cues.get("no_update")
        ):
            risk.append("stale_update")
        if (
            relation.coverage >= COVERAGE_THRESHOLD
            and relation.support < RESIDUAL_SUPPORT_THRESHOLD
        ):
            risk.append("covered_memory")
        if relation.coverage >= 0.55 and not cues.get("has_code"):
            risk.append("covered_worker_context")
        if relation.coverage >= COVERAGE_THRESHOLD and card.answer:
            risk.append("repeated_same_answer_content")
        if (
            cues.get("has_answer")
            and not cues.get("has_evidence")
            and relation.receiver_value < 0.35
        ):
            risk.append("answer_without_local_evidence")
        if (
            message.channel == "temporal"
            and not cues.get("has_code")
            and (
                not local_signal
                or relation.coverage >= 0.35
                or relation.receiver_value < 0.35
            )
        ):
            risk.append("weak_worker_temporal_update")
        if relation.receiver_value < 0.2:
            risk.append("receiver_mismatch")
        if relation.conflict > 0.0 and not cues.get("has_evidence"):
            risk.append("answer_conflict_without_evidence")
        if cues.get("format_risk"):
            risk.append("format_risk")

        if not relation.relation_reason:
            support = []
        support_predicates = tuple(sorted(set(support)))
        risk_predicates = tuple(sorted(set(risk)))
        support_score = float(len(support_predicates))
        risk_score = float(len(risk_predicates))
        margin = support_score - risk_score
        admitted = bool(relation.relation_reason) and margin >= GATE_THRESHOLD
        admission = AdmissionDecision(
            admitted=admitted,
            support_score=support_score,
            risk_score=risk_score,
            margin=margin,
            support_predicates=support_predicates,
            risk_predicates=risk_predicates,
        )

        if not relation.route_excluded:
            return admission
        risks = tuple(
            sorted({*admission.risk_predicates, "same_source_artifact_repetition"})
        )
        risk_score = max(float(len(risks)), admission.support_score + 1.0)
        return replace(
            admission,
            admitted=False,
            risk_score=risk_score,
            margin=admission.support_score - risk_score,
            risk_predicates=risks,
        )
