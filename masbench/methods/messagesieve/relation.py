from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...core.agent import AgentProfile
from ...core.message import Message
from .card import (
    MessageSieveCard,
    _clamp,
    _clean_extracted_answer,
    _cosine,
    _safe_text,
    _tokens,
    _vectorize,
)

LEXICAL_WEIGHT = 0.8
ANSWER_WEIGHT = 0.6
TOPOLOGY_PENALTY = 0.4
TOP_K_SUPPORT = 2
DUPLICATE_SIMILARITY = 0.8
LOW_NOVELTY_THRESHOLD = 0.6


@dataclass(frozen=True, slots=True)
class MessageSieveRelation:
    coverage: float
    support: float
    conflict: float
    receiver_value: float
    novelty: float = 0.0
    relation_score: int = 0
    relation_reason: str = ""
    # Explicit route exclusions take precedence over support.
    route_excluded: bool = False


def _answer_key(answer: str, *, bucket: str) -> str:
    text = _clean_extracted_answer(
        answer, normalize_wrapped_option=bucket == "math"
    ).lower()
    text = re.sub(r"\s+", " ", text).strip()
    if bucket == "qa":
        match = re.search(r"\b([a-e])\b", text)
        if match:
            return match.group(1)
    return text


def _agent_index(agent_id: str) -> int | None:
    match = re.search(r"worker_(\d+)", _safe_text(agent_id))
    if not match:
        return None
    return int(match.group(1))


class RelationMapper:
    """Map receiver-local relations and select task-family representatives."""

    def _map_relations(
        self,
        *,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        receiver: AgentProfile,
        bucket: str,
    ) -> dict[str, MessageSieveRelation]:
        answer_counts = Counter(
            _answer_key(card.answer, bucket=card.bucket)
            for card in cards.values()
            if card.answer
        )
        relation_values: dict[str, dict[str, Any]] = {}
        context_vector = self._context_vector(
            receiver=receiver, messages=messages, cards=cards
        )
        for message in messages:
            card = cards.get(message.message_id)
            if card is None:
                continue
            coverage_values: list[float] = []
            support_values: list[float] = []
            conflict_values: list[float] = []
            for other in messages:
                if other.message_id == message.message_id:
                    continue
                other_card = cards.get(other.message_id)
                if other_card is None:
                    continue
                lexical = _cosine(
                    card.evidence_vector,
                    other_card.evidence_vector,
                )
                agreement = self._agreement(card, other_card)
                repeated = self._topology_repetition(
                    message,
                    other,
                    card,
                    other_card,
                )
                evidence_overlap = _clamp(
                    LEXICAL_WEIGHT * lexical + ANSWER_WEIGHT * agreement
                )
                support_signal = _clamp(evidence_overlap - TOPOLOGY_PENALTY * repeated)
                coverage_signal = _clamp(evidence_overlap * repeated)
                support_values.append(support_signal)
                coverage_values.append(coverage_signal)
                conflict_values.append(self._conflict(card, other_card))
            support_values.sort(reverse=True)
            top_k = TOP_K_SUPPORT
            support = (
                sum(support_values[:top_k]) / min(len(support_values), top_k)
                if support_values
                else 0.0
            )
            coverage = max(coverage_values, default=0.0)
            conflict = max(conflict_values, default=0.0)
            receiver_value = self._receiver_value(
                message=message, card=card, receiver=receiver
            )
            if (
                card.answer
                and answer_counts[_answer_key(card.answer, bucket=card.bucket)] == 1
            ):
                support = max(
                    support,
                    0.6,
                )
            novelty = self._message_novelty(
                card=card,
                context_vector=context_vector,
            )
            relation_score = self._relation_score(
                message=message, card=card, receiver=receiver, novelty=novelty
            )
            relation_values[message.message_id] = {
                "coverage": _clamp(coverage),
                "support": _clamp(support),
                "conflict": _clamp(conflict),
                "receiver_value": _clamp(receiver_value),
                "novelty": _clamp(novelty),
                "relation_score": relation_score,
                "relation_reason": "",
            }
        relation_plan = self._build_family_relation_plan(
            bucket=bucket,
            messages=messages,
            cards=cards,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )
        for message_id, reason in relation_plan.items():
            values = relation_values.get(message_id)
            if values is None:
                continue
            values["relation_reason"] = reason
        relations: dict[str, MessageSieveRelation] = {}
        for message_id, values in relation_values.items():
            relations[message_id] = MessageSieveRelation(
                coverage=float(values["coverage"]),
                support=float(values["support"]),
                conflict=float(values["conflict"]),
                receiver_value=float(values["receiver_value"]),
                novelty=float(values["novelty"]),
                relation_score=int(values["relation_score"]),
                relation_reason=_safe_text(values["relation_reason"]),
                route_excluded=(
                    _safe_text(values["relation_reason"])
                    == "code_same_source_repetition"
                ),
            )
        return relations

    def _build_family_relation_plan(
        self,
        *,
        bucket: str,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        if bucket == "code":
            return self._code_relation_plan(
                messages=messages,
                cards=cards,
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            )
        if bucket == "math":
            return self._math_relation_plan(
                messages=messages,
                cards=cards,
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            )
        return self._qa_relation_plan(
            messages=messages,
            cards=cards,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )

    def _qa_relation_plan(
        self,
        *,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        rows = [
            (message, cards[message.message_id])
            for message in messages
            if message.message_id in cards
            and not cards[message.message_id].cues.get("no_update")
        ]
        spatial_rows = [
            (message, card) for message, card in rows if message.channel == "spatial"
        ]
        temporal_rows = [
            (message, card) for message, card in rows if message.channel == "temporal"
        ]
        plan = self._qa_spatial_relation_plan(
            rows=spatial_rows,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )
        plan.update(
            self._qa_temporal_relation_plan(
                rows=temporal_rows,
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
                all_rows=rows,
            )
        )
        return plan

    def _code_relation_plan(
        self,
        *,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        rows = [
            (message, cards[message.message_id])
            for message in messages
            if message.message_id in cards
            and not cards[message.message_id].cues.get("no_update")
        ]
        if not rows:
            return {}
        ranked = sorted(
            rows,
            key=lambda row: self._code_receiver_rank(
                message=row[0],
                card=row[1],
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            ),
            reverse=True,
        )
        plan: dict[str, str] = {}
        selected_vectors: list[Mapping[str, float]] = []
        selected_code_vectors_by_source: dict[str, list[Mapping[str, float]]] = {}
        noncode_lite_kept = 0
        noncode_lite_limit = 3
        for message, card in ranked:
            verification_pass = card.cues.get("verification_pass")
            if card.cues.get("has_code"):
                source_vectors = selected_code_vectors_by_source.get(message.source, [])
                same_source_repetition = (
                    verification_pass is not True
                    and self._redundant_with_selected(
                        card=card,
                        selected_vectors=source_vectors,
                        context_vector=context_vector,
                    )
                )
                has_failed_feedback = verification_pass is False and bool(
                    _safe_text(card.sections.get("Feedback"))
                )
                if same_source_repetition and not (
                    has_failed_feedback
                    and not any(
                        plan.get(item.message_id) == "code_failed_feedback_relation"
                        for item, _ in rows
                        if item.source == message.source
                    )
                ):
                    plan[message.message_id] = "code_same_source_repetition"
                    continue
                globally_redundant = self._redundant_with_selected(
                    card=card,
                    selected_vectors=selected_vectors,
                    context_vector=context_vector,
                )
                if verification_pass is False:
                    reason = "code_failed_feedback_relation"
                elif globally_redundant and not source_vectors:
                    reason = "code_cross_source_artifact_relation"
                else:
                    reason = "code_artifact_relation"
                plan[message.message_id] = reason
                if card.evidence_vector:
                    selected_vectors.append(card.evidence_vector)
                    selected_code_vectors_by_source.setdefault(
                        message.source, []
                    ).append(card.evidence_vector)
                continue

            if noncode_lite_kept >= noncode_lite_limit:
                continue
            if not self._passes_code_receiver_gate(
                message=message,
                card=card,
                relation_values=relation_values,
                context_vector=context_vector,
            ):
                continue
            if self._redundant_with_selected(
                card=card,
                selected_vectors=selected_vectors,
                context_vector=context_vector,
            ):
                continue
            if message.channel == "temporal":
                reason = (
                    "temporal_self_relation"
                    if message.source == message.target == receiver.agent_id
                    else "temporal_representative_relation"
                )
            else:
                novelty = float(
                    relation_values.get(message.message_id, {}).get("novelty", 0.0)
                )
                reason = (
                    "novelty_relation"
                    if novelty >= LOW_NOVELTY_THRESHOLD
                    else "spatial_representative_relation"
                )
            plan.setdefault(message.message_id, reason)
            if card.evidence_vector:
                selected_vectors.append(card.evidence_vector)
                noncode_lite_kept += 1
        return plan

    def _math_anchor_plan(
        self,
        *,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        rows = [
            (message, cards[message.message_id])
            for message in messages
            if message.message_id in cards
            and not cards[message.message_id].cues.get("no_update")
        ]
        if not rows:
            return {}
        plan: dict[str, str] = {}

        for message, card in rows:
            if (
                message.channel == "temporal"
                and message.source == message.target == receiver.agent_id
                and (card.answer or card.cues.get("has_evidence"))
            ):
                plan[message.message_id] = "math_self_temporal_anchor"

        answer_clusters = self._answer_clusters(rows=rows)
        answered_clusters = [
            cluster for cluster in answer_clusters if cluster.get("answer")
        ]
        has_disagreement = len(answered_clusters) > 1
        if has_disagreement:
            for cluster in answered_clusters:
                representative = self._best_representative(
                    rows=cluster["items"],
                    relation_values=relation_values,
                    receiver=receiver,
                    context_vector=context_vector,
                )
                if representative is not None:
                    plan.setdefault(
                        representative[0].message_id, "math_disagreement_representative"
                    )
            return plan

        candidates = [
            (message, card)
            for message, card in rows
            if message.message_id not in plan
            and message.source != receiver.agent_id
            and (card.answer or card.cues.get("has_evidence") or card.sections)
        ]
        ranked = sorted(
            candidates,
            key=lambda row: self._math_context_rank(
                message=row[0],
                card=row[1],
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            ),
            reverse=True,
        )
        for message, _card in ranked[:1]:
            plan.setdefault(message.message_id, "math_consensus_context")
        return plan

    def _math_relation_plan(
        self,
        *,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        plan = self._math_anchor_plan(
            messages=messages,
            cards=cards,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )
        anchor_reasons = {
            "math_self_temporal_anchor",
            "math_disagreement_representative",
        }
        anchor_rows = [
            (message, cards[message.message_id])
            for message in messages
            if message.message_id in cards
            and plan.get(message.message_id) in anchor_reasons
        ]
        if not anchor_rows:
            return plan

        self_rows = [
            row
            for row in anchor_rows
            if row[0].channel == "temporal"
            and row[0].source == row[0].target == receiver.agent_id
        ]
        ranked_pool = self_rows or anchor_rows
        ranked = sorted(
            ranked_pool,
            key=lambda row: self._representative_rank(
                message=row[0],
                card=row[1],
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            ),
            reverse=True,
        )
        full_ids = {message.message_id for message, _card in ranked[:1]}
        for message, _card in anchor_rows:
            plan[message.message_id] = (
                "math_bounded_full_anchor"
                if message.message_id in full_ids
                else "math_bounded_secondary_anchor"
            )
        return plan

    def _math_context_rank(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> tuple[int, int, int, float, float, float, float, int]:
        values = relation_values.get(message.message_id, {})
        sender_role = _safe_text(card.topology.get("sender_role")).lower()
        verifier_role = (
            1 if re.search(r"\b(verifier|analyst|solver)\b", sender_role) else 0
        )
        return (
            1 if card.cues.get("has_evidence") else 0,
            1 if card.sections else 0,
            verifier_role,
            float(values.get("receiver_value", 0.0)),
            float(values.get("novelty", 0.0)),
            float(values.get("support", 0.0)),
            self._topology_need(message=message, receiver=receiver)
            + _cosine(card.evidence_vector, context_vector),
            -card.raw_tokens,
        )

    def _code_receiver_rank(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> tuple[int, int, int, int, int, float, float, float, float, int]:
        verification_pass = card.cues.get("verification_pass")
        values = relation_values.get(message.message_id, {})
        return (
            1 if verification_pass is True else 0,
            1 if card.cues.get("has_code") and verification_pass is not False else 0,
            self._code_algorithm_note_role_bonus(card=card),
            1 if card.sections else 0,
            1 if card.answer else 0,
            self._topology_need(message=message, receiver=receiver),
            _cosine(card.evidence_vector, context_vector),
            float(values.get("novelty", 0.0)),
            float(values.get("support", 0.0)),
            -card.raw_tokens,
        )

    def _code_algorithm_note_role_bonus(self, *, card: MessageSieveCard) -> int:
        if self.task_family != "code":
            return 0
        if card.cues.get("has_code"):
            return 0
        sender_role = _safe_text(card.topology.get("sender_role")).lower()
        receiver_role = _safe_text(card.topology.get("receiver_role")).lower()
        algorithm_sender = bool(
            re.search(r"\b(algorithm|designer|design|planner)\b", sender_role)
        )
        implementation_receiver = bool(
            re.search(r"\b(programming|expert|bug|fix|test|analyst)\b", receiver_role)
        )
        if algorithm_sender and implementation_receiver:
            return 1
        return 0

    def _passes_code_receiver_gate(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation_values: Mapping[str, Mapping[str, Any]],
        context_vector: Mapping[str, float],
    ) -> bool:
        values = relation_values.get(message.message_id, {})
        if card.sections or card.answer or card.cues.get("has_feedback"):
            return True
        if float(values.get("support", 0.0)) >= 0.45:
            return True
        if (
            float(values.get("novelty", 0.0)) >= 0.45
            and _cosine(card.evidence_vector, context_vector) >= 0.20
        ):
            return True
        return False

    def _redundant_with_selected(
        self,
        *,
        card: MessageSieveCard,
        selected_vectors: Sequence[Mapping[str, float]],
        context_vector: Mapping[str, float],
    ) -> bool:
        if not selected_vectors or not card.evidence_vector:
            return False
        novelty = self._message_novelty(card=card, context_vector=context_vector)
        similarity = max(
            (_cosine(card.evidence_vector, vector) for vector in selected_vectors),
            default=0.0,
        )
        return similarity >= DUPLICATE_SIMILARITY and novelty < LOW_NOVELTY_THRESHOLD

    def _qa_spatial_relation_plan(
        self,
        *,
        rows: Sequence[tuple[Message, MessageSieveCard]],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> dict[str, str]:
        plan: dict[str, str] = {}
        if not rows:
            return plan
        answer_clusters = self._answer_clusters(rows=rows)
        answered_clusters = [
            cluster for cluster in answer_clusters if cluster["answer"]
        ]
        if len(answered_clusters) > 1:
            for cluster in answered_clusters:
                representative = self._best_representative(
                    rows=cluster["items"],
                    relation_values=relation_values,
                    receiver=receiver,
                    context_vector=context_vector,
                )
                if representative is not None:
                    plan[representative[0].message_id] = "disagreement_relation"
            return plan
        representative = self._best_representative(
            rows=rows,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )
        if representative is None:
            return plan
        message, card = representative
        if not self._is_sendable_representative(
            message=message,
            card=card,
            relation_values=relation_values,
        ):
            return plan
        if self._topology_need(message=message, receiver=receiver) < 0.50:
            return plan
        novelty = float(relation_values.get(message.message_id, {}).get("novelty", 0.0))
        plan[message.message_id] = (
            "novelty_relation"
            if novelty >= LOW_NOVELTY_THRESHOLD
            else "spatial_representative_relation"
        )
        return plan

    def _qa_temporal_relation_plan(
        self,
        *,
        rows: Sequence[tuple[Message, MessageSieveCard]],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
        all_rows: Sequence[tuple[Message, MessageSieveCard]],
    ) -> dict[str, str]:
        plan: dict[str, str] = {}
        if not rows:
            return plan
        answer_clusters = self._answer_clusters(rows=all_rows)
        answered_clusters = [
            cluster for cluster in answer_clusters if cluster["answer"]
        ]
        has_disagreement = len(answered_clusters) > 1
        if has_disagreement:
            for cluster in answered_clusters:
                representative = self._best_representative(
                    rows=[
                        row for row in cluster["items"] if row[0].channel == "temporal"
                    ],
                    relation_values=relation_values,
                    receiver=receiver,
                    context_vector=context_vector,
                )
                if representative is None:
                    continue
                message, card = representative
                if message.source == message.target == receiver.agent_id:
                    if self._stable_self_temporal(
                        message=message,
                        card=card,
                        answer_clusters=answer_clusters,
                        has_disagreement=True,
                    ):
                        plan[message.message_id] = "temporal_self_relation"
            return plan
        self_rows = [
            (message, card)
            for message, card in rows
            if message.source == message.target == receiver.agent_id and card.answer
        ]
        representative = self._best_representative(
            rows=self_rows,
            relation_values=relation_values,
            receiver=receiver,
            context_vector=context_vector,
        )
        if representative is None:
            return plan
        message, card = representative
        if self._stable_self_temporal(
            message=message,
            card=card,
            answer_clusters=answer_clusters,
            has_disagreement=False,
        ):
            plan[message.message_id] = "temporal_self_relation"
        return plan

    def _best_representative(
        self,
        *,
        rows: Sequence[tuple[Message, MessageSieveCard]],
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> tuple[Message, MessageSieveCard] | None:
        usable = [
            (message, card) for message, card in rows if not card.cues.get("no_update")
        ]
        if not usable:
            return None
        return max(
            usable,
            key=lambda row: self._representative_rank(
                message=row[0],
                card=row[1],
                relation_values=relation_values,
                receiver=receiver,
                context_vector=context_vector,
            ),
        )

    def _representative_rank(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation_values: Mapping[str, Mapping[str, Any]],
        receiver: AgentProfile,
        context_vector: Mapping[str, float],
    ) -> tuple[int, int, int, int, float, float, float, int]:
        values = relation_values.get(message.message_id, {})
        return (
            1 if card.cues.get("has_code") else 0,
            1 if card.answer else 0,
            1 if card.sections else 0,
            int(values.get("relation_score", 0)),
            float(values.get("novelty", 0.0)),
            self._topology_need(message=message, receiver=receiver),
            _cosine(card.evidence_vector, context_vector),
            -card.raw_tokens,
        )

    def _is_sendable_representative(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        relation_values: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        values = relation_values.get(message.message_id, {})
        if card.cues.get("no_update"):
            return False
        if card.cues.get("has_code") or card.answer:
            return True
        if card.sections and int(values.get("relation_score", 0)) >= 3:
            return True
        return float(values.get("novelty", 0.0)) >= LOW_NOVELTY_THRESHOLD

    def _stable_self_temporal(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        answer_clusters: Sequence[Mapping[str, Any]],
        has_disagreement: bool,
    ) -> bool:
        if card.cues.get("no_update") or not card.answer:
            return False
        cluster_sizes = [
            len(cluster["items"])
            for cluster in answer_clusters
            if cluster.get("answer")
        ]
        if not cluster_sizes:
            return False
        self_cluster_size = 0
        for cluster in answer_clusters:
            if not cluster.get("answer"):
                continue
            if any(
                peer.message_id == message.message_id for peer, _ in cluster["items"]
            ):
                self_cluster_size = len(cluster["items"])
                break
        if self_cluster_size <= 0:
            return False
        max_cluster_size = max(cluster_sizes)
        if has_disagreement:
            return self_cluster_size == max_cluster_size and self_cluster_size >= 2
        answered_messages = sum(cluster_sizes)
        if answered_messages <= 1:
            return False
        return self_cluster_size >= 2

    @staticmethod
    def _answer_clusters(
        *,
        rows: Sequence[tuple[Message, MessageSieveCard]],
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for message, card in rows:
            answer = _safe_text(card.answer)
            if not answer:
                clusters.append({"answer": "", "items": [(message, card)]})
                continue
            answer_key = _answer_key(answer, bucket=card.bucket)
            for cluster in clusters:
                if not cluster["answer"]:
                    continue
                cluster_key = _answer_key(
                    _safe_text(cluster["answer"]), bucket=card.bucket
                )
                if answer_key and answer_key == cluster_key:
                    cluster["items"].append((message, card))
                    break
            else:
                clusters.append({"answer": answer, "items": [(message, card)]})
        return clusters

    def _context_vector(
        self,
        *,
        receiver: AgentProfile,
        messages: Sequence[Message],
        cards: Mapping[str, MessageSieveCard],
    ) -> Mapping[str, float]:
        self_memory = ""
        for message in messages:
            if message.channel == "temporal" and message.source == receiver.agent_id:
                card = cards.get(message.message_id)
                if card is not None:
                    self_memory = card.raw_text
                    break
        return _vectorize(
            "\n".join(
                part
                for part in (receiver.role_name, receiver.system_prompt, self_memory)
                if part
            )
        )

    @staticmethod
    def _message_novelty(
        *, card: MessageSieveCard, context_vector: Mapping[str, float]
    ) -> float:
        if card.cues.get("no_update"):
            return 0.0
        if not card.evidence_vector or not context_vector:
            return card.freshness
        return _clamp(1.0 - _cosine(card.evidence_vector, context_vector))

    def _relation_score(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        receiver: AgentProfile,
        novelty: float,
    ) -> int:
        role = self._role_need(card=card, self_edge=message.source == message.target)
        topology = self._topology_need(message=message, receiver=receiver)
        score = 0
        score += 1 if card.relevance >= 0.35 else 0
        score += 1 if role >= 0.65 else 0
        score += 1 if topology >= 0.7 else 0
        score += 1 if card.answer else 0
        score += 1 if card.sections else 0
        score += 1 if (novelty >= 0.5 or card.freshness >= 0.5) else 0
        score += 2 if card.cues.get("has_code") else 0
        score += 1 if card.cues.get("verification_pass") is True else 0
        score -= 1 if card.cues.get("no_update") else 0
        return max(score, 0)

    def _role_need(self, *, card: MessageSieveCard, self_edge: bool) -> float:
        if self_edge:
            return 1.0
        sender_tokens = set(_tokens(_safe_text(card.topology.get("sender_role"))))
        receiver_tokens = set(_tokens(_safe_text(card.topology.get("receiver_role"))))
        if not sender_tokens or not receiver_tokens:
            return 0.62
        overlap = len(sender_tokens & receiver_tokens) / max(
            len(sender_tokens | receiver_tokens), 1
        )
        return _clamp(0.72 - 0.2 * overlap)

    def _topology_need(self, *, message: Message, receiver: AgentProfile) -> float:
        if message.channel == "temporal":
            return 1.0 if message.source == message.target else 0.72
        sender_idx = _agent_index(message.source)
        receiver_idx = _agent_index(receiver.agent_id)
        if sender_idx is None or receiver_idx is None:
            return 0.65
        if sender_idx >= receiver_idx:
            return 0.35
        return 1.0 / max(receiver_idx - sender_idx, 1)

    @staticmethod
    def _agreement(left: MessageSieveCard, right: MessageSieveCard) -> float:
        if left.answer and right.answer:
            return (
                1.0
                if _answer_key(left.answer, bucket=left.bucket)
                == _answer_key(right.answer, bucket=right.bucket)
                else 0.0
            )
        if left.cues.get("has_code") and right.cues.get("has_code"):
            return 0.5
        return 0.0

    @staticmethod
    def _conflict(left: MessageSieveCard, right: MessageSieveCard) -> float:
        if not left.answer or not right.answer:
            return 0.0
        return (
            0.0
            if _answer_key(left.answer, bucket=left.bucket)
            == _answer_key(right.answer, bucket=right.bucket)
            else 1.0
        )

    def _topology_repetition(
        self,
        left_msg: Message,
        right_msg: Message,
        left: MessageSieveCard,
        right: MessageSieveCard,
    ) -> float:
        score = 0.0
        if left_msg.source == right_msg.source:
            score += 0.3
        if left.topology.get("sender_role") == right.topology.get("sender_role"):
            score += 0.2
        if left_msg.channel == right_msg.channel:
            score += 0.2
        if left_msg.target == right_msg.target:
            score += 0.2
        if (
            left_msg.source != right_msg.source
            and left_msg.channel != right_msg.channel
        ):
            score -= 0.1
        return _clamp(score)

    def _receiver_value(
        self,
        *,
        message: Message,
        card: MessageSieveCard,
        receiver: AgentProfile,
    ) -> float:
        value = 0.55 * card.relevance
        sender_role = _safe_text(card.topology.get("sender_role")).lower()
        receiver_role = receiver.role_name.lower()
        if message.source == receiver.agent_id:
            value += 0.2
        if "critic" in receiver_role and (
            card.cues.get("has_feedback") or card.cues.get("has_evidence")
        ):
            value += 0.25
        if "judge" in receiver_role and (
            card.cues.get("has_answer") or card.cues.get("has_code")
        ):
            value += 0.35
        if (
            "program" in receiver_role
            or "bug" in receiver_role
            or "code" in receiver_role
        ):
            if card.cues.get("has_code") or card.cues.get("has_feedback"):
                value += 0.3
        if sender_role and sender_role in receiver_role:
            value += 0.1
        return _clamp(value)
