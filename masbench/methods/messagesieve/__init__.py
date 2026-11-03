from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from ...core.agent import AgentProfile
from ...core.message import Message
from ..base import BaseMethod
from .card import CardBuilder, MessageSieveCard, _safe_text
from .gate import AdmissionGate
from .relation import RelationMapper
from .view import ViewProjector

_DATASET_FAMILIES = {
    "gpqa_diamond": "qa",
    "aqua": "math",
    "math": "math",
    "livecodebench_codegen_lite": "code",
    "mbppplus": "code",
}


def task_family_for_dataset(dataset_id: str) -> str:
    normalized = str(dataset_id or "").strip().lower()
    if not normalized:
        raise ValueError("dataset_id must be explicit for MessageSieve")
    try:
        return _DATASET_FAMILIES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_DATASET_FAMILIES))
        raise ValueError(
            f"Unsupported MessageSieve dataset_id={dataset_id!r}; supported IDs: {supported}"
        ) from exc


class MessageSieveMethod(
    CardBuilder, RelationMapper, AdmissionGate, ViewProjector, BaseMethod
):
    """Run the four MessageSieve stages for each receiver."""

    method_id = "messagesieve"
    display_name = "MessageSieve"

    def __init__(self) -> None:
        super().__init__()
        self.task_family = ""
        self.previous_pair_vectors: dict[tuple[str, str], Mapping[str, float]] = {}
        self.cards: dict[str, MessageSieveCard] = {}

    def configure(self, params: Mapping[str, Any] | None) -> None:
        supplied = dict(params or {})
        family = str(supplied.get("task_family") or "").strip().lower()
        if family not in {"qa", "math", "code"}:
            raise ValueError("task_family must be one of: code, math, qa")
        unknown = set(supplied) - {"dataset_id", "task_family", "base_graph_variant"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported MessageSieve parameter(s): {names}")
        if supplied.get("base_graph_variant", "full") != "full":
            raise ValueError("MessageSieve uses the full candidate communication graph")

        resolved = {"task_family": family}
        dataset_id = str(supplied.get("dataset_id") or "").strip().lower()
        if dataset_id:
            resolved["dataset_id"] = dataset_id
        if "base_graph_variant" in supplied:
            resolved["base_graph_variant"] = supplied["base_graph_variant"]
        super().configure(resolved)
        self.task_family = family

    def reset_for_run(self) -> None:
        self.previous_pair_vectors = {}
        self.cards = {}

    def filter_messages(
        self,
        *,
        preset,
        receiver: AgentProfile,
        round_index: int,
        task_input: str,
        candidates: Sequence[Message],
    ) -> list[Message]:
        messages = [message for message in candidates if _safe_text(message.content)]
        if not messages:
            return []
        bucket = self._bucket()
        cards = {}
        for message in messages:
            card = self.cards.get(message.message_id)
            if card is not None:
                cards[message.message_id] = card
        relations = self._map_relations(
            messages=messages,
            cards=cards,
            receiver=receiver,
            bucket=bucket,
        )
        realized_by_id: dict[str, Message] = {}
        for message in messages:
            card = cards.get(message.message_id)
            if card is None:
                continue
            relation = relations[message.message_id]
            admission = self._admit(
                message=message,
                card=card,
                relation=relation,
                bucket=bucket,
            )
            if not admission.admitted:
                continue
            projected = self._project_view(
                card=card, relation=relation, admission=admission
            )
            realized = replace(message, content=projected)
            realized_by_id[message.message_id] = realized
        return [
            realized_by_id[message.message_id]
            for message in messages
            if message.message_id in realized_by_id
        ]

    def filter_judge_messages(
        self,
        *,
        preset,
        judge: AgentProfile,
        task_input: str,
        candidates: Sequence[Message],
    ) -> list[Message]:
        return [message for message in candidates if _safe_text(message.content)]

    def _bucket(self) -> str:
        if not self.task_family:
            raise RuntimeError("MessageSieve must be configured before routing")
        return self.task_family


__all__ = ["MessageSieveMethod", "task_family_for_dataset"]
