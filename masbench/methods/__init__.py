from __future__ import annotations

from typing import Sequence

from ..core.agent import AgentProfile
from ..presets.catalog import SINGLE_PRESETS
from .base import BaseMethod, MethodSpec
from .messagesieve import MessageSieveMethod


class SingleMethod(BaseMethod):
    method_id = "single"
    display_name = "Single"

    def select_worker_agents(
        self, agents: Sequence[AgentProfile]
    ) -> list[AgentProfile]:
        if not agents:
            return []
        dataset_id = str(agents[0].metadata.get("dataset_id") or "").strip().lower()
        try:
            role_name, system_prompt = SINGLE_PRESETS[dataset_id]
        except KeyError as exc:
            raise KeyError(
                f"Single is not configured for dataset: {dataset_id}"
            ) from exc
        return [
            AgentProfile(
                agent_id="direct_solver",
                display_name=role_name,
                role_name=role_name,
                system_prompt=system_prompt,
                metadata={
                    "dataset_id": dataset_id,
                },
            )
        ]

    def use_judge(self) -> bool:
        return False

    def allow_temporal_edge(self, *, sender, receiver, round_index: int) -> bool:
        return False


class FullMASMethod(BaseMethod):
    method_id = "full_connected_mas"
    display_name = "Full-MAS"


METHOD_REGISTRY = {
    "single": MethodSpec(
        method_id="single",
        display_name="Single",
        description="One direct-answer agent without a judge or inter-agent messages.",
        method_cls=SingleMethod,
    ),
    "full_connected_mas": MethodSpec(
        method_id="full_connected_mas",
        display_name="Full-MAS",
        description="The two-round multi-agent system with all candidate messages passed through.",
        method_cls=FullMASMethod,
    ),
    "messagesieve": MethodSpec(
        method_id="messagesieve",
        display_name="MessageSieve",
        description="Receiver-local MessageSieve with the paper's default family policy.",
        method_cls=MessageSieveMethod,
    ),
}

METHOD_ALIASES = {
    "single": "single",
    "full_mas": "full_connected_mas",
    "fullmas": "full_connected_mas",
    "full_connected_mas": "full_connected_mas",
    "messagesieve": "messagesieve",
    "message_sieve": "messagesieve",
}


def canonical_method_id(method_id: str) -> str:
    key = method_id.strip().lower().replace(" ", "_").replace("-", "_")
    canonical = METHOD_ALIASES.get(key)
    if canonical is None:
        raise KeyError(
            f"Unknown method: {method_id}. Available: single, full_connected_mas, messagesieve"
        )
    return canonical


def get_method(method_id: str) -> BaseMethod:
    return METHOD_REGISTRY[canonical_method_id(method_id)].method_cls()


__all__ = [
    "BaseMethod",
    "METHOD_REGISTRY",
    "MethodSpec",
    "canonical_method_id",
    "get_method",
    "SingleMethod",
]
