from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from ..core.agent import AgentProfile
from ..core.message import Message


class BaseMethod:
    method_id: str = ""
    display_name: str = ""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def select_worker_agents(self, agents: Sequence[AgentProfile]) -> List[AgentProfile]:
        return list(agents)

    def use_judge(self) -> bool:
        return True

    def filter_messages(
        self,
        *,
        preset,
        receiver: AgentProfile,
        round_index: int,
        task_input: str,
        candidates: Sequence[Message],
    ) -> List[Message]:
        return list(candidates)

    def filter_judge_messages(
        self,
        *,
        preset,
        judge: AgentProfile,
        task_input: str,
        candidates: Sequence[Message],
    ) -> List[Message]:
        return list(candidates)

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
        return message

    def configure(self, params: Mapping[str, Any] | None) -> None:
        self.params = dict(params or {})

    def reset_for_run(self) -> None:
        return None

    def allow_spatial_edge(
        self,
        *,
        sender: AgentProfile,
        receiver: AgentProfile,
        round_index: int,
    ) -> bool:
        del round_index
        return sender.agent_id != receiver.agent_id

    def allow_temporal_edge(
        self,
        *,
        sender: AgentProfile,
        receiver: AgentProfile,
        round_index: int,
    ) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    display_name: str
    description: str
    method_cls: type[BaseMethod]
