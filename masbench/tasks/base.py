from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.agent import AgentProfile
from ..core.message import Message


def _sender_role(message: Message) -> str:
    sender_role = message.metadata.get("sender_role")
    if isinstance(sender_role, str) and sender_role:
        return sender_role
    return message.source


@dataclass(frozen=True, slots=True)
class WorkerPromptInputs:
    task_input: str
    agent: AgentProfile
    round_index: int
    state: Mapping[str, Any]
    spatial_messages: Sequence[Message]
    temporal_messages: Sequence[Message]


@dataclass(frozen=True, slots=True)
class JudgePromptInputs:
    task_input: str
    judge: AgentProfile
    worker_messages: Sequence[Message]


class BaseTaskPack:
    dataset_id: str = ""

    def initialize_state(
        self, *, task_input: str, agent: AgentProfile
    ) -> Mapping[str, Any]:
        return {}

    def update_state(
        self,
        *,
        task_input: str,
        agent: AgentProfile,
        round_index: int,
        previous_state: Mapping[str, Any],
        response: str,
    ) -> Mapping[str, Any]:
        return dict(previous_state)

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        raise NotImplementedError

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        raise NotImplementedError

    def postprocess_worker_response(
        self,
        *,
        task_input: str,
        agent: AgentProfile,
        round_index: int,
        response: str,
    ) -> str:
        return response

    def prepare_judge_message(
        self,
        *,
        task_input: str,
        sender: AgentProfile,
        judge: AgentProfile,
        message: Message,
    ) -> Message:
        return message
