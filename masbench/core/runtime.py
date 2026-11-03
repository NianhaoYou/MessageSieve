from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .agent import AgentProfile
from .graph import FullCommunicationGraph
from .message import Message, TokenUsage
from ..llm.base import BaseLLM, ChatMessage, LLMResult
from ..methods.base import BaseMethod
from ..presets.catalog import DatasetPreset
from ..tasks.base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs


@dataclass(slots=True)
class RoundTurn:
    agent_id: str
    round_index: int
    prompt: str
    response: str
    usage: TokenUsage
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MASRunResult:
    final_text: str
    worker_turns: List[RoundTurn]
    judge_turn: RoundTurn | None
    exchanged_messages: List[Message]
    usage: TokenUsage


class MASRuntime:
    """MessageSieve scheduling and delivery over a full candidate worker graph."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        preset: DatasetPreset,
        task_pack: BaseTaskPack,
        method: BaseMethod,
        agents: Sequence[AgentProfile],
        judge: AgentProfile | None = None,
        graph: FullCommunicationGraph | None = None,
    ) -> None:
        if not agents:
            raise ValueError("MASRuntime requires at least one worker agent.")
        self.llm = llm
        self.preset = preset
        self.task_pack = task_pack
        self.method = method
        self._base_agents = tuple(agents)
        self._base_judge = judge
        self.agents = list(self._base_agents)
        self.agent_map = {agent.agent_id: agent for agent in self.agents}
        self.judge = judge
        self.graph = graph or FullCommunicationGraph.from_agent_ids(
            agent.agent_id for agent in self.agents
        )
        self._observed_usage = TokenUsage()
        self._observed_calls: list[dict[str, Any]] = []

    def run(
        self, task_input: str, *, task_metadata: Mapping[str, Any] | None = None
    ) -> MASRunResult:
        self._observed_usage = TokenUsage()
        self._observed_calls = []
        task_metadata = dict(task_metadata or {})
        enriched_workers: list[AgentProfile] = []
        for agent in self._base_agents:
            metadata = dict(agent.metadata)
            metadata.setdefault("method_id", self.method.method_id)
            metadata.update(task_metadata)
            enriched_workers.append(replace(agent, metadata=metadata))
        self.agents = enriched_workers
        self.agent_map = {agent.agent_id: agent for agent in self.agents}
        if self._base_judge is not None:
            judge_metadata = dict(self._base_judge.metadata)
            judge_metadata.setdefault("method_id", self.method.method_id)
            judge_metadata.update(task_metadata)
            self.judge = replace(self._base_judge, metadata=judge_metadata)
        self.method.reset_for_run()
        turns: List[RoundTurn] = []
        messages: List[Message] = []
        usage = TokenUsage()
        previous_round_outputs: List[RoundTurn] = []
        agent_states: dict[str, Mapping[str, Any]] = {
            agent.agent_id: self.task_pack.initialize_state(
                task_input=task_input, agent=agent
            )
            for agent in self.agents
        }

        for round_index in range(self.preset.num_rounds):
            current_round_outputs: List[RoundTurn] = []
            active_agents = self.agents
            realized_edges = self._realize_spatial_edges(
                active_agents=active_agents, round_index=round_index
            )
            execution_order = self._topological_order(
                active_agents=active_agents,
                realized_edges=realized_edges,
            )
            temporal_incoming = self._build_temporal_incoming(
                active_agents=active_agents,
                round_index=round_index,
                task_input=task_input,
                previous_round_outputs=previous_round_outputs,
                agent_states=agent_states,
            )
            spatial_incoming: dict[str, list[Message]] = defaultdict(list)
            successors = self._build_successor_map(realized_edges)
            for agent_id in execution_order:
                agent = self.agent_map[agent_id]
                inbound_candidates = [
                    *spatial_incoming.get(agent.agent_id, []),
                    *temporal_incoming.get(agent.agent_id, []),
                ]
                filtered_messages = self.method.filter_messages(
                    preset=self.preset,
                    receiver=agent,
                    round_index=round_index,
                    task_input=task_input,
                    candidates=inbound_candidates,
                )
                messages.extend(filtered_messages)
                spatial_messages = [
                    message
                    for message in filtered_messages
                    if message.channel == "spatial"
                ]
                temporal_messages = [
                    message
                    for message in filtered_messages
                    if message.channel == "temporal"
                ]
                prompt = self.task_pack.build_worker_prompt(
                    WorkerPromptInputs(
                        task_input=task_input,
                        agent=agent,
                        round_index=round_index,
                        state=agent_states.get(agent.agent_id, {}),
                        spatial_messages=spatial_messages,
                        temporal_messages=temporal_messages,
                    )
                )
                result = self._generate(agent.system_prompt, prompt)
                response_text = self.task_pack.postprocess_worker_response(
                    task_input=task_input,
                    agent=agent,
                    round_index=round_index,
                    response=result.text,
                )
                turn = RoundTurn(
                    agent_id=agent.agent_id,
                    round_index=round_index,
                    prompt=prompt,
                    response=response_text,
                    usage=result.usage,
                    metadata=self._llm_result_metadata(result),
                )
                usage = usage + result.usage
                turns.append(turn)
                current_round_outputs.append(turn)
                updated_state = self.task_pack.update_state(
                    task_input=task_input,
                    agent=agent,
                    round_index=round_index,
                    previous_state=agent_states.get(agent.agent_id, {}),
                    response=response_text,
                )
                state_payload = dict(updated_state or {})
                agent_states[agent.agent_id] = state_payload
                for receiver_id in successors.get(agent.agent_id, []):
                    receiver = self.agent_map[receiver_id]
                    sender_state = agent_states.get(agent.agent_id, {})
                    message = Message(
                        source=agent.agent_id,
                        target=receiver_id,
                        round_index=round_index,
                        channel="spatial",
                        content=turn.response,
                        metadata=self._message_metadata(turn, agent, sender_state),
                    )
                    prepared = self.method.prepare_message(
                        preset=self.preset,
                        sender=agent,
                        receiver=receiver,
                        round_index=round_index,
                        task_input=task_input,
                        message=message,
                    )
                    spatial_incoming[receiver_id].append(prepared)
            previous_round_outputs = current_round_outputs

        judge_turn: RoundTurn | None = None
        final_text: str
        if self.judge is None:
            final_text = (
                previous_round_outputs[-1].response if previous_round_outputs else ""
            )
        else:
            judge_candidates = []
            for turn in previous_round_outputs:
                sender = self.agent_map[turn.agent_id]
                sender_state = agent_states.get(turn.agent_id, {})
                raw_message = Message(
                    source=turn.agent_id,
                    target=self.judge.agent_id,
                    round_index=self.preset.num_rounds,
                    channel="spatial",
                    content=turn.response,
                    metadata=self._message_metadata(turn, sender, sender_state),
                )
                raw_message = self.task_pack.prepare_judge_message(
                    task_input=task_input,
                    sender=sender,
                    judge=self.judge,
                    message=raw_message,
                )
                prepared = self.method.prepare_message(
                    preset=self.preset,
                    sender=sender,
                    receiver=self.judge,
                    round_index=self.preset.num_rounds,
                    task_input=task_input,
                    message=raw_message,
                )
                judge_candidates.append(prepared)
            judge_candidates = self.method.filter_judge_messages(
                preset=self.preset,
                judge=self.judge,
                task_input=task_input,
                candidates=judge_candidates,
            )
            messages.extend(judge_candidates)
            judge_prompt = self.task_pack.build_judge_prompt(
                JudgePromptInputs(
                    task_input=task_input,
                    judge=self.judge,
                    worker_messages=judge_candidates,
                )
            )
            judge_result = self._generate(self.judge.system_prompt, judge_prompt)
            judge_turn = RoundTurn(
                agent_id=self.judge.agent_id,
                round_index=self.preset.num_rounds,
                prompt=judge_prompt,
                response=judge_result.text,
                usage=judge_result.usage,
                metadata=self._llm_result_metadata(judge_result),
            )
            usage = usage + judge_result.usage
            final_text = judge_result.text

        return MASRunResult(
            final_text=final_text,
            worker_turns=turns,
            judge_turn=judge_turn,
            exchanged_messages=messages,
            usage=usage,
        )

    @staticmethod
    def _message_metadata(
        turn: RoundTurn,
        sender: AgentProfile,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        verification_pass = state.get("internal_test_pass")
        if verification_pass is None:
            verification_pass = state.get("public_test_pass")
        verification_feedback = state.get("internal_test_feedback")
        if verification_feedback in {None, ""}:
            verification_feedback = state.get("public_test_feedback")
        return {
            "sender_round": turn.round_index,
            "sender_role": sender.role_name,
            "internal_test_pass": verification_pass,
            "internal_test_feedback": verification_feedback,
            "public_test_pass": state.get("public_test_pass"),
            "public_test_feedback": state.get("public_test_feedback"),
            **dict(turn.metadata or {}),
        }

    def _build_temporal_incoming(
        self,
        *,
        active_agents: Sequence[AgentProfile],
        round_index: int,
        task_input: str,
        previous_round_outputs: Iterable[RoundTurn],
        agent_states: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, List[Message]]:
        incoming: Dict[str, List[Message]] = defaultdict(list)
        if round_index == 0:
            return incoming

        previous_by_agent = {turn.agent_id: turn for turn in previous_round_outputs}
        for receiver in active_agents:
            for sender_id, sender_turn in previous_by_agent.items():
                sender = self.agent_map.get(sender_id)
                if sender is None:
                    continue
                sender_state = agent_states.get(sender_id, {})
                if not self.method.allow_temporal_edge(
                    sender=sender,
                    receiver=receiver,
                    round_index=round_index,
                ):
                    continue
                message = Message(
                    source=sender_id,
                    target=receiver.agent_id,
                    round_index=round_index,
                    channel="temporal",
                    content=sender_turn.response,
                    metadata=self._message_metadata(sender_turn, sender, sender_state),
                )
                incoming[receiver.agent_id].append(
                    self.method.prepare_message(
                        preset=self.preset,
                        sender=sender,
                        receiver=receiver,
                        round_index=round_index,
                        task_input=task_input,
                        message=message,
                    )
                )
        return incoming

    def _generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        messages: List[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self.llm.generate(messages)
        self._observed_usage = self._observed_usage + result.usage
        self._observed_calls.append(
            {
                "call_index": len(self._observed_calls),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": result.text,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "metadata": self._llm_result_metadata(result),
            }
        )
        return result

    def observed_call_evidence(self) -> Mapping[str, Any]:
        return {
            "prompt_tokens": self._observed_usage.prompt_tokens,
            "completion_tokens": self._observed_usage.completion_tokens,
            "total_tokens": self._observed_usage.total_tokens,
            "calls": [dict(row) for row in self._observed_calls],
        }

    @staticmethod
    def _llm_result_metadata(result: LLMResult) -> Mapping[str, Any]:
        raw = result.raw_response
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            if raw.get("blocked") is True:
                return {
                    "blocked": True,
                    "block_reason": "provider_content_filter",
                }
            return {}
        choices = getattr(raw, "choices", None)
        if not choices:
            return {}
        first_choice = choices[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        if finish_reason is None:
            return {}
        finish_reason_text = str(finish_reason)
        return {
            "finish_reason": finish_reason_text,
            "truncated_by_length": finish_reason_text.lower() == "length",
        }

    def _realize_spatial_edges(
        self,
        *,
        active_agents: Sequence[AgentProfile],
        round_index: int,
    ) -> List[tuple[str, str]]:
        active_ids = {agent.agent_id for agent in active_agents}
        successors: dict[str, list[str]] = defaultdict(list)
        realized_edges: list[tuple[str, str]] = []

        for sender in self.agents:
            if sender.agent_id not in active_ids:
                continue
            for receiver_id in self.graph.outbound_neighbors(sender.agent_id):
                if receiver_id not in active_ids:
                    continue
                receiver = self.agent_map[receiver_id]
                if not self.method.allow_spatial_edge(
                    sender=sender,
                    receiver=receiver,
                    round_index=round_index,
                ):
                    continue
                if self._path_exists(receiver_id, sender.agent_id, successors):
                    continue
                successors[sender.agent_id].append(receiver_id)
                realized_edges.append((sender.agent_id, receiver_id))
        return realized_edges

    def _topological_order(
        self,
        *,
        active_agents: Sequence[AgentProfile],
        realized_edges: Sequence[tuple[str, str]],
    ) -> List[str]:
        active_ids = {agent.agent_id for agent in active_agents}
        indegree = {agent.agent_id: 0 for agent in active_agents}
        successors: dict[str, list[str]] = defaultdict(list)

        for sender_id, receiver_id in realized_edges:
            if sender_id not in active_ids or receiver_id not in active_ids:
                continue
            indegree[receiver_id] += 1
            successors[sender_id].append(receiver_id)

        queue = deque(
            agent.agent_id
            for agent in self.agents
            if agent.agent_id in active_ids and indegree[agent.agent_id] == 0
        )
        ordered: list[str] = []
        while queue:
            agent_id = queue.popleft()
            ordered.append(agent_id)
            for receiver_id in successors.get(agent_id, []):
                indegree[receiver_id] -= 1
                if indegree[receiver_id] == 0:
                    queue.append(receiver_id)

        if len(ordered) != len(active_agents):
            for agent in self.agents:
                if agent.agent_id in active_ids and agent.agent_id not in ordered:
                    ordered.append(agent.agent_id)
        return ordered

    def _build_successor_map(
        self, realized_edges: Sequence[tuple[str, str]]
    ) -> Dict[str, List[str]]:
        successors: Dict[str, List[str]] = defaultdict(list)
        for sender_id, receiver_id in realized_edges:
            successors[sender_id].append(receiver_id)
        return successors

    def _path_exists(
        self,
        start_id: str,
        target_id: str,
        successors: Dict[str, List[str]],
        seen: set[str] | None = None,
    ) -> bool:
        if start_id == target_id:
            return True
        seen = seen or set()
        if start_id in seen:
            return False
        seen.add(start_id)
        for next_id in successors.get(start_id, []):
            if self._path_exists(next_id, target_id, successors, seen):
                return True
        return False
