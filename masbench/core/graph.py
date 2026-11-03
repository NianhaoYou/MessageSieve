from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True, slots=True)
class FullCommunicationGraph:
    """The full candidate worker graph for MessageSieve.

    Spatial candidates connect every worker to every other worker.
    The runtime realizes an acyclic same-round subgraph from these candidates.
    Temporal delivery, including self-memory, is handled by the runtime.
    """

    agent_ids: Tuple[str, ...]

    @classmethod
    def from_agent_ids(cls, agent_ids: Iterable[str]) -> "FullCommunicationGraph":
        normalized = tuple(dict.fromkeys(agent_ids))
        if not normalized:
            raise ValueError("At least one agent is required to build a graph.")
        return cls(agent_ids=normalized)

    def outbound_neighbors(self, sender_id: str) -> List[str]:
        self._validate_agent(sender_id)
        return [agent_id for agent_id in self.agent_ids if agent_id != sender_id]

    def _validate_agent(self, agent_id: str) -> None:
        if agent_id not in self.agent_ids:
            raise KeyError(f"Unknown agent_id: {agent_id}")
