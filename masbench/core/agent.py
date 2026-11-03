from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AgentProfile:
    agent_id: str
    display_name: str
    role_name: str
    system_prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
