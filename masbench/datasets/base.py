from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TaskSample:
    task_id: str
    dataset_id: str
    prompt: str
    target: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    dataset_id: str
    path: str
    split: str | None = None
    start_index: int = 0
    limit: int | None = None
