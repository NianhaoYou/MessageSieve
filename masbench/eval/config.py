from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from ..datasets import DatasetConfig
from ..llm.base import LLMConfig
from ..utils.config_loader import load_yaml_with_bases


@dataclass(slots=True)
class RuntimeConfig:
    output_dir: str = "outputs"
    save_predictions: bool = True
    save_trajectories: bool = True
    write_progress: bool = True
    progress_every: int = 1
    parallel_samples: int = 1
    num_rounds_override: int | None = None
    skip_failed_samples: bool = False
    contract: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalConfig:
    name: str
    method: str
    dataset: DatasetConfig
    model: LLMConfig
    method_params: Dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_eval_config(path: str) -> EvalConfig:
    config_path = Path(path)
    raw = load_yaml_with_bases(config_path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid config file: {path}")

    dataset = raw.get("dataset") or {}
    model = raw.get("model") or {}
    runtime = raw.get("runtime") or {}

    return EvalConfig(
        name=str(raw.get("name") or config_path.stem),
        method=str(raw["method"]),
        dataset=DatasetConfig(
            dataset_id=str(dataset["dataset_id"]),
            path=str(dataset["path"]),
            split=dataset.get("split"),
            start_index=int(dataset.get("start_index", 0) or 0),
            limit=int(dataset["limit"]) if dataset.get("limit") is not None else None,
        ),
        model=LLMConfig.from_dict(model),
        method_params=dict(raw.get("method_params") or {}),
        runtime=RuntimeConfig(
            output_dir=str(runtime.get("output_dir", "outputs")),
            save_predictions=bool(runtime.get("save_predictions", True)),
            save_trajectories=bool(runtime.get("save_trajectories", True)),
            write_progress=bool(runtime.get("write_progress", True)),
            progress_every=max(int(runtime.get("progress_every", 1)), 1),
            parallel_samples=max(int(runtime.get("parallel_samples", 1)), 1),
            num_rounds_override=(
                max(int(runtime["num_rounds_override"]), 1)
                if runtime.get("num_rounds_override") is not None
                else None
            ),
            skip_failed_samples=bool(runtime.get("skip_failed_samples", False)),
            contract=dict(runtime.get("contract") or {}),
        ),
    )
