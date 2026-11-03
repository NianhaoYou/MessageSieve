from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_with_bases(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return _load_yaml_with_bases(config_path, stack=[])


def _load_yaml_with_bases(path: Path, *, stack: list[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in [*stack, resolved])
        raise ValueError(f"Config inheritance cycle detected: {cycle}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid config file: {path}")

    merged: dict[str, Any] = {}
    base_entries = raw.get("_base")
    if base_entries is None:
        base_list: list[str] = []
    elif isinstance(base_entries, str):
        base_list = [base_entries]
    elif isinstance(base_entries, list):
        base_list = [str(item) for item in base_entries]
    else:
        raise TypeError(f"_base must be a string or a list of strings: {path}")

    next_stack = [*stack, resolved]
    for base_entry in base_list:
        base_path = (path.parent / base_entry).resolve()
        base_payload = _load_yaml_with_bases(base_path, stack=next_stack)
        merged = _deep_merge(merged, base_payload)

    own_payload = {key: value for key, value in raw.items() if key != "_base"}
    merged = _deep_merge(merged, own_payload)
    return merged


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if key == "method_params" and isinstance(value, Mapping) and not value:
            merged[key] = {}
            continue
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, dict(value))
            continue
        merged[key] = value
    return merged
