from __future__ import annotations

from .base import BaseTaskPack
from .aqua.pack import AQuATaskPack
from .gpqa.pack import GPQADiamondTaskPack
from .mbppplus.pack import MBPPPlusTaskPack
from .livecodebench.pack import LiveCodeBenchCodeGenLiteTaskPack
from .math.pack import MathTaskPack


TASK_PACKS: dict[str, BaseTaskPack] = {
    "aqua": AQuATaskPack(),
    "gpqa_diamond": GPQADiamondTaskPack(),
    "mbppplus": MBPPPlusTaskPack(),
    "livecodebench_codegen_lite": LiveCodeBenchCodeGenLiteTaskPack(),
    "math": MathTaskPack(),
}


def get_task_pack(dataset_id: str) -> BaseTaskPack:
    key = dataset_id.strip().lower()
    if key not in TASK_PACKS:
        raise KeyError(f"Unknown task pack: {dataset_id}")
    return TASK_PACKS[key]
