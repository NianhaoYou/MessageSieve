from __future__ import annotations

from ..base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs, _sender_role


def _render_messages(messages) -> str:
    if not messages:
        return ""
    chunks: list[str] = []
    for message in messages:
        sender_role = _sender_role(message)
        chunks.append(
            f"Agent {message.source}, role is {sender_role}, output is:\n\n{message.content}\n"
        )
    return "\n".join(chunks).strip()


def _build_mas_worker_prompt(inputs: WorkerPromptInputs) -> str:
    prompt = (
        "The task is a graduate-level science multiple-choice question.\n"
        "Only one option is correct.\n"
        "Read the question carefully, reason from your assigned scientific role, and identify the most defensible option.\n\n"
        f"The task is:\n\n{inputs.task_input}\n"
    )
    spatial = _render_messages(inputs.spatial_messages)
    temporal = _render_messages(inputs.temporal_messages)
    if spatial:
        prompt += f"\nAt the same time, the outputs of other agents are as follows:\n\n{spatial}\n"
    if temporal:
        prompt += f"\nIn the last round of dialogue, the outputs of other agents were:\n\n{temporal}\n"
    return (
        prompt.strip() + "\nGive a brief scientific justification first. "
        "End with a final line exactly in this format: Final Answer: <A/B/C/D>."
    )


def _build_single_worker_prompt(inputs: WorkerPromptInputs) -> str:
    return (
        "The task is a graduate-level science multiple-choice question.\n"
        "Only one option is correct.\n\n"
        f"The task is:\n\n{inputs.task_input}\n\n"
        "Do not include explanation. Reply with one final line exactly in this format: Final Answer: <A/B/C/D>."
    ).strip()


def _build_mas_judge_prompt(inputs: JudgePromptInputs) -> str:
    worker_outputs = _render_messages(inputs.worker_messages)
    prompt = (
        "The task is a graduate-level science multiple-choice question.\n"
        "Only one option is correct.\n\n"
        f"The task is:\n\n{inputs.task_input}\n"
    )
    if worker_outputs:
        prompt += f"\nAt the same time, the output of other agents is as follows:\n\n{worker_outputs}\n"
    prompt += (
        "\nGive a brief final decision rationale first. "
        "End with a final line exactly in this format: Final Answer: <A/B/C/D>."
    )
    return prompt.strip()


class GPQADiamondTaskPack(BaseTaskPack):
    dataset_id = "gpqa_diamond"

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        if (
            str(inputs.agent.metadata.get("method_id") or "").strip().lower()
            == "single"
        ):
            return _build_single_worker_prompt(inputs)
        return _build_mas_worker_prompt(inputs)

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        return _build_mas_judge_prompt(inputs)
