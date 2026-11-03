from __future__ import annotations

from ..base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs


def _render_messages(messages, *, suffix: str) -> str:
    if not messages:
        return ""
    chunks: list[str] = []
    for message in messages:
        sender_role = message.metadata.get("sender_role") or message.source
        content = str(message.content or "").strip()
        chunks.append(
            f"Agent {message.source} as a {sender_role} his answer to this question {suffix}:\n\n{content}\n\n"
        )
    return "\n".join(chunks).strip()


def _append_message_sections(
    prompt: str, *, spatial_messages, temporal_messages
) -> str:
    spatial = _render_messages(spatial_messages, suffix="is")
    temporal = _render_messages(temporal_messages, suffix="was")
    if spatial:
        prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial} \n\n"
    if temporal:
        prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal}"
    return prompt


def _build_mas_worker_prompt(inputs: WorkerPromptInputs) -> str:
    role_name = inputs.agent.role_name
    if (
        inputs.round_index <= 0
        and not inputs.spatial_messages
        and not inputs.temporal_messages
    ):
        prompt = (
            f"Role: {role_name}\n"
            "Solve the MATH500 problem carefully from your role perspective. "
            "Show the reasoning needed to reach the answer, and keep the final answer in exact mathematical notation. "
            "The last line of your output must be exactly: The answer is <value>\n\n"
            f"Problem:\n{inputs.task_input}"
        )
        return prompt.strip()

    discussion_note = (
        "Earlier agents in this round have already provided messages. "
        if inputs.round_index <= 0
        else "This is the second discussion round. Use previous-round and current messages to improve or verify your answer. "
    )
    prompt = (
        f"Role: {role_name}\n"
        f"{discussion_note}"
        "Use other agents' messages as references, but verify the mathematical reasoning before adopting their answer. "
        "If you find a concrete error, correct it. Otherwise continue with the most reliable solution. "
        "The last line of your output must be exactly: The answer is <value>\n\n"
        f"Problem:\n{inputs.task_input}"
    )
    return _append_message_sections(
        prompt,
        spatial_messages=inputs.spatial_messages,
        temporal_messages=inputs.temporal_messages,
    ).strip()


def _build_mas_judge_prompt(inputs: JudgePromptInputs) -> str:
    worker_outputs = _render_messages(inputs.worker_messages, suffix="is")
    return (
        "You are the final decision-maker for a MATH500 multi-agent math system.\n"
        "Read the agents' solutions and choose the most reliable final answer. "
        "Compare the reasoning, calculations, assumptions, and candidate answers. "
        "Do not blindly follow a majority vote; verify the answer against the original problem. "
        "Preserve exact notation, including fractions, radicals, tuples, intervals, signs, and complex numbers. "
        "Give concise reasons for the decision, and make the last line exactly: The answer is <value>\n\n"
        f"Problem:\n{inputs.task_input}\n\n"
        "Agent outputs:\n"
        f"{worker_outputs if worker_outputs else 'No available worker messages.'}"
    ).strip()


class MathTaskPack(BaseTaskPack):
    dataset_id = "math"

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        if (
            str(inputs.agent.metadata.get("method_id") or "").strip().lower()
            == "single"
        ):
            return (
                f"Role: {inputs.agent.role_name}\n"
                "Solve the problem directly.\n"
                "Identify the exact quantity requested before answering.\n"
                "Preserve complete mathematical notation in the final answer.\n"
                "End with exactly one final line: The answer is <value>\n\n"
                f"Problem:\n{inputs.task_input}"
            ).strip()
        return _build_mas_worker_prompt(inputs)

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        return _build_mas_judge_prompt(inputs)
