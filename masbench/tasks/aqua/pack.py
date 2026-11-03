from __future__ import annotations

import re

from ..base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs


def _render_messages(messages) -> str:
    if not messages:
        return ""
    chunks: list[str] = []
    for message in messages:
        sender_role = message.metadata.get("sender_role") or message.source
        chunks.append(
            f"Agent {message.source}, role is {sender_role}, output is:\n\n{message.content}\n\n"
        )
    return "\n".join(chunks).strip()


def _render_choice_hints(messages) -> str:
    if not messages:
        return ""
    hints: list[str] = []
    for message in messages:
        text = str(message.content or "")
        matches = re.findall(
            r"(?im)(?:final\s+answer|answer)\s*:\s*[\(\[\{]?\s*([A-E])\s*[\)\]\}]?",
            text,
        )
        if matches:
            hints.append(matches[-1].upper())
            continue
        option_matches = re.findall(r"\b([A-E])\b", text.upper())
        if option_matches:
            hints.append(option_matches[-1].upper())
    return " ".join(hints)


def _build_single_worker_prompt(task_input: str) -> str:
    return (
        "Answer this multiple-choice math problem directly.\n"
        "Keep the response short. Use brief working only if needed, but do not write a full step-by-step solution.\n\n"
        f"Q: {task_input}\n\n"
        "End with exactly one final answer line in this format: Final Answer: <A/B/C/D/E>."
    ).strip()


class AQuATaskPack(BaseTaskPack):
    dataset_id = "aqua"

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        if (
            str(inputs.agent.metadata.get("method_id") or "").strip().lower()
            == "single"
        ):
            return _build_single_worker_prompt(inputs.task_input)
        role_name = getattr(inputs.agent, "role_name", "")
        prompt = (
            f"You are a {role_name}.\n"
            "You will be given a multiple-choice question and hints from other agents.\n"
            "Use the hints only as references; reject arithmetic or option-matching mistakes.\n"
            "Before solving, identify whether the key variables describe absolute amounts, rates, or percentages.\n"
            "The last line of your response must be exactly: The answer is <A/B/C/D/E>.\n\n"
            f"Q: {inputs.task_input}\n"
        )
        if role_name == "Math Solver":
            hint_letters = _render_choice_hints(
                list(inputs.spatial_messages) + list(inputs.temporal_messages)
            )
            if hint_letters:
                prompt += (
                    f"\n(Hint: The answer from other agents are list: {hint_letters}.)"
                )
            prompt += "\nGive your own solving process step by step based on the hints."
        elif role_name == "Mathematical Analyst":
            prompt += "\nFirst analyze the problem-solving process step by step, where the variables are represented by letters. Determine whether the variables are absolute amounts, rates, or percentages. Then substitute the values into the analysis process to perform calculations and get the results."
            spatial = _render_messages(inputs.spatial_messages)
            temporal = _render_messages(inputs.temporal_messages)
            if spatial:
                prompt += f"\nAt the same time, the outputs of other agents are as follows:\n\n{spatial}\n"
            if temporal:
                prompt += f"\nIn the last round of dialogue, the outputs of other agents were:\n\n{temporal}\n"
        else:
            prompt += "\nIntegrate step-by-step reasoning with the available hints, determine the variable type, and compute the final choice carefully."
            spatial = _render_messages(inputs.spatial_messages)
            temporal = _render_messages(inputs.temporal_messages)
            if spatial:
                prompt += f"\nAt the same time, the outputs of other agents are as follows:\n\n{spatial}\n"
            if temporal:
                prompt += f"\nIn the last round of dialogue, the outputs of other agents were:\n\n{temporal}\n"
        return prompt.strip()

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        worker_outputs = _render_messages(inputs.worker_messages)
        prompt = (
            "The task is an AQuA multiple-choice math word problem.\n"
            "First solve the task yourself from the original question and choices.\n"
            "Then compare the worker analyses, reject arithmetic or option-matching errors, and choose the single correct option.\n"
            "Do not accept worker consensus if they treat rates or percentages as absolute counts.\n\n"
            f"Q: {inputs.task_input}\n"
        )
        if worker_outputs:
            prompt += f"\nAt the same time, the output of other agents is as follows:\n\n{worker_outputs}\n"
        prompt += (
            "\nGive brief reasons for the decision when useful. "
            "The last line of your output must be exactly: The answer is <A/B/C/D/E>."
        )
        return prompt.strip()
