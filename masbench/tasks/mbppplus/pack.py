from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ...core.agent import AgentProfile
from ...core.message import Message
from ..base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs, _sender_role
from .judge import extract_python_code, run_mbppplus_tests


class MBPPPlusTaskPack(BaseTaskPack):
    dataset_id = "mbppplus"

    def initialize_state(
        self, *, task_input: str, agent: AgentProfile
    ) -> Mapping[str, Any]:
        del task_input
        return {
            "mode": self._infer_mode(agent),
            "public_tests": tuple(self._load_public_tests(agent)),
            "public_test_pass": None,
            "public_test_feedback": "",
        }

    def update_state(
        self,
        *,
        task_input: str,
        agent: AgentProfile,
        round_index: int,
        previous_state: Mapping[str, Any],
        response: str,
    ) -> Mapping[str, Any]:
        del task_input, round_index
        next_state = dict(previous_state)
        tests = tuple(
            previous_state.get("public_tests") or self._load_public_tests(agent)
        )
        mode = str(previous_state.get("mode") or self._infer_mode(agent))
        next_state["mode"] = mode
        next_state["public_tests"] = tests
        extracted_code = extract_python_code(response)
        code = extracted_code or response.strip()
        if mode == "single" or extracted_code:
            passed, feedback = self._run_public_tests(code, tests, agent=agent)
            next_state["public_test_pass"] = passed
            next_state["public_test_feedback"] = feedback
            return next_state
        next_state["public_test_pass"] = None
        next_state["public_test_feedback"] = ""
        return next_state

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        if str(inputs.state.get("mode") or self._infer_mode(inputs.agent)) == "single":
            return self._build_single_prompt(inputs)
        return self._build_analysis_prompt(inputs)

    @staticmethod
    def _build_single_prompt(inputs: WorkerPromptInputs) -> str:
        return (
            f"The Python function task is:\n\n{inputs.task_input}\n\n"
            "Write Python code for this task. Use the public assert tests to preserve the expected function name, interface, and return behavior. "
            "Return exactly one Python code block and no extra explanation."
        ).strip()

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        prompt = f"The Python function task is:\n\n{inputs.task_input}\n"
        worker_str = self._render_peer_messages(inputs.worker_messages)
        if worker_str:
            prompt += "\nWorker outputs from the team are as follows:\n\n"
            prompt += worker_str
            prompt += "\n"
        prompt += (
            "\nWrite the final accepted Python solution now. "
            "First reconstruct the intended algorithm from the original task and public assert tests, then use the agents as evidence. "
            "Use the strongest design, algorithm, implementation, testing, and bug-fixing signals above. "
            "Prefer worker implementations that are explicitly reported as passing the public tests, but still repair them if they violate the task. "
            "Reject any plan that changes the expected function name, switches to stdin/stdout, returns the wrong type, or broadens behavior beyond the task and tests without explicit support. "
            "Prefer the simplest implementation consistent with the task and public tests. "
            "Return exactly one Python code block containing the full final function implementation."
        )
        return prompt.strip()

    def _build_analysis_prompt(self, inputs: WorkerPromptInputs) -> str:
        role_name = str(inputs.agent.role_name or "").strip()
        implementation_roles = {"Programming Expert", "Bug Fixer"}
        if role_name in implementation_roles:
            role_output_instruction = (
                "- if the implementation is clear, include one full Python function in a code block\n"
                "- preserve the expected function name and interface from the public tests\n"
            )
        else:
            role_output_instruction = (
                "- send a concise role-specific note for the final judge, preferably within fifty words\n"
                "- do not include a full implementation unless it is necessary to make your point\n"
            )
        prompt = (
            f"The Python function task is:\n\n{inputs.task_input}\n\n"
            f"Your role is: {role_name}.\n"
            "You are part of a multi-agent code-generation team using the Project Manager, Algorithm Designer, Programming Expert, Test Analyst, and Bug Fixer role split.\n"
            "Write naturally from your role's perspective. Do not follow a required schema.\n"
            "Guidelines:\n"
            f"{role_output_instruction}"
            "- be concrete and implementation-oriented\n"
            "- do not claim an algorithm unless you can justify it from the prompt\n"
            "- use the public assert tests to infer the expected function name and interface\n"
            "- do not switch to stdin/stdout\n"
            "- prefer implementation-relevant details over generic explanation\n"
            "- prefer the simplest behavior consistent with the task and public tests\n"
            "- do not invent broader input types, conversions, nested structures, or regex semantics unless explicitly required by the task\n"
        )
        spatial_str = self._render_peer_messages(inputs.spatial_messages)
        temporal_str = self._render_peer_messages(inputs.temporal_messages)
        if spatial_str:
            prompt += f"\nSame-round predecessor messages:\n\n{spatial_str}\n"
        if temporal_str:
            prompt += f"\nPrevious-round messages:\n\n{temporal_str}\n"
        return prompt.strip()

    def _render_peer_messages(self, messages: Sequence[Message]) -> str:
        if not messages:
            return ""
        rendered: list[str] = []
        for message in messages:
            sender_role = _sender_role(message)
            status_lines: list[str] = []
            if message.metadata.get("public_test_pass") is not None:
                status_lines.append(
                    f"Public-test pass: {message.metadata.get('public_test_pass')}"
                )
            feedback = str(message.metadata.get("public_test_feedback") or "").strip()
            if feedback:
                status_lines.append(f"Public-test feedback: {feedback}")
            status = ("\n" + "\n".join(status_lines) + "\n") if status_lines else ""
            rendered.append(
                f"Agent {message.source} as a {sender_role}:{status}\n{str(message.content or '').strip()}\n"
            )
        return "\n".join(rendered).strip()

    def _load_public_tests(self, agent: AgentProfile) -> list[str]:
        raw = agent.metadata.get("test_list")
        if raw is None:
            raw = agent.metadata.get("public_test_cases")
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item).strip()]
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(payload, list):
                return [str(item) for item in payload if str(item).strip()]
        return []

    @staticmethod
    def _infer_mode(agent: AgentProfile) -> str:
        return "single" if str(agent.metadata.get("method_id") or "").strip().lower() == "single" else "analysis"

    def _run_public_tests(
        self, code: str, tests: Sequence[str], *, agent: AgentProfile
    ) -> tuple[bool | None, str]:
        if not code.strip() or not tests:
            return None, ""
        setup_code = str(agent.metadata.get("test_setup_code") or "")
        test_imports = agent.metadata.get("test_imports")
        if isinstance(test_imports, (str, bytes)):
            try:
                parsed_imports = json.loads(test_imports)
            except json.JSONDecodeError:
                parsed_imports = [
                    line for line in str(test_imports).splitlines() if line.strip()
                ]
        elif isinstance(test_imports, Sequence):
            parsed_imports = list(test_imports)
        else:
            parsed_imports = None
        try:
            passed = (
                run_mbppplus_tests(
                    code,
                    tests=tests,
                    setup_code=setup_code,
                    imports=parsed_imports,
                    timeout_sec=6,
                )
                >= 1.0
            )
        except Exception as exc:
            return False, f"public-test execution failed: {exc}"
        return (
            passed,
            "all public tests passed" if passed else "failed at least one public test",
        )
