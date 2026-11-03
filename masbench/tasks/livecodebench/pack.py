from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ...core.agent import AgentProfile
from ...core.message import Message
from ...datasets.base import TaskSample
from ..base import BaseTaskPack, JudgePromptInputs, WorkerPromptInputs, _sender_role
from .judge import run_livecodebench_call_tests, run_livecodebench_stdin_tests

_PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_python_code(text: str) -> str | None:
    match = _PYTHON_BLOCK_PATTERN.search(str(text or ""))
    if not match:
        return None
    return match.group(1).strip()


class LiveCodeBenchCodeGenLiteTaskPack(BaseTaskPack):
    dataset_id = "livecodebench_codegen_lite"

    def initialize_state(
        self, *, task_input: str, agent: AgentProfile
    ) -> Mapping[str, Any]:
        del task_input
        return {
            "previous_output": "",
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
        next_state["previous_output"] = response
        tests = tuple(
            previous_state.get("public_tests") or self._load_public_tests(agent)
        )
        mode = str(previous_state.get("mode") or self._infer_mode(agent))
        next_state["mode"] = mode
        next_state["public_tests"] = tests
        candidate = self._extract_candidate_code(response)
        if mode == "single" and not candidate:
            candidate = response.strip()
        passed, feedback = self._run_public_tests(candidate, tests, agent=agent)
        next_state["public_test_pass"] = passed
        next_state["public_test_feedback"] = feedback
        return next_state

    def build_worker_prompt(self, inputs: WorkerPromptInputs) -> str:
        if str(inputs.state.get("mode") or self._infer_mode(inputs.agent)) == "single":
            if self._uses_starter_code(inputs.agent):
                return self._build_single_prompt_function(inputs)
            return self._build_single_prompt_stdin(inputs)
        return self._build_analysis_prompt(inputs)

    def _build_single_prompt_stdin(self, inputs: WorkerPromptInputs) -> str:
        return (
            f"The programming task is:\n\n{inputs.task_input}\n\n"
            "Write Python code for this task. Read from standard input and print the answer to standard output. "
            "Return exactly one Python code block and no extra explanation."
        ).strip()

    def _build_single_prompt_function(self, inputs: WorkerPromptInputs) -> str:
        return (
            f"The programming task is:\n\n{inputs.task_input}\n\n"
            "Write Python code for this task. Preserve the exact starter class/function signature shown above. "
            "Return exactly one Python code block and no extra explanation."
        ).strip()

    def build_judge_prompt(self, inputs: JudgePromptInputs) -> str:
        worker_str = self._render_worker_messages(inputs.worker_messages)
        prompt = f"The programming task is:\n\n{inputs.task_input}\n"
        if worker_str:
            prompt += "\nWorker outputs from the team are as follows:\n\n"
            prompt += worker_str
            prompt += "\n"
        if self._uses_starter_code(inputs.judge):
            prompt += (
                "\nWrite the final accepted Python solution now. "
                "First reconstruct the intended algorithm from the original task, then use the agents as evidence. "
                "Use the strongest algorithmic ideas above, reject any plan that violates the constraints or only passes samples, "
                "preserve the exact starter class/function signature, and return exactly one Python code block "
                "containing the full final implementation."
            )
        else:
            prompt += (
                "\nWrite the final accepted Python solution now. "
                "First reconstruct the intended algorithm from the original task, then use the agents as evidence. "
                "Use the strongest algorithmic ideas above, reject any plan that violates the constraints or only passes samples, "
                "write a complete Python program that reads from standard input and writes to standard output, "
                "and return exactly one Python code block containing the full final solution."
            )
        return prompt.strip()

    def _build_analysis_prompt(self, inputs: WorkerPromptInputs) -> str:
        implementation_worker = inputs.agent.role_name == "Implementation Strategist"
        audit_worker = inputs.agent.role_name in {
            "Constraint Auditor",
            "Edge-Case Analyst",
        }
        late_round = inputs.round_index >= 1
        uses_starter_code = self._uses_starter_code(inputs.agent)
        interface_rule = (
            "- preserve the exact starter class/function signature and do not switch to stdin/stdout\n"
            if uses_starter_code
            else "- the final solution should be a complete stdin/stdout program rather than a class-based wrapper\n"
        )
        prompt = (
            f"The programming task is:\n\n{inputs.task_input}\n\n"
            f"Your role is: {inputs.agent.role_name}.\n"
            "You are part of a competitive-programming team.\n"
        )
        prompt += (
            "Send a concise role-specific note for the final judge. "
            "Do not follow a required schema; write naturally from your role's perspective.\n"
            "Guidelines:\n"
            "- be concrete and contest-oriented\n"
            "- do not claim an algorithm unless you can justify it from the prompt\n"
            f"{interface_rule}"
            "- prefer implementation-relevant details over generic explanation\n"
            "- do not trust public samples as proof; reason about hidden cases from the constraints\n"
        )
        if implementation_worker and late_round:
            prompt += "- as the implementation strategist in a later round, include one concise candidate implementation if the algorithm is clear\n"
        elif audit_worker:
            prompt += "- mention concrete constraint or edge-case concerns rather than restating the full solution\n"
        else:
            prompt += "- do not include a full program\n"
        previous_output = str(inputs.state.get("previous_output") or "").strip()
        if previous_output:
            prompt += "\nIn your own previous round attempt, your analysis was:\n\n"
            prompt += previous_output
            public_pass = inputs.state.get("public_test_pass")
            feedback = str(inputs.state.get("public_test_feedback") or "").strip()
            if public_pass is not None:
                prompt += f"\n\nWhether your code hint passes the public tests? {public_pass}."
            if feedback:
                prompt += (
                    f"\n\nPublic-test feedback on your previous hint:\n\n{feedback}\n"
                )
        spatial_str = self._render_peer_messages(inputs.spatial_messages)
        temporal_str = self._render_peer_messages(inputs.temporal_messages)
        if spatial_str:
            prompt += f"\nSame-round predecessor messages:\n\n{spatial_str}\n"
        if temporal_str:
            prompt += f"\nPrevious-round messages:\n\n{temporal_str}\n"
        prompt += (
            "\nFocus on the real contest bottleneck. "
            "If another agent proposes an O(N^2) or O(N^3) idea under large constraints, explicitly reject it. "
            "If another agent gives only vague commentary, replace it with an implementation-usable statement."
        )
        return prompt.strip()

    def _render_peer_messages(self, messages: Sequence[Message]) -> str:
        if not messages:
            return ""
        rendered: list[str] = []
        for message in messages:
            sender_role = _sender_role(message)
            content = str(message.content or "").strip()
            block = [f"Agent {message.source} as a {sender_role}:\n\n{content}"]
            passed = message.metadata.get("public_test_pass")
            feedback = str(message.metadata.get("public_test_feedback") or "").strip()
            if passed is not None:
                block.append(f"PUBLIC_TEST_PASS: {passed}")
            if feedback:
                block.append(f"PUBLIC_TEST_FEEDBACK:\n{feedback}")
            rendered.append("\n\n".join(block))
        return "\n".join(rendered).strip()

    def _render_worker_messages(self, messages: Sequence[Message]) -> str:
        if not messages:
            return ""
        rendered_all: list[str] = []
        for message in messages:
            sender_role = _sender_role(message)
            content = str(message.content or "").strip()
            passed = message.metadata.get("public_test_pass")
            feedback = str(message.metadata.get("public_test_feedback") or "").strip()
            if passed is False:
                content = self._omit_failed_executable_candidate(content)
            block = [
                f"Agent {message.source} as a {sender_role}:",
                "",
                content,
            ]
            if passed is not None:
                block.extend(["", f"PUBLIC_TEST_PASS: {passed}"])
            if feedback:
                block.extend(["", f"PUBLIC_TEST_FEEDBACK:\n{feedback}"])
            rendered = "\n".join(block).strip()
            rendered_all.append(rendered)
        return "\n\n".join(rendered_all).strip()

    @staticmethod
    def _infer_mode(agent: AgentProfile) -> str:
        return "single" if str(agent.metadata.get("method_id") or "").strip().lower() == "single" else "analysis"

    @staticmethod
    def _uses_starter_code(agent: AgentProfile) -> bool:
        starter_code = agent.metadata.get("starter_code")
        return bool(str(starter_code or "").strip()) or bool(
            LiveCodeBenchCodeGenLiteTaskPack._func_name(agent)
        )

    @staticmethod
    def _io_mode(agent: AgentProfile) -> str:
        explicit = str(agent.metadata.get("io_mode") or "").strip().lower()
        if explicit in {"functional", "stdin"}:
            return explicit
        return (
            "functional"
            if LiveCodeBenchCodeGenLiteTaskPack._uses_starter_code(agent)
            else "stdin"
        )

    @staticmethod
    def _func_name(agent: AgentProfile) -> str:
        direct = str(agent.metadata.get("func_name") or "").strip()
        if direct:
            return direct
        raw_metadata = agent.metadata.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata = raw_metadata
        else:
            try:
                metadata = json.loads(str(raw_metadata or "{}"))
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, Mapping):
            return ""
        for key in ("func_name", "function_name", "entry_point"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _load_public_tests(self, agent: AgentProfile) -> list[dict[str, Any]]:
        raw = agent.metadata.get("public_test_cases")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        return []

    def _run_public_tests(
        self, code: str, tests: Sequence[dict[str, Any]], *, agent: AgentProfile
    ) -> tuple[bool | None, str]:
        if not code.strip() or not tests:
            return None, ""
        try:
            sample = TaskSample(
                task_id=str(agent.metadata.get("task_id") or ""),
                dataset_id=self.dataset_id,
                prompt="",
                metadata=dict(agent.metadata),
            )
            if self._io_mode(agent) == "functional":
                func_name = self._func_name(agent)
                if not func_name:
                    return False, "functional task is missing func_name"
                passed = (
                    run_livecodebench_call_tests(
                        code,
                        starter_code=str(agent.metadata.get("starter_code") or ""),
                        func_name=func_name,
                        tests=list(tests),
                        timeout_sec=6,
                    )
                    >= 1.0
                )
            else:
                passed = (
                    run_livecodebench_stdin_tests(
                        code,
                        tests=list(tests),
                        timeout_sec=6,
                        sample=sample,
                    )
                    >= 1.0
                )
        except Exception as exc:
            return False, f"public-test execution failed: {exc}"
        return (
            passed,
            "all public tests passed" if passed else "failed at least one public test",
        )

    @staticmethod
    def _extract_candidate_code(summary: str) -> str:
        text = str(summary or "")
        marker = "FINAL_CODE_CANDIDATE:"
        if marker in text:
            candidate = text.split(marker, 1)[1].strip()
            if candidate.upper().startswith("NONE"):
                return ""
            code = _extract_python_code(candidate)
            if code is not None:
                return code
            return candidate
        code = _extract_python_code(text)
        if code is not None:
            return code
        return ""

    @staticmethod
    def _omit_failed_executable_candidate(summary: str) -> str:
        text = str(summary or "")
        marker = "FINAL_CODE_CANDIDATE:"
        if marker not in text:
            return re.sub(
                r"```python\s*.*?```",
                "```python\n<omitted because this executable candidate failed public tests; preserve only the analysis above>\n```",
                text,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )
        before, after = text.split(marker, 1)
        tail = ""
        final_marker = "## Final Answer"
        if final_marker in after:
            tail = final_marker + after.split(final_marker, 1)[1]
        return (
            before.rstrip()
            + "\n"
            + marker
            + "\n<omitted because this executable candidate failed public tests; preserve only the analysis above>\n"
            + tail
        ).strip()
