from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime
import re
from time import monotonic
from typing import Any, Dict, List, Mapping, Sequence

from ..core.runtime import MASRuntime
from ..datasets import load_task_samples
from ..llm import build_llm
from ..methods import get_method
from ..presets import apply_prompt_protocol, get_dataset_preset
from ..tasks import get_task_pack
from .config import EvalConfig
from .io import append_jsonl, append_text_line, ensure_dir, write_json, write_jsonl
from .scoring import build_utility_fn, validate_code_scorer_sample


def _cancel_future_before_start(future: Any) -> bool:
    """Cancel only queued work; running provider work must remain tracked."""
    return bool(future.cancel())


class _SampleEvaluationError(RuntimeError):
    def __init__(self, cause: Exception, observed_call_evidence: Mapping[str, Any]) -> None:
        super().__init__(f"{cause.__class__.__name__}: {cause}")
        self.cause_type = cause.__class__.__name__
        self.observed_call_evidence = dict(observed_call_evidence)


def _messagesieve_method_params(config: EvalConfig) -> Dict[str, Any]:
    params = dict(config.method_params or {})
    from ..methods.messagesieve import task_family_for_dataset

    dataset_id = config.dataset.dataset_id.strip().lower()
    expected_family = task_family_for_dataset(dataset_id)
    declared_dataset_id = str(params.get("dataset_id") or "").strip().lower()
    declared_family = str(params.get("task_family") or "").strip().lower()
    if declared_dataset_id and declared_dataset_id != dataset_id:
        raise ValueError(
            "MessageSieve method_params.dataset_id must match "
            f"config.dataset.dataset_id ({declared_dataset_id!r} != {dataset_id!r})"
        )
    if not declared_family:
        raise ValueError("MessageSieve method_params.task_family must be explicit")
    if declared_family != expected_family:
        raise ValueError(
            f"MessageSieve dataset_id={dataset_id!r} requires "
            f"task_family={expected_family!r}; received {declared_family!r}"
        )
    params["dataset_id"] = dataset_id
    params["task_family"] = expected_family
    return params


def _validate_messagesieve_samples(config: EvalConfig, samples: Sequence[Any]) -> None:
    dataset_id = config.dataset.dataset_id.strip().lower()
    for sample in samples:
        if str(sample.dataset_id).strip().lower() != dataset_id:
            raise ValueError(
                f"MessageSieve sample {sample.task_id!r} has dataset_id={sample.dataset_id!r}; "
                f"expected {dataset_id!r}"
            )
        embedded_dataset_id = str(sample.metadata.get("dataset_id") or "").strip().lower()
        if embedded_dataset_id and embedded_dataset_id != dataset_id:
            raise ValueError(
                f"MessageSieve sample {sample.task_id!r} embeds dataset_id="
                f"{embedded_dataset_id!r}; expected {dataset_id!r}"
            )


def _serialize_agent(agent) -> Dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "display_name": agent.display_name,
        "role_name": agent.role_name,
        "system_prompt": agent.system_prompt,
        "metadata": dict(agent.metadata),
    }


def _runtime_task_metadata(config: EvalConfig) -> Dict[str, Any]:
    raw = config.runtime.contract.get("task_metadata")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


_SECRET_ARTIFACT_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}")
_URL_ARTIFACT_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)


def _sanitize_artifact_error(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    text = _SECRET_ARTIFACT_RE.sub("[REDACTED_CREDENTIAL]", text)
    return _URL_ARTIFACT_RE.sub("[REDACTED_ENDPOINT]", text)


def _is_timeout_exception(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "deadline exceeded" in text


def _build_run_contract(
    *,
    config: EvalConfig,
    method,
    preset,
    runtime: MASRuntime,
    num_samples: int,
    task_samples: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    return {
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "path": config.dataset.path,
            "split": config.dataset.split,
            "start_index": config.dataset.start_index,
            "limit": config.dataset.limit,
            "num_samples": num_samples,
            "selected_sample_ids": [str(sample.task_id) for sample in (task_samples or [])],
        },
        "runtime": {
            "num_rounds": preset.num_rounds,
            "worker_count": len(runtime.agents),
            "judge_enabled": runtime.judge is not None,
        },
        "task_metadata": _runtime_task_metadata(config),
        "decoding_config": {
            "provider": config.model.provider,
            "model": config.model.model,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "timeout": config.model.timeout,
            "max_retries": config.model.max_retries,
            "disable_thinking": config.model.disable_thinking,
        },
        "method": {
            "method_id": method.method_id,
            "display_name": method.display_name,
            "method_params": dict(method.params),
        },
        "worker_roster": [_serialize_agent(agent) for agent in runtime.agents],
        "judge": None if runtime.judge is None else _serialize_agent(runtime.judge),
    }


def _build_runtime(config: EvalConfig) -> tuple[Any, Any, MASRuntime]:
    method = get_method(config.method)
    method_params = (
        _messagesieve_method_params(config)
        if method.method_id == "messagesieve"
        else config.method_params
    )
    method.configure(method_params)
    runtime_task_metadata = _runtime_task_metadata(config)
    preset = apply_prompt_protocol(get_dataset_preset(config.dataset.dataset_id), runtime_task_metadata)
    if config.runtime.num_rounds_override is not None:
        preset = replace(preset, num_rounds=config.runtime.num_rounds_override)
    task_pack = get_task_pack(config.dataset.dataset_id)
    llm = build_llm(config.model)
    workers = method.select_worker_agents(preset.build_workers())
    judge = preset.build_judge() if method.use_judge() else None
    expected_workers = config.runtime.contract.get("expected_worker_count")
    if expected_workers is not None and len(workers) != expected_workers:
        raise ValueError(f"Run config expected {expected_workers} workers; received {len(workers)}")
    expected_judge = config.runtime.contract.get("expected_judge_enabled")
    if expected_judge is not None and (judge is not None) != expected_judge:
        raise ValueError(
            f"Run config expected judge_enabled={expected_judge}; received {judge is not None}"
        )
    runtime = MASRuntime(
        llm=llm,
        preset=preset,
        task_pack=task_pack,
        method=method,
        agents=workers,
        judge=judge,
    )
    return method, preset, runtime


def run_eval(config: EvalConfig) -> Dict[str, Any]:
    method, preset, runtime = _build_runtime(config)
    task_samples = load_task_samples(config.dataset)
    if method.method_id == "messagesieve":
        _validate_messagesieve_samples(config, task_samples)
    for sample in task_samples:
        validate_code_scorer_sample(config.dataset.dataset_id, sample)
    utility_fn = build_utility_fn(config.dataset.dataset_id)
    dataset_key = config.dataset.dataset_id.strip().lower()
    score_name = "solve_rate" if dataset_key in {"math", "mbppplus"} else "acc"
    parallel_samples = max(config.runtime.parallel_samples, 1)
    started_at = datetime.now().isoformat()

    prediction_rows_by_index: List[Dict[str, Any] | None] = [None] * len(task_samples)
    trajectory_rows_by_index: List[Dict[str, Any] | None] = [None] * len(task_samples)
    eval_prompt_tokens = 0
    eval_completion_tokens = 0
    utility_sum = 0.0
    output_root = ensure_dir(config.runtime.output_dir)
    run_dir = ensure_dir(output_root / config.name)
    progress_path = run_dir / "progress.json"
    progress_log_path = run_dir / "progress.log"
    partial_predictions_path = run_dir / "predictions.partial.jsonl"
    partial_trajectories_path = run_dir / "trajectories.partial.jsonl"
    run_contract_path = run_dir / "run_contract.json"
    write_json(
        run_contract_path,
        _build_run_contract(
            config=config,
            method=method,
            preset=preset,
            runtime=runtime,
            num_samples=len(task_samples),
            task_samples=task_samples,
        ),
    )

    def write_progress(
        *,
        status: str,
        done_samples: int,
        current_task_id: str | None = None,
        last_utility: float | None = None,
        error: str | None = None,
    ) -> None:
        current_score = utility_sum / done_samples if done_samples else 0.0
        progress = {
            "name": config.name,
            "method": method.display_name,
            "dataset_id": config.dataset.dataset_id,
            "dataset_name": preset.display_name,
            "status": status,
            "done_samples": done_samples,
            "total_samples": len(task_samples),
            "score_name": score_name,
            "current_score": current_score,
            score_name: current_score,
            "eval_tokens": eval_prompt_tokens + eval_completion_tokens,
            "prompt_tokens": eval_prompt_tokens,
            "completion_tokens": eval_completion_tokens,
            "total_tokens": eval_prompt_tokens + eval_completion_tokens,
            "num_rounds": preset.num_rounds,
            "worker_count": len(runtime.agents),
            "judge_enabled": runtime.judge is not None,
            "skip_failed_samples": bool(config.runtime.skip_failed_samples),
            "run_contract_path": str(run_contract_path),
            "current_task_id": current_task_id,
            "last_utility": last_utility,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
        }
        if error:
            progress["error"] = error
        write_json(progress_path, progress)

    if config.runtime.write_progress:
        write_progress(status="running", done_samples=0)

    def evaluate_sample(index: int, sample):
        task_metadata = dict(sample.metadata)
        task_metadata.setdefault("task_id", sample.task_id)
        task_metadata.setdefault("dataset_id", sample.dataset_id)
        task_metadata.update(_runtime_task_metadata(config))
        if parallel_samples == 1:
            current_runtime = runtime
        else:
            _, _, current_runtime = _build_runtime(config)
        try:
            result = current_runtime.run(sample.prompt, task_metadata=task_metadata)
            judge_metadata = dict(result.judge_turn.metadata or {}) if result.judge_turn else {}
            blocked = judge_metadata.get("blocked") is True
            utility = 0.0 if blocked else float(utility_fn(result.final_text, sample))
        except Exception as exc:
            raise _SampleEvaluationError(
                exc,
                current_runtime.observed_call_evidence(),
            ) from exc
        return index, sample, result, utility

    def record_completed_sample(*, index: int, sample, result, utility: float, done_samples: int) -> None:
        nonlocal eval_prompt_tokens, eval_completion_tokens, utility_sum
        eval_prompt_tokens += result.usage.prompt_tokens
        eval_completion_tokens += result.usage.completion_tokens
        utility_sum += utility
        prediction_row = {
            "task_id": sample.task_id,
            "dataset_id": sample.dataset_id,
            "prediction": result.final_text,
            "target": sample.target,
            "utility": utility,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
        judge_metadata = dict(result.judge_turn.metadata or {}) if result.judge_turn else {}
        if judge_metadata.get("blocked") is True:
            prediction_row["blocked"] = True
            prediction_row["block_reason"] = "provider_content_filter"
        trajectory_row = {
            "task_id": sample.task_id,
            "worker_turns": [
                {
                    "agent_id": turn.agent_id,
                    "round_index": turn.round_index,
                    "prompt": turn.prompt,
                    "response": turn.response,
                    "prompt_tokens": turn.usage.prompt_tokens,
                    "completion_tokens": turn.usage.completion_tokens,
                    "metadata": dict(turn.metadata or {}),
                }
                for turn in result.worker_turns
            ],
            "judge_turn": None
            if result.judge_turn is None
            else {
                "agent_id": result.judge_turn.agent_id,
                "round_index": result.judge_turn.round_index,
                "prompt": result.judge_turn.prompt,
                "response": result.judge_turn.response,
                "prompt_tokens": result.judge_turn.usage.prompt_tokens,
                "completion_tokens": result.judge_turn.usage.completion_tokens,
                "metadata": dict(result.judge_turn.metadata or {}),
            },
            "messages": [
                {
                    "source": message.source,
                    "target": message.target,
                    "round_index": message.round_index,
                    "channel": message.channel,
                    "content": message.content,
                    "metadata": dict(message.metadata),
                }
                for message in result.exchanged_messages
            ],
        }
        prediction_rows_by_index[index] = prediction_row
        trajectory_rows_by_index[index] = trajectory_row

        if config.runtime.write_progress:
            if config.runtime.save_predictions:
                append_jsonl(partial_predictions_path, prediction_row)
            if config.runtime.save_trajectories:
                append_jsonl(partial_trajectories_path, trajectory_row)
            if done_samples % config.runtime.progress_every == 0 or done_samples == len(task_samples):
                current_score = utility_sum / done_samples
                log_line = (
                    f"{datetime.now().isoformat()} "
                    f"sample={done_samples}/{len(task_samples)} "
                    f"{score_name}={current_score:.6f} "
                    f"eval_tokens={eval_prompt_tokens + eval_completion_tokens} "
                    f"last_task_id={sample.task_id} "
                    f"last_utility={utility:.6f}"
                )
                append_text_line(progress_log_path, log_line)
                print(log_line, flush=True)
                write_progress(
                    status="running",
                    done_samples=done_samples,
                    current_task_id=sample.task_id,
                    last_utility=utility,
                )

    def record_failed_sample(*, index: int, sample, exc: Exception, done_samples: int) -> None:
        nonlocal eval_prompt_tokens, eval_completion_tokens, utility_sum
        error_text = _sanitize_artifact_error(exc)
        error_type = str(getattr(exc, "cause_type", exc.__class__.__name__))
        observed = dict(getattr(exc, "observed_call_evidence", {}) or {})
        prompt_tokens = int(observed.get("prompt_tokens") or 0)
        completion_tokens = int(observed.get("completion_tokens") or 0)
        total_tokens = prompt_tokens + completion_tokens
        partial_calls = list(observed.get("calls") or [])
        eval_prompt_tokens += prompt_tokens
        eval_completion_tokens += completion_tokens
        utility = 0.0
        utility_sum += utility
        prediction_row = {
            "task_id": sample.task_id,
            "dataset_id": sample.dataset_id,
            "prediction": "",
            "target": sample.target,
            "utility": utility,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "sample_error": True,
            "error_type": error_type,
            "error": error_text,
            "known_partial_call_count": len(partial_calls),
        }
        trajectory_row = {
            "task_id": sample.task_id,
            "worker_turns": [],
            "judge_turn": None,
            "messages": [],
            "partial_llm_calls": partial_calls,
            "sample_error": True,
            "error_type": error_type,
            "error": error_text,
            "known_partial_call_count": len(partial_calls),
            "known_partial_prompt_tokens": prompt_tokens,
            "known_partial_completion_tokens": completion_tokens,
        }
        prediction_rows_by_index[index] = prediction_row
        trajectory_rows_by_index[index] = trajectory_row

        if config.runtime.write_progress:
            if config.runtime.save_predictions:
                append_jsonl(partial_predictions_path, prediction_row)
            if config.runtime.save_trajectories:
                append_jsonl(partial_trajectories_path, trajectory_row)
            log_line = (
                f"{datetime.now().isoformat()} "
                f"sample={done_samples}/{len(task_samples)} "
                f"{score_name}={utility_sum / done_samples:.6f} "
                f"eval_tokens={eval_prompt_tokens + eval_completion_tokens} "
                f"last_task_id={sample.task_id} "
                f"last_utility=0.000000 "
                f"sample_error={exc.__class__.__name__}: {error_text}"
            )
            append_text_line(progress_log_path, log_line)
            print(log_line, flush=True)
            write_progress(
                status="running",
                done_samples=done_samples,
                current_task_id=sample.task_id,
                last_utility=utility,
                error=f"sample skipped after error: {error_text}",
            )

    def log_sample_started(*, sample_position: int, sample) -> None:
        if not config.runtime.write_progress:
            return
        log_line = (
            f"{datetime.now().isoformat()} "
            f"status=sample_started "
            f"sample={sample_position}/{len(task_samples)} "
            f"task_id={sample.task_id}"
        )
        append_text_line(progress_log_path, log_line)
        print(log_line, flush=True)
        write_progress(
            status="running",
            done_samples=sum(1 for row in prediction_rows_by_index if row is not None),
            current_task_id=sample.task_id,
        )

    try:
        if parallel_samples == 1:
            for index, sample in enumerate(task_samples):
                log_sample_started(sample_position=index + 1, sample=sample)
                try:
                    _, completed_sample, result, utility = evaluate_sample(index, sample)
                except Exception as exc:
                    if not config.runtime.skip_failed_samples and not _is_timeout_exception(exc):
                        raise
                    record_failed_sample(index=index, sample=sample, exc=exc, done_samples=index + 1)
                else:
                    record_completed_sample(
                        index=index,
                        sample=completed_sample,
                        result=result,
                        utility=utility,
                        done_samples=index + 1,
                    )
        else:
            per_sample_timeout_seconds = max(float(config.model.timeout) * 2.0, 300.0)
            with ThreadPoolExecutor(max_workers=parallel_samples) as executor:
                future_to_index: dict[Any, int] = {}
                future_started_at: dict[Any, float] = {}
                next_index = 0
                while next_index < len(task_samples) and len(future_to_index) < parallel_samples:
                    sample = task_samples[next_index]
                    log_sample_started(sample_position=next_index + 1, sample=sample)
                    future = executor.submit(evaluate_sample, next_index, sample)
                    future_to_index[future] = next_index
                    future_started_at[future] = monotonic()
                    next_index += 1
                done_samples = 0
                while future_to_index:
                    completed_futures, _ = wait(
                        future_to_index,
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed_futures:
                        index = future_to_index.pop(future)
                        future_started_at.pop(future, None)
                        result = None
                        caught_exc: BaseException | None = None
                        try:
                            index, completed_sample, result, utility = future.result()
                        except Exception as exc:
                            caught_exc = exc
                            if not config.runtime.skip_failed_samples and not _is_timeout_exception(exc):
                                raise
                            completed_sample = task_samples[index]
                            utility = 0.0
                        done_samples += 1
                        if result is None:
                            record_failed_sample(
                                index=index,
                                sample=completed_sample,
                                exc=caught_exc or RuntimeError("sample returned no result"),
                                done_samples=done_samples,
                            )
                        else:
                            record_completed_sample(
                                index=index,
                                sample=completed_sample,
                                result=result,
                                utility=utility,
                                done_samples=done_samples,
                            )
                        if next_index < len(task_samples):
                            sample = task_samples[next_index]
                            log_sample_started(sample_position=next_index + 1, sample=sample)
                            future = executor.submit(evaluate_sample, next_index, sample)
                            future_to_index[future] = next_index
                            future_started_at[future] = monotonic()
                            next_index += 1
                    now = monotonic()
                    timed_out_futures = [
                        future
                        for future, started_at in list(future_started_at.items())
                        if now - started_at >= per_sample_timeout_seconds
                    ]
                    for future in timed_out_futures:
                        index = future_to_index.get(future)
                        future_started_at.pop(future, None)
                        if index is None:
                            continue
                        if not _cancel_future_before_start(future):
                            # Python cannot stop a running thread safely. Keep it
                            # tracked and wait for its provider-level timeouts
                            # instead of submitting extra unaccounted work.
                            continue
                        future_to_index.pop(future, None)
                        done_samples += 1
                        record_failed_sample(
                            index=index,
                            sample=task_samples[index],
                            exc=TimeoutError(
                                f"sample timeout after {per_sample_timeout_seconds:.1f} seconds"
                            ),
                            done_samples=done_samples,
                        )
                        if next_index < len(task_samples):
                            sample = task_samples[next_index]
                            log_sample_started(sample_position=next_index + 1, sample=sample)
                            replacement = executor.submit(evaluate_sample, next_index, sample)
                            future_to_index[replacement] = next_index
                            future_started_at[replacement] = monotonic()
                            next_index += 1
    except Exception as exc:
        if config.runtime.write_progress:
            safe_error = _sanitize_artifact_error(exc)
            log_line = f"{datetime.now().isoformat()} status=failed error={safe_error}"
            append_text_line(
                progress_log_path,
                log_line,
            )
            print(log_line, flush=True)
            write_progress(
                status="failed",
                done_samples=sum(1 for row in prediction_rows_by_index if row is not None),
                error=safe_error,
            )
        raise

    prediction_rows = [row for row in prediction_rows_by_index if row is not None]
    trajectory_rows = [row for row in trajectory_rows_by_index if row is not None]

    score = utility_sum / len(task_samples) if task_samples else 0.0
    num_sample_errors = sum(bool(row.get("sample_error")) for row in prediction_rows)
    num_correct = sum(float(row.get("utility") or 0.0) >= 1.0 for row in prediction_rows)
    num_blocks = sum(bool(row.get("blocked")) for row in prediction_rows)
    num_empty_predictions = sum(not str(row.get("prediction") or "").strip() for row in prediction_rows)

    summary = {
        "name": config.name,
        "method": method.display_name,
        "dataset_id": config.dataset.dataset_id,
        "dataset_name": preset.display_name,
        "num_samples": len(task_samples),
        "num_scored": len(prediction_rows),
        "num_correct": num_correct,
        "num_incorrect": len(prediction_rows) - num_correct,
        "num_sample_errors": num_sample_errors,
        "num_blocks": num_blocks,
        "num_empty_predictions": num_empty_predictions,
        "denominator_policy": "all_loaded_samples_count",
        "score_name": score_name,
        "score": score,
        score_name: score,
        "eval_tokens": eval_prompt_tokens + eval_completion_tokens,
        "prompt_tokens": eval_prompt_tokens,
        "completion_tokens": eval_completion_tokens,
        "total_tokens": eval_prompt_tokens + eval_completion_tokens,
        "num_rounds": preset.num_rounds,
        "worker_count": len(runtime.agents),
        "judge_enabled": runtime.judge is not None,
        "run_contract_path": str(run_contract_path),
        "timestamp": datetime.now().isoformat(),
    }

    write_json(run_dir / "summary.json", summary)
    if config.runtime.save_predictions:
        write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    if config.runtime.save_trajectories:
        write_jsonl(run_dir / "trajectories.jsonl", trajectory_rows)
    if config.runtime.write_progress:
        log_line = (
            f"{datetime.now().isoformat()} "
            f"status=completed "
            f"samples={len(task_samples)}/{len(task_samples)} "
            f"{score_name}={score:.6f} "
            f"eval_tokens={eval_prompt_tokens + eval_completion_tokens}"
        )
        append_text_line(
            progress_log_path,
            log_line,
        )
        print(log_line, flush=True)
        write_progress(status="completed", done_samples=len(task_samples))

    return {
        "summary": summary,
        "predictions": prediction_rows,
        "trajectories": trajectory_rows,
        "run_dir": str(run_dir),
    }
