from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer

from paichecker.run.baselines import run_single

app = typer.Typer()

MAX_RETRIES = 3
BASELINE_METHODS = ("zero-shot", "few-shot", "cot")
ALL_METHODS = [*BASELINE_METHODS, "mini-swe-agent", "paichecker"]
ALL_MODELS = ["gpt", "claude", "gemini", "qwen"]
FULL_MODEL_NAMES = {
    "gpt": "openai/gpt-5.3-codex",
    "claude": "openai/claude-sonnet-4-6",
    "gemini": "openai/gemini-3.1-pro-preview",
    "qwen": "openai/qwen3.5-plus",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "minisweagent_sample.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
DEFAULT_SUB_AGENT_DIR = PROJECT_ROOT / "data" / "sub_agent_outputs"


@dataclass
class Experiment:
    method: str
    model: str
    output_path: Path


def _run_experiment_worker(
    method: str,
    model: str,
    output_path_str: str,
    records: list[dict],
    input_path_str: str,
    sub_agent_dir_str: str,
    include_assistant_messages: bool,
    missing_indices: list[int],
    max_retries: int = MAX_RETRIES,
    attempt_offsets: dict[str, int] | None = None,
) -> str:
    """Worker function for parallel execution. Returns status string."""
    tag = f"[{method} / {model}]"
    output_path = Path(output_path_str)
    input_path = Path(input_path_str)
    sub_agent_dir = Path(sub_agent_dir_str)
    fails = 0
    cached = 0
    for position, index in enumerate(missing_indices, start=1):
        iid = records[index]["instance_id"]
        print(f"{tag} [{position}/{len(missing_indices)}] {iid}")
        # Cached paichecker records need no retry
        if method == "paichecker":
            cached_record = _try_build_from_cache(sub_agent_dir, FULL_MODEL_NAMES[model], iid)
            if cached_record:
                cached_record["attempt"] = 1
                _append_jsonl(output_path, cached_record)
                cached += 1
                print(f"{tag} [CACHED] {iid}")
                continue
        existing_fails = (attempt_offsets or {}).get(iid, 0)
        remaining = max_retries - existing_fails
        succeeded = False
        for attempt in range(existing_fails + 1, existing_fails + remaining + 1):
            try:
                if method in BASELINE_METHODS:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    result = run_single(records[index], method, FULL_MODEL_NAMES[model])
                    result["attempt"] = attempt
                    _append_jsonl(output_path, result)
                    last_record: dict | None = result
                elif method == "mini-swe-agent":
                    last_record = _run_subprocess_with_attempt(
                        _run_subprocess_mini,
                        attempt=attempt, output_path=output_path,
                        input_path=input_path, index=index,
                        model_name=FULL_MODEL_NAMES[model],
                        include_assistant_messages=include_assistant_messages,
                    )
                else:
                    last_record = _run_subprocess_with_attempt(
                        _run_subprocess_multi,
                        attempt=attempt, output_path=output_path,
                        input_path=input_path, index=index,
                        model_name=FULL_MODEL_NAMES[model],
                        sub_agent_dir=sub_agent_dir,
                        include_assistant_messages=include_assistant_messages,
                    )
            except Exception as e:
                _append_jsonl(output_path, {
                    "instance_id": iid, "status": "attempt_failed",
                    "attempt": attempt, "error": str(e),
                })
                print(f"{tag} [attempt {attempt}/{max_retries} FAIL] {iid}: {e}")
                continue
            # if last_record is None or _result_is_empty(last_record):
            #     _append_jsonl(output_path, {
            #         "instance_id": iid, "status": "attempt_failed",
            #         "attempt": attempt, "error": "empty classifications and final_output",
            #     })
            #     print(f"{tag} [attempt {attempt}/{max_retries} EMPTY] {iid}")
            #     continue
            succeeded = True
            if attempt > 1:
                print(f"{tag} [OK on attempt {attempt}] {iid}")
            break
        if not succeeded:
            fails += 1
    done = len(missing_indices) - fails - cached
    return f"{tag} finished: {done} ok, {cached} cached, {fails} failed (of {len(missing_indices)})"


@app.command()
def main(
    input_path: Path = typer.Option(DEFAULT_INPUT_PATH, "--input", help="Path to JSONL file"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", help="Output root directory"),
    sub_agent_dir: Path = typer.Option(
        DEFAULT_SUB_AGENT_DIR,
        "--sub-agent-dir",
        help="Directory for multi-agent sub-agent outputs",
    ),
    methods: str = typer.Option(
        ",".join(ALL_METHODS),
        "--methods",
        help="Comma-separated methods: zero-shot,few-shot,cot,mini-swe-agent,paichecker",
    ),
    models: str = typer.Option(
        ",".join(ALL_MODELS),
        "--models",
        help="Comma-separated models: gpt,claude,gemini,qwen",
    ),
    include_assistant_messages: bool = typer.Option(
        False,
        "--include-assistant-messages",
        help="Include assistant traces for mini/multi-agent outputs",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report missing experiments"),
    retry_empty_classifications: bool = typer.Option(
        False,
        "--retry-empty-classifications",
        "--retry-skipped",
        help="Retry instances where classifications are empty (keeps --retry-skipped as alias)",
    ),
    max_retries: int = typer.Option(MAX_RETRIES, "--max-retries", help="Max retries per instance"),
    parallel: int = typer.Option(20, "--parallel", "-j", help="Max parallel workers (0=sequential)"),
    batch_size: int = typer.Option(20, "--batch-size", "-b", help="Instances per batch (0=one batch per experiment)"),
) -> None:
    records = _load_input_records(input_path)
    selected_methods = _parse_csv(methods, ALL_METHODS, "methods")
    selected_models = _parse_csv(models, ALL_MODELS, "models")
    experiments = [
        Experiment(method=method, model=model, output_path=_resolve_output_path(output_dir, method, model))
        for method in selected_methods
        for model in selected_models
    ]

    # Scan all experiments, collect work items
    work: list[tuple[Experiment, list[int], dict[str, int]]] = []
    total_missing = 0
    for experiment in experiments:
        done_ids = _load_completed_ids(
            experiment.output_path,
            exclude_empty_classifications=retry_empty_classifications,
            max_retries=max_retries,
        )
        attempt_counts = _load_attempt_counts(experiment.output_path)
        missing_indices = [
            index for index, record in enumerate(records) if record["instance_id"] not in done_ids
        ]
        if retry_empty_classifications and missing_indices:
            empty_ids = _load_ids_with_empty_classifications(experiment.output_path)
            rerun_empty_ids = {
                records[index]["instance_id"] for index in missing_indices if records[index]["instance_id"] in empty_ids
            }
            removed = _remove_records_with_empty_classifications(experiment.output_path, rerun_empty_ids)
            if removed:
                print(f"[{experiment.method} / {experiment.model}] removed_empty_classification_records={removed}")
        offsets = {records[i]["instance_id"]: attempt_counts.get(records[i]["instance_id"], 0) for i in missing_indices}
        total_missing += len(missing_indices)
        cache_info = ""
        if experiment.method == "paichecker" and missing_indices:
            full, partial = _count_cache_status(
                sub_agent_dir, FULL_MODEL_NAMES[experiment.model],
                [records[i]["instance_id"] for i in missing_indices],
            )
            cache_info = f" cache_hit={full} partial_cache={partial}"
        print(
            f"[{experiment.method} / {experiment.model}] "
            f"done={len(done_ids)} missing={len(missing_indices)}{cache_info} output={experiment.output_path}"
        )
        if missing_indices:
            work.append((experiment, missing_indices, offsets))

    if dry_run or not work:
        print(f"\nTotal missing: {total_missing}")
        return

    # Split work into batches for instance-level parallelism
    batches: list[tuple[Experiment, list[int], dict[str, int]]] = []
    for experiment, missing_indices, offsets in work:
        if batch_size > 0:
            for i in range(0, len(missing_indices), batch_size):
                chunk = missing_indices[i:i + batch_size]
                chunk_offsets = {records[j]["instance_id"]: offsets.get(records[j]["instance_id"], 0) for j in chunk}
                batches.append((experiment, chunk, chunk_offsets))
        else:
            batches.append((experiment, missing_indices, offsets))

    # Run experiments
    workers = min(parallel, len(batches)) if parallel > 0 else 1
    print(f"\nLaunching {len(batches)} batches with {workers} parallel workers...")
    start = time.time()

    if workers <= 1:
        # Sequential mode
        for experiment, indices, offsets in batches:
            print(
                _run_experiment_worker(
                    experiment.method, experiment.model, str(experiment.output_path),
                    records, str(input_path), str(sub_agent_dir),
                    include_assistant_messages, indices, max_retries, offsets,
                )
            )
    else:
        # Parallel mode: each batch runs in its own process
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_experiment_worker,
                    exp.method, exp.model, str(exp.output_path),
                    records, str(input_path), str(sub_agent_dir),
                    include_assistant_messages, indices, max_retries, offsets,
                ): f"{exp.method}/{exp.model}[{i}]"
                for i, (exp, indices, offsets) in enumerate(batches)
            }
            failures: list[str] = []
            for future in as_completed(futures):
                tag = futures[future]
                try:
                    print(future.result())
                except Exception as exc:
                    failures.append(f"{tag}: {exc}")
                    print(f"[FAIL] {tag}: {exc}")

        elapsed = time.time() - start
        print(f"\nAll done in {elapsed:.0f}s. {len(batches)} batches, {workers} parallel workers.")
        if failures:
            print("Failures:\n" + "\n".join(failures))


def _load_input_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_csv(raw: str, allowed: list[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise typer.BadParameter(f"Invalid {label}: {', '.join(invalid)}")
    return values


def _resolve_output_path(output_dir: Path, method: str, model: str) -> Path:
    folder = "checker" if method == "paichecker" else method
    candidates = [
        output_dir / folder / f"misalign_outputs_{model}_all.jsonl",
        output_dir / folder / f"misalign_outputs_{model}_lite.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _record_has_empty_classifications(record: dict) -> bool:
    if "classifications" not in record:
        return False
    classifications = record.get("classifications")
    return classifications is None or (isinstance(classifications, list) and not classifications)


def _load_completed_ids(
    path: Path,
    *,
    exclude_empty_classifications: bool = False,
    max_retries: int = MAX_RETRIES,
) -> set[str]:
    if not path.exists():
        return set()
    successful: set[str] = set()
    fail_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        instance_id = record.get("instance_id")
        if not instance_id:
            continue
        if exclude_empty_classifications and _record_has_empty_classifications(record):
            continue
        if record.get("status") == "attempt_failed":
            fail_counts[instance_id] = fail_counts.get(instance_id, 0) + 1
        else:
            successful.add(instance_id)
    return successful | {iid for iid, c in fail_counts.items() if c >= max_retries}


def _load_attempt_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "attempt_failed":
            iid = record.get("instance_id")
            if iid:
                counts[iid] = counts.get(iid, 0) + 1
    return counts


def _load_ids_with_status(path: Path, status: str) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != status:
            continue
        instance_id = record.get("instance_id")
        if instance_id:
            ids.add(instance_id)
    return ids


def _load_ids_with_empty_classifications(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _record_has_empty_classifications(record):
            continue
        instance_id = record.get("instance_id")
        if instance_id:
            ids.add(instance_id)
    return ids


def _remove_records_by_status(path: Path, status: str, instance_ids: set[str]) -> int:
    if not path.exists() or not instance_ids:
        return 0
    kept_lines: list[str] = []
    removed = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(raw_line)
            continue
        if record.get("status") == status and record.get("instance_id") in instance_ids:
            removed += 1
            continue
        kept_lines.append(raw_line)
    if kept_lines:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")
    return removed


def _remove_records_with_empty_classifications(path: Path, instance_ids: set[str]) -> int:
    if not path.exists() or not instance_ids:
        return 0
    kept_lines: list[str] = []
    removed = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(raw_line)
            continue
        if _record_has_empty_classifications(record) and record.get("instance_id") in instance_ids:
            removed += 1
            continue
        kept_lines.append(raw_line)
    if kept_lines:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")
    return removed


def _model_output_dirname(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip("/"))


def _extract_classifications(text: str) -> list[dict[str, str]]:
    blocks = re.findall(r"<classification>\s*(.*?)\s*</classification>", text, re.DOTALL)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        label_m = re.search(r"<label>\s*(.*?)\s*</label>", block, re.DOTALL)
        reason_m = re.search(r"<reason>\s*(.*?)\s*</reason>", block, re.DOTALL)
        if not label_m:
            continue
        label = label_m.group(1).strip()
        if label in seen:
            continue
        seen.add(label)
        result.append({"label": label, "reason": reason_m.group(1).strip() if reason_m else ""})
    return result


def _try_build_from_cache(sub_agent_dir: Path, model_name: str, instance_id: str) -> dict | None:
    """Build run record from cached code_validator.txt if it exists."""
    cv_path = sub_agent_dir / _model_output_dirname(model_name) / instance_id / "code_validator.txt"
    if not cv_path.exists():
        return None
    content = cv_path.read_text(encoding="utf-8").strip()
    classifications = _extract_classifications(content)
    if not classifications:
        return None
    final_output = "\n".join(
        f"<classification>\n<label>{c['label']}</label>\n<reason>{c['reason']}</reason>\n</classification>"
        for c in classifications
    )
    return {
        "instance_id": instance_id,
        "status": "Submitted",
        "final_output": final_output,
        "classifications": classifications,
        "cached": True,
        "model_calls": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "estimated_cost_usd": 0.0,
    }


def _count_cache_status(
    sub_agent_dir: Path, model_name: str, instance_ids: list[str],
) -> tuple[int, int]:
    """Count instances with full cache (code_validator) and partial cache (any sub-agent file)."""
    model_dir = sub_agent_dir / _model_output_dirname(model_name)
    full = partial = 0
    for iid in instance_ids:
        d = model_dir / iid
        if not d.exists():
            continue
        if (d / "code_validator.txt").exists():
            full += 1
        else:
            partial += 1
    return full, partial


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _result_is_empty(record: dict) -> bool:
    classifications = record.get("classifications")
    final_output = record.get("final_output")
    return not classifications and not final_output


def _run_subprocess_with_attempt(
    subprocess_fn, *, attempt: int, output_path: Path, **kwargs,
) -> dict | None:
    """Run a subprocess method, capture its output in a temp file, tag with attempt number."""
    tmp_path = Path(tempfile.mktemp(suffix=".jsonl"))
    last_record: dict | None = None
    try:
        subprocess_fn(output_path=tmp_path, **kwargs)
        if tmp_path.exists():
            for line in tmp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec["attempt"] = attempt
                    _append_jsonl(output_path, rec)
                    last_record = rec
    finally:
        tmp_path.unlink(missing_ok=True)
    return last_record


def _run_subprocess_mini(
    *,
    input_path: Path,
    index: int,
    model_name: str,
    output_path: Path,
    include_assistant_messages: bool,
) -> None:
    cmd = [
        sys.executable, "-m", "paichecker.run.mini_swe_detector",
        "--input", str(input_path),
        "--index", str(index),
        "--model", model_name,
        "--output", str(output_path),
    ]
    if include_assistant_messages:
        cmd.append("--include-assistant-messages")
    subprocess.run(cmd, check=True, timeout=360)


def _run_subprocess_multi(
    *,
    input_path: Path,
    index: int,
    model_name: str,
    output_path: Path,
    sub_agent_dir: Path,
    include_assistant_messages: bool,
) -> None:
    cmd = [
        sys.executable, "-m", "paichecker.run.multi_swe_detector",
        "--input", str(input_path),
        "--index", str(index),
        "--model", model_name,
        "--output", str(output_path),
        "--output-dir", str(sub_agent_dir),
    ]
    if include_assistant_messages:
        cmd.append("--include-assistant-messages")
    subprocess.run(cmd, check=True, timeout=360)


if __name__ == "__main__":
    app()
