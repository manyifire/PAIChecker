## Multi-agent misalignment detector entry point.

import json
import os
import re
import signal
from pathlib import Path

import typer
import yaml

from paichecker import package_dir
from paichecker.agents.multi_agent import CoordinatorAgent, NoMatchError
from paichecker.environments.local import LocalEnvironment
from paichecker.models.litellm_model import LitellmModel

app = typer.Typer()

COST_LIMIT = 0.3
TIME_LIMIT = 300  # 5 minutes


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("timeout")


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", help="Path to JSONL file"),
    index: int = typer.Option(0, "--index", help="Line index in JSONL (0-based)"),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME"), "--model", help="Model name (defaults to MSWEA_MODEL_NAME env var)",
    ),
    output_path: Path = typer.Option(
        Path("multi_swe_detector_outputs.jsonl"), "--output", help="Where to append JSONL output",
    ),
    output_dir: Path = typer.Option(
        Path(__file__).resolve().parents[3] / "data" / "sub_agent_outputs",
        "--output-dir", help="Directory to save sub-agent outputs for verification",
    ),
    include_assistant_messages: bool = typer.Option(
        False, "--include-assistant-messages", help="Include assistant message trace",
    ),
) -> dict:
    record = _read_jsonl_record(input_path, index)
    instance_id = _require(record, "instance_id")
    if _is_already_done(output_path, instance_id):
        typer.echo(f"[SKIP] {instance_id}: already in output, skipping.")
        return {}

    config = yaml.safe_load(Path(package_dir / "config" / "multi_swe_detector.yaml").read_text())

    model = LitellmModel(model_name=model_name, model_kwargs={"caching": True})
    env = LocalEnvironment()
    model_output_dir = output_dir / _model_output_dirname(model_name)

    coordinator = CoordinatorAgent(
        model,
        env,
        coordinator_config=config["coordinator"],
        sub_agent_configs=config["sub_agents"],
        curl_examples=config["shared"]["curl_examples"],
        output_dir=model_output_dir,
    )

    task = "Detect misalignment between a GitHub Issue Description and the corresponding Pull Request Implementation."

    template_vars = dict(
        instance_id=instance_id,
        issue_number=_require(record, "issue_number"),
        problem_statement=_require(record, "problem_statement"),
        hints_text=_require(record, "hints_text"),
        is_issue_mentioned=_require(record, "is_issue_mentioned"),
        pr_number=_extract_pr_number(instance_id),
        pr_description=_require(record, "pr_description"),
        pr_comments=_require(record, "pr_comments"),
        commit_message=_require(record, "commit_message"),
        review_comments=_require(record, "review_comments"),
        is_pr_mentioned=_require(record, "is_pr_mentioned"),
        patch=_require(record, "patch"),
        test_patch=_require(record, "test_patch"),
        files=_require(record, "files"),
    )

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIME_LIMIT)
    try:
        status, final_output = coordinator.run(task, **template_vars)
    except _Timeout:
        status, final_output = "skipped", coordinator.recover_partial_output()
        skip_reason = "timeout"
    except NoMatchError as e:
        status, final_output = "skipped", coordinator.recover_partial_output()
        skip_reason = str(e)
    except Exception as e:
        status, final_output = "skipped", coordinator.recover_partial_output()
        skip_reason = f"error: {e}"
    else:
        skip_reason = ""
    finally:
        signal.alarm(0)

    run_record = coordinator.build_run_record(
        instance_id=instance_id,
        status=status,
        final_output=final_output,
        include_assistant_messages=include_assistant_messages,
    )
    if skip_reason:
        run_record["status"] = "skipped"
        run_record["skip_reason"] = skip_reason
    if run_record.get("estimated_cost_usd", 0) > COST_LIMIT and not run_record.get("classifications"):
        run_record["status"] = "skipped"
        run_record["skip_reason"] = f"cost_exceeded ({run_record['estimated_cost_usd']:.4f} > {COST_LIMIT})"

    _append_jsonl(output_path, run_record)
    typer.echo(f"Saved output to {output_path}")
    usage = run_record.get("token_usage", {})
    typer.echo(
        "Token usage: "
        f"prompt={usage.get('prompt_tokens', 0)}, "
        f"cached_input={usage.get('cached_input_tokens', 0)}, "
        f"output={usage.get('completion_tokens', 0)}"
    )
    typer.echo(f"Estimated cost (USD): {run_record.get('estimated_cost_usd', 0.0)}")
    typer.echo(f"Sub-agent outputs dir: {model_output_dir}")
    return run_record


def _read_jsonl_record(path: Path, index: int) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if index < 0 or index >= len(lines):
        raise typer.BadParameter(f"Index {index} out of range for {path}")
    try:
        record = json.loads(lines[index])
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON at line {index}: {exc}") from exc
    if not isinstance(record, dict):
        raise typer.BadParameter("JSONL line must be an object")
    return record


def _require(record: dict, key: str):
    if key not in record:
        raise typer.BadParameter(f"Missing required field: {key}")
    return record[key]


def _is_already_done(output_path: Path, instance_id: str) -> bool:
    if not output_path.exists():
        return False
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("instance_id") == instance_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _extract_pr_number(instance_id: str) -> int:
    try:
        return int(str(instance_id).rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid instance_id for PR number: {instance_id}") from exc


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_output_dirname(model_name: str | None) -> str:
    if not model_name:
        return "unknown_model"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip("/"))


if __name__ == "__main__":
    app()
