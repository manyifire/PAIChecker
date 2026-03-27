## mini-swe-agent baseline. 用脚本run_mini_swe_detector跑就行

import json
import os
import re
import signal
from pathlib import Path

import typer
import yaml

from paichecker import package_dir
from paichecker.agents.default import DefaultAgent
from paichecker.environments.local import LocalEnvironment
from paichecker.models.litellm_model import LitellmModel

app = typer.Typer()

COST_LIMIT = 0.3
TIME_LIMIT = 300  # 5 minutes


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("timeout")


def _recover_from_messages(agent: DefaultAgent) -> str:
    """Extract classification XML from agent messages as fallback."""
    for msg in reversed(agent.messages):
        if msg.get("role") != "assistant":
            continue
        blocks = re.findall(r"<classification>.*?</classification>", msg.get("content", ""), re.DOTALL)
        if blocks:
            return "\n".join(blocks)
    return ""


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", help="Path to JSONL file", show_default=True, prompt=False),
    index: int = typer.Option(
        0, "--index", help="Line index in JSONL (0-based)", show_default=True, prompt=False
    ),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME"),
        "--model",
        help="Model name (defaults to MSWEA_MODEL_NAME env var)",
        prompt=False,
    ),
    output_path: Path = typer.Option(
        Path("mini_swe_detector_outputs.jsonl"),
        "--output",
        help="Where to append structured model outputs as JSONL",
        show_default=True,
        prompt=False,
    ),
    include_assistant_messages: bool = typer.Option(
        False,
        "--include-assistant-messages",
        help="Include assistant message trace in output JSONL",
        show_default=True,
        prompt=False,
    ),
) -> dict:
    record = _read_jsonl_record(input_path, index)
    instance_id = _require(record, "instance_id")
    if output_path.exists():
        done_ids = {json.loads(l)["instance_id"] for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        if instance_id in done_ids:
            typer.echo(f"Skipping {instance_id} (already done)")
            return {}
    agent = DefaultAgent(
        LitellmModel(model_name=model_name, model_kwargs={"caching": True}),
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "mini_swe_detector.yaml").read_text())["agent"],
    )
    task = "Detect misalignment between a GitHub Issue description (problem_statement) and the corresponding Pull Request (PR) using the provided fields."
    run_kwargs = dict(
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
        status, final_output = agent.run(task, **run_kwargs)
    except _Timeout:
        status, final_output = "skipped", _recover_from_messages(agent)
        skip_reason = "timeout"
    except Exception as e:
        status, final_output = "skipped", _recover_from_messages(agent)
        skip_reason = f"error: {e}"
    else:
        skip_reason = ""
    finally:
        signal.alarm(0)

    run_record = agent.build_run_record(
        instance_id=instance_id,
        status=status,
        final_output=final_output,
        include_assistant_messages=include_assistant_messages,
    )
    if skip_reason:
        run_record["status"] = "skipped"
        run_record["skip_reason"] = skip_reason
    if run_record.get("estimated_cost_usd", 0) > COST_LIMIT:
        run_record["status"] = "skipped"
        run_record["skip_reason"] = f"cost_exceeded ({run_record['estimated_cost_usd']:.4f} > {COST_LIMIT})"

    agent.append_jsonl(output_path, run_record)
    typer.echo(f"Saved output to {output_path}")
    usage = run_record.get("token_usage", {})
    typer.echo(
        "Token usage: "
        f"prompt={usage.get('prompt_tokens', 0)}, "
        f"cached_input={usage.get('cached_input_tokens', 0)}, "
        f"output={usage.get('completion_tokens', 0)}"
    )
    typer.echo(f"Estimated cost (USD): {run_record.get('estimated_cost_usd', 0.0)}")
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


def _extract_pr_number(instance_id: str) -> int:
    try:
        return int(str(instance_id).rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid instance_id for PR number: {instance_id}") from exc


if __name__ == "__main__":
    app()
