"""Evaluate ablation results for the PAIChecker pipeline.

Ablations:
  - without_part2: remove coordinator
  - without_part3: remove code_validator
  - without_issue_analyzer: remove issue_analyzer only
  - without_pr_scope_analyzer: remove pr_scope_analyzer only
  - without_pr_connection_analyzer: remove pr_connection_analyzer only

Usage:
  python evaluate_ablation.py
  python evaluate_ablation.py --ablations without_part1,without_part3 --models gemini
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from evaluate import (
    EvalResult,
    compute_metrics,
    load_human_labels,
    print_comparison_table,
    print_single_result,
    write_detail_excel,
)

app = typer.Typer()
ABLATIONS = [
    "without_part2", "without_part3",
    "without_issue_analyzer", "without_pr_scope_analyzer", "without_pr_connection_analyzer",
]

ABLATION_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs" / "ablation"


def _find_output_file(ablation: str, model: str) -> Path | None:
    path = ABLATION_DIR / ablation / f"misalign_outputs_{model}_lite.jsonl"
    return path if path.exists() else None


def _normalize_record(rec: dict) -> dict:
    if "classifications" in rec:
        return rec
    final_output = str(rec.get("final_output", "") or "")
    classifications = []
    for block in final_output.split("<classification>"):
        if "</classification>" not in block:
            continue
        label = _extract_tag(block, "label")
        reason = _extract_tag(block, "reason")
        if label:
            classifications.append({"label": label, "reason": reason})
    rec["classifications"] = classifications
    return rec


def _extract_tag(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start < 0 or end < 0:
        return ""
    return text[start + len(tag) + 2:end].strip()


def _load_ablation_outputs(path: Path) -> dict[str, dict]:
    return {
        rec["instance_id"]: _normalize_record(rec)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for rec in [json.loads(line)]
    }


def _write_ablation_excel(human: dict[str, set[str]], results: list[EvalResult], output_path: Path) -> None:
    write_detail_excel(human, results, output_path)


@app.command()
def main(
    output_xlsx: Path = typer.Option(Path(__file__).resolve().parent.parent / "data" / "results" / "evaluation_ablation_results.xlsx", "--output", help="Output Excel path"),
    ablations: str = typer.Option(",".join(ABLATIONS), "--ablations", help="Comma-separated ablations"),
    models: str = typer.Option("gemini", "--models", help="Comma-separated models"),
) -> None:
    ablation_list = [item.strip() for item in ablations.split(",") if item.strip()]
    model_list = [item.strip() for item in models.split(",") if item.strip()]

    human = load_human_labels()
    print(f"Loaded {len(human)} human-labeled instances")

    all_results: list[EvalResult] = []
    for ablation in ablation_list:
        for model in model_list:
            path = _find_output_file(ablation, model)
            if path is None:
                print(f"  [{ablation}/{model}] No output file found, skipping")
                continue
            outputs = _load_ablation_outputs(path)
            if not outputs:
                print(f"  [{ablation}/{model}] Empty output file, evaluating against all instances with empty predictions")
            result = compute_metrics(human, outputs, ablation, model)
            all_results.append(result)
            print_single_result(result)

    if all_results:
        print_comparison_table(all_results)
        _write_ablation_excel(human, all_results, output_xlsx)


if __name__ == "__main__":
    app()
