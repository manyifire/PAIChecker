"""Zero-shot, few-shot, and CoT baselines for misalignment detection."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

from paichecker.models.litellm_model import LitellmModel
from paichecker.utils.pricing import estimate_cost_usd

app = typer.Typer()

COST_LIMIT = 0.3
TIME_LIMIT = 300  # 5 minutes

ALL_LABELS = ["SC", "FP", "DP", "IS", "UL", "Others", "No Misalignment"]

MODEL_MAP = {
    "gpt": "openai/gpt-5.3-codex",
    "claude": "openai/claude-sonnet-4-6",
    "gemini": "openai/gemini-3.1-pro-preview",
    "qwen": "openai/qwen3.5-plus",
}

# ── Label definitions (shared across prompts) ─────────────────────────────

LABEL_DEFINITIONS = """\
## Core Concept: Misalignment Labels
Your task is to classify any misalignment between the issue and PR into one or more of the following categories.

### 1. SC (Scope Creep)
- **Concept:** The PR contains extra scope beyond this issue description scope.
- **Applies when:** SC applies only if at least one condition is met:
    1. **Multi-issue closure in PR text**: PR description/commits explicitly indicate fixes for multiple distinct issues, beyond the target issue scope.
    2. **Additional unrelated enhancement/fix**: PR text shows unrelated functionality, bugfix, or enhancement not requested by this issue.
    3. **Cross-issue additions in commits/tests**: Commit messages or review context explicitly reference additional issue work (including separate tests) outside the issue-description scope.

### 2. FP (Follow-up PR)
- **Concept:** This PR is the later same-issue patch that supplements/fixes an earlier PR.
- **Applies when:** FP applies only if ALL conditions are met:
    1. There is an earlier merged PR for the exact same target issue.
    2. This PR is the later supplementary/corrective PR for that same issue chain.
    3. **Multi-issue check:** If `issue_number` contains multiple issues, identify the single target issue that matches the issue description. Search for earlier merged PRs specifically for *that* target issue. 
- **Does NOT apply when:** This PR is the very first or only merged PR for the issue. An older PR merely cross-referenced the issue but didn't actually attempt to fix it.

### 3. DP (Defective PR)
- **Concept:** This PR is the earlier problematic one (buggy or incomplete) that later needed correction.
- **Applies when:** 
    DP applies only if at least one condition is met:
    1. **Bug introduced by this PR**:
    After merge, a subsequent issue or PR explicitly states this PR introduced a bug, and a merged fix exists.
    The follow-up fix may target a different issue. The bug claim must be clearly confirmed (not speculative like `probably` or `maybe`) and closed with a merged fix.
    2. **Incomplete earlier PR with functional follow-up**:
    A subsequent merged PR supplements missing behavior from this PR and adds/updates regression tests.
    The follow-up does not need to be linked to the same issue but must map back to missing behavior in this PR.
    Patch-only follow-ups without tests, or non-functional-only updates (docs/comments/`.gitignore`/`.rst`), do NOT qualify.
    3. **Same-issue discussion-confirmed failure with a later PR**:
    The same issue has multiple PRs; this PR is earlier, and discussion explicitly says it did not fully fix the issue or introduced a bug.
    A later PR for the same issue is merged. Discussion evidence + merge chronology can establish DP when evidence is explicit and non-speculative.

- **Does NOT apply when:** Later PRs/issues casually cross-referencing this PR without claiming it was the source of a defect or regression.

### 4. IS (Incomplete Specification)
- **Concept:** Incomplete Specification means the core problem scope evolved during the discussion, and the PR implemented that updated/sanctioned scope.
- **Applies when:** (Match ANY ONE to classify as IS)
    1. **Maintainer-Requested Supplementation**:
        - A maintainer explicitly asks for more information/clarification AND the reporter responds with the requested details.
        - *Note: Maintainers supplementing their own issues does NOT count.*
    2. **Reporter Proactive Supplementation**:
        - The reporter proactively replies to their own issue to add critical missing pieces.
        - *Valid supplements are limited strictly to: reproduction code/steps, actual behavior, expected behavior, or fixing typos in the original issue description.*
    3. **Discussion-Added Problem Scope or Finalized Requirement**:
        - The discussion introduces a **new, distinct problem to be fixed** that was not in the baseline.
        - OR the baseline raises an ambiguity (e.g., inconsistent names/formats) but doesn't prescribe the exact solution, and the discussion explicitly finalizes the expected behavior/requirement to be implemented.
        - This must be a change in *what to solve* (new expected/actual behavior, new failing edge cases, or finalizing an undecided interface), NOT *how to solve it* (implementation details, root cause analysis, or coding hints).

- **Does NOT apply when:** Just acknowledging or repeating the original requirement.

### 5. UL (Unspecified Literal Implementation)
- **Concept:** Test assertions contain specific literal values not specified in the issue.
- **Applies when:** A completely new hardcoded literal string/value is introduced in the `patch` AND specifically asserted against in `test_patch`, but it is completely absent from the issue description/discussion.
- **Does NOT apply when:** The exact literal string was stated in the issue. Or, the patch is just forwarding pre-existing metadata properties (like accessing `__name__` or `__doc__`).

### 6. Others
- **Concept:** There is clear TEXTUAL evidence of misalignment that fails to fit SC, FP, DP, IS, or UL.

### 7. No Misalignment
- **Concept:** The PR strictly and perfectly addresses the issue at hand.
- **Applies when:** No extra scope, no follow-ups, no defective prs, no unspecified literals, no evolution.
- **Rule of Thumb:** Non-functional differences (e.g., changes to `.gitignore`, comments, documentation) without functional code changes should be treated as No Misalignment. This label is MUTUALLY EXCLUSIVE.

## Strict Decision Rules
1. **Evidence is Mandatory:** You must be able to quote the text or point to the specific component in the patch.
2. **Text is Primary:** Never guess misalignment labels from pure code/diff size alone. There must be textual corroboration (e.g. PR description, discussion, commit message).
3. **No Misalignment is Exclusive:** If you output `No Misalignment`, you cannot output anything else."""

# ── Output format (shared) ─────────────────────────────────────────────────

OUTPUT_FORMAT = """\
## Output Format
Output one or more XML blocks, each with exactly these tags: `<classification>`, `<label>`, `<reason>`.
- `label` must be one of: SC, FP, DP, IS, UL, Others, No Misalignment.
- `reason` must be 2-3 sentences with explicit textual evidence.
- If no misalignment, output exactly one block with `<label>No Misalignment</label>`.
- If `No Misalignment` is present, do not output any other labels.

Example:
<classification>
<label>SC</label>
<reason>PR description states "also fixes #1234", while the target issue is #5678. Commit message includes unrelated bugfix scope.</reason>
</classification>"""

# ── Input data formatting ──────────────────────────────────────────────────

def _format_input(rec: dict) -> str:
    pr_number = _extract_pr_number(rec["instance_id"])
    return f"""\
## Input Data
- Instance ID: {rec["instance_id"]}
- Issue Number: {rec.get("issue_number", "N/A")}
- Issue Description: {rec["problem_statement"]}
- Issue Discussion: {rec.get("hints_text", "N/A")}
- PRs that Mentioned this issue: {rec.get("is_issue_mentioned", "N/A")}
- PR Number: {pr_number}
- PR Description: {rec.get("pr_description", "N/A")}
- PR Comments: {rec.get("pr_comments", "N/A")}
- PR Commit Messages: {rec.get("commit_message", "N/A")}
- PR Code Review Comments: {rec.get("review_comments", "N/A")}
- PRs/Issues that Mentioned this PR After Merging: {rec.get("is_pr_mentioned", "N/A")}
- PR Patch: {rec["patch"]}
- PR Test Patch: {rec["test_patch"]}
- PR Changed Files: {rec.get("files", "N/A")}"""


def _extract_pr_number(instance_id: str) -> int:
    return int(str(instance_id).rsplit("-", 1)[1])


# ── Prompts ────────────────────────────────────────────────────────────────

def _zero_shot_prompt(rec: dict) -> str:
    return f"""\
You are an expert software engineer. Your task is to detect if there is any misalignment between a GitHub Issue description and the corresponding Pull Request (PR) implementation.

{LABEL_DEFINITIONS}

{_format_input(rec)}

{OUTPUT_FORMAT}"""


def _few_shot_prompt(rec: dict) -> str:
    return f"""\
You are an expert software engineer. Your task is to detect if there is any misalignment between a GitHub Issue description and the corresponding Pull Request (PR) implementation.

{LABEL_DEFINITIONS}

## Reference Examples

**Example A — SC (Scope Creep):**
- Issue: "SVG inheritance-diagram produces 404 links for external classes."
- PR description: Explicitly lists multiple closures (e.g., "closing #865, #3176, and #10570 together").
- Classification: `SC` — PR claims broader scope than the single target issue.

**Example B — IS (Incomplete Specification):**
- Issue description: "`default_factory` value appears in generated schema."
- Later discussion: Participants actively add a second related requirement ("`default: None` should also appear in schema").
- Classification: `IS` — Issue scope evolved before merge and PR follows those newly sanctioned details.

**Example C — UL (Unspecified Literal):**
- `patch` introduces: `raise ClientException("Error executing request: Instance role required.")`
- `test_patch` asserts: `assert "Instance role required" in err["Message"]`
- The issue description: Never explicitly specified this exact string or wording.
- Classification: `UL` — Literal is newly introduced and strictly asserted, but not specified in issue text.

**Example D — FP (Follow-up PR):**
- Issue mentions: An older PR (#123) that was merged months ago, which partially fixed this target issue.
- This PR (#456): Description states "Finally completing the fix started in #123".
- Classification: `FP` — This PR is the later same-issue patch supplementing an earlier merged PR.

**Example E — NO_FP (Cross-reference only):**
- Issue mentions lists an earlier PR (#123).
- But that earlier PR says "Fixes #A, also see #B". It did not attempt to fix #B.
- Classification: `No Misalignment` (NO FP) — A mere cross-reference does not make this PR a follow-up.

**Example F — DP (Defective PR):**
- Later artifacts: A merged PR (#789) explicitly states "Fixes regression introduced by #current_pr".
- Classification: `DP` — This PR is the problematic earlier one that later needed correction.

**Example G — No Misalignment:**
- Issue reports a specific 404 bug. PR fixes exactly that 404 bug with matching tests.
- Classification: `No Misalignment` — No extra scope, no hardcoded unspec literals, no follow-ups.

{_format_input(rec)}

{OUTPUT_FORMAT}"""


def _cot_prompt(rec: dict) -> str:
    return f"""\
You are an expert software engineer. Your task is to detect if there is any misalignment between a GitHub Issue description and the corresponding Pull Request (PR) implementation.

{LABEL_DEFINITIONS}

{_format_input(rec)}

## Step-by-Step Analysis Required

Please strictly follow this step-by-step Execution Workflow before giving your final answer:

**Step 1: Check the Baseline Issue Scope (IS)**
- Read the original issue description. What was the exact bug/feature requested?
- Read the issue discussion/comments (`hints_text`). Did the requirements evolve or clarify after the initial post? If yes, and the PR implemented the new requirements → note `IS`.
- Did the author just fix a typo stated in the issue comments? If yes → note `IS`.

**Step 2: Check the PR Text Claims (SC)**
- Read the PR Description, commit messages, and review comments.
- Does the author explicitly say "also fixes #XYZ" or "while I was at it I refactored..."? If yes → note `SC`.
- REMEMBER: Do NOT flag `SC` just because the code patch looks long or adds a lot of tests, unless there's textual evidence.

**Step 3: Check PR Connections (FP/DP)**
- Does `is_issue_mentioned` contain an OLDER, merged PR for the exact same target issue? And does this current PR fix a shortcoming in that older one? If yes → note `FP`.
- Does `is_pr_mentioned` contain a NEWER, merged PR or issue that explicitly blames this PR for a regression (e.g. "regression from #<this_pr>")? If yes → note `DP`.

**Step 4: Check Unspecified Literals (UL)**
- Look at `test_patch`. Are there highly specific hardcoded literal strings/numbers?
- Were these exact literals specified anywhere in the original issue description or discussion? 
- If they are completely new runtime literals introduced in `patch` and tested in `test_patch` without textual specification → note `UL`.

**Step 5: Final Verdict**
- Consolidate all the noted labels. Evaluate `Others` only if something very abnormal happened textually.
- If absolutely zero labels match, you MUST output exactly one `No Misalignment` label.

{OUTPUT_FORMAT}"""


PROMPT_BUILDERS = {
    "zero-shot": _zero_shot_prompt,
    "few-shot": _few_shot_prompt,
    "cot": _cot_prompt,
}

# ── Parse model response ──────────────────────────────────────────────────

def parse_classifications(text: str) -> list[dict]:
    results = []
    for m in re.finditer(r"<classification>\s*<label>(.*?)</label>\s*<reason>(.*?)</reason>\s*</classification>", text, re.DOTALL):
        label = m.group(1).strip()
        reason = m.group(2).strip()
        if label in ALL_LABELS:
            results.append({"label": label, "reason": reason})
    return results or [{"label": "No Misalignment", "reason": "No classification extracted from model response."}]


def _build_token_cost(model: LitellmModel) -> dict[str, Any]:
    prompt_tokens = int(getattr(model, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(model, "completion_tokens", 0) or 0)
    cached_input_tokens = int(getattr(model, "cached_input_tokens", 0) or 0)
    estimated_cost_usd, non_cached_input_tokens, pricing = estimate_cost_usd(
        prompt_tokens=prompt_tokens,
        cached_input_tokens=cached_input_tokens,
        completion_tokens=completion_tokens,
        model_name=getattr(getattr(model, "config", None), "model_name", None),
    )
    return {
        "model_calls": int(getattr(model, "n_calls", 0) or 0),
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "cached_input_tokens": cached_input_tokens,
            "non_cached_input_tokens": non_cached_input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "pricing_usd_per_1m_tokens": {
            "input": pricing.input_per_1m,
            "cached_input": pricing.cached_input_per_1m,
            "output": pricing.output_per_1m,
        },
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "litellm_reported_cost_usd": round(float(getattr(model, "cost", 0.0) or 0.0), 8),
    }


# ── Run one instance ──────────────────────────────────────────────────────

def run_single(rec: dict, method: str, model_name: str) -> dict:
    prompt = PROMPT_BUILDERS[method](rec)
    model = LitellmModel(model_name=model_name, model_kwargs={"caching": True, "temperature": 0, "drop_params": True, "timeout": TIME_LIMIT})
    try:
        response = model.query([{"role": "user", "content": prompt}])
    except Exception as e:
        return _skipped_record(rec["instance_id"], f"error: {e}", model)
    content = response["content"]
    classifications = parse_classifications(content)
    record = {
        "instance_id": rec["instance_id"],
        "status": "completed",
        "final_output": content,
        "classifications": classifications,
    }
    record |= _build_token_cost(model)
    if record.get("estimated_cost_usd", 0) > COST_LIMIT:
        record["status"] = "skipped"
        record["skip_reason"] = f"cost_exceeded ({record['estimated_cost_usd']:.4f} > {COST_LIMIT})"
    return record


def _skipped_record(instance_id: str, reason: str, model: LitellmModel | None = None) -> dict:
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "status": "skipped",
        "skip_reason": reason,
        "final_output": "",
        "classifications": [],
    }
    if model:
        record |= _build_token_cost(model)
    return record


# ── CLI ────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", help="Path to constructed JSONL"),
    method: str = typer.Option(..., "--method", help="zero-shot, few-shot, or cot"),
    model: str = typer.Option("gpt", "--model", help="Model key: gpt, claude, gemini, qwen"),
    output_dir: Path = typer.Option(
        Path(__file__).resolve().parents[3] / "data" / "outputs", "--output-dir", help="Output directory",
    ),
    start: int = typer.Option(0, "--start", help="Start index (inclusive)"),
    end: int = typer.Option(-1, "--end", help="End index (exclusive, -1 = all)"),
) -> None:
    if method not in PROMPT_BUILDERS:
        raise typer.BadParameter(f"Method must be one of: {list(PROMPT_BUILDERS.keys())}")
    model_name = MODEL_MAP.get(model, model)
    model_short = model if model in MODEL_MAP else model.split("/")[-1]

    lines = input_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if end == -1:
        end = len(records)
    records = records[start:end]

    output_path = output_dir / method / f"misalign_outputs_{model_short}_all.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["instance_id"])

    typer.echo(f"Running {method} with {model_name} on {len(records)} instances ({len(done_ids)} already done)")
    for i, rec in enumerate(records):
        if rec["instance_id"] in done_ids:
            typer.echo(f"  [{i+1}/{len(records)}] {rec['instance_id']} — skipped (already done)")
            continue
        typer.echo(f"  [{i+1}/{len(records)}] {rec['instance_id']}")
        result = run_single(rec, method, model_name)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    typer.echo(f"Saved → {output_path}")


if __name__ == "__main__":
    app()
