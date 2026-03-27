"""Ablation study entry point for multi-agent misalignment detector.

Ablation modes:
  without_part2 — skip Coordinator synthesis;
                                    existing Tier 1 sub-agent results are passed directly to code_validator.
  without_part3 — skip code_validator;
                  coordinator output is the final classification.
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
import yaml

from paichecker import package_dir
from paichecker.agents.multi_agent import CoordinatorAgent, NoMatchError
from paichecker.environments.local import LocalEnvironment
from paichecker.models.litellm_model import LitellmModel

MODEL_MAP = {
    "gpt": "openai/gpt-5.3-codex",
    "claude": "openai/claude-sonnet-4-6",
    "gemini": "openai/gemini-3.1-pro-preview",
    "qwen": "openai/qwen3.5-plus",
}

ABLATION_MODES = (
    "without_part2",
    "without_part3",
    "without_issue_analyzer",
    "without_pr_scope_analyzer",
    "without_pr_connection_analyzer",
)

ABLATION_GROUPS: dict[str, list[str]] = {
    "all": list(ABLATION_MODES),
    "phase1": ["without_issue_analyzer", "without_pr_scope_analyzer", "without_pr_connection_analyzer"],
    "parallel-core": ["without_part2", "without_part3", "without_issue_analyzer", "without_pr_scope_analyzer", "without_pr_connection_analyzer"],
}

_jsonl_lock = threading.Lock()

app = typer.Typer()


class AblationCoordinatorAgent(CoordinatorAgent):
    """CoordinatorAgent with ablation support — selectively skips pipeline stages."""

    SINGLE_AGENT_ABLATIONS = {
        "without_issue_analyzer": "issue_analyzer",
        "without_pr_scope_analyzer": "pr_scope_analyzer",
        "without_pr_connection_analyzer": "pr_connection_analyzer",
    }

    def __init__(self, *args, ablation_mode: str, existing_output_dir: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert ablation_mode in ABLATION_MODES, f"Invalid ablation mode: {ablation_mode}"
        self.ablation_mode = ablation_mode
        self.existing_output_dir = existing_output_dir

    def _load_existing_coordinator_output(self, instance_id: str | None) -> str:
        if not instance_id or not self.existing_output_dir:
            raise NoMatchError("without_part3 requires existing coordinator outputs")
        coordinator_path = self.existing_output_dir / instance_id / "coordinator.txt"
        if not coordinator_path.exists():
            raise NoMatchError(f"Missing existing coordinator output: {coordinator_path}")
        return coordinator_path.read_text(encoding="utf-8").strip()

    def _load_existing_tier1_results(self, instance_id: str | None) -> dict[str, str]:
        if not instance_id or not self.existing_output_dir:
            raise NoMatchError("without_part2 requires existing Tier 1 outputs")
        base = self.existing_output_dir / instance_id
        names = ("issue_analyzer", "pr_scope_analyzer", "pr_connection_analyzer")
        missing = [name for name in names if not (base / f"{name}.txt").exists()]
        if missing:
            raise NoMatchError(
                f"Missing existing Tier 1 outputs for {instance_id}: {', '.join(missing)}",
            )
        return {
            name: (base / f"{name}.txt").read_text(encoding="utf-8").strip()
            for name in names
        }

    def _build_raw_data_summary(self, **kwargs) -> str:
        fields = [
            ("Instance ID", "instance_id"),
            ("Issue Number", "issue_number"),
            ("Issue Description", "problem_statement"),
            ("Issue Discussion", "hints_text"),
            ("PRs Mentioning Issue", "is_issue_mentioned"),
            ("PR Number", "pr_number"),
            ("PR Description", "pr_description"),
            ("PR Comments", "pr_comments"),
            ("Commit Messages", "commit_message"),
            ("Review Comments", "review_comments"),
            ("PRs/Issues Mentioning PR", "is_pr_mentioned"),
            ("Patch", "patch"),
            ("Test Patch", "test_patch"),
            ("Changed Files", "files"),
        ]
        lines = ["### Raw Data (no sub-agent pre-analysis performed)"]
        lines.extend(f"- **{label}**: {kwargs.get(key, 'N/A')}" for label, key in fields)
        return "\n".join(lines)

    def _coordinator_synthesize_from_raw_data(self, raw_data_summary: str) -> str:
        system_msg = self.coordinator_config["system_template"]
        instance_msg = (
            "You are running an ablation where Tier 1 sub-agents are removed.\n"
            "The input below is raw issue/PR data, not Tier 1 agent outputs.\n\n"
            "Task:\n"
            "1. Analyze the raw data directly.\n"
            "2. Infer supported labels from this evidence only (SC, FP, DP, IS, UL, Others, No Misalignment).\n"
            "3. Output one or more <classification> blocks with 2-3 sentence reasons.\n"
            "4. Output No Misalignment only if no other label applies.\n\n"
            "Raw Data:\n"
            f"{raw_data_summary}"
        )
        self.messages = [
            {"role": "system", "content": system_msg, "timestamp": time.time()},
            {"role": "user", "content": instance_msg, "timestamp": time.time()},
        ]
        response = self.model.query(self.messages)
        self.messages.append({"role": "assistant", "content": response["content"], "timestamp": time.time()})
        return response["content"]

    def _can_reuse_code_validator(self, new_coordinator_labels: str, instance_id: str | None) -> str | None:
        """If ablated coordinator labels match the full pipeline, reuse existing code_validator output."""
        if not instance_id or not self.existing_output_dir:
            return None
        existing_coord_path = self.existing_output_dir / instance_id / "coordinator.txt"
        existing_cv_path = self.existing_output_dir / instance_id / "code_validator.txt"
        if not existing_coord_path.exists() or not existing_cv_path.exists():
            return None
        new_labels = {c["label"] for c in self._extract_classifications(new_coordinator_labels)}
        existing_labels = {c["label"] for c in self._extract_classifications(
            existing_coord_path.read_text(encoding="utf-8"),
        )}
        if new_labels == existing_labels:
            print(f"[Ablation] Coordinator labels match full pipeline — reusing code_validator output.")
            reused = existing_cv_path.read_text(encoding="utf-8").strip()
            self._save_sub_agent_output("code_validator", reused, instance_id)
            return reused
        print(f"[Ablation] Coordinator labels differ (new={new_labels}, existing={existing_labels}) — running code_validator.")
        return None

    def _can_reuse_full_pipeline(self, instance_id: str | None) -> str | None:
        """Reuse full pipeline code_validator output directly."""
        if not instance_id or not self.existing_output_dir:
            return None
        existing_cv_path = self.existing_output_dir / instance_id / "code_validator.txt"
        if not existing_cv_path.exists():
            return None
        reused = existing_cv_path.read_text(encoding="utf-8").strip()
        classifications = self._extract_classifications(reused)
        if not classifications:
            return None
        self._save_sub_agent_output("code_validator", reused, instance_id)
        return reused

    AGENT_LABEL_MAP = {
        "issue_analyzer": {"IS"},
        "pr_scope_analyzer": {"SC", "UL"},
        "pr_connection_analyzer": {"DP", "FP"},
    }
    AGENT_JUDGMENT_TAGS = {
        "issue_analyzer": ["judgment"],
        "pr_scope_analyzer": ["sc_judgment", "ul_judgment"],
        "pr_connection_analyzer": ["dp_judgment", "fp_judgment"],
    }

    def _agent_detects_positive(self, agent_name: str, agent_output: str) -> set[str]:
        """Return the set of positive labels detected by a sub-agent."""
        positive: set[str] = set()
        for tag in self.AGENT_JUDGMENT_TAGS.get(agent_name, []):
            m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", agent_output, re.DOTALL)
            if m:
                val = m.group(1).strip().upper()
                if not val.startswith("NO"):
                    label_candidates = self.AGENT_LABEL_MAP.get(agent_name, set())
                    for lbl in label_candidates:
                        if lbl.upper() in val or val in (lbl.upper(),):
                            positive.add(lbl)
        return positive

    def _can_skip_single_agent_ablation(self, skip_agent: str, instance_id: str | None) -> str | None:
        """If the ablated agent didn't affect the outcome, reuse the full pipeline result."""
        if not instance_id or not self.existing_output_dir:
            return None
        agent_path = self.existing_output_dir / instance_id / f"{skip_agent}.txt"
        coord_path = self.existing_output_dir / instance_id / "coordinator.txt"
        cv_path = self.existing_output_dir / instance_id / "code_validator.txt"
        if not agent_path.exists() or not cv_path.exists():
            return None
        agent_output = agent_path.read_text(encoding="utf-8")
        positive_labels = self._agent_detects_positive(skip_agent, agent_output)
        if not positive_labels:
            print(f"[Ablation] {skip_agent} detected nothing positive — reusing full pipeline.")
            return self._can_reuse_full_pipeline(instance_id)
        if coord_path.exists():
            coord_labels = {c["label"] for c in self._extract_classifications(
                coord_path.read_text(encoding="utf-8"),
            )}
            if not (positive_labels & coord_labels):
                print(f"[Ablation] {skip_agent} labels {positive_labels} not in coordinator {coord_labels} — reusing full pipeline.")
                return self._can_reuse_full_pipeline(instance_id)
        print(f"[Ablation] {skip_agent} labels {positive_labels} affected outcome — running API.")
        return None

    def _extract_tier1_labels(self, tier1_results: dict[str, str]) -> set[str]:
        """Extract positive labels from tier1 sub-agent judgments."""
        positive: set[str] = set()
        for name, output in tier1_results.items():
            positive |= self._agent_detects_positive(name, output)
        return positive

    def _can_reuse_validator_for_without_part2(self, tier1_results: dict[str, str], instance_id: str | None) -> str | None:
        """If tier1 labels match the existing coordinator labels, reuse existing code_validator."""
        if not instance_id or not self.existing_output_dir:
            return None
        coord_path = self.existing_output_dir / instance_id / "coordinator.txt"
        cv_path = self.existing_output_dir / instance_id / "code_validator.txt"
        if not coord_path.exists() or not cv_path.exists():
            return None
        tier1_labels = self._extract_tier1_labels(tier1_results)
        coord_labels = {c["label"] for c in self._extract_classifications(
            coord_path.read_text(encoding="utf-8"),
        )}
        # Both have no positive labels → both "No Misalignment" signal
        # Or both have matching positive labels → same signal to code_validator
        if tier1_labels == coord_labels or (not tier1_labels and coord_labels == {"No Misalignment"}):
            print(f"[Ablation] Tier1 labels {tier1_labels or {'No Misalignment'}} match coordinator {coord_labels} — reusing code_validator.")
            return self._can_reuse_full_pipeline(instance_id)
        print(f"[Ablation] Tier1 labels {tier1_labels} differ from coordinator {coord_labels} — running code_validator.")
        return None

    def _run_tier1_without(self, skip_agent: str, **kwargs) -> dict[str, str]:
        """Load Tier 1 outputs with one sub-agent excluded."""
        instance_id = kwargs.get("instance_id")
        existing = self._load_existing_tier1_results(instance_id)
        results: dict[str, str] = {}
        for name in ("issue_analyzer", "pr_scope_analyzer", "pr_connection_analyzer"):
            if name == skip_agent:
                print(f"[Ablation] Skipping {name}.")
                continue
            output = existing[name]
            self._save_sub_agent_output(name, output, instance_id)
            results[name] = output
            print(f"[Coordinator] Reused {name} output.")
        return results

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        instance_id = kwargs.get("instance_id")

        if self.ablation_mode == "without_part1":
            print("[Ablation] Skipping Part 1 (sub-agents). Coordinator uses raw data.")
            raw_summary = self._build_raw_data_summary(**kwargs)
            coordinator_labels = self._coordinator_synthesize_from_raw_data(raw_summary)
            self._save_sub_agent_output("coordinator", coordinator_labels, instance_id)
            reused = self._can_reuse_code_validator(coordinator_labels, instance_id)
            if reused:
                return "Submitted", reused
            final_output = self._run_tier2(task, coordinator_labels, **kwargs) or coordinator_labels
            return "Submitted", final_output

        if self.ablation_mode == "without_part2":
            print("[Ablation] Skipping Part 2 (coordinator). Reusing existing Tier 1 → code_validator.")
            tier1_results = self._load_existing_tier1_results(instance_id)
            for name, output in tier1_results.items():
                self._save_sub_agent_output(name, output, instance_id)
            reused = self._can_reuse_validator_for_without_part2(tier1_results, instance_id)
            if reused:
                return "Submitted", reused
            tier1_summary = self._build_tier1_summary(tier1_results)
            final_output = self._run_tier2(task, tier1_summary, **kwargs) or tier1_summary
            return "Submitted", final_output

        if self.ablation_mode == "without_part3":
            print("[Ablation] Skipping Part 3 (code_validator). Reusing existing coordinator output.")
            coordinator_labels = self._load_existing_coordinator_output(instance_id)
            self._save_sub_agent_output("coordinator", coordinator_labels, instance_id)
            return "Submitted", coordinator_labels

        if self.ablation_mode in self.SINGLE_AGENT_ABLATIONS:
            skip = self.SINGLE_AGENT_ABLATIONS[self.ablation_mode]
            print(f"[Ablation] Skipping single sub-agent: {skip}.")
            early_reuse = self._can_skip_single_agent_ablation(skip, instance_id)
            if early_reuse:
                return "Submitted", early_reuse
            tier1_results = self._run_tier1_without(skip, **kwargs)
            tier1_summary = self._build_tier1_summary(tier1_results)
            coordinator_labels = self._coordinator_synthesize(tier1_summary)
            self._save_sub_agent_output("coordinator", coordinator_labels, instance_id)
            reused = self._can_reuse_code_validator(coordinator_labels, instance_id)
            if reused:
                return "Submitted", reused
            final_output = self._run_tier2(task, coordinator_labels, **kwargs) or coordinator_labels
            return "Submitted", final_output

        raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")


def _run_single(
    record: dict,
    *,
    model_name: str,
    ablation: str,
    config: dict,
    output_path: Path,
    output_dir: Path,
    include_assistant_messages: bool,
    completed_ids: set[str] | None = None,
) -> dict:
    """Core logic for one (ablation, instance) pair. Shared by CLI and batch."""
    instance_id = _require(record, "instance_id")
    if completed_ids is not None:
        if instance_id in completed_ids:
            print(f"[SKIP] {ablation} already completed: {instance_id}")
            return {}
    elif _has_existing_result(output_path, instance_id):
        print(f"[SKIP] {ablation} already completed: {instance_id}")
        return {}

    agent = AblationCoordinatorAgent(
        LitellmModel(model_name=model_name, model_kwargs={"caching": True}),
        LocalEnvironment(),
        coordinator_config=config["coordinator"],
        sub_agent_configs=config["sub_agents"],
        curl_examples=config["shared"]["curl_examples"],
        output_dir=output_dir / f"{ablation}_{_model_short(model_name)}",
        existing_output_dir=output_dir / _model_short(model_name),
        ablation_mode=ablation,
    )

    template_vars = {
        "instance_id": instance_id,
        "issue_number": _require(record, "issue_number"),
        "problem_statement": _require(record, "problem_statement"),
        "hints_text": _require(record, "hints_text"),
        "is_issue_mentioned": _require(record, "is_issue_mentioned"),
        "pr_number": _extract_pr_number(instance_id),
        "pr_description": _require(record, "pr_description"),
        "pr_comments": _require(record, "pr_comments"),
        "commit_message": _require(record, "commit_message"),
        "review_comments": _require(record, "review_comments"),
        "is_pr_mentioned": _require(record, "is_pr_mentioned"),
        "patch": _require(record, "patch"),
        "test_patch": _require(record, "test_patch"),
        "files": _require(record, "files"),
    }

    try:
        status, final_output = agent.run(
            "Detect misalignment between a GitHub Issue Description and the corresponding Pull Request Implementation.",
            **template_vars,
        )
    except NoMatchError as e:
        print(f"[SKIP] {instance_id}: {e}")
        return {}

    run_record = agent.build_run_record(
        instance_id=instance_id,
        status=status,
        final_output=final_output,
        include_assistant_messages=include_assistant_messages,
    )
    run_record["ablation_mode"] = ablation

    _append_jsonl(output_path, run_record)
    usage = run_record.get("token_usage", {})
    print(
        f"[{ablation}] {instance_id} | "
        f"prompt={usage.get('prompt_tokens', 0)}, output={usage.get('completion_tokens', 0)} | "
        f"cost=${run_record.get('estimated_cost_usd', 0.0):.4f}"
    )
    return run_record


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", help="Path to JSONL file"),
    index: int = typer.Option(0, "--index", help="Line index in JSONL (0-based)"),
    model: str = typer.Option("gemini", "--model", help="Model key: gpt, claude, gemini, qwen"),
    ablation: str = typer.Option(..., "--ablation", help="Ablation mode: without_part2, without_part3"),
    output_path: Path = typer.Option(
        Path("ablation_outputs.jsonl"), "--output", help="Where to append JSONL output",
    ),
    output_dir: Path = typer.Option(
        Path("sub_agent_outputs"), "--output-dir", help="Directory to save sub-agent outputs",
    ),
    include_assistant_messages: bool = typer.Option(
        False, "--include-assistant-messages", help="Include assistant message trace",
    ),
) -> dict:
    config = yaml.safe_load(Path(package_dir / "config" / "multi_swe_detector.yaml").read_text())
    return _run_single(
        _read_jsonl_record(input_path, index),
        model_name=MODEL_MAP.get(model, model),
        ablation=ablation,
        config=config,
        output_path=output_path,
        output_dir=output_dir,
        include_assistant_messages=include_assistant_messages,
    )


@app.command()
def batch(
    input_path: Path = typer.Option(..., "--input", help="Path to JSONL file"),
    model: str = typer.Option("all", "--model", help="Model key (gpt/claude/gemini/qwen) or 'all'"),
    ablation: str = typer.Option("all", "--ablation", help="Ablation mode, group name (all/phase1/parallel-core), or single mode"),
    output_base: Path = typer.Option(
        Path("misalign_output_data"), "--output-base", help="Base directory for output files",
    ),
    output_dir: Path = typer.Option(
        Path("sub_agent_outputs"), "--output-dir", help="Directory to save sub-agent outputs",
    ),
    jobs: int = typer.Option(1, "--jobs", "-j", help="Number of parallel workers"),
    include_assistant_messages: bool = typer.Option(
        False, "--include-assistant-messages", help="Include assistant message trace",
    ),
    gemini_filter: Path = typer.Option(
        None, "--gemini-filter",
        help="Path to Gemini main experiment JSONL; only run instances with non-empty Gemini results",
    ),
) -> None:
    """Run ablation study in batch — all instances × ablations × models in one process."""
    records = _read_all_jsonl(input_path)
    config = yaml.safe_load(Path(package_dir / "config" / "multi_swe_detector.yaml").read_text())

    gemini_nonempty_ids: set[str] | None = None
    if gemini_filter and gemini_filter.exists():
        gemini_nonempty_ids = set()
        for line in gemini_filter.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("classifications"):
                gemini_nonempty_ids.add(rec["instance_id"])
        print(f"[Filter] Gemini non-empty filter: {len(gemini_nonempty_ids)} instances")

    ablations = ABLATION_GROUPS.get(ablation, [ablation])
    models = list(MODEL_MAP.keys()) if model == "all" else [model]

    tasks: list[tuple[str, str, str, Path]] = []
    skipped_gemini = 0
    for abl in ablations:
        for mdl in models:
            model_name = MODEL_MAP.get(mdl, mdl)
            out_path = output_base / abl / f"misalign_outputs_{mdl}_lite.jsonl"
            completed = _load_completed_ids(out_path)
            for record in records:
                instance_id = record.get("instance_id", "")
                if instance_id in completed:
                    continue
                if gemini_nonempty_ids is not None and instance_id not in gemini_nonempty_ids:
                    skipped_gemini += 1
                    continue
                tasks.append((abl, mdl, model_name, out_path, record))
    if skipped_gemini:
        print(f"[Filter] Skipped {skipped_gemini} tasks (empty Gemini results)")

    print(f"\n{'='*50}")
    print(f"  Total tasks: {len(tasks)}  |  Workers: {jobs}")
    print(f"{'='*50}\n")

    if not tasks:
        print("Nothing to do.")
        return

    failed = 0

    def _do(args: tuple) -> str:
        abl, mdl, model_name, out_path, record = args
        try:
            _run_single(
                record,
                model_name=model_name,
                ablation=abl,
                config=config,
                output_path=out_path,
                output_dir=output_dir,
                include_assistant_messages=include_assistant_messages,
            )
            return ""
        except Exception as exc:
            iid = record.get("instance_id", "?")
            msg = f"[ERROR] {abl}/{mdl} {iid}: {exc}"
            print(msg)
            return msg

    if jobs <= 1:
        for t in tasks:
            if _do(t):
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for result in as_completed([pool.submit(_do, t) for t in tasks]):
                if result.result():
                    failed += 1

    print(f"\nDone! {len(tasks) - failed}/{len(tasks)} succeeded.")
    if failed:
        print(f"[WARN] {failed} tasks failed.")


# ── helpers ────────────────────────────────────────────────────────────────

def _read_jsonl_record(path: Path, index: int) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if index < 0 or index >= len(lines):
        raise typer.BadParameter(f"Index {index} out of range for {path}")
    return json.loads(lines[index])


def _require(record: dict, key: str):
    if key not in record:
        raise typer.BadParameter(f"Missing required field: {key}")
    return record[key]


def _extract_pr_number(instance_id: str) -> int:
    try:
        return int(str(instance_id).rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid instance_id for PR number: {instance_id}") from exc


def _read_all_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _append_jsonl(path: Path, record: dict) -> None:
    with _jsonl_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _model_short(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip("/"))


def _has_existing_result(path: Path, instance_id: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("instance_id") == instance_id:
            return True
    return False


if __name__ == "__main__":
    app()
