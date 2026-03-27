"""Multi-agent system for misalignment detection.

Architecture:
  Coordinator (Python orchestrator + LLM synthesis)
  ├── Tier 1: Text-level analysis (serial)
  │   ├── Issue Analyzer              → IS
  │   ├── PR Scope & Literal Analyzer → SC, UL
  │   └── PR Connection Analyzer      → DP, FP
  ├── Coordinator LLM synthesizes Tier 1 → preliminary labels (incl. Others / No Misalignment)
  └── Tier 2: Code-level validation
      └── Code Validator              → validates all labels → FINAL output
"""

import re
import time
from pathlib import Path
from typing import Any

from paichecker import Environment, Model
from paichecker.agents.default import (
    DefaultAgent,
    FormatError,
    LimitsExceeded,
    Submitted,
)
from paichecker.utils.pricing import estimate_cost_usd

from jinja2 import StrictUndefined, Template


class NoMatchError(Exception):
    """Raised when a sub-agent returns empty output (no classification match)."""


class SubAgent(DefaultAgent):
    """A sub-agent that runs independently and returns structured results to the coordinator."""

    def __init__(self, model: Model, env: Environment, *, name: str, **kwargs):
        super().__init__(model, env, **kwargs)
        self.name = name
        self._start_n_calls = 0

    def _estimated_cost(self) -> float:
        cost, _, _ = estimate_cost_usd(
            prompt_tokens=int(getattr(self.model, "prompt_tokens", 0) or 0),
            cached_input_tokens=int(getattr(self.model, "cached_input_tokens", 0) or 0),
            completion_tokens=int(getattr(self.model, "completion_tokens", 0) or 0),
            model_name=getattr(getattr(self.model, "config", None), "model_name", None),
        )
        return cost

    def query(self) -> dict:
        """Override to enforce per-agent step limit instead of cumulative."""
        agent_calls = self.model.n_calls - self._start_n_calls
        if 0 < self.config.step_limit <= agent_calls or 0 < self.config.cost_limit <= self._estimated_cost():
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def run_and_extract(self, task: str, **kwargs) -> str:
        """Run the sub-agent and return the raw final output text."""
        self._start_n_calls = self.model.n_calls
        _status, final_output = self.run(task, **kwargs)
        if final_output.strip():
            return final_output
        for message in reversed(self.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            match = re.search(r"<sub_agent_result>.*?</sub_agent_result>", content, re.DOTALL)
            if match:
                return match.group(0)
            classifications = re.findall(r"<classification>.*?</classification>", content, re.DOTALL)
            if classifications:
                return "\n".join(classifications)
        return final_output


class CoordinatorAgent:
    """Coordinator that orchestrates sub-agents and produces the final classification."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        coordinator_config: dict[str, Any],
        sub_agent_configs: dict[str, dict[str, Any]],
        curl_examples: str = "",
        output_dir: Path | None = None,
    ):
        self.model = model
        self.env = env
        self.coordinator_config = coordinator_config
        self.sub_agent_configs = sub_agent_configs
        self.curl_examples = curl_examples
        self.output_dir = output_dir
        self.messages: list[dict] = []

    def _create_sub_agent(self, name: str) -> SubAgent:
        return SubAgent(
            self.model,
            self.env,
            name=name,
            **self.sub_agent_configs[name],
        )

    def _save_sub_agent_output(self, name: str, output: str, instance_id: str | None) -> None:
        if not self.output_dir:
            return
        dest = self.output_dir / (instance_id or "unknown")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{name}.txt").write_text(output, encoding="utf-8")

    def _load_cached_output(self, name: str, instance_id: str | None) -> str | None:
        if not self.output_dir or not instance_id:
            return None
        path = self.output_dir / instance_id / f"{name}.txt"
        if path.exists():
            print(f"[Coordinator] Reusing cached {name} for {instance_id}")
            return path.read_text(encoding="utf-8")
        return None

    def _run_sub_agent(self, name: str, task: str, max_retries: int = 3, **kwargs) -> str | None:
        """Run a sub-agent with retry. Attaches error info on retry, clears on success."""
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                retry_kwargs = dict(kwargs)
                if last_error:
                    retry_kwargs["previous_error"] = (
                        f"[Retry {attempt}/{max_retries}] Previous attempt failed: {last_error}"
                    )
                agent = self._create_sub_agent(name)
                output = agent.run_and_extract(task, **retry_kwargs)
                if output and output.strip():
                    return output
                last_error = "empty output"
            except Exception as e:
                last_error = str(e)
            print(f"[Coordinator] {name} attempt {attempt}/{max_retries} failed: {last_error}")
        print(f"[Coordinator] {name} exhausted all {max_retries} retries.")
        return None

    def _run_tier1(self, task: str, **kwargs) -> dict[str, str]:
        """Run all Tier 1 sub-agents serially. Any failure → raise NoMatchError (skip instance)."""
        results: dict[str, str] = {}
        instance_id = kwargs.get("instance_id")
        for name in ("issue_analyzer", "pr_scope_analyzer", "pr_connection_analyzer"):
            cached = self._load_cached_output(name, instance_id)
            if cached:
                results[name] = cached
                print(f"[Coordinator] {name} finished (cached).")
                continue
            output = self._run_sub_agent(
                name, task, curl_examples=self.curl_examples, **kwargs,
            )
            if not output:
                raise NoMatchError(f"Tier 1 sub-agent '{name}' failed or produced no output")
            self._save_sub_agent_output(name, output, instance_id)
            results[name] = output
            print(f"[Coordinator] {name} finished.")
        return results

    def _build_tier1_summary(self, tier1_results: dict[str, str]) -> str:
        parts = []
        for name, output in tier1_results.items():
            parts.append(f"### {name}\n{output}")
        return "\n\n".join(parts)

    def _run_tier2(self, task: str, coordinator_labels: str, **kwargs) -> str | None:
        """Run Code Validator. Returns output or None on failure."""
        output = self._run_sub_agent(
            "code_validator", task,
            coordinator_labels=coordinator_labels, curl_examples=self.curl_examples, **kwargs,
        )
        if output:
            self._save_sub_agent_output("code_validator", output, kwargs.get("instance_id"))
        return output

    def _coordinator_synthesize(self, tier1_summary: str) -> str:
        """Have the coordinator LLM synthesize Tier 1 results into preliminary labels."""
        system_msg = Template(
            self.coordinator_config["system_template"], undefined=StrictUndefined,
        ).render()
        instance_msg = Template(
            self.coordinator_config["instance_template"], undefined=StrictUndefined,
        ).render(tier1_summary=tier1_summary)

        self.messages = [
            {"role": "system", "content": system_msg, "timestamp": time.time()},
            {"role": "user", "content": instance_msg, "timestamp": time.time()},
        ]
        response = self.model.query(self.messages)
        self.messages.append({"role": "assistant", "content": response["content"], "timestamp": time.time()})
        return response["content"]

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Full multi-agent pipeline. Returns (status, final_output)."""
        self._current_instance_id = kwargs.get("instance_id")
        instance_id = kwargs.get("instance_id")

        cached_final = self._load_cached_output("code_validator", instance_id)
        if cached_final:
            print("[Coordinator] Reusing fully cached pipeline result.")
            return "Submitted", cached_final

        print("[Coordinator] Starting Tier 1 analysis...")
        tier1_results = self._run_tier1(task, **kwargs)
        tier1_summary = self._build_tier1_summary(tier1_results)

        cached_coordinator = self._load_cached_output("coordinator", instance_id)
        if cached_coordinator:
            coordinator_labels = cached_coordinator
        else:
            print("[Coordinator] Synthesizing preliminary labels...")
            coordinator_labels = self._coordinator_synthesize(tier1_summary)
            self._save_sub_agent_output("coordinator", coordinator_labels, instance_id)
        print("[Coordinator] Preliminary labels ready.")

        print("[Coordinator] Starting Tier 2 code validation...")
        final_output = self._run_tier2(task, coordinator_labels, **kwargs)
        if final_output:
            print("[Coordinator] Code Validator finished. Pipeline complete.")
        else:
            print("[Coordinator] Code Validator failed — falling back to Coordinator output.")
            final_output = coordinator_labels
        return "Submitted", final_output

    def recover_partial_output(self) -> str:
        """Try to recover classification labels from coordinator messages or saved files."""
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            classifications = self._extract_classifications(msg["content"])
            if classifications:
                return self._classifications_to_output(self._dedupe_classifications(classifications))
        if self.output_dir and getattr(self, "_current_instance_id", None):
            instance_dir = self.output_dir / self._current_instance_id
            for name in ("coordinator", "code_validator"):
                path = instance_dir / f"{name}.txt"
                if path.exists():
                    classifications = self._extract_classifications(path.read_text(encoding="utf-8"))
                    if classifications:
                        return self._classifications_to_output(self._dedupe_classifications(classifications))
        return ""

    # Reuse DefaultAgent's classification extraction / dedup / record-building logic

    def _extract_classifications(self, text: str) -> list[dict[str, str]]:
        blocks = re.findall(r"<classification>\s*(.*?)\s*</classification>", text, re.DOTALL)
        classifications: list[dict[str, str]] = []
        for block in blocks:
            label_match = re.search(r"<label>\s*(.*?)\s*</label>", block, re.DOTALL)
            reason_match = re.search(r"<reason>\s*(.*?)\s*</reason>", block, re.DOTALL)
            if not label_match:
                continue
            classifications.append({
                "label": label_match.group(1).strip(),
                "reason": reason_match.group(1).strip() if reason_match else "",
            })
        return classifications

    def _dedupe_classifications(self, classifications: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in classifications:
            label = item["label"].strip()
            if label in seen:
                continue
            seen.add(label)
            deduped.append(item)
        return deduped

    def _classifications_to_output(self, classifications: list[dict[str, str]]) -> str:
        return "\n".join(
            f"<classification>\n<label>{c['label']}</label>\n<reason>{c['reason']}</reason>\n</classification>"
            for c in classifications
        )

    def _build_token_cost(self) -> dict[str, Any]:
        prompt_tokens = int(getattr(self.model, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(self.model, "completion_tokens", 0) or 0)
        cached_input_tokens = int(getattr(self.model, "cached_input_tokens", 0) or 0)
        estimated_cost_usd, non_cached_input_tokens, pricing = estimate_cost_usd(
            prompt_tokens=prompt_tokens,
            cached_input_tokens=cached_input_tokens,
            completion_tokens=completion_tokens,
            model_name=getattr(getattr(self.model, "config", None), "model_name", None),
        )
        return {
            "model_calls": int(getattr(self.model, "n_calls", 0) or 0),
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
            "litellm_reported_cost_usd": round(float(getattr(self.model, "cost", 0.0) or 0.0), 8),
        }

    def build_run_record(
        self,
        *,
        instance_id: str | None,
        status: str,
        final_output: str,
        include_assistant_messages: bool = False,
    ) -> dict[str, Any]:
        classifications = self._dedupe_classifications(self._extract_classifications(final_output))
        record: dict[str, Any] = {
            "instance_id": instance_id,
            "status": status,
            "final_output": self._classifications_to_output(classifications) if classifications else final_output,
            "classifications": classifications,
        }
        record |= self._build_token_cost()
        if include_assistant_messages:
            record["assistant_messages"] = [
                m["content"] for m in self.messages if m.get("role") == "assistant"
            ]
        return record
