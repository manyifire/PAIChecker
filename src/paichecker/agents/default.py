"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from paichecker import Environment, Model
from paichecker.utils.pricing import estimate_cost_usd


class AgentConfig(BaseModel):
    # Check the config files in paichecker/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

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

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def extract_classifications(self, final_output: str) -> list[dict[str, str]]:
        blocks = re.findall(r"<classification>\s*(.*?)\s*</classification>", final_output, re.DOTALL)
        classifications: list[dict[str, str]] = []
        for block in blocks:
            label_match = re.search(r"<label>\s*(.*?)\s*</label>", block, re.DOTALL)
            reason_match = re.search(r"<reason>\s*(.*?)\s*</reason>", block, re.DOTALL)
            if not label_match:
                continue
            classifications.append(
                {
                    "label": label_match.group(1).strip(),
                    "reason": reason_match.group(1).strip() if reason_match else "",
                }
            )
        return classifications

    def _normalize_for_dedup(self, text: str) -> str:
        return (
            text.strip()
            .replace(r'\"', '"')
            .replace(r"\'", "'")
            .replace(r"\$", "$")
        )

    def _dedupe_classifications(self, classifications: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in classifications:
            label = item.get("label", "").strip()
            reason = item.get("reason", "").strip()
            key = (label, self._normalize_for_dedup(reason))
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"label": label, "reason": self._normalize_for_dedup(reason)})
        return deduped

    def _classifications_to_output(self, classifications: list[dict[str, str]]) -> str:
        return "\n".join(
            (
                "<classification>\n"
                f"<label>{item['label']}</label>\n"
                f"<reason>{item['reason']}</reason>\n"
                "</classification>"
            )
            for item in classifications
        )

    def _extract_classification_xml_blocks(self, text: str) -> str:
        blocks = re.findall(r"<classification>\s*.*?\s*</classification>", text, re.DOTALL)
        return "\n".join(block.strip() for block in blocks)

    def _try_parse_echo_classification(self, text: str) -> list[dict[str, str]]:
        """Try to extract classifications from malformed echo output that doesn't use proper XML."""
        valid_labels = {"SC", "FP", "DP", "IS", "UL", "Others", "No Misalignment"}
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        patterns = [
            # "Label: SC" / "label: DP" style
            r"(?:^|\n)\s*[Ll]abel\s*[:：]\s*(.+?)(?:\n|$).*?(?:[Rr]eason\s*[:：]\s*(.+?)(?:\n\n|\n[Ll]abel|$))",
            # "**Label**: SC" markdown style
            r"\*\*[Ll]abel\*\*\s*[:：]\s*(.+?)(?:\n|$).*?(?:\*\*[Rr]eason\*\*\s*[:：]\s*(.+?)(?:\n\n|\n\*\*|$))",
            # "- Label: SC" list style
            r"-\s*[Ll]abel\s*[:：]\s*(.+?)(?:\n|$).*?(?:-\s*[Rr]eason\s*[:：]\s*(.+?)(?:\n\n|\n-|$))",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.DOTALL):
                label = m.group(1).strip().strip("*`'\"")
                reason = m.group(2).strip().strip("*`'\"") if m.group(2) else ""
                if label in valid_labels and label not in seen:
                    seen.add(label)
                    results.append({"label": label, "reason": reason})
        return results

    def _recover_final_output_from_messages(self, final_output: str) -> str:
        if final_output.strip():
            return final_output
        assistant_contents = [
            message.get("content", "") for message in self.messages if message.get("role") == "assistant"
        ]
        # First pass: look at the marker message and its predecessor (original logic)
        for idx in range(len(assistant_contents) - 1, -1, -1):
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in assistant_contents[idx]:
                continue
            recovered = self._extract_classification_xml_blocks(assistant_contents[idx])
            if recovered:
                return recovered
            if idx > 0:
                recovered = self._extract_classification_xml_blocks(assistant_contents[idx - 1])
                if recovered:
                    return recovered
            break
        # Second pass: search ALL assistant messages for classification XML blocks
        for idx in range(len(assistant_contents) - 1, -1, -1):
            recovered = self._extract_classification_xml_blocks(assistant_contents[idx])
            if recovered:
                return recovered
        # Third pass: check echo output in user (observation) messages for classification content
        for message in reversed(self.messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if "<output>" not in content:
                continue
            output_match = re.search(r"<output>\s*(.*?)\s*</output>", content, re.DOTALL)
            if not output_match:
                continue
            echo_text = output_match.group(1)
            recovered = self._extract_classification_xml_blocks(echo_text)
            if recovered:
                return recovered
            # Try parsing non-XML format from echo output
            parsed = self._try_parse_echo_classification(echo_text)
            if parsed:
                return self._classifications_to_output(parsed)
        # Fourth pass: try non-XML format from any assistant message
        for idx in range(len(assistant_contents) - 1, -1, -1):
            parsed = self._try_parse_echo_classification(assistant_contents[idx])
            if parsed:
                return self._classifications_to_output(parsed)
        return final_output

    def build_run_record(
        self,
        *,
        instance_id: str | None,
        status: str,
        final_output: str,
        include_assistant_messages: bool = False,
    ) -> dict[str, Any]:
        resolved_final_output = self._recover_final_output_from_messages(final_output)
        deduped_classifications = self._dedupe_classifications(self.extract_classifications(resolved_final_output))
        record: dict[str, Any] = {
            "instance_id": instance_id,
            "status": status,
            "final_output": self._classifications_to_output(deduped_classifications)
            if deduped_classifications
            else resolved_final_output,
            "classifications": deduped_classifications,
        }
        record |= self._build_token_cost()
        if include_assistant_messages:
            record["assistant_messages"] = [
                message.get("content", "") for message in self.messages if message.get("role") == "assistant"
            ]
        return record

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(existing + json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def run_and_save(
        self,
        task: str,
        *,
        instance_id: str | None,
        output_path: Path,
        include_assistant_messages: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        run_kwargs = kwargs | ({"instance_id": instance_id} if instance_id is not None else {})
        status, final_output = self.run(task, **run_kwargs)
        record = self.build_run_record(
            instance_id=instance_id,
            status=status,
            final_output=final_output,
            include_assistant_messages=include_assistant_messages,
        )
        self.append_jsonl(output_path, record)
        return record

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def _estimated_cost(self) -> float:
        from paichecker.utils.pricing import estimate_cost_usd
        cost, _, _ = estimate_cost_usd(
            prompt_tokens=int(getattr(self.model, "prompt_tokens", 0) or 0),
            cached_input_tokens=int(getattr(self.model, "cached_input_tokens", 0) or 0),
            completion_tokens=int(getattr(self.model, "completion_tokens", 0) or 0),
            model_name=getattr(getattr(self.model, "config", None), "model_name", None),
        )
        return cost

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self._estimated_cost():
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        # print(response)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            print(f"Parsed action: {actions[0].strip()}")
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                after = "".join(lines[i + 1:])
                if after.strip():
                    raise Submitted(after)
                before = "".join(lines[:i])
                if before.strip():
                    raise Submitted(before)
                # Marker found but nothing before/after: check full output for classification blocks
                full = output.get("output", "")
                blocks = re.findall(r"<classification>\s*.*?\s*</classification>", full, re.DOTALL)
                if blocks:
                    raise Submitted("\n".join(b.strip() for b in blocks))
                raise Submitted("")
