"""Deterministic validation and bounded repair for Workflow Planner drafts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from route_schema import RouteDecision, RouteValidationError
from workflow_planner import (
    WorkflowPlan,
    WorkflowPlanStep,
    WorkflowPlanningError,
)


_PLACEHOLDER_PATTERN = re.compile(r"\{[A-Za-z0-9_]+\}|\[[A-Za-z][A-Za-z0-9_-]*\]")


class DAGValidationError(ValueError):
    """A workflow draft failed one or more deterministic checks."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class DAGRepairError(DAGValidationError):
    """A draft remained invalid after the bounded repair budget."""

    def __init__(self, errors: list[str], attempts: list[dict]):
        self.attempts = list(attempts)
        super().__init__(errors)


@dataclass(frozen=True)
class ValidatedWorkflow:
    """A validated plan and its executor-ready RouteDecision."""

    plan: WorkflowPlan
    route_decision: RouteDecision
    repair_attempts: list[dict]

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "route_decision": self.route_decision.to_dict(),
            "repair_attempts": list(self.repair_attempts),
        }


class DAGValidator:
    """Perform deterministic checks against Agent contracts and executor schema."""

    _REF_PATTERN = re.compile(r"^([^\.\s]+)\.(.+)$")
    _VISUAL_TERMS = ("画图", "绘图", "图表", "可视化", "chart", "plot", "visual")
    _ANALYSIS_TERMS = ("分析", "趋势", "analyze", "analysis", "trend", "anomaly")
    _RETRIEVAL_TERMS = (
        "查询",
        "查找",
        "获取",
        "数据库",
        "sql",
        "销售",
        "订单",
        "retrieve",
        "query",
        "database",
    )
    _REPORT_TERMS = ("报告", "报表", "总结", "周报", "report", "summary")

    def __init__(self, agent_specs: Mapping[str, dict], max_steps: int = 6):
        self.agent_specs = dict(agent_specs or {})
        self.max_steps = max(1, int(max_steps))

    def validate(
        self,
        plan: WorkflowPlan | Mapping,
        *,
        original_query: str = "",
    ) -> ValidatedWorkflow:
        normalized = self._coerce_plan(plan)
        errors = self._collect_errors(normalized, original_query=original_query)
        if errors:
            raise DAGValidationError(errors)

        try:
            decision = self._to_route_decision(normalized, original_query=original_query)
        except (RouteValidationError, ValueError) as exc:
            raise DAGValidationError([f"executor schema: {exc}"]) from exc
        return ValidatedWorkflow(
            plan=normalized,
            route_decision=decision,
            repair_attempts=[],
        )

    def validate_or_repair(
        self,
        plan: WorkflowPlan | Mapping,
        *,
        original_query: str = "",
        repair_llm: Optional[Callable[..., Mapping]] = None,
        max_repair_attempts: int = 3,
        use_llm: bool = False,
    ) -> ValidatedWorkflow:
        """Validate, then ask an injected/default repairer to fix errors at most N times."""
        current = self._coerce_plan(plan)
        attempts: list[dict] = []
        budget = max(0, min(3, int(max_repair_attempts)))
        for attempt in range(budget + 1):
            try:
                result = self.validate(current, original_query=original_query)
                return ValidatedWorkflow(
                    plan=result.plan,
                    route_decision=result.route_decision,
                    repair_attempts=attempts,
                )
            except DAGValidationError as exc:
                if attempt >= budget:
                    raise DAGRepairError(exc.errors, attempts) from exc
                if repair_llm is None and not use_llm:
                    raise DAGRepairError(exc.errors, attempts) from exc
                try:
                    payload = self._call_repairer(
                        current,
                        errors=exc.errors,
                        original_query=original_query,
                        attempt=attempt + 1,
                        repair_llm=repair_llm,
                    )
                    current = self._coerce_plan(payload)
                    attempts.append({
                        "attempt": attempt + 1,
                        "status": "repaired",
                        "errors": list(exc.errors),
                    })
                except Exception as repair_exc:
                    attempts.append({
                        "attempt": attempt + 1,
                        "status": "repair_failed",
                        "errors": list(exc.errors),
                        "repair_error": str(repair_exc)[:300],
                    })
                    if attempt >= budget - 1:
                        raise DAGRepairError(exc.errors, attempts) from repair_exc
        raise DAGRepairError(["repair loop exhausted"], attempts)

    def _collect_errors(self, plan: WorkflowPlan, *, original_query: str) -> list[str]:
        errors: list[str] = []
        if not isinstance(plan.dag, list):
            return ["dag must be a list"]
        if not isinstance(plan.atomic_intents, list) or not all(
            isinstance(intent, str) and intent.strip() for intent in plan.atomic_intents
        ):
            errors.append("atomic_intents must be a list of non-empty strings")
        if not isinstance(plan.candidate_agents, list) or not all(
            isinstance(agent_id, str) and agent_id.strip()
            for agent_id in plan.candidate_agents
        ):
            errors.append("candidate_agents must be a list of non-empty strings")
        if len(plan.dag) > self.max_steps:
            errors.append(f"dag has {len(plan.dag)} steps; maximum is {self.max_steps}")
        if isinstance(plan.atomic_intents, list) and len(plan.atomic_intents) != len(plan.dag):
            errors.append("atomic_intents must contain exactly one intent per dag step")
        elif isinstance(plan.atomic_intents, list) and plan.atomic_intents != [step.intent for step in plan.dag]:
            errors.append("atomic_intents must match dag step intents in order")

        for agent_id in plan.candidate_agents if isinstance(plan.candidate_agents, list) else []:
            if agent_id not in self.agent_specs:
                errors.append(f"candidate Agent '{agent_id}' is not registered")

        by_id: dict[str, WorkflowPlanStep] = {}
        for index, step in enumerate(plan.dag):
            path = f"dag[{index}]"
            if not all(
                isinstance(value, str) and value.strip()
                for value in (step.call_id, step.agent, step.intent, step.instruction)
            ):
                errors.append(f"{path} requires non-empty call_id, agent, intent, instruction")
            if _PLACEHOLDER_PATTERN.search(step.instruction or ""):
                errors.append(f"{path}.instruction contains unresolved placeholder")
            if step.call_id in by_id:
                errors.append(f"duplicate call_id '{step.call_id}'")
            by_id[step.call_id] = step
            if step.agent not in self.agent_specs:
                errors.append(f"{path}.agent '{step.agent}' is not registered")
                continue
            if step.agent not in plan.candidate_agents:
                errors.append(
                    f"{path}.agent '{step.agent}' is missing from candidate_agents"
                )
            supported = set(self.agent_specs[step.agent].get("intents", []) or [])
            if step.intent not in supported:
                errors.append(
                    f"{path}.intent '{step.intent}' is not supported by Agent '{step.agent}'"
                )
            if self._looks_like_multiple_intents(step.intent):
                errors.append(f"{path}.intent must contain exactly one atomic intent")
            if not isinstance(step.input_refs, list) or not all(
                isinstance(ref, str) and ref.strip() for ref in step.input_refs
            ):
                errors.append(f"{path}.input_refs must be a list of strings")
            if not isinstance(step.depends_on, list) or not all(
                isinstance(dep, str) and dep.strip() for dep in step.depends_on
            ):
                errors.append(f"{path}.depends_on must be a list of strings")
            elif len(step.depends_on) != len(set(step.depends_on)):
                errors.append(f"{path}.depends_on contains duplicates")

        known = set(by_id)
        for step in plan.dag:
            for dependency in step.depends_on:
                if dependency not in known:
                    errors.append(
                        f"step '{step.call_id}' depends on unknown step '{dependency}'"
                    )
                if dependency == step.call_id:
                    errors.append(f"step '{step.call_id}' cannot depend on itself")

        errors.extend(self._cycle_errors(plan.dag, by_id))
        errors.extend(self._input_ref_errors(plan.dag, by_id))
        errors.extend(self._missing_intermediate_errors(plan, original_query, by_id))
        return _unique(errors)

    def _input_ref_errors(
        self,
        steps: list[WorkflowPlanStep],
        by_id: Mapping[str, WorkflowPlanStep],
    ) -> list[str]:
        errors = []
        for step in steps:
            for ref in step.input_refs:
                match = self._REF_PATTERN.match(ref)
                if not match:
                    errors.append(
                        f"step '{step.call_id}' input_ref '{ref}' must use '<step>.<export>'"
                    )
                    continue
                source_id, export_path = match.groups()
                source = by_id.get(source_id)
                if source is None:
                    errors.append(f"input_ref '{ref}' references unknown step")
                    continue
                if source_id not in step.depends_on:
                    errors.append(
                        f"input_ref '{ref}' source must be a dependency of step '{step.call_id}'"
                    )
                source_spec = self.agent_specs.get(source.agent, {})
                exports = set(source_spec.get("exports", []) or [])
                exports.update(
                    (source_spec.get("output_schema", {}) or {}).get("exports", []) or []
                )
                if export_path not in exports:
                    errors.append(
                        f"input_ref '{ref}' is not exported by Agent '{source.agent}'"
                    )
        return errors

    @staticmethod
    def _cycle_errors(steps: list[WorkflowPlanStep], by_id: Mapping[str, WorkflowPlanStep]) -> list[str]:
        errors = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(call_id: str) -> None:
            if call_id in visiting:
                errors.append("DAG dependency graph contains a cycle")
                return
            if call_id in visited or call_id not in by_id:
                return
            visiting.add(call_id)
            for dependency in by_id[call_id].depends_on:
                visit(dependency)
            visiting.remove(call_id)
            visited.add(call_id)

        for step in steps:
            visit(step.call_id)
        return errors

    def _missing_intermediate_errors(
        self,
        plan: WorkflowPlan,
        query: str,
        by_id: Mapping[str, WorkflowPlanStep],
    ) -> list[str]:
        lowered = (query or "").lower()
        if not lowered:
            return []
        has_retrieval = self._contains(lowered, self._RETRIEVAL_TERMS)
        has_report = self._contains(lowered, self._REPORT_TERMS)
        if not (has_retrieval and has_report):
            return []

        required_groups = []
        if self._contains(lowered, self._VISUAL_TERMS):
            required_groups.append(("data_visualization", "画图/可视化步骤"))
        if self._contains(lowered, self._ANALYSIS_TERMS):
            required_groups.append(("data_analysis", "分析步骤"))

        errors = []
        for intent, label in required_groups:
            choices = {intent}
            if intent == "data_analysis":
                choices.add("trend_analysis")
            matches = [step for step in plan.dag if step.intent in choices]
            if not matches:
                errors.append(
                    f"missing required intermediate {label}: add an Agent step with intent '{intent}'"
                )
                continue
            retrieval_ids = {
                step.call_id
                for step in plan.dag
                if step.intent in {"database_query", "data_retrieval", "knowledge_retrieval"}
                or step.agent == "nl2sql"
            }
            if retrieval_ids and not any(
                self._has_ancestor(step.call_id, retrieval_ids, by_id)
                for step in matches
            ):
                errors.append(
                    f"{label} step must consume an upstream retrieval step"
                )
        return errors

    @staticmethod
    def _has_ancestor(call_id: str, targets: set[str], by_id: Mapping[str, WorkflowPlanStep]) -> bool:
        pending = list(by_id.get(call_id, WorkflowPlanStep("", "", "")).depends_on)
        seen = set()
        while pending:
            current = pending.pop()
            if current in targets:
                return True
            if current in seen or current not in by_id:
                continue
            seen.add(current)
            pending.extend(by_id[current].depends_on)
        return False

    def _to_route_decision(self, plan: WorkflowPlan, *, original_query: str) -> RouteDecision:
        calls = []
        for step in plan.dag:
            calls.append({
                "call_id": step.call_id,
                "intent": step.intent,
                "function": {
                    "name": step.agent,
                    "arguments": {
                        "instruction": step.instruction or original_query,
                        "input_refs": list(step.input_refs),
                    },
                },
                "depends_on": list(step.depends_on),
                "confidence": 1.0,
            })
        payload = {
            "mode": "single" if len(calls) == 1 else "workflow",
            "calls": calls,
            "candidates": [],
            "overall_confidence": 1.0,
            "method": "workflow_planner_validated",
            "reason": "Deterministic DAG validation passed",
        }
        return RouteDecision.from_dict(payload, agent_specs=self.agent_specs)

    def _call_repairer(
        self,
        plan: WorkflowPlan,
        *,
        errors: list[str],
        original_query: str,
        attempt: int,
        repair_llm: Optional[Callable[..., Mapping]],
    ) -> Mapping:
        if repair_llm is not None:
            return repair_llm(
                query=original_query,
                plan=plan.to_dict(),
                errors=list(errors),
                schema=self.repair_schema(self.max_steps),
                attempt=attempt,
            )
        from config import get_llm_config
        from openai import OpenAI

        config = get_llm_config()
        if not config.get("api_key"):
            raise DAGRepairError(
                ["LLM repair requested but DEEPSEEK_API_KEY is not configured"],
                [],
            )
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        prompt = (
            "Repair this Workflow Planner DAG. Return only JSON matching the schema. "
            "Keep every step atomic, use only registered Agent contracts, and fix "
            "all listed deterministic errors.\n"
            f"Current plan: {json.dumps(plan.to_dict(), ensure_ascii=False)}\n"
            f"Validation errors: {json.dumps(errors, ensure_ascii=False)}\n"
            f"Schema: {json.dumps(self.repair_schema(self.max_steps), ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": original_query},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        text = response.choices[0].message.content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        return json.loads(text)

    @staticmethod
    def repair_schema(max_steps: int = 6) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["atomic_intents", "candidate_agents", "dag"],
            "properties": {
                "atomic_intents": {"type": "array", "items": {"type": "string"}},
                "candidate_agents": {"type": "array", "items": {"type": "string"}},
                "dag": {
                    "type": "array",
                    "maxItems": max_steps,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["call_id", "agent", "intent", "input_refs", "depends_on", "instruction"],
                        "properties": {
                            "call_id": {"type": "string"},
                            "agent": {"type": "string"},
                            "intent": {"type": "string"},
                            "input_refs": {"type": "array", "items": {"type": "string"}},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "instruction": {"type": "string"},
                        },
                    },
                },
            },
        }

    @staticmethod
    def _coerce_plan(value: WorkflowPlan | Mapping) -> WorkflowPlan:
        if isinstance(value, WorkflowPlan):
            return value
        if not isinstance(value, Mapping):
            raise DAGValidationError(["workflow plan must be an object"])
        try:
            raw_steps = value.get("dag")
            if not isinstance(raw_steps, list):
                raise WorkflowPlanningError("dag must be a list")
            steps = []
            for raw in raw_steps:
                if not isinstance(raw, Mapping):
                    raise WorkflowPlanningError("each dag step must be an object")
                steps.append(
                    WorkflowPlanStep(
                        call_id=raw.get("call_id", ""),
                        agent=raw.get("agent", ""),
                        intent=raw.get("intent", ""),
                        input_refs=raw.get("input_refs", []),
                        depends_on=raw.get("depends_on", []),
                        instruction=raw.get("instruction", ""),
                    )
                )
            return WorkflowPlan(
                atomic_intents=value.get("atomic_intents", []),
                candidate_agents=value.get("candidate_agents", []),
                dag=steps,
                method=value.get("method", "external"),
                reason=value.get("reason", ""),
            )
        except (TypeError, ValueError) as exc:
            raise DAGValidationError([f"invalid workflow plan shape: {exc}"]) from exc

    @staticmethod
    def _looks_like_multiple_intents(intent: str) -> bool:
        return bool(re.search(r"[,，;；|/]|\band\b|\u548c|\u5e76\u4e14", intent, re.I))

    @staticmethod
    def _contains(text: str, terms: tuple[str, ...]) -> bool:
        return any(term.lower() in text for term in terms)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "DAGRepairError",
    "DAGValidationError",
    "DAGValidator",
    "ValidatedWorkflow",
]
