"""Workflow Planner: turn a compound request into a structured DAG draft.

The planner is intentionally separate from route execution and DAG validation.
It consumes the Agent capability contracts registered by :mod:`agent_registry`
and produces a draft that can be handed to the validator in the next stage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional


class WorkflowPlanningError(ValueError):
    """Raised when a planner proposal cannot be represented as a draft."""


@dataclass(frozen=True)
class WorkflowPlanStep:
    """One atomic draft operation in a workflow."""

    call_id: str
    agent: str
    intent: str
    input_refs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    instruction: str = ""

    def to_dict(self) -> dict:
        result = {
            "call_id": self.call_id,
            "agent": self.agent,
            "intent": self.intent,
            "input_refs": list(self.input_refs),
        }
        if self.depends_on:
            result["depends_on"] = list(self.depends_on)
        if self.instruction:
            result["instruction"] = self.instruction
        return result


@dataclass(frozen=True)
class WorkflowPlan:
    """Structured workflow draft emitted by :class:`WorkflowPlanner`."""

    atomic_intents: list[str]
    candidate_agents: list[str]
    dag: list[WorkflowPlanStep]
    method: str = "contract_fallback"
    reason: str = ""

    def to_dict(self) -> dict:
        # Keep the wire format aligned with the planner schema. Diagnostic
        # fields remain available as ``plan.method`` and ``plan.reason``.
        return {
            "atomic_intents": list(self.atomic_intents),
            "candidate_agents": list(self.candidate_agents),
            "dag": [step.to_dict() for step in self.dag],
        }


class WorkflowPlanner:
    """Create a draft DAG from Agent contracts.

    ``llm_planner`` is an injectable callable with the same spirit as Router's
    structured LLM override. It receives the request, candidate profiles and
    JSON schema, and returns a decoded mapping. The default local planner keeps
    tests and offline development deterministic; callers can opt into the LLM
    path with ``use_llm=True``.
    """

    _CLAUSE_SPLIT = re.compile(
        r"(?:，|,|；|;|\n|\s+then\s+|\s+and\s+then\s+|\s+finally\s+|"
        r"然后|接着|最后|再|并且|同时|以及|之后|随后|并)"
    )
    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
    _EXPORT_PREFERENCE = (
        "query_result",
        "dataframe",
        "data.structured",
        "data.artifacts",
        "data.report",
        "sources",
        "summary",
        "answer",
        "sql",
    )
    _INTENT_HINTS = {
        "database_query": ("查询", "数据库", "sql", "query", "sales", "订单"),
        "data_retrieval": ("查询", "获取", "retrieve", "data"),
        "trend_analysis": ("趋势", "trend", "变化", "走势"),
        "data_analysis": ("analyze", "analysis", "统计"),
        "data_visualization": ("画图", "绘图", "图表", "chart", "plot", "可视化"),
        "report_generation": ("报告", "report", "周报", "汇总"),
        "summary_writing": ("总结", "摘要", "summary", "概括"),
    }

    def __init__(
        self,
        registry=None,
        agent_specs: Optional[Mapping[str, dict]] = None,
        llm_planner: Optional[Callable[..., dict]] = None,
        client=None,
        model: str = "",
    ):
        self.registry = registry
        self._agent_specs_override = dict(agent_specs or {})
        self.llm_planner = llm_planner
        self.client = client
        self.model = model

    @staticmethod
    def json_schema(max_steps: int = 6) -> dict:
        """Return the function-calling-like schema for planner drafts."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["atomic_intents", "candidate_agents", "dag"],
            "properties": {
                "atomic_intents": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "candidate_agents": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "dag": {
                    "type": "array",
                    "maxItems": max_steps,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "call_id",
                            "agent",
                            "intent",
                            "input_refs",
                        ],
                        "properties": {
                            "call_id": {"type": "string"},
                            "agent": {"type": "string"},
                            "intent": {"type": "string"},
                            "input_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "instruction": {"type": "string"},
                        },
                    },
                },
            },
        }

    def plan(
        self,
        user_request: str,
        *,
        context: str = "",
        use_llm: bool = False,
        max_steps: int = 6,
    ) -> WorkflowPlan:
        if not isinstance(user_request, str) or not user_request.strip():
            raise WorkflowPlanningError("user_request must be a non-empty string")
        if max_steps < 1:
            raise WorkflowPlanningError("max_steps must be positive")

        specs = self._get_specs()
        if not specs:
            raise WorkflowPlanningError("no registered Agent contracts are available")
        candidates = self._rank_candidates(user_request, specs)

        if use_llm or self.llm_planner is not None:
            payload = self._call_llm(
                user_request,
                context=context,
                candidates=candidates,
                specs=specs,
                max_steps=max_steps,
            )
            return self._parse_payload(
                payload,
                specs=specs,
                max_steps=max_steps,
                method="llm_planner",
            )

        return self._contract_fallback(
            user_request,
            specs=specs,
            candidates=candidates,
            max_steps=max_steps,
        )

    def _get_specs(self) -> dict[str, dict]:
        if self._agent_specs_override:
            return dict(self._agent_specs_override)
        if self.registry is not None:
            for method_name in ("get_router_config", "list_all"):
                method = getattr(self.registry, method_name, None)
                if callable(method):
                    specs = method() or {}
                    if specs:
                        return dict(specs)
        return {}

    def _rank_candidates(
        self,
        query: str,
        specs: Mapping[str, dict],
        limit: int = 6,
    ) -> list[str]:
        query_lower = query.lower()
        query_tokens = set(self._TOKEN_PATTERN.findall(query_lower))
        scored = []
        for agent_id, spec in specs.items():
            if spec.get("is_fallback") or agent_id == "general":
                continue
            terms = [
                *(spec.get("keywords", []) or []),
                *(spec.get("intents", []) or []),
                *(spec.get("examples", []) or []),
                *(item.get("query", "") for item in (spec.get("few_shots", []) or [])),
                spec.get("description", ""),
            ]
            score = 0.0
            for term in terms:
                term_text = str(term).lower()
                if term_text and term_text in query_lower:
                    score += 2.0 if len(term_text) > 1 else 0.25
                term_tokens = set(self._TOKEN_PATTERN.findall(term_text))
                if term_tokens:
                    score += min(1.0, len(query_tokens & term_tokens) * 0.15)
            scored.append((score, agent_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        positive = [agent_id for score, agent_id in scored if score > 0]
        ranked = positive or [agent_id for _, agent_id in scored]
        return ranked[: max(1, int(limit))]

    def _contract_fallback(
        self,
        query: str,
        *,
        specs: Mapping[str, dict],
        candidates: list[str],
        max_steps: int,
    ) -> WorkflowPlan:
        clauses = [item.strip() for item in self._CLAUSE_SPLIT.split(query) if item.strip()]
        if not clauses:
            clauses = [query.strip()]
        clauses = clauses[:max_steps]

        steps: list[WorkflowPlanStep] = []
        used_agents: list[str] = []
        for index, clause in enumerate(clauses, start=1):
            agent_id = self._best_agent_for_clause(clause, candidates, specs)
            if agent_id is None:
                continue
            spec = specs[agent_id]
            intent = self._best_intent(clause, spec)
            depends_on = [steps[-1].call_id] if steps else []
            source_steps = steps[-1:] if steps else []
            if agent_id == "report_writer" and steps:
                depends_on = [step.call_id for step in steps]
                source_steps = steps
            input_refs = []
            for source_step in source_steps:
                source_spec = specs[source_step.agent]
                export = self._preferred_export(source_spec)
                if export:
                    input_refs.append(f"{source_step.call_id}.{export}")
            call_id = f"step_{index}"
            steps.append(
                WorkflowPlanStep(
                    call_id=call_id,
                    agent=agent_id,
                    intent=intent,
                    input_refs=input_refs,
                    depends_on=depends_on,
                    instruction=clause,
                )
            )
            if agent_id not in used_agents:
                used_agents.append(agent_id)

        if not steps:
            raise WorkflowPlanningError("no Agent matched the request")
        atomic_intents = [step.intent for step in steps]
        candidate_agents = [agent_id for agent_id in candidates if agent_id in specs]
        for agent_id in used_agents:
            if agent_id not in candidate_agents:
                candidate_agents.append(agent_id)
        return WorkflowPlan(
            atomic_intents=atomic_intents,
            candidate_agents=candidate_agents,
            dag=steps,
            reason="contract keyword/few-shot draft; pending DAG validation",
        )

    def _best_agent_for_clause(
        self,
        clause: str,
        candidates: list[str],
        specs: Mapping[str, dict],
    ) -> Optional[str]:
        ranked = self._rank_candidates(clause, specs)
        for agent_id in ranked:
            if agent_id in candidates:
                return agent_id
        return candidates[0] if candidates else None

    def _best_intent(self, clause: str, spec: Mapping[str, dict]) -> str:
        intents = list(spec.get("intents", []) or [])
        if not intents:
            return "general_task"
        lowered = clause.lower()
        for intent in intents:
            if intent.lower() in lowered:
                return intent

        scored = []
        for index, intent in enumerate(intents):
            hints = self._INTENT_HINTS.get(intent, ())
            score = sum(2 if len(hint) > 1 else 1 for hint in hints if hint.lower() in lowered)
            for example in spec.get("few_shots", []) or []:
                if example.get("expected_intent") != intent:
                    continue
                example_terms = set(self._TOKEN_PATTERN.findall(str(example.get("query", "")).lower()))
                clause_terms = set(self._TOKEN_PATTERN.findall(lowered))
                score += min(2, len(example_terms & clause_terms)) * 0.25
            scored.append((score, -index, intent))
        return max(scored)[2]

    def _preferred_export(self, spec: Mapping[str, dict]) -> str:
        exports = list(spec.get("exports", []) or [])
        if not exports:
            exports = list((spec.get("output_schema", {}) or {}).get("exports", []) or [])
        intents = set(spec.get("intents", []) or [])
        if "data_visualization" in intents and "data.artifacts" in exports:
            return "data.artifacts"
        if {"data_analysis", "trend_analysis"} & intents and "summary" in exports:
            return "summary"
        for export in self._EXPORT_PREFERENCE:
            if export in exports:
                return export
        return exports[0] if exports else ""

    def _call_llm(
        self,
        query: str,
        *,
        context: str,
        candidates: list[str],
        specs: Mapping[str, dict],
        max_steps: int,
    ) -> dict:
        candidate_specs = {
            agent_id: {
                "id": agent_id,
                "description": specs[agent_id].get("description", "")[:240],
                "intents": specs[agent_id].get("intents", []),
                "keywords": specs[agent_id].get("keywords", []),
                "few_shots": (specs[agent_id].get("few_shots", []) or [])[:3],
                "requires": specs[agent_id].get("requires", []),
                "exports": specs[agent_id].get(
                    "exports",
                    (specs[agent_id].get("output_schema", {}) or {}).get("exports", []),
                ),
            }
            for agent_id in candidates[:8]
            if agent_id in specs
        }
        schema = self.json_schema(max_steps=max_steps)
        if self.llm_planner is not None:
            return self.llm_planner(
                query=query,
                context=context,
                candidates=candidates,
                candidate_specs=candidate_specs,
                schema=schema,
            )

        if self.client is None:
            from config import get_llm_config
            from openai import OpenAI

            config = get_llm_config()
            if not config.get("api_key"):
                raise WorkflowPlanningError(
                    "LLM planner requested but DEEPSEEK_API_KEY is not configured"
                )
            self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
            self.model = self.model or config["model"]

        prompt = (
            "You are the Mini Thalamus Workflow Planner. Split the request into atomic "
            "single-intent steps, select only registered Agents, and return JSON "
            "matching the supplied schema. This is a draft: do not invent Agents.\n"
            "Each dag step instruction must be self-contained and directly searchable. "
            "Never leave placeholders such as {city}, [birthplace], or 'the identified X'. "
            "When a step depends on an earlier step, write the concrete value from that "
            "earlier step into the instruction (for example 'Find the tallest building in "
            "Bangkok' instead of 'Find the tallest building in the city identified in "
            "step_1'). Do not invent words not supported by the user request or an "
            "earlier confirmed value.\n"
            f"Agent profiles: {json.dumps(candidate_specs, ensure_ascii=False)}\n"
            f"JSON schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        messages = [{"role": "system", "content": prompt}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context[:2000]}"})
        messages.append({"role": "user", "content": query})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=900,
        )
        text = response.choices[0].message.content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkflowPlanningError("LLM planner did not return valid JSON") from exc
        return payload

    def _parse_payload(
        self,
        payload: dict,
        *,
        specs: Mapping[str, dict],
        max_steps: int,
        method: str,
    ) -> WorkflowPlan:
        if not isinstance(payload, dict):
            raise WorkflowPlanningError("planner output must be an object")
        required = {"atomic_intents", "candidate_agents", "dag"}
        missing = sorted(required - set(payload))
        if missing:
            raise WorkflowPlanningError(
                "planner output is missing fields: " + ", ".join(missing)
            )
        atomic_intents = self._string_list(payload["atomic_intents"], "atomic_intents")
        candidate_agents = self._string_list(payload["candidate_agents"], "candidate_agents")
        raw_dag = payload["dag"]
        if not isinstance(raw_dag, list) or not raw_dag:
            raise WorkflowPlanningError("planner output dag must be a non-empty list")
        if len(raw_dag) > max_steps:
            raise WorkflowPlanningError(f"planner output supports at most {max_steps} steps")

        steps = []
        seen_ids: set[str] = set()
        for raw_step in raw_dag:
            if not isinstance(raw_step, dict):
                raise WorkflowPlanningError("each dag step must be an object")
            for field_name in ("call_id", "agent", "intent", "input_refs"):
                if field_name not in raw_step:
                    raise WorkflowPlanningError(f"dag step is missing '{field_name}'")
            call_id = raw_step["call_id"]
            agent_id = raw_step["agent"]
            intent = raw_step["intent"]
            if not all(isinstance(value, str) and value.strip() for value in (call_id, agent_id, intent)):
                raise WorkflowPlanningError("dag call_id, agent, and intent must be non-empty strings")
            if call_id in seen_ids:
                raise WorkflowPlanningError(f"duplicate dag call_id: {call_id}")
            seen_ids.add(call_id)
            if agent_id not in specs:
                raise WorkflowPlanningError(f"dag agent '{agent_id}' is not registered")
            input_refs = self._string_list(raw_step["input_refs"], f"{call_id}.input_refs")
            depends_on = self._string_list(raw_step.get("depends_on", []), f"{call_id}.depends_on")
            instruction = raw_step.get("instruction", "")
            if not isinstance(instruction, str):
                raise WorkflowPlanningError(f"{call_id}.instruction must be a string")
            steps.append(
                WorkflowPlanStep(
                    call_id=call_id,
                    agent=agent_id,
                    intent=intent,
                    input_refs=input_refs,
                    depends_on=depends_on,
                    instruction=instruction,
                )
            )

        unknown_candidates = [agent_id for agent_id in candidate_agents if agent_id not in specs]
        if unknown_candidates:
            raise WorkflowPlanningError(
                "candidate Agent(s) not registered: " + ", ".join(unknown_candidates)
            )
        if not candidate_agents:
            candidate_agents = list(dict.fromkeys(step.agent for step in steps))
        if not atomic_intents:
            atomic_intents = [step.intent for step in steps]
        return WorkflowPlan(
            atomic_intents=atomic_intents,
            candidate_agents=candidate_agents,
            dag=steps,
            method=method,
            reason="LLM-generated workflow draft; pending DAG validation",
        )

    @staticmethod
    def _string_list(value, field_name: str) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise WorkflowPlanningError(f"{field_name} must be a list of strings")
        return list(value)


__all__ = [
    "WorkflowPlan",
    "WorkflowPlanStep",
    "WorkflowPlanner",
    "WorkflowPlanningError",
]
