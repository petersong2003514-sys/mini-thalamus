"""Cascaded Router: deterministic rules -> semantics -> schema-bound LLM."""

import json
import logging
import re
import uuid

from openai import OpenAI

from config import (
    ROUTER_AGENTS,
    ROUTER_SEMANTIC_FALLBACK_THRESHOLD,
    ROUTER_SEMANTIC_MARGIN,
    ROUTER_SEMANTIC_THRESHOLD,
    ROUTER_TOP_K,
    get_llm_config,
)
from route_schema import RouteDecision, RouteValidationError
from semantic_router import SemanticMatcher


_log = logging.getLogger(__name__)
FALLBACK_AGENT = next(
    (
        agent_id
        for agent_id, info in ROUTER_AGENTS.items()
        if info.get("is_fallback")
    ),
    "general",
)

_COMPOUND_PATTERN = re.compile(
    r"(?:然后|并且|并|最后|接着|再(?:把|将|生成|写|画|分析)|同时|以及|->|→|;|；)"
)
_EXPLICIT_AGENT_PATTERN = re.compile(
    r"(?:@|/agent\s+)([a-zA-Z0-9_:-]+)",
    re.IGNORECASE,
)


class Router:
    """Route requests through progressively more expensive decision stages."""

    def __init__(
        self,
        dynamic_agents: dict = None,
        registry=None,
        semantic_matcher=None,
        llm_router=None,
        semantic_threshold: float = ROUTER_SEMANTIC_THRESHOLD,
        semantic_margin: float = ROUTER_SEMANTIC_MARGIN,
        semantic_fallback_threshold: float = ROUTER_SEMANTIC_FALLBACK_THRESHOLD,
        top_k: int = ROUTER_TOP_K,
    ):
        self.dynamic_agents = dynamic_agents or {}
        self._registry = None
        self._keyword_stats: dict[str, int] = {}
        self.semantic_matcher = semantic_matcher or SemanticMatcher()
        self.semantic_threshold = semantic_threshold
        self.semantic_margin = semantic_margin
        self.semantic_fallback_threshold = semantic_fallback_threshold
        self.top_k = max(1, int(top_k))
        self._llm_router_override = llm_router

        self.client = None
        self.model = ""
        if llm_router is None:
            config = get_llm_config()
            self.client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )
            self.model = config["model"]

        if registry is not None:
            self.attach_registry(registry)

    def attach_registry(self, registry) -> None:
        if self._registry is registry:
            return
        self._registry = registry
        self.refresh_from_registry()
        try:
            registry.on_register(
                lambda agent_id, meta: self.refresh_from_registry(),
                owner="router",
            )
        except TypeError:
            registry.on_register(
                lambda agent_id, meta: self.refresh_from_registry()
            )

    def refresh_from_registry(self) -> None:
        if self._registry is None:
            return
        try:
            self.dynamic_agents = self._registry.get_router_config()
        except Exception:
            _log.warning("Unable to refresh Router from registry", exc_info=True)

    def route(
        self,
        user_message: str,
        session_context: str = "",
        file_context: dict = None,
    ) -> dict:
        """Return the legacy dict shape backed by a strict RouteDecision."""
        return self.route_decision(
            user_message,
            session_context=session_context,
            file_context=file_context,
        ).to_legacy_dict()

    def route_decision(
        self,
        user_message: str,
        session_context: str = "",
        file_context: dict = None,
    ) -> RouteDecision:
        specs = self._agent_specs()
        trace_id = uuid.uuid4().hex
        file_context = file_context or {}
        compound = bool(_COMPOUND_PATTERN.search(user_message))
        self._record_keyword_stats(user_message, specs)

        rule_agent, rule_reason = self._rule_match(
            user_message,
            file_context,
            specs,
        )
        if rule_agent and not compound:
            return self._single_decision(
                agent_id=rule_agent,
                instruction=user_message,
                method="rule",
                reason=rule_reason,
                confidence=1.0,
                candidates=[
                    {
                        "agent_id": rule_agent,
                        "score": 1.0,
                        "evidence": rule_reason,
                    }
                ],
                specs=specs,
                trace_id=trace_id,
            )

        semantic_candidates = self._semantic_candidates(user_message, specs)
        if not compound and self._semantic_is_decisive(semantic_candidates):
            best = semantic_candidates[0]
            return self._single_decision(
                agent_id=best["agent_id"],
                instruction=user_message,
                method="semantic",
                reason=best.get("evidence") or "semantic capability match",
                confidence=best["score"],
                candidates=semantic_candidates,
                specs=specs,
                trace_id=trace_id,
            )

        llm_candidates = semantic_candidates or self._lexical_candidates(
            user_message,
            specs,
        )
        try:
            payload = self._call_llm_router(
                query=user_message,
                context=session_context,
                file_context=file_context,
                candidates=llm_candidates[: self.top_k],
                specs=specs,
            )
            if not isinstance(payload, dict):
                raise RouteValidationError("LLM route must be an object")
            payload = dict(payload)
            payload["method"] = "llm"
            payload["trace_id"] = trace_id
            if not payload.get("candidates"):
                payload["candidates"] = llm_candidates[: self.top_k]
            return RouteDecision.from_dict(payload, agent_specs=specs)
        except Exception as exc:
            _log.warning("Structured Router LLM failed: %s", str(exc)[:200])

        if semantic_candidates and (
            semantic_candidates[0]["score"] >= self.semantic_fallback_threshold
        ):
            best = semantic_candidates[0]
            return self._single_decision(
                agent_id=best["agent_id"],
                instruction=user_message,
                method="semantic_fallback",
                reason="LLM route failed; using the strongest semantic candidate",
                confidence=best["score"],
                candidates=semantic_candidates,
                specs=specs,
                trace_id=trace_id,
            )
        return self._fallback_decision(
            reason="No validated Agent route was available",
            candidates=semantic_candidates,
            specs=specs,
            trace_id=trace_id,
        )

    def _agent_specs(self) -> dict:
        specs = {}
        if self._registry is not None:
            try:
                specs.update(self._registry.list_all())
            except Exception:
                _log.warning("Unable to read Agent registry", exc_info=True)

        for agent_id, info in ROUTER_AGENTS.items():
            if info.get("is_fallback") or agent_id in specs:
                continue
            specs[agent_id] = self._normalize_spec(agent_id, info)
        for agent_id, info in self.dynamic_agents.items():
            if agent_id == FALLBACK_AGENT:
                continue
            existing = specs.get(agent_id, {})
            merged = dict(existing)
            merged.update(info)
            specs[agent_id] = self._normalize_spec(agent_id, merged)

        specs[FALLBACK_AGENT] = self._normalize_spec(
            FALLBACK_AGENT,
            {
                "name": "General Agent",
                "description": "通用问答和安全降级",
                "intents": ["general_question"],
                "keywords": [],
                "examples": [],
                "capabilities": {"side_effects": "none"},
            },
        )
        return specs

    @staticmethod
    def _normalize_spec(agent_id: str, info: dict) -> dict:
        spec = dict(info)
        spec["id"] = agent_id
        spec.setdefault("name", agent_id)
        spec.setdefault("description", "")
        spec.setdefault("keywords", [])
        spec.setdefault("intents", [])
        spec.setdefault("examples", [])
        spec.setdefault("capabilities", {})
        spec.setdefault(
            "input_schema",
            {"required": ["instruction"], "properties": {}},
        )
        spec.setdefault(
            "output_schema",
            {
                "exports": [
                    "answer",
                    "summary",
                    "data.structured",
                    "data.artifacts",
                    "data.report",
                ]
            },
        )
        return spec

    def _rule_match(self, query: str, file_context: dict, specs: dict):
        explicit = _EXPLICIT_AGENT_PATTERN.search(query)
        if explicit:
            agent_id = explicit.group(1)
            if agent_id in specs and agent_id != FALLBACK_AGENT:
                return agent_id, "explicit Agent selection"

        if file_context.get("has_files") and "docparse" in specs:
            if re.search(r"文件|文档|附件|PDF|Word|Excel|表格", query, re.I):
                return "docparse", "uploaded file rule"
        return None, ""

    def _semantic_candidates(self, query: str, specs: dict) -> list[dict]:
        routable = {
            agent_id: spec
            for agent_id, spec in specs.items()
            if agent_id != FALLBACK_AGENT
        }
        try:
            raw = self.semantic_matcher.rank(
                query,
                routable,
                top_k=self.top_k,
            )
        except Exception as exc:
            _log.warning("Semantic Router failed: %s", exc)
            return []

        result = []
        for item in raw or []:
            if isinstance(item, dict):
                agent_id = item.get("agent_id")
                score = item.get("score", 0.0)
                evidence = item.get("evidence", "semantic_similarity")
            else:
                agent_id = getattr(item, "agent_id", "")
                score = getattr(item, "score", 0.0)
                evidence = getattr(item, "evidence", "semantic_similarity")
            if agent_id not in routable:
                continue
            try:
                score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                continue
            result.append({
                "agent_id": agent_id,
                "score": score,
                "evidence": str(evidence),
            })
        return sorted(result, key=lambda item: item["score"], reverse=True)[
            : self.top_k
        ]

    def _semantic_is_decisive(self, candidates: list[dict]) -> bool:
        if not candidates or candidates[0]["score"] < self.semantic_threshold:
            return False
        if len(candidates) == 1:
            return True
        return (
            candidates[0]["score"] - candidates[1]["score"]
            >= self.semantic_margin
        )

    def _lexical_candidates(self, query: str, specs: dict) -> list[dict]:
        lowered = query.lower()
        scored = []
        for agent_id, spec in specs.items():
            if agent_id == FALLBACK_AGENT:
                continue
            terms = [
                *(spec.get("keywords", []) or []),
                *(spec.get("intents", []) or []),
                *(spec.get("examples", []) or []),
            ]
            hits = sum(1 for term in terms if str(term).lower() in lowered)
            score = min(0.7, hits * 0.2)
            scored.append({
                "agent_id": agent_id,
                "score": score,
                "evidence": "candidate_prefilter",
            })
        return sorted(scored, key=lambda item: (-item["score"], item["agent_id"]))[
            : self.top_k
        ]

    def _call_llm_router(
        self,
        *,
        query: str,
        context: str,
        file_context: dict,
        candidates: list[dict],
        specs: dict,
    ) -> dict:
        candidate_ids = [item["agent_id"] for item in candidates]
        if not candidate_ids:
            candidate_ids = [
                agent_id
                for agent_id in specs
                if agent_id != FALLBACK_AGENT
            ][: self.top_k]
        candidate_specs = {
            agent_id: specs[agent_id]
            for agent_id in candidate_ids
            if agent_id in specs
        }
        schema = self._route_json_schema(candidate_ids)
        if self._llm_router_override is not None:
            return self._llm_router_override(
                query=query,
                candidates=candidates,
                candidate_specs=candidate_specs,
                context=context,
                file_context=file_context,
                schema=schema,
            )

        compact_agents = [
            {
                "id": agent_id,
                "description": spec.get("description", "")[:160],
                "intents": spec.get("intents", []),
                "examples": (spec.get("examples", []) or [])[:3],
                "exports": spec.get("output_schema", {}).get("exports", []),
            }
            for agent_id, spec in candidate_specs.items()
        ]
        system_prompt = (
            "你是 ShuShu Agent 的结构化路由器。把复合请求拆为原子步骤；"
            "每一步只能有一个主要 intent 和一个 function。"
            "calls 必须形成无环依赖图，input_refs 只能读取依赖步骤的 exports。"
            "只输出符合 JSON Schema 的 JSON，不要 Markdown。\n"
            f"候选 Agent: {json.dumps(compact_agents, ensure_ascii=False)}\n"
            f"JSON Schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({
                "role": "system",
                "content": f"会话上下文（仅辅助判断）:\n{context[:2000]}",
            })
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
        return json.loads(text)

    @staticmethod
    def _route_json_schema(agent_ids: list[str]) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "mode",
                "calls",
                "candidates",
                "overall_confidence",
                "method",
                "reason",
            ],
            "properties": {
                "mode": {"enum": ["single", "workflow", "clarify", "fallback"]},
                "calls": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "call_id",
                            "intent",
                            "function",
                            "depends_on",
                            "confidence",
                        ],
                        "properties": {
                            "call_id": {"type": "string"},
                            "intent": {"type": "string"},
                            "function": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "arguments"],
                                "properties": {
                                    "name": {"enum": agent_ids},
                                    "arguments": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["instruction", "input_refs"],
                                        "properties": {
                                            "instruction": {"type": "string"},
                                            "input_refs": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                },
                "candidates": {"type": "array"},
                "overall_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "method": {"type": "string"},
                "reason": {"type": "string"},
            },
        }

    def _single_decision(
        self,
        *,
        agent_id: str,
        instruction: str,
        method: str,
        reason: str,
        confidence: float,
        candidates: list[dict],
        specs: dict,
        trace_id: str,
    ) -> RouteDecision:
        spec = specs[agent_id]
        intents = spec.get("intents", []) or ["general_task"]
        payload = {
            "mode": "single",
            "calls": [
                {
                    "call_id": "step_1",
                    "intent": intents[0],
                    "function": {
                        "name": agent_id,
                        "arguments": {
                            "instruction": instruction,
                            "input_refs": [],
                        },
                    },
                    "depends_on": [],
                    "confidence": confidence,
                }
            ],
            "candidates": candidates,
            "overall_confidence": confidence,
            "method": method,
            "reason": reason,
            "trace_id": trace_id,
        }
        return RouteDecision.from_dict(payload, agent_specs=specs)

    def _fallback_decision(
        self,
        *,
        reason: str,
        candidates: list[dict],
        specs: dict,
        trace_id: str,
    ) -> RouteDecision:
        return RouteDecision.from_dict(
            {
                "mode": "fallback",
                "calls": [],
                "candidates": candidates,
                "overall_confidence": 0.0,
                "method": "fallback",
                "reason": reason,
                "trace_id": trace_id,
            },
            agent_specs=specs,
        )

    def _record_keyword_stats(self, message: str, specs: dict) -> None:
        lowered = message.lower()
        for agent_id, info in specs.items():
            if agent_id == FALLBACK_AGENT:
                continue
            if any(str(keyword).lower() in lowered for keyword in info.get("keywords", [])):
                self._keyword_stats[agent_id] = self._keyword_stats.get(agent_id, 0) + 1

    def get_stats(self) -> dict:
        return dict(self._keyword_stats)


__all__ = ["FALLBACK_AGENT", "Router"]
