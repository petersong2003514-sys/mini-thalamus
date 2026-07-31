"""Strict routing contracts shared by Router and Orchestrator."""

from dataclasses import dataclass, field
from typing import Callable, Mapping


class RouteValidationError(ValueError):
    """Raised when a route proposal violates the platform contract."""


def _require_mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise RouteValidationError(f"{path} must be an object")
    return value


def _reject_extra(value: Mapping, allowed: set[str], path: str) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        raise RouteValidationError(
            f"{path} has unexpected fields: {', '.join(extra)}"
        )


def _require_fields(value: Mapping, required: set[str], path: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise RouteValidationError(
            f"{path} is missing required fields: {', '.join(missing)}"
        )


def _confidence(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteValidationError(f"{path} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise RouteValidationError(f"{path} must be between 0 and 1")
    return result


@dataclass
class RouteCandidate:
    agent_id: str
    score: float
    evidence: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "RouteCandidate":
        value = _require_mapping(value, "candidate")
        _reject_extra(value, {"agent_id", "score", "evidence"}, "candidate")
        _require_fields(value, {"agent_id", "score"}, "candidate")
        if not isinstance(value["agent_id"], str) or not value["agent_id"]:
            raise RouteValidationError(
                "candidate.agent_id must be a non-empty string"
            )
        evidence = value.get("evidence", "")
        if not isinstance(evidence, str):
            raise RouteValidationError("candidate.evidence must be a string")
        return cls(
            agent_id=value["agent_id"],
            score=_confidence(value["score"], "candidate.score"),
            evidence=evidence,
        )

    def to_dict(self) -> dict:
        result = {"agent_id": self.agent_id, "score": self.score}
        if self.evidence:
            result["evidence"] = self.evidence
        return result


@dataclass
class RouteFunction:
    name: str
    arguments: dict

    @classmethod
    def from_dict(
        cls,
        value: dict,
        *,
        agent_specs: Mapping[str, dict],
    ) -> "RouteFunction":
        value = _require_mapping(value, "function")
        _reject_extra(value, {"name", "arguments"}, "function")
        _require_fields(value, {"name", "arguments"}, "function")

        name = value["name"]
        if not isinstance(name, str) or not name:
            raise RouteValidationError("function.name must be a non-empty string")
        if name not in agent_specs:
            raise RouteValidationError(f"function.name '{name}' is not registered")

        arguments = _require_mapping(value["arguments"], "function.arguments")
        spec = agent_specs.get(name, {})
        input_schema = spec.get("input_schema", {}) or {}
        declared = set((input_schema.get("properties", {}) or {}).keys())
        allowed = {"instruction", "input_refs"} | declared
        _reject_extra(arguments, allowed, "function.arguments")
        _require_fields(arguments, {"instruction", "input_refs"}, "function.arguments")

        if not isinstance(arguments["instruction"], str) or not arguments[
            "instruction"
        ].strip():
            raise RouteValidationError(
                "function.arguments.instruction must be a non-empty string"
            )
        input_refs = arguments["input_refs"]
        if not isinstance(input_refs, list) or not all(
            isinstance(ref, str) and ref for ref in input_refs
        ):
            raise RouteValidationError(
                "function.arguments.input_refs must be a list of strings"
            )
        return cls(name=name, arguments=dict(arguments))

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass
class RouteCall:
    call_id: str
    intent: str
    function: RouteFunction
    depends_on: list[str]
    confidence: float
    policy: dict = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        value: dict,
        *,
        agent_specs: Mapping[str, dict],
    ) -> "RouteCall":
        value = _require_mapping(value, "call")
        _reject_extra(
            value,
            {
                "call_id",
                "intent",
                "function",
                "depends_on",
                "confidence",
                "policy",
            },
            "call",
        )
        _require_fields(
            value,
            {"call_id", "intent", "function", "depends_on", "confidence"},
            "call",
        )

        call_id = value["call_id"]
        intent = value["intent"]
        if not isinstance(call_id, str) or not call_id:
            raise RouteValidationError("call.call_id must be a non-empty string")
        if not isinstance(intent, str) or not intent.strip():
            raise RouteValidationError("call.intent must be a non-empty string")

        depends_on = value["depends_on"]
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) and dep for dep in depends_on
        ):
            raise RouteValidationError("call.depends_on must be a list of strings")
        if len(depends_on) != len(set(depends_on)):
            raise RouteValidationError(f"call '{call_id}' has duplicate dependencies")

        policy = value.get("policy", {}) or {}
        policy = _require_mapping(policy, "call.policy")
        _reject_extra(
            policy,
            {
                "action",
                "risk_level",
                "requires_approval",
                "sandboxed",
                "reason",
            },
            "call.policy",
        )

        return cls(
            call_id=call_id,
            intent=intent,
            function=RouteFunction.from_dict(
                value["function"],
                agent_specs=agent_specs,
            ),
            depends_on=list(depends_on),
            confidence=_confidence(value["confidence"], "call.confidence"),
            policy=dict(policy),
        )

    def to_dict(self) -> dict:
        result = {
            "call_id": self.call_id,
            "intent": self.intent,
            "function": self.function.to_dict(),
            "depends_on": list(self.depends_on),
            "confidence": self.confidence,
        }
        if self.policy:
            result["policy"] = dict(self.policy)
        return result


@dataclass
class RouteDecision:
    mode: str
    calls: list[RouteCall]
    candidates: list[RouteCandidate]
    overall_confidence: float
    method: str
    reason: str
    trace_id: str = ""
    metadata: dict = field(default_factory=dict)
    _agent_specs: Mapping[str, dict] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_dict(
        cls,
        value: dict,
        *,
        agent_specs: Mapping[str, dict],
        max_calls: int = 6,
    ) -> "RouteDecision":
        value = _require_mapping(value, "route")
        _reject_extra(
            value,
            {
                "mode",
                "calls",
                "candidates",
                "overall_confidence",
                "method",
                "reason",
                "trace_id",
                "metadata",
            },
            "route",
        )
        _require_fields(
            value,
            {
                "mode",
                "calls",
                "candidates",
                "overall_confidence",
                "method",
                "reason",
            },
            "route",
        )

        mode = value["mode"]
        if mode not in {"single", "workflow", "clarify", "fallback"}:
            raise RouteValidationError(f"unsupported route mode: {mode!r}")
        if not isinstance(value["calls"], list):
            raise RouteValidationError("route.calls must be a list")
        if len(value["calls"]) > max_calls:
            raise RouteValidationError(f"route.calls supports at most {max_calls} calls")

        calls = [
            RouteCall.from_dict(call, agent_specs=agent_specs)
            for call in value["calls"]
        ]
        if mode == "single" and len(calls) != 1:
            raise RouteValidationError("single mode requires exactly one call")
        if mode == "workflow" and not calls:
            raise RouteValidationError("workflow mode requires at least one call")
        if mode in {"clarify", "fallback"} and len(calls) > 1:
            raise RouteValidationError(f"{mode} mode supports at most one call")

        candidates_value = value["candidates"]
        if not isinstance(candidates_value, list):
            raise RouteValidationError("route.candidates must be a list")
        candidates = [
            RouteCandidate.from_dict(candidate)
            for candidate in candidates_value
        ]
        for candidate in candidates:
            if candidate.agent_id not in agent_specs:
                raise RouteValidationError(
                    f"candidate agent '{candidate.agent_id}' is not registered"
                )

        method = value["method"]
        reason = value["reason"]
        trace_id = value.get("trace_id", "")
        metadata = value.get("metadata", {}) or {}
        if not isinstance(method, str) or not method:
            raise RouteValidationError("route.method must be a non-empty string")
        if not isinstance(reason, str):
            raise RouteValidationError("route.reason must be a string")
        if not isinstance(trace_id, str):
            raise RouteValidationError("route.trace_id must be a string")
        if not isinstance(metadata, dict):
            raise RouteValidationError("route.metadata must be an object")

        decision = cls(
            mode=mode,
            calls=calls,
            candidates=candidates,
            overall_confidence=_confidence(
                value["overall_confidence"],
                "route.overall_confidence",
            ),
            method=method,
            reason=reason,
            trace_id=trace_id,
            metadata=dict(metadata),
            _agent_specs=agent_specs,
        )
        decision._validate_graph()
        decision._validate_input_refs()
        return decision

    def _validate_graph(self) -> None:
        call_ids = [call.call_id for call in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise RouteValidationError("route has duplicate call_id values")

        known = set(call_ids)
        for call in self.calls:
            for dependency in call.depends_on:
                if dependency not in known:
                    raise RouteValidationError(
                        f"call '{call.call_id}' has unknown dependency '{dependency}'"
                    )

        graph = {call.call_id: list(call.depends_on) for call in self.calls}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(call_id: str) -> None:
            if call_id in visiting:
                raise RouteValidationError("route dependency graph contains a cycle")
            if call_id in visited:
                return
            visiting.add(call_id)
            for dependency in graph[call_id]:
                visit(dependency)
            visiting.remove(call_id)
            visited.add(call_id)

        for call_id in call_ids:
            visit(call_id)

    def _validate_input_refs(self) -> None:
        by_id = {call.call_id: call for call in self.calls}
        for call in self.calls:
            for ref in call.function.arguments.get("input_refs", []):
                source_id, separator, export_path = ref.partition(".")
                if not separator or not export_path:
                    raise RouteValidationError(
                        f"input_ref '{ref}' must use '<call_id>.<export>' format"
                    )
                if source_id not in by_id:
                    raise RouteValidationError(
                        f"input_ref '{ref}' references unknown call"
                    )
                if source_id not in call.depends_on:
                    raise RouteValidationError(
                        f"input_ref source '{source_id}' must be a dependency "
                        f"of call '{call.call_id}'"
                    )
                source_agent = by_id[source_id].function.name
                source_spec = self._agent_specs.get(source_agent, {})
                output_schema = source_spec.get("output_schema", {}) or {}
                exports = set(output_schema.get("exports", []) or [])
                # Agent contracts expose the canonical planner exports; keep
                # legacy output_schema exports valid for older manifests.
                exports.update(source_spec.get("exports", []) or [])
                if export_path not in exports:
                    raise RouteValidationError(
                        f"input_ref '{ref}' is not exported by agent '{source_agent}'"
                    )

    def apply_policy(
        self,
        resolver: Callable[[RouteCall], Mapping],
    ) -> "RouteDecision":
        for call in self.calls:
            policy = resolver(call)
            if not isinstance(policy, Mapping):
                raise RouteValidationError("policy resolver must return an object")
            call.policy = dict(policy)
        return self

    def to_dict(self) -> dict:
        result = {
            "mode": self.mode,
            "calls": [call.to_dict() for call in self.calls],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "overall_confidence": self.overall_confidence,
            "method": self.method,
            "reason": self.reason,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    def to_legacy_dict(self) -> dict:
        agents = [call.function.name for call in self.calls]
        instructions = {
            call.function.name: call.function.arguments["instruction"]
            for call in self.calls
        }
        return {
            "agent": agents[0] if agents else "general",
            "agents": agents or ["general"],
            "instructions": instructions,
            "steps": [
                {
                    "agent": call.function.name,
                    "instruction": call.function.arguments["instruction"],
                }
                for call in self.calls
            ],
            "confidence": self.overall_confidence,
            "method": self.method,
            "reason": self.reason,
            "route_decision": self.to_dict(),
        }

