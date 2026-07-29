"""Validated execution of atomic Agent calls and their data dependencies."""

import time

from base_agent import AgentContext, AgentResult
from route_schema import RouteDecision


class WorkflowExecutor:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(
        self,
        decision: RouteDecision,
        *,
        original_query: str,
        context: dict = None,
        approved_call_ids=None,
        context_provider=None,
    ) -> dict:
        approved = set(approved_call_ids or [])
        base_context = dict(context or {})
        results: dict[str, dict] = {}
        trace = []

        for route_call in self._topological_calls(decision):
            failed_dependencies = [
                dependency
                for dependency in route_call.depends_on
                if results.get(dependency, {}).get("status") != "success"
            ]
            if failed_dependencies:
                result = {
                    "status": "skipped",
                    "answer": "",
                    "summary": (
                        "依赖步骤失败: " + ", ".join(failed_dependencies)
                    ),
                    "data": {},
                    "metadata": {
                        "agent_id": route_call.function.name,
                        "call_id": route_call.call_id,
                    },
                }
                results[route_call.call_id] = result
                trace.append({
                    "call_id": route_call.call_id,
                    "agent_id": route_call.function.name,
                    "intent": route_call.intent,
                    "status": "skipped",
                    "latency_ms": 0.0,
                    "attempt": 0,
                    "retry": False,
                    "fallback": False,
                })
                continue

            if route_call.policy.get("action") == "deny":
                result = {
                    "status": "blocked",
                    "answer": route_call.policy.get("reason", "执行策略拒绝该步骤"),
                    "summary": "执行策略拒绝",
                    "data": {},
                    "metadata": {
                        "agent_id": route_call.function.name,
                        "call_id": route_call.call_id,
                    },
                }
                results[route_call.call_id] = result
                trace.append({
                    "call_id": route_call.call_id,
                    "agent_id": route_call.function.name,
                    "intent": route_call.intent,
                    "status": "blocked",
                    "latency_ms": 0.0,
                    "attempt": 0,
                    "retry": False,
                    "fallback": False,
                })
                continue

            if (
                route_call.policy.get("requires_approval")
                and route_call.call_id not in approved
            ):
                return {
                    "status": "checkpoint",
                    "answer": route_call.policy.get(
                        "reason",
                        "该步骤需要批准后才能执行。",
                    ),
                    "summary": "等待用户批准",
                    "results": results,
                    "trace": trace,
                    "pending_call": route_call.to_dict(),
                    "route_decision": decision.to_dict(),
                }

            workflow_inputs = {
                ref: self._resolve_ref(ref, results)
                for ref in route_call.function.arguments.get("input_refs", [])
            }
            run_context = dict(base_context)
            run_context.update({
                "workflow_inputs": workflow_inputs,
                "workflow": {
                    "trace_id": decision.trace_id,
                    "call_id": route_call.call_id,
                    "intent": route_call.intent,
                    "depends_on": list(route_call.depends_on),
                    "original_query": original_query,
                },
            })
            if route_call.depends_on:
                run_context["previous_result"] = results[
                    route_call.depends_on[-1]
                ]

            if context_provider is not None:
                provided = context_provider(route_call, results)
                if not isinstance(provided, dict):
                    raise TypeError("context_provider must return a dict")
                run_context.update(provided)

            started = time.perf_counter()
            attempt = 0
            while attempt < 2:
                attempt += 1
                try:
                    agent = self.runtime.get_agent(route_call.function.name)
                    agent_query = route_call.function.arguments["instruction"]
                    memory_payload = run_context.get("memory")
                    if isinstance(memory_payload, dict):
                        agent_query = AgentContext.from_dict(
                            memory_payload
                        ).render_for_agent()
                    raw_result = agent.run(
                        agent_query,
                        context=run_context,
                    )
                    result = self._normalize_result(
                        raw_result,
                        agent_id=route_call.function.name,
                        call_id=route_call.call_id,
                        trace_id=decision.trace_id,
                    )
                except Exception as exc:
                    error_type = (
                        "transient_error"
                        if isinstance(exc, (ConnectionError, TimeoutError))
                        else "internal_error"
                    )
                    result = {
                        "status": "error",
                        "answer": str(exc),
                        "summary": str(exc),
                        "data": {},
                        "metadata": {
                            "agent_id": route_call.function.name,
                            "call_id": route_call.call_id,
                            "trace_id": decision.trace_id,
                            "error_type": error_type,
                        },
                    }
                is_transient = (
                    result.get("status") == "error"
                    and result.get("metadata", {}).get("error_type")
                    in {"transient_error", "timeout"}
                )
                if not is_transient or attempt == 2:
                    break
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            results[route_call.call_id] = result
            trace.append({
                "call_id": route_call.call_id,
                "agent_id": route_call.function.name,
                "intent": route_call.intent,
                "status": result["status"],
                "latency_ms": latency_ms,
                "attempt": attempt,
                "retry": attempt > 1,
                "fallback": False,
            })

        successful = [
            result
            for result in results.values()
            if result.get("status") == "success"
        ]
        unsuccessful = [
            result
            for result in results.values()
            if result.get("status") != "success"
        ]
        if successful and unsuccessful:
            status = "partial"
        elif successful:
            status = "success"
        else:
            status = "error"
        final_result = successful[-1] if successful else (
            list(results.values())[-1] if results else {}
        )
        return {
            "status": status,
            "answer": final_result.get("answer", ""),
            "summary": final_result.get("summary", ""),
            "data": final_result.get("data", {}),
            "results": results,
            "trace": trace,
            "route_decision": decision.to_dict(),
        }

    @staticmethod
    def _topological_calls(decision: RouteDecision):
        pending = {call.call_id: call for call in decision.calls}
        completed = set()
        ordered = []
        while pending:
            ready = [
                call
                for call in pending.values()
                if set(call.depends_on).issubset(completed)
            ]
            if not ready:
                raise RuntimeError("validated workflow unexpectedly contains a cycle")
            for call in ready:
                ordered.append(call)
                completed.add(call.call_id)
                del pending[call.call_id]
        return ordered

    @staticmethod
    def _resolve_ref(ref: str, results: dict):
        call_id, _, path = ref.partition(".")
        value = results[call_id]
        for segment in path.split("."):
            if not isinstance(value, dict) or segment not in value:
                raise KeyError(f"workflow input '{ref}' is missing at runtime")
            value = value[segment]
        return value

    @staticmethod
    def _normalize_result(
        raw_result,
        *,
        agent_id: str,
        call_id: str,
        trace_id: str,
    ) -> dict:
        if isinstance(raw_result, AgentResult):
            result = raw_result.to_dict()
        elif isinstance(raw_result, dict):
            result = AgentResult.from_dict(raw_result).to_dict()
        else:
            raise TypeError("Agent must return AgentResult or dict")

        metadata = dict(result.get("metadata", {}))
        metadata.update({
            "agent_id": agent_id,
            "call_id": call_id,
            "trace_id": trace_id,
        })
        result["metadata"] = metadata
        return result


__all__ = ["WorkflowExecutor"]
