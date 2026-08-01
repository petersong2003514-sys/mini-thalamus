"""Validated execution of atomic Agent calls and their data dependencies."""

import time
from concurrent.futures import ThreadPoolExecutor

from base_agent import AgentResult
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
    ) -> dict:
        approved = set(approved_call_ids or [])
        base_context = dict(context or {})
        results: dict[str, dict] = {}
        trace = []
        ordered = self._topological_calls(decision)
        pending_ids = [call.call_id for call in ordered]
        completed: set[str] = set()

        while pending_ids:
            ready = [
                call
                for call in ordered
                if call.call_id in pending_ids
                and set(call.depends_on).issubset(completed)
            ]
            if not ready:
                raise RuntimeError("validated workflow unexpectedly contains a cycle")

            executable = []
            checkpoint_call = None
            for call in ready:
                failed_dependencies = [
                    dependency
                    for dependency in call.depends_on
                    if results.get(dependency, {}).get("status") != "success"
                ]
                if failed_dependencies:
                    results[call.call_id] = {
                        "status": "skipped",
                        "answer": "",
                        "summary": (
                            "依赖步骤失败: " + ", ".join(failed_dependencies)
                        ),
                        "data": {},
                        "metadata": {
                            "agent_id": call.function.name,
                            "call_id": call.call_id,
                        },
                    }
                    trace.append({
                        "call_id": call.call_id,
                        "agent_id": call.function.name,
                        "intent": call.intent,
                        "status": "skipped",
                        "latency_ms": 0.0,
                        "attempt": 0,
                        "retry": False,
                        "fallback": False,
                    })
                    continue

                if call.policy.get("action") == "deny":
                    results[call.call_id] = {
                        "status": "blocked",
                        "answer": call.policy.get("reason", "执行策略拒绝该步骤"),
                        "summary": "执行策略拒绝",
                        "data": {},
                        "metadata": {
                            "agent_id": call.function.name,
                            "call_id": call.call_id,
                        },
                    }
                    trace.append({
                        "call_id": call.call_id,
                        "agent_id": call.function.name,
                        "intent": call.intent,
                        "status": "blocked",
                        "latency_ms": 0.0,
                        "attempt": 0,
                        "retry": False,
                        "fallback": False,
                    })
                    continue

                if (
                    call.policy.get("requires_approval")
                    and call.call_id not in approved
                ):
                    checkpoint_call = call
                    break
                executable.append(call)

            if checkpoint_call is not None:
                self._run_batch(
                    executable,
                    base_context=base_context,
                    original_query=original_query,
                    results=results,
                    trace=trace,
                    trace_id=decision.trace_id,
                )
                return {
                    "status": "checkpoint",
                    "answer": checkpoint_call.policy.get(
                        "reason",
                        "该步骤需要批准后才能执行。",
                    ),
                    "summary": "等待用户批准",
                    "results": results,
                    "trace": trace,
                    "pending_call": checkpoint_call.to_dict(),
                    "route_decision": decision.to_dict(),
                }

            self._run_batch(
                executable,
                base_context=base_context,
                original_query=original_query,
                results=results,
                trace=trace,
                trace_id=decision.trace_id,
            )

            for call in ready:
                completed.add(call.call_id)
                pending_ids.remove(call.call_id)

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

    def _run_batch(
        self,
        calls,
        *,
        base_context: dict,
        original_query: str,
        results: dict,
        trace: list,
        trace_id: str,
    ) -> None:
        if not calls:
            return

        prepared = []
        agents = {}
        for call in calls:
            agent_id = call.function.name
            if agent_id not in agents:
                agents[agent_id] = self.runtime.get_agent(agent_id)
            workflow_inputs = {
                ref: self._resolve_ref(ref, results)
                for ref in call.function.arguments.get("input_refs", [])
            }
            run_context = dict(base_context)
            run_context.update({
                "workflow_inputs": workflow_inputs,
                "workflow": {
                    "trace_id": trace_id,
                    "call_id": call.call_id,
                    "intent": call.intent,
                    "depends_on": list(call.depends_on),
                    "original_query": original_query,
                },
            })
            if call.depends_on:
                run_context["previous_result"] = results[call.depends_on[-1]]
            prepared.append((call, agents[agent_id], run_context))

        groups = {}
        for call, agent, run_context in prepared:
            groups.setdefault(call.function.name, []).append(
                (call, agent, run_context)
            )

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {
                agent_id: pool.submit(
                    self._run_group,
                    items,
                    original_query=original_query,
                    trace_id=trace_id,
                )
                for agent_id, items in groups.items()
            }
            group_results = {
                agent_id: future.result()
                for agent_id, future in futures.items()
            }
            for call, _, _ in prepared:
                result, entry = group_results[call.function.name].pop(0)
                results[call.call_id] = result
                trace.append(entry)

    def _run_group(self, items, *, original_query, trace_id):
        return [
            self._run_call(call, agent, run_context, original_query, trace_id)
            for call, agent, run_context in items
        ]

    def _run_call(self, call, agent, run_context, original_query, trace_id):
        started = time.perf_counter()
        attempt = 0
        while attempt < 2:
            attempt += 1
            try:
                raw_result = agent.run(
                    call.function.arguments["instruction"],
                    context=run_context,
                )
                result = self._normalize_result(
                    raw_result,
                    agent_id=call.function.name,
                    call_id=call.call_id,
                    trace_id=trace_id,
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
                        "agent_id": call.function.name,
                        "call_id": call.call_id,
                        "trace_id": trace_id,
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
        return result, {
            "call_id": call.call_id,
            "agent_id": call.function.name,
            "intent": call.intent,
            "status": result["status"],
            "latency_ms": latency_ms,
            "attempt": attempt,
            "retry": attempt > 1,
            "fallback": False,
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
