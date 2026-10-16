"""Validated execution of atomic Agent calls and their data dependencies."""

import time
from concurrent.futures import ThreadPoolExecutor

from mini_thalamus.base_agent import AgentContext, AgentResult
from mini_thalamus.route_schema import RouteDecision


class WorkflowExecutor:
    # 单次请求最多重规划次数(安全护栏); 是否启用由 Orchestrator 配置决定
    _REPLAN_LIMIT = 3

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
        on_stage=None,
        replan_fn=None,
    ) -> dict:
        """执行 DAG。

        replan_fn 为可选的运行时重规划回调: 每批执行完后被调用,入参为
        dict(decision/original_query/results/batch_call_ids/pending_calls/
        completed_ids/attempt),若返回一个新 RouteDecision,则用新计划替换
        剩余节点并重新开始拓扑执行(已完成结果保留,供新计划引用)。
        """
        approved = set(approved_call_ids or [])
        base_context = dict(context or {})
        results: dict[str, dict] = {}
        trace = []
        ordered = self._topological_calls(decision)
        pending_ids = [call.call_id for call in ordered]
        completed: set[str] = set()
        replan_count = 0

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
                    if on_stage is not None:
                        on_stage("step_start", {
                            "call_id": call.call_id,
                            "agent_id": call.function.name,
                            "intent": call.intent,
                            "instruction": call.function.arguments["instruction"],
                        })
                        on_stage("step_end", {
                            "call_id": call.call_id,
                            "agent_id": call.function.name,
                            "status": "skipped",
                            "summary": "依赖步骤失败",
                        })
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
                    if on_stage is not None:
                        on_stage("step_start", {
                            "call_id": call.call_id,
                            "agent_id": call.function.name,
                            "intent": call.intent,
                            "instruction": call.function.arguments["instruction"],
                        })
                        on_stage("step_end", {
                            "call_id": call.call_id,
                            "agent_id": call.function.name,
                            "status": "blocked",
                            "summary": "执行策略拒绝",
                        })
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
                    context_provider=context_provider,
                    on_stage=on_stage,
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
                context_provider=context_provider,
                on_stage=on_stage,
            )

            # ── 运行时重规划钩子(MAO): 依据本批结果重写下游节点 ──
            if (
                replan_fn is not None
                and pending_ids
                and replan_count < self._REPLAN_LIMIT
            ):
                new_decision = replan_fn({
                    "decision": decision,
                    "original_query": original_query,
                    "results": results,
                    "batch_call_ids": [call.call_id for call in ready],
                    "pending_calls": [
                        call for call in ordered if call.call_id in pending_ids
                    ],
                    "completed_ids": sorted(completed),
                    "attempt": replan_count,
                })
                if new_decision is not None and new_decision.calls:
                    if on_stage is not None:
                        on_stage("replan", {
                            "from": len(completed),
                            "to": len(new_decision.calls),
                            "reason": "部分步骤未达预期，重新规划剩余任务",
                        })
                    ordered = self._topological_calls(new_decision)
                    pending_ids = [call.call_id for call in ordered]
                    completed = set()
                    decision = new_decision
                    replan_count += 1
                    continue

            for call in ready:
                completed.add(call.call_id)
                pending_ids.remove(call.call_id)

        # 最终状态只统计最终计划(可能是重规划后的)的结果;
        # 重规划覆盖掉的旧失败步骤不再计入 partial。
        final_ids = {call.call_id for call in decision.calls}
        relevant = {
            call_id: result
            for call_id, result in results.items()
            if call_id in final_ids
        }
        successful = [
            result for result in relevant.values()
            if result.get("status") == "success"
        ]
        unsuccessful = [
            result for result in relevant.values()
            if result.get("status") != "success"
        ]
        if successful and unsuccessful:
            status = "partial"
        elif successful:
            status = "success"
        else:
            status = "error"
        final_result = successful[-1] if successful else (
            list(relevant.values())[-1] if relevant else {}
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
        context_provider=None,
        on_stage=None,
    ) -> None:
        if not calls:
            return

        prepared = []
        agents = {}
        for call in calls:
            agent_id = call.function.name
            try:
                if agent_id not in agents:
                    agents[agent_id] = self.runtime.get_agent(agent_id)
                if agents[agent_id] is None:
                    raise RuntimeError(f"Agent '{agent_id}' is not available")
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

                if context_provider is not None:
                    provided = context_provider(call, results)
                    if not isinstance(provided, dict):
                        raise TypeError("context_provider must return a dict")
                    run_context.update(provided)
            except Exception as exc:
                error_result = {
                    "status": "error",
                    "answer": "该步骤暂不可执行，已自动跳过。",
                    "summary": "步骤不可执行，已自动跳过",
                    "data": {},
                    "metadata": {
                        "agent_id": agent_id,
                        "call_id": call.call_id,
                        "error_type": "preparation_error",
                        "error": str(exc)[:200],
                    },
                }
                results[call.call_id] = error_result
                trace.append({
                    "call_id": call.call_id,
                    "agent_id": agent_id,
                    "intent": call.intent,
                    "status": "error",
                    "latency_ms": 0.0,
                    "attempt": 0,
                    "retry": False,
                    "fallback": True,
                })
                if on_stage is not None:
                    on_stage("step_start", {
                        "call_id": call.call_id,
                        "agent_id": agent_id,
                        "intent": call.intent,
                        "instruction": call.function.arguments["instruction"],
                    })
                    on_stage("step_end", {
                        "call_id": call.call_id,
                        "agent_id": agent_id,
                        "intent": call.intent,
                        "status": "error",
                        "summary": "步骤不可执行，已自动跳过",
                    })
                continue
            prepared.append((call, agents[agent_id], run_context))

        if not prepared:
            return

        groups = {}
        for call, agent, run_context in prepared:
            groups.setdefault(call.function.name, []).append(
                (call, agent, run_context)
            )

        if on_stage is not None:
            for call, agent, run_context in prepared:
                on_stage("step_start", {
                    "call_id": call.call_id,
                    "agent_id": call.function.name,
                    "intent": call.intent,
                    "instruction": call.function.arguments["instruction"],
                })

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
            group_results = {}
            for agent_id, future in futures.items():
                try:
                    group_results[agent_id] = future.result()
                except Exception as exc:
                    group_results[agent_id] = [
                        {
                            "status": "error",
                            "answer": "该步骤执行异常，已自动跳过。",
                            "summary": "agent execution failed",
                            "data": {},
                            "metadata": {
                                "agent_id": agent_id,
                                "error_type": "batch_error",
                                "error": str(exc)[:200],
                            },
                        }
                        for _ in groups[agent_id]
                    ]
            for call, _, _ in prepared:
                result, entry = group_results[call.function.name].pop(0)
                results[call.call_id] = result
                trace.append(entry)
                if on_stage is not None:
                    on_stage("step_end", {
                        "call_id": call.call_id,
                        "agent_id": call.function.name,
                        "status": result.get("status"),
                        "summary": result.get("summary", "")[:200],
                        "answer": result.get("answer", "")[:120],
                    })

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
                agent_query = call.function.arguments["instruction"]
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
