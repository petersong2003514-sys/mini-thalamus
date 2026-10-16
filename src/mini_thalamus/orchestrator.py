"""Main Agent orchestration over validated RouteDecision workflows."""

import os

from mini_thalamus.base_agent import BaseAgent
from mini_thalamus.config import (
    ALLOW_LOCAL_EXECUTION,
    AUTO_APPROVE,
    REPLAN_ENABLED,
    REPLAN_MAX,
    ROUTER_AGENTS,
    SANDBOX_USE_DOCKER,
    SELF_EVAL_ENABLED,
)
from mini_thalamus.route_schema import RouteDecision
from mini_thalamus.router import Router
from mini_thalamus.sandbox import ExecutionPolicy
from mini_thalamus.session_manager import SessionManager
from mini_thalamus.workflow_executor import WorkflowExecutor


FALLBACK_AGENT = next(
    (
        agent_id
        for agent_id, info in ROUTER_AGENTS.items()
        if info.get("is_fallback")
    ),
    "general",
)


class _AgentMapRuntime:
    def __init__(self, agents: dict):
        self.agents = agents

    def get_agent(self, agent_id: str):
        return self.agents[agent_id]


class Orchestrator:
    """The only component allowed to route and coordinate child Agents."""

    def __init__(
        self,
        agents: dict = None,
        session_mgr: SessionManager = None,
        registry=None,
        runtime=None,
        router: Router = None,
        execution_policy=None,
        memory_service=None,
    ):
        self.agents = agents or {}
        self.registry = registry or getattr(runtime, "registry", None)
        self.runtime = runtime or _AgentMapRuntime(self.agents)
        self.router = router or Router(registry=self.registry)
        from mini_thalamus.memory_service import get_memory_service

        self.memory_service = memory_service or get_memory_service()
        self.session_mgr = session_mgr or self.memory_service.sessions
        self.execution_policy = execution_policy or ExecutionPolicy(
            workspace_root=os.path.dirname(os.path.abspath(__file__)),
            local_execution_enabled=ALLOW_LOCAL_EXECUTION,
            docker_backend_enabled=SANDBOX_USE_DOCKER,
            auto_approve=AUTO_APPROVE,
        )
        self.workflow_executor = WorkflowExecutor(self.runtime)
        self.memory = self.memory_service.store

    def process(
        self,
        user_message: str,
        user_id: str = "default",
        session_id: str = None,
        approved_call_ids=None,
        decision_override: RouteDecision = None,
        on_stage=None,
        files: list = None,
    ) -> dict:
        session = self.memory_service.resolve_session(user_id, session_id)
        sid = session.session_id
        turn = self.memory_service.start_turn(session, user_message)
        try:
            uploaded_files = list(files or [])
            if uploaded_files and isinstance(uploaded_files[0], dict):
                file_names = [f.get("name", "") for f in uploaded_files]
            else:
                uploaded_files = []
                file_names = list(self.memory.get_files(sid))
            file_context = None
            if file_names:
                file_context = {
                    "has_files": True,
                    "file_names": file_names,
                    "session_id": sid,
                }
            decision = decision_override or self.router.route_decision(
                user_message,
                session_context=session.get_context_for("router"),
                file_context=file_context,
            )
            decision.apply_policy(self._resolve_policy)

            route_trace = {
                "step": "route",
                "trace_id": decision.trace_id,
                "mode": decision.mode,
                "method": decision.method,
                "confidence": decision.overall_confidence,
                "reason": decision.reason,
                "candidates": [
                    item.to_dict()
                    for item in decision.candidates
                ],
            }
            if on_stage is not None:
                on_stage("route", {
                    "mode": decision.mode,
                    "method": decision.method,
                    "confidence": decision.overall_confidence,
                    "reason": decision.reason,
                    "agents": [call.function.name for call in decision.calls],
                })

            if not decision.calls:
                answer = self._general_reply(user_message, session)
                self.memory_service.complete_turn(
                    session,
                    turn.turn_id,
                    answer=answer,
                    agent_id=FALLBACK_AGENT,
                    status="success",
                    result={"summary": answer[:300]},
                )
                return {
                    "status": "success",
                    "answer": answer,
                    "agent_used": FALLBACK_AGENT,
                    "trace": [route_trace],
                    "session_id": sid,
                    "route_decision": decision.to_dict(),
                }

            def context_provider(route_call, workflow_results):
                capabilities = dict(BaseAgent.capabilities)
                if self.registry is not None:
                    if hasattr(self.registry, "get_capabilities"):
                        capabilities.update(
                            self.registry.get_capabilities(
                                route_call.function.name
                            )
                            or {}
                        )
                    else:
                        entry = self.registry.get(route_call.function.name)
                        if entry:
                            capabilities.update(
                                entry.get("meta", {}).get(
                                    "capabilities",
                                    {},
                                )
                            )
                workflow_inputs = {
                    ref: WorkflowExecutor._resolve_ref(
                        ref,
                        workflow_results,
                    )
                    for ref in route_call.function.arguments.get(
                        "input_refs",
                        [],
                    )
                }
                previous_result = None
                if route_call.depends_on:
                    previous_result = workflow_results.get(
                        route_call.depends_on[-1]
                    )
                memory_context = self.memory_service.build_context(
                    user_id=user_id,
                    session_id=sid,
                    agent_id=route_call.function.name,
                    capabilities=capabilities,
                    current_query=route_call.function.arguments[
                        "instruction"
                    ],
                    previous_result=previous_result,
                    workflow_inputs=workflow_inputs,
                )
                return {
                    "memory": memory_context.to_dict(),
                    "session_state": self.memory_service.session_state(sid),
                }

            execution_error = None
            try:
                execution = self.workflow_executor.execute(
                    decision,
                    original_query=user_message,
                    context={
                        "session_id": sid,
                        "files": uploaded_files or file_names,
                    },
                    approved_call_ids=approved_call_ids,
                    context_provider=context_provider,
                    on_stage=on_stage,
                    replan_fn=self._make_replan_fn(),
                )
            except Exception as exc:
                execution = None
                execution_error = exc
            if execution is None or execution.get("status") == "error":
                answer = self._general_reply(user_message, session)
                fallback_reason = (
                    str(execution_error)[:300]
                    if execution_error is not None
                    else str(execution.get("summary", ""))[:300]
                )
                self.memory_service.complete_turn(
                    session,
                    turn.turn_id,
                    answer=answer,
                    agent_id=FALLBACK_AGENT,
                    status="fallback",
                    result={
                        "summary": answer[:300],
                        "error": fallback_reason,
                    },
                )
                return {
                    "status": "fallback",
                    "answer": answer,
                    "agent_used": FALLBACK_AGENT,
                    "trace": [
                        route_trace,
                        {
                            "step": "fallback",
                            "reason": fallback_reason or "workflow execution failed",
                        },
                    ],
                    "session_id": sid,
                    "route_decision": decision.to_dict(),
                }
            execution_trace = [route_trace, *execution.get("trace", [])]
            if execution["status"] == "checkpoint":
                pending_agent = execution["pending_call"]["function"]["name"]
                self.memory_service.complete_turn(
                    session,
                    turn.turn_id,
                    answer=execution.get("answer", "等待用户批准"),
                    agent_id=pending_agent,
                    status="checkpoint",
                    result=execution,
                )
                return {
                    **execution,
                    "agent_used": pending_agent,
                    "trace": execution_trace,
                    "session_id": sid,
                }

            answer = (
                execution.get("answer")
                or execution.get("summary")
                or "处理完成"
            )
            executed_agents = [
                item["agent_id"]
                for item in execution.get("trace", [])
                if item.get("status") != "skipped"
            ]
            agent_used = (
                executed_agents[-1]
                if executed_agents
                else FALLBACK_AGENT
            )
            turn_status = (
                "success"
                if execution["status"] == "success"
                else "error"
            )
            self.memory_service.complete_turn(
                session,
                turn.turn_id,
                answer=answer,
                agent_id=agent_used,
                status=turn_status,
                result=execution,
            )
            return {
                **execution,
                "answer": answer,
                "agent_used": agent_used,
                "trace": execution_trace,
                "session_id": sid,
            }
        except Exception as exc:
            current = self.memory.get_turn(turn.turn_id)
            if current is not None and current.status == "running":
                try:
                    self.memory_service.complete_turn(
                        session,
                        turn.turn_id,
                        answer=str(exc),
                        agent_id=FALLBACK_AGENT,
                        status="error",
                        result={"summary": str(exc)[:300]},
                    )
                except Exception:
                    pass
            raise

    def _resolve_policy(self, route_call):
        evaluator = getattr(
            self.execution_policy,
            "evaluate_call",
            self.execution_policy,
        )
        spec = {}
        if self.registry is not None:
            entry = self.registry.get(route_call.function.name)
            if entry:
                spec = entry.get("meta", {})
        return evaluator(route_call, spec)

    def _make_replan_fn(self):
        """构造运行时重规划回调: 失败/假成功 → 重写下游 DAG(MAO)。

        - 收集本批失败(error/blocked)与"假成功"(SelfEvaluator 判未达目标)步骤;
        - 用 MetaReasoner 为失败步骤补充替代 Agent 建议;
        - Replanner 复用 Router.plan_workflow + DAGValidator 重生成剩余 DAG;
        - 对新决策重新 apply_policy(保证审批检查点不因重规划被绕过)。
        """
        if not REPLAN_ENABLED:
            return None
        if getattr(self, "_replan_ctx", None) is None:
            from mini_thalamus.config import get_llm_config
            from mini_thalamus.meta_reasoner import MetaReasoner
            from mini_thalamus.replanner import Replanner
            from mini_thalamus.self_eval import SelfEvaluator

            self._replan_ctx = {
                "replanner": Replanner(router=self.router),
                "reasoner": MetaReasoner(get_llm_config()),
                "evaluator": (
                    SelfEvaluator(get_llm_config())
                    if SELF_EVAL_ENABLED
                    else None
                ),
            }

        replanner = self._replan_ctx["replanner"]
        reasoner = self._replan_ctx["reasoner"]
        evaluator = self._replan_ctx["evaluator"]

        def available_agents() -> list:
            agent_ids = []
            if self.registry is not None:
                try:
                    agent_ids = list(self.registry.list_all() or {})
                except Exception:
                    agent_ids = []
            agent_ids += [aid for aid in self.agents if aid not in agent_ids]
            return agent_ids or list(ROUTER_AGENTS)

        def replan_fn(info: dict):
            if info.get("attempt", 0) >= REPLAN_MAX:
                return None
            results = info["results"]
            decision = info["decision"]
            original = info["original_query"]
            by_id = {call.call_id: call for call in decision.calls}
            failures: dict = {}
            for call_id in info.get("batch_call_ids", []):
                result = results.get(call_id) or {}
                status = result.get("status")
                if status in ("error", "blocked"):
                    failures[call_id] = {
                        "status": status,
                        "error": str(
                            result.get("answer")
                            or result.get("summary")
                            or ""
                        )[:400],
                    }
            if evaluator is not None:
                for call_id in info.get("batch_call_ids", []):
                    result = results.get(call_id) or {}
                    if result.get("status") != "success":
                        continue
                    call = by_id.get(call_id)
                    agent_name = call.function.name if call else "?"
                    evaluation = evaluator.evaluate(
                        agent_name, original, result
                    )
                    if not evaluation.get("passed"):
                        failures[call_id] = {
                            "status": "semantic_failure",
                            "error": str(evaluation.get("reason"))[:200],
                        }
            if not failures:
                return None
            agents = available_agents()
            for call_id, failure in failures.items():
                call = by_id.get(call_id)
                if call is None:
                    continue
                alternative = reasoner.suggest_alternative(
                    failure.get("error", ""),
                    call.function.name,
                    agents,
                    original,
                )
                if alternative:
                    failure["suggested_alternative"] = alternative
            new_decision = replanner.replan(
                original_query=original,
                results=results,
                failures=failures,
                attempt=info.get("attempt", 0),
            )
            if new_decision is None:
                return None
            try:
                new_decision.apply_policy(self._resolve_policy)
            except Exception:
                return None
            return new_decision

        return replan_fn

    def _build_run_context(
        self,
        sid: str,
        files: list = None,
        previous_result: dict = None,
        memory_struct=None,
        agent_id: str = None,
    ) -> dict:
        """Compatibility helper for existing web and extension code."""
        context = {"session_id": sid}
        if files:
            context["files"] = files
        if previous_result is not None:
            context["previous_result"] = previous_result
        if memory_struct is not None:
            context["memory"] = memory_struct
        if agent_id:
            try:
                agent = self.agents.get(agent_id)
                needs = (
                    getattr(agent, "capabilities", {}).get("needs_credentials", [])
                    if agent
                    else []
                )
                if needs:
                    from mini_thalamus.credentials import ScopedCredView, get_store

                    context["credentials"] = ScopedCredView(get_store(), agent_id)
            except Exception:
                pass
        return context

    def _general_reply(self, message: str, session) -> str:
        try:
            from openai import OpenAI
            from mini_thalamus.config import get_llm_config

            config = get_llm_config()
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 Mini Thalamus 助手。简洁回答；能力不足时明确说明限制。"
                        ),
                    },
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return (
                f"收到。关于「{message[:100]}」，您可以尝试更具体地描述需求，"
                "例如查询数据库、分析文件或生成报告。"
            )


class MultiStepOrchestrator(Orchestrator):
    """Compatibility name; multi-step planning now belongs to Router."""

    def process_multi(self, user_message: str, user_id: str = "default") -> dict:
        return self.process(user_message, user_id=user_id)

    def process_chain(
        self,
        user_message: str,
        plan: list,
        user_id: str = "default",
    ) -> dict:
        """Execute an already selected legacy Agent list through WorkflowExecutor."""
        specs = self._legacy_specs(plan)
        calls = []
        previous_call = None
        for index, agent_id in enumerate(plan, start=1):
            call_id = f"step_{index}"
            calls.append({
                "call_id": call_id,
                "intent": "legacy_step",
                "function": {
                    "name": agent_id,
                    "arguments": {
                        "instruction": user_message,
                        "input_refs": [],
                    },
                },
                "depends_on": [previous_call] if previous_call else [],
                "confidence": 1.0,
            })
            previous_call = call_id
        decision = RouteDecision.from_dict(
            {
                "mode": "workflow" if len(calls) > 1 else "single",
                "calls": calls,
                "candidates": [],
                "overall_confidence": 1.0,
                "method": "legacy_explicit_plan",
                "reason": "caller supplied an Agent plan",
            },
            agent_specs=specs,
        )
        return self.process(
            user_message,
            user_id=user_id,
            decision_override=decision,
        )

    def _legacy_specs(self, plan: list) -> dict:
        if self.registry is not None:
            specs = self.registry.list_all()
        else:
            specs = {}
        for agent_id in plan:
            specs.setdefault(agent_id, {
                "id": agent_id,
                "name": agent_id,
                "description": "legacy Agent",
                "input_schema": {
                    "required": ["instruction"],
                    "properties": {},
                },
                "output_schema": {
                    "exports": ["answer", "summary", "data.structured"],
                },
                "capabilities": {},
            })
        return specs


__all__ = ["FALLBACK_AGENT", "MultiStepOrchestrator", "Orchestrator"]
