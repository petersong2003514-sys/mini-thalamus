"""Main Agent orchestration over validated RouteDecision workflows."""

import os

from config import ALLOW_LOCAL_EXECUTION, ROUTER_AGENTS, SANDBOX_USE_DOCKER
from route_schema import RouteDecision
from router import Router
from sandbox import ExecutionPolicy
from session_manager import SessionManager
from workflow_executor import WorkflowExecutor


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
    ):
        self.agents = agents or {}
        self.registry = registry or getattr(runtime, "registry", None)
        self.runtime = runtime or _AgentMapRuntime(self.agents)
        self.router = router or Router(registry=self.registry)
        self.session_mgr = session_mgr or SessionManager()
        self.execution_policy = execution_policy or ExecutionPolicy(
            workspace_root=os.path.dirname(os.path.abspath(__file__)),
            local_execution_enabled=ALLOW_LOCAL_EXECUTION,
            docker_backend_enabled=SANDBOX_USE_DOCKER,
        )
        self.workflow_executor = WorkflowExecutor(self.runtime)
        from memory_store import get_store

        self.memory = get_store()

    def process(
        self,
        user_message: str,
        user_id: str = "default",
        session_id: str = None,
        approved_call_ids=None,
    ) -> dict:
        session = self.session_mgr.get_or_create(user_id, session_id)
        session.add_user_message(user_message)
        sid = session.session_id

        files = self.memory.get_files(sid)
        file_context = None
        if files:
            file_context = {
                "has_files": True,
                "file_names": files,
                "session_id": sid,
            }
        decision = self.router.route_decision(
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
            "candidates": [item.to_dict() for item in decision.candidates],
        }
        if not decision.calls:
            answer = self._general_reply(user_message, session)
            session.add_assistant_message(answer)
            self.memory.record_turn(sid, user_message, answer, FALLBACK_AGENT)
            return {
                "status": "success",
                "answer": answer,
                "agent_used": FALLBACK_AGENT,
                "trace": [route_trace],
                "session_id": sid,
                "route_decision": decision.to_dict(),
            }

        execution = self.workflow_executor.execute(
            decision,
            original_query=user_message,
            context={
                "session_id": sid,
                "files": files,
                "memory": self.memory.build_agent_context(
                    sid,
                    decision.calls[0].function.name,
                    user_message,
                    as_struct=True,
                ).to_dict(),
            },
            approved_call_ids=approved_call_ids,
        )
        execution_trace = [route_trace, *execution.get("trace", [])]
        if execution["status"] == "checkpoint":
            return {
                **execution,
                "agent_used": execution["pending_call"]["function"]["name"],
                "trace": execution_trace,
                "session_id": sid,
            }

        answer = execution.get("answer") or execution.get("summary") or "处理完成"
        executed_agents = [
            item["agent_id"]
            for item in execution.get("trace", [])
            if item.get("status") != "skipped"
        ]
        agent_used = executed_agents[-1] if executed_agents else FALLBACK_AGENT
        session.add_agent_trace(agent_used, execution["status"], execution.get("summary", ""))
        session.set_last_result(execution)
        session.add_assistant_message(answer)
        self.memory.record_turn(sid, user_message, answer, agent_used)
        return {
            **execution,
            "answer": answer,
            "agent_used": agent_used,
            "trace": execution_trace,
            "session_id": sid,
        }

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
                    from credentials import ScopedCredView, get_store

                    context["credentials"] = ScopedCredView(get_store(), agent_id)
            except Exception:
                pass
        return context

    def _general_reply(self, message: str, session) -> str:
        try:
            from openai import OpenAI
            from config import get_llm_config

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
                            "你是 ShuShu Agent 助手。简洁回答；能力不足时明确说明限制。"
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
        session = self.session_mgr.get_or_create(user_id)
        result = self.workflow_executor.execute(
            decision,
            original_query=user_message,
            context={"session_id": session.session_id},
        )
        result["session_id"] = session.session_id
        return result

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
