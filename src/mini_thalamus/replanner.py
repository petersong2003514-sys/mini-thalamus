"""Replanner —— 失败后基于中间结果重写下游 DAG(MAO 运行时重规划)。

背景: WorkflowExecutor 按依赖分批执行 DAG。当某一步失败(或"假成功"),
单纯替换当前 Agent 重跑时,下游步骤仍建立在错误结果上;真正的 MAO 是
根据中间执行状态实时修正后续调用节点。

本模块把「已完成中间结果 + 失败原因」作为上下文,复用 Router.plan_workflow
(LLM 规划 + DAGValidator 校验/修复)为剩余目标重新生成一张自包含 DAG:
- 已完成的中间结果值直接内联进后续指令(与项目"指令自包含、不留占位符"一致);
- 新 call_id 命名空间化(如 rp0_step_1),避免与已完成步骤冲突;
- 无法重规划时返回 None,由执行器保留原计划(失败依赖按原有规则跳过)。
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

from mini_thalamus.route_schema import RouteDecision

_log = logging.getLogger(__name__)


class Replanner:
    """基于中间结果生成新 RouteDecision。"""

    def __init__(
        self,
        router=None,
        agent_specs: Mapping[str, dict] = None,
        max_steps: int = 6,
    ):
        self.router = router
        self._specs_override = dict(agent_specs or {})
        self.max_steps = max(1, int(max_steps))

    # ── 对外接口 ──
    def replan(
        self,
        *,
        original_query: str,
        results: Mapping[str, dict],
        failures: Mapping[str, dict],
        attempt: int = 0,
    ) -> Optional[RouteDecision]:
        """重规划剩余目标,返回新 RouteDecision(未 apply_policy,由调用方负责)。

        results:   call_id -> 步骤结果 dict(status/answer/summary/...)
        failures:  call_id -> {"status":..., "error":..., "suggested_alternative":...}
        attempt:   第几次重规划(用于 call_id 命名空间前缀)

        无法重规划时返回 None。
        """
        if self.router is None:
            _log.warning("Replanner: 无 Router,跳过重规划")
            return None
        context = self._build_context(results, failures)
        try:
            workflow = self.router.plan_workflow(
                original_query,
                session_context=context,
                use_llm=True,
                max_steps=self.max_steps,
                validate=True,
            )
        except Exception as exc:  # LLM 不可用 / 校验修复失败 → 不重规划
            _log.warning("Replanner: 重新规划失败: %s", str(exc)[:200])
            return None
        decision = workflow.route_decision
        if not decision.calls:
            return None
        self._namespace(decision, results, attempt)
        return decision

    # ── 内部 ──
    def _build_context(self, results: Mapping[str, dict], failures: Mapping[str, dict]) -> str:
        parts = ["以下 Agent 步骤已经完成,其最终结果(值)可直接用于后续步骤,不要再重复执行:", ""]
        completed = [
            (call_id, result)
            for call_id, result in results.items()
            if result.get("status") == "success"
        ]
        for call_id, result in completed:
            answer = str(result.get("answer") or result.get("summary") or "")[:300]
            agent = (result.get("metadata") or {}).get("agent_id", "?")
            parts.append(f"- {call_id} [{agent}] (成功): {answer}")
        if failures:
            parts.append("")
            parts.append("以下步骤失败/未完成,需要重做或换一个可行的 Agent 完成对应目标:")
            for call_id, failure in failures.items():
                detail = str(
                    failure.get("error")
                    or failure.get("reason")
                    or failure.get("status")
                )[:300]
                parts.append(f"- {call_id}: {detail}")
                alt = failure.get("suggested_alternative")
                if alt:
                    parts.append(f"  (可考虑替代 Agent: {alt})")
        parts.append("")
        parts.append(
            "请针对用户的原始请求,只规划【尚未完成/失败后需要重做】的步骤;"
            "把已完成的中间结果值直接写进后续步骤的 instruction(不要用占位符),"
            "不要重复规划已经成功的步骤。"
        )
        return "\n".join(parts)

    def _namespace(self, decision: RouteDecision, results: Mapping, attempt: int) -> None:
        """把新计划的所有 call_id 命名空间化,避免与已完成结果冲突。"""
        prefix = f"rp{attempt}_"
        used = set(results)
        rename: dict[str, str] = {}
        for call in decision.calls:
            new_id = prefix + call.call_id
            while new_id in used:
                new_id = "_" + new_id
            used.add(new_id)
            rename[call.call_id] = new_id
        for call in decision.calls:
            call.call_id = rename[call.call_id]
            call.depends_on = [rename.get(dep, dep) for dep in call.depends_on]
            refs = call.function.arguments.get("input_refs", [])
            call.function.arguments["input_refs"] = [
                self._rename_ref(ref, rename) for ref in refs
            ]

    @staticmethod
    def _rename_ref(ref: str, rename: Mapping[str, str]) -> str:
        source, sep, rest = ref.partition(".")
        if not sep:
            return ref
        return rename.get(source, source) + "." + rest


__all__ = ["Replanner"]
