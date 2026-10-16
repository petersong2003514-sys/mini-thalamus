"""SelfEvaluator —— Agent 步骤轻量自我评估

每步 Agent 执行后，调一次 LLM 判断是否真正解决了问题。
技术成功但实际未完成（如命令运行但文件不存在）时标记为需要重做。
"""
from __future__ import annotations
import logging
from typing import Optional

_log = logging.getLogger(__name__)

# 明显失败信号：无需 LLM 就能判断（用词边界/上下文收紧，避免误判正常描述）
_OBVIOUS_FAIL_SIGNALS = [
    "file not found", "no such file", "command not found",
    "error:", "错误:", "exception:", "traceback", "failed:",
    "执行失败", "操作失败", "运行失败",
    "cannot be", "cannot find", "unable to",
    "无法执行", "无法完成", "无法找到", "无法访问",
    "拒绝访问", "请求被拒绝",
    "blocked:",
]

# 明显成功信号：有这些词时跳过 LLM 评估，节省 token
# 注意：不要加 "found" —— 它是 "not found"/"no such file ... found" 的子串，
# 在"成功优先"判定下会把"文件未找到"类失败误判为成功（见 _OBVIOUS_FAIL_SIGNALS）。
_OBVIOUS_SUCCESS_SIGNALS = [
    "成功", "完成", "已抓取", "已提取", "已生成", "successfully",
    "created", "written", "saved",
]


class SelfEvaluator:
    """Agent 步骤自我评估器。"""

    def __init__(self, llm_config: dict = None):
        self._llm_config = llm_config or {}
        self._client = None

    def _llm(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._llm_config.get("api_key", ""),
                base_url=self._llm_config.get("base_url", ""),
            )
        except Exception:
            pass
        return self._client

    def evaluate(
        self,
        agent_name: str,
        user_goal: str,
        result: dict,
    ) -> dict:
        """评估一步 Agent 的结果是否真正完成了用户目标。

        Returns:
            {"passed": bool, "reason": str}
        """
        answer = (result.get("answer") or result.get("summary") or "")[:600]
        status = result.get("status", "success")

        # 1. 状态本身就是失败
        if status == "error":
            return {"passed": False, "reason": f"Agent 返回 error 状态: {answer[:100]}"}

        answer_lower = answer.lower()

        # 2. 明显成功信号（优先：命中即 pass，避免答案中同时含成功和失败词被误判）
        for sig in _OBVIOUS_SUCCESS_SIGNALS:
            if sig in answer_lower:
                return {"passed": True, "reason": "output contains success signal"}

        # 3. 明显失败信号（不调 LLM）
        for sig in _OBVIOUS_FAIL_SIGNALS:
            if sig in answer_lower:
                return {"passed": False, "reason": f"输出包含失败信号: '{sig}'"}

        # 4. LLM 判断模糊地带
        return self._llm_eval(agent_name, user_goal, answer)

    def _llm_eval(self, agent_name: str, user_goal: str, answer: str) -> dict:
        client = self._llm()
        if client is None:
            return {"passed": True, "reason": "llm unavailable, assume pass"}
        try:
            resp = client.chat.completions.create(
                model=self._llm_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": (
                        "你是执行结果评估者。判断 Agent 是否真正完成了用户目标。"
                        "只输出 JSON: {\"passed\": true/false, \"reason\": \"一句话\"}"
                    )},
                    {"role": "user", "content": (
                        f"Agent: {agent_name}\n"
                        f"用户目标: {user_goal[:200]}\n"
                        f"Agent 输出: {answer}\n"
                        "技术上执行了不代表真正完成了目标。判断是否真正完成："
                    )},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            import json, re
            text = resp.choices[0].message.content.strip()
            if "```" in text:
                text = re.sub(r"```\w*\n?", "", text).replace("```", "")
            ev = json.loads(text)
            return {"passed": bool(ev.get("passed", True)), "reason": ev.get("reason", "")}
        except Exception as e:
            _log.debug("SelfEvaluator LLM err: %s", str(e)[:80])
            return {"passed": True, "reason": "eval failed, assume pass"}


__all__ = ["SelfEvaluator"]
