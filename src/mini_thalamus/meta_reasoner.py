"""MetaReasoner —— Agent 链执行失败时分析原因并建议替代 Agent

当 Agent 链某步失败，MetaReasoner 分析失败原因，
从 Agent 池中选择替代方案，避免整条链直接中断。
规则快速匹配优先，规则未命中再走 LLM 兜底。
"""
from __future__ import annotations
import re, logging
from typing import Optional

_log = logging.getLogger(__name__)

# 错误类型 → 候选替代 Agent 的偏好顺序（规则快速匹配）
_ERROR_ROUTING = [
    (re.compile(r"file.not.found|no such file|不存在|找不到.*文件", re.I),
     ["api_caller", "shell", "url_fetch", "websearch"]),
    (re.compile(r"connection.refused|timeout|connect.*failed|网络.*错误|超时", re.I),
     ["websearch", "url_fetch"]),
    (re.compile(r"permission.denied|403|401|unauthorized|权限.*拒绝", re.I),
     ["api_caller", "browser"]),
    (re.compile(r"\b404\b|page.not.found|页面不存在", re.I),
     ["websearch", "url_fetch"]),
    (re.compile(r"syntax.error|parse.error|invalid.json|解析失败|unexpected token|<\!DOCTYPE", re.I),
     ["api_caller", "code", "general"]),
    (re.compile(r"command.not.found|not.installed|未安装|没有.*命令", re.I),
     ["api_caller", "url_fetch"]),
    (re.compile(r"api.*error|request.failed|请求失败", re.I),
     ["browser", "websearch"]),
]


class MetaReasoner:
    """链失败恢复器：规则匹配 + LLM 兜底，推荐替代 Agent。"""

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

    def suggest_alternative(
        self,
        error: str,
        original_agent: str,
        available_agents: list,
        context: str = "",
    ) -> Optional[str]:
        """分析失败原因，返回替代 Agent ID；无合适选择返回 None。"""
        # 1. 规则快速匹配
        for pattern, candidates in _ERROR_ROUTING:
            if pattern.search(error or ""):
                for c in candidates:
                    if c != original_agent and c in available_agents:
                        _log.info("MetaReasoner rule: %s -> %s (%.60s)",
                                  original_agent, c, error)
                        return c
        # 2. LLM 兜底
        return self._llm_suggest(error, original_agent, available_agents, context)

    def _llm_suggest(
        self,
        error: str,
        original_agent: str,
        available_agents: list,
        context: str,
    ) -> Optional[str]:
        client = self._llm()
        if client is None:
            return None
        try:
            candidates = [a for a in available_agents if a != original_agent]
            resp = client.chat.completions.create(
                model=self._llm_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": (
                        "你是 Agent 链恢复决策者。分析失败原因，从可用 Agent 中选出最合适的替代方案。"
                        "只输出 agent_id（单个字符串）或 null。"
                    )},
                    {"role": "user", "content": (
                        f"失败 Agent: {original_agent}\n"
                        f"错误信息: {(error or '')[:400]}\n"
                        f"用户意图: {context[:200]}\n"
                        f"候选 Agent: {', '.join(candidates)}\n"
                        "选择替代 Agent（只输出 agent_id 或 null）:"
                    )},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            s = resp.choices[0].message.content.strip().strip('"').strip("'")
            if s.lower() in ("null", "none", ""):
                return None
            if s in available_agents and s != original_agent:
                _log.info("MetaReasoner LLM: %s -> %s", original_agent, s)
                return s
        except Exception as e:
            _log.debug("MetaReasoner LLM err: %s", str(e)[:80])
        return None


__all__ = ["MetaReasoner"]
