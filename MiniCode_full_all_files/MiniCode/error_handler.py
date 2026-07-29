"""统一错误处理 —— 降级链"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentError:
    agent: str
    code: str
    message: str


class ErrorHandler:
    """通用降级链 —— 不针对特定 Agent 硬编码，按错误类型降级"""

    # 错误码 → 降级策略优先级（和 Agent 无关，只和错误类型有关）
    ERROR_FALLBACK = {
        "schema_not_found":   ["retry_hint", "ask_user", "fail"],
        "empty_result":       ["retry_broader", "ask_user", "fail"],
        "exec_failed":        ["retry_simpler", "ask_user", "fail"],
        "timeout":            ["retry_simpler", "fail"],
        "permission_denied":  ["fail"],
        "command_not_found":  ["retry_hint", "ask_user", "fail"],
        "ssh_auth_failed":    ["ask_user", "fail"],
        "cwd_denied":         ["fail"],
        "user_cancelled":     ["fail"],
    }
    DEFAULT_CHAIN = ["retry_hint", "ask_user", "fail"]

    # NOTE: 此策略链是"单选器"而非"自动回退链"——ErrorHandler 遍历 chain
    # 列表，命中第一个策略即立即返回，不会在一个策略失败后尝试下一个。
    # 若需要自动级联回退（如 retry_hint 失败后自动尝试 ask_user），需在此处实现。
    def handle(self, agent: str, error: AgentError, session) -> Dict:
        chain = self.ERROR_FALLBACK.get(error.code, self.DEFAULT_CHAIN)

        for strategy in chain:
            if strategy == "retry_hint":
                return {"action": "retry", "agent": agent,
                        "hint": f"执行失败（{error.code}），尝试换一种写法"}

            elif strategy == "retry_broader":
                return {"action": "retry", "agent": agent,
                        "hint": "上次无结果，尝试放宽条件"}

            elif strategy == "retry_simpler":
                return {"action": "retry", "agent": agent,
                        "hint": "执行失败，尝试简化输入"}

            elif strategy == "ask_user":
                return {"action": "ask_user",
                        "message": self._clarify(error)}

            elif strategy == "fail":
                return {"action": "fail",
                        "message": f"抱歉，尝试多种方式后仍无法完成: {error.message}"}

        return {"action": "fail", "message": error.message}

    def _clarify(self, error: AgentError) -> str:
        prompts = {
            "schema_not_found": "没有找到相关数据表。您能提供更多关于要查询的数据的信息吗？",
            "no_results":     "没有找到匹配结果。您能缩小或调整一下查询范围吗？",
            "ambiguous":      "您的问题有几种理解方式，可以再具体一点吗？",
            "exec_failed":    f"执行失败了（{error.message}）。换一种方式描述，我换个角度帮您解决？",
        }
        return prompts.get(error.code, f"遇到了一点困难: {error.message}。换个方式试试？")
