"""BaseAgent ABC + AgentResult + AgentContext — 统一 Agent 协议（向后兼容）"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AgentResult:
    """Agent 标准返回 —— 与既有 dict 返回完全等价（通过 to_dict 互转）"""
    status: str = "success"  # "success" | "error"
    answer: str = ""
    data: dict = field(default_factory=dict)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "answer": self.answer,
                "data": self.data, "summary": self.summary,
                "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> "AgentResult":
        return cls(
            status=d.get("status", "success"),
            answer=d.get("answer", ""),
            data=d.get("data", {}) or {},
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class AgentContext:
    """结构化记忆消费接口 —— 取代原来"把所有内容拼成一个大字符串"的做法。

    向后兼容设计:
      - Agent 仍可以收到拼好的 prompt 字符串作为 query 参数（旧路径）
      - 想用结构化字段的 Agent 通过 context.relevant_history / context.long_term 取
      - 通过 to_prompt_string() 可一键还原成旧格式拼接的字符串
    """
    current_query: str = ""
    relevant_history: List[dict] = field(default_factory=list)  # [{q,a,agent,tags}]
    tags: List[str] = field(default_factory=list)
    long_term: dict = field(default_factory=dict)               # {tag: gist}
    files: List[str] = field(default_factory=list)
    session_id: str = ""
    previous_result: Optional[dict] = None                       # 多步链中前一步结果
    conversation_history: str = ""                               # 完整 Q&A 拼接（兼容旧 agent）

    def to_prompt_string(self) -> str:
        """把结构化字段拼成单字符串，等价于旧的 build_agent_context 返回值"""
        parts = []
        if self.tags:
            parts.append(f"[记忆标签: {'·'.join(self.tags)}]")
        if self.relevant_history:
            hist = "\n---\n".join(
                f"Q: {h.get('q', '')[:80]}\nA: {h.get('a', '')[:150]}"
                for h in self.relevant_history[-3:]
            )
            parts.append(f"[相关历史]\n{hist}")
        if self.current_query:
            parts.append(self.current_query)
        if self.conversation_history:
            parts.append(self.conversation_history)
        if self.long_term:
            ltm = "\n".join(f"[长期记忆·{k}] {v}" for k, v in self.long_term.items())
            parts.append(ltm)
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "current_query": self.current_query,
            "relevant_history": self.relevant_history,
            "tags": self.tags,
            "long_term": self.long_term,
            "files": self.files,
            "session_id": self.session_id,
            "previous_result": self.previous_result,
        }


class BaseAgent(ABC):
    """所有 Agent 的基类（推荐继承，但不强制 —— 向后兼容老 agent）"""

    agent_id: str = ""
    capabilities: dict = {
        "needs_history": True,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        # ─── FEATURE_PLAN.md §1 P0.2（audit R3）保守默认 ───
        # 任何未显式覆盖的 Agent（含 wrapper/skill/mcp）都被当作:
        #   - 有外部副作用 → memory_store 不自动 crystallize
        #   - 不允许 auto-chain → 中间步骤仍走 checkpoint 暂停
        #   - 不需要凭证视图 → orchestrator 不注入 ScopedCredView
        "side_effects": "external",
        "auto_chainable": False,
        "needs_credentials": [],
    }

    @abstractmethod
    def run(self, query: str, context: dict = None) -> dict:
        """执行 Agent 逻辑。

        参数:
          query: 既可以是用户原始 query，也可以是 build_agent_context 拼好的字符串（向后兼容）
          context: 可以是普通 dict 或 AgentContext（用 isinstance 区分），也可以是 None
        返回:
          dict —— 兼容现有调用方；新 Agent 也可以返回 AgentResult().to_dict()
        """
        ...
