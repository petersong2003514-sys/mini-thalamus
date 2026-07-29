"""BaseAgent ABC + AgentResult + AgentContext — 统一 Agent 协议（向后兼容）"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentResult:
    """Agent 标准返回 —— 与既有 dict 返回完全等价（通过 to_dict 互转）"""

    status: str = "success"
    answer: str = ""
    data: dict = field(default_factory=dict)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "answer": self.answer,
            "data": self.data,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AgentResult":
        return cls(
            status=value.get("status", "success"),
            answer=value.get("answer", ""),
            data=value.get("data", {}) or {},
            summary=value.get("summary", ""),
            metadata=value.get("metadata", {}) or {},
        )


@dataclass
class AgentContext:
    """Structured, lossless context contract shared by every Agent."""

    current_query: str = ""
    relevant_history: List[dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    long_term: dict = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    session_id: str = ""
    previous_result: Optional[dict] = None
    conversation_history: str = ""
    workflow_inputs: dict = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        parts = []
        if self.tags:
            parts.append(f"[记忆标签: {'·'.join(self.tags)}]")
        if self.relevant_history:
            history = "\n---\n".join(
                f"Q: {item.get('q', '')[:80]}\n"
                f"A: {item.get('a', '')[:150]}"
                for item in self.relevant_history[-3:]
            )
            parts.append(f"[相关历史]\n{history}")
        if self.current_query:
            parts.append(self.current_query)
        if self.conversation_history:
            parts.append(self.conversation_history)
        if self.long_term:
            memories = "\n".join(
                f"[长期记忆·{key}] {value}"
                for key, value in self.long_term.items()
            )
            parts.append(memories)
        return "\n\n".join(part for part in parts if part)

    def to_dict(self) -> dict:
        return {
            "current_query": self.current_query,
            "relevant_history": self.relevant_history,
            "tags": self.tags,
            "long_term": self.long_term,
            "files": self.files,
            "session_id": self.session_id,
            "previous_result": self.previous_result,
            "conversation_history": self.conversation_history,
            "workflow_inputs": self.workflow_inputs,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AgentContext":
        return cls(
            current_query=value.get("current_query", ""),
            relevant_history=list(value.get("relevant_history", [])),
            tags=list(value.get("tags", [])),
            long_term=dict(value.get("long_term", {})),
            files=list(value.get("files", [])),
            session_id=value.get("session_id", ""),
            previous_result=value.get("previous_result"),
            conversation_history=value.get("conversation_history", ""),
            workflow_inputs=dict(value.get("workflow_inputs", {})),
        )


class BaseAgent(ABC):
    """所有 Agent 的基类（推荐继承，但不强制，兼容旧 Agent）。"""

    agent_id: str = ""
    capabilities: dict = {
        "needs_history": True,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        "side_effects": "external",
        "auto_chainable": False,
        "needs_credentials": [],
    }

    @abstractmethod
    def run(self, query: str, context: dict = None) -> dict:
        """Execute the Agent and return an AgentResult-compatible dict."""
        ...
