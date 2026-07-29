"""统一会话管理 —— 跨 Agent 上下文共享 + 磁盘持久化"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time
import os
import json


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    messages: List[Dict] = field(default_factory=list)
    agent_history: List[str] = field(default_factory=list)
    summary: str = ""
    last_result: Optional[Dict] = None
    max_context_tokens: int = 8000

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._compact_if_needed()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._compact_if_needed()

    def add_agent_trace(self, agent: str, action: str, result_summary: str):
        self.agent_history.append(agent)
        trace = f"[AgentTrace] {agent} 完成: {action} → {result_summary[:300]}"
        self.messages.append({"role": "system", "content": trace})
        self._compact_if_needed()

    def _compact_if_needed(self):
        """Deterministically compact old turns before the context budget is exceeded."""
        max_chars = max(80, int(self.max_context_tokens) * 4)
        current_chars = len(self.summary) + sum(
            len(str(message.get("content", ""))) for message in self.messages
        )
        if current_chars <= max_chars or len(self.messages) <= 2:
            return

        keep_count = 2
        while keep_count < len(self.messages):
            candidate = self.messages[-(keep_count + 1):]
            candidate_chars = sum(
                len(str(message.get("content", ""))) for message in candidate
            )
            if candidate_chars > int(max_chars * 0.75):
                break
            keep_count += 1
        old_messages = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]
        compacted = "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in old_messages
        )
        combined = "\n".join(part for part in (self.summary, compacted) if part)
        kept_chars = sum(
            len(str(message.get("content", ""))) for message in self.messages
        )
        summary_budget = max(0, max_chars - kept_chars)
        self.summary = combined[-summary_budget:] if summary_budget else ""

    def get_context_for(self, agent: str) -> str:
        ctx = f"Session: {self.session_id}\n"
        ctx += f"历史 Agent: {' → '.join(self.agent_history)}\n"
        if self.summary:
            ctx += f"对话摘要: {self.summary}\n"
        if self.last_result:
            ctx += f"上一步结论: {self.last_result.get('summary', '')}\n"
        return ctx

    def set_summary(self, text: str):
        self.summary = text

    def set_last_result(self, result: Dict):
        self.last_result = result

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "user_id": self.user_id,
            "created_at": self.created_at,
            "messages": self.messages[-20:],
            "agent_history": self.agent_history,
            "summary": self.summary,
        }


class SessionManager:
    """会话管理器 —— 内存 + 磁盘持久化"""

    def __init__(self, persist_dir: str = None):
        self._sessions: Dict[str, Session] = {}
        self._persist_dir = persist_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "sessions")
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
        except OSError:
            self._persist_dir = None  # read-only FS: skip disk persistence

    def create(self, user_id: str = "default") -> Session:
        sid = str(uuid.uuid4())[:8]
        session = Session(session_id=sid, user_id=user_id)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, user_id: str, session_id: str = None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        # 尝试从磁盘恢复
        if session_id:
            restored = self._load_from_disk(session_id)
            if restored:
                self._sessions[session_id] = restored
                return restored
        return self.create(user_id)

    def save(self, session: Session):
        """持久化 session 到磁盘（原子写入，防并发损坏）"""
        path = os.path.join(self._persist_dir, session.session_id, "session.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass

    def _load_from_disk(self, sid: str) -> Optional[Session]:
        path = os.path.join(self._persist_dir, sid, "session.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session(
                session_id=data["session_id"], user_id=data.get("user_id", "default"),
                created_at=data.get("created_at", time.time()),
                messages=data.get("messages", []),
                agent_history=data.get("agent_history", []),
                summary=data.get("summary", ""),
            )
        except Exception:
            return None

    def list_sessions(self, user_id: str = None) -> List[Session]:
        if user_id:
            return [s for s in self._sessions.values() if s.user_id == user_id]
        return list(self._sessions.values())

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)
