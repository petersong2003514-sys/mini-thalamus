"""统一会话管理 —— 跨 Agent 上下文共享 + SQLite 持久化。"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os
import time
import uuid

from memory_db import MemoryDatabase


class SessionOwnershipError(PermissionError):
    """Raised when a session belongs to another user."""


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
        """Deterministically compact old turns before the context budget."""
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

    def clone(self) -> "Session":
        return deepcopy(self)

    def restore_from(self, snapshot: "Session"):
        self.session_id = snapshot.session_id
        self.user_id = snapshot.user_id
        self.created_at = snapshot.created_at
        self.messages = deepcopy(snapshot.messages)
        self.agent_history = list(snapshot.agent_history)
        self.summary = snapshot.summary
        self.last_result = deepcopy(snapshot.last_result)
        self.max_context_tokens = snapshot.max_context_tokens

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
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "messages": deepcopy(self.messages[-20:]),
            "agent_history": list(self.agent_history),
            "summary": self.summary,
            "last_result": deepcopy(self.last_result),
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_record(cls, row) -> "Session":
        return cls(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=float(row["created_at"]),
            messages=json.loads(row["recent_messages_json"] or "[]"),
            agent_history=json.loads(row["agent_history_json"] or "[]"),
            summary=row["summary"] or "",
            last_result=(
                json.loads(row["last_result_json"])
                if row["last_result_json"]
                else None
            ),
            max_context_tokens=int(row["max_context_tokens"]),
        )


class SessionManager:
    """Session identity and window repository backed by SQLite."""

    def __init__(
        self,
        persist_dir: str = None,
        database: MemoryDatabase = None,
    ):
        self._sessions: Dict[str, Session] = {}
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "agent_memory.db",
        )
        if persist_dir:
            default_path = os.path.join(persist_dir, "agent_memory.db")
        self.database = database or MemoryDatabase(default_path)
        self.database.initialize()

    def create(
        self,
        user_id: str = "default",
        session_id: str = None,
    ) -> Session:
        sid = session_id or uuid.uuid4().hex[:8]
        session = Session(session_id=sid, user_id=user_id)
        self._sessions[sid] = session
        self.save(session)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, user_id: str, session_id: str = None) -> Session:
        return self.resolve(user_id, session_id)

    def resolve(self, user_id: str, session_id: str = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:8]
        cached = self._sessions.get(sid)
        if cached is not None:
            if cached.user_id != user_id:
                raise SessionOwnershipError(sid)
            return cached

        row = self._load_row(sid)
        if row is not None:
            if row["user_id"] != user_id:
                raise SessionOwnershipError(sid)
            session = Session.from_record(row)
            self._sessions[sid] = session
            return session
        return self.create(user_id, sid)

    def save(self, session: Session, conn=None):
        """Persist a complete session, optionally in a caller transaction."""
        payload = session.to_dict()
        parameters = (
            session.session_id,
            session.user_id,
            session.summary,
            json.dumps(payload["messages"], ensure_ascii=False),
            (
                json.dumps(payload["last_result"], ensure_ascii=False)
                if payload["last_result"] is not None
                else None
            ),
            json.dumps(payload["agent_history"], ensure_ascii=False),
            int(payload["max_context_tokens"]),
            float(session.created_at),
            time.time(),
        )
        statement = """
            INSERT INTO sessions (
                session_id, user_id, summary, recent_messages_json,
                last_result_json, agent_history_json, max_context_tokens,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_id=excluded.user_id,
                summary=excluded.summary,
                recent_messages_json=excluded.recent_messages_json,
                last_result_json=excluded.last_result_json,
                agent_history_json=excluded.agent_history_json,
                max_context_tokens=excluded.max_context_tokens,
                updated_at=excluded.updated_at
        """
        if conn is not None:
            conn.execute(statement, parameters)
            return
        with self.database.transaction() as owned:
            owned.execute(statement, parameters)

    def _load_row(self, session_id: str):
        with self.database.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()

    def list_sessions(self, user_id: str = None) -> List[Session]:
        if user_id:
            return [
                session
                for session in self._sessions.values()
                if session.user_id == user_id
            ]
        return list(self._sessions.values())

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)


__all__ = ["Session", "SessionManager", "SessionOwnershipError"]
