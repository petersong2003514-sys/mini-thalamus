"""Atomic application service for session, turn and context memory."""

import os
import threading
import time

from context_assembler import ContextAssembler
from memory_db import MemoryDatabase
from memory_store import MemoryStore
from session_manager import Session, SessionManager


class MemoryService:
    def __init__(
        self,
        database: MemoryDatabase,
        sessions: SessionManager,
        store: MemoryStore,
        assembler: ContextAssembler,
    ):
        self.database = database
        self.sessions = sessions
        self.store = store
        self.assembler = assembler

    @classmethod
    def for_path(
        cls,
        db_path: str,
        *,
        data_root: str = None,
        migrate_legacy: bool = False,
    ) -> "MemoryService":
        root = data_root or os.path.dirname(os.path.abspath(db_path))
        database = MemoryDatabase(db_path)
        database.initialize()
        sessions = SessionManager(database=database)
        store = MemoryStore(data_root=root, database=database)
        if migrate_legacy:
            store.migrate_legacy_json(root)
        database.recover_interrupted_turns(time.time() - 1800)
        assembler = ContextAssembler(store, sessions)
        return cls(database, sessions, store, assembler)

    def resolve_session(
        self,
        user_id: str,
        session_id: str = None,
    ) -> Session:
        return self.sessions.resolve(user_id, session_id)

    def start_turn(
        self,
        session: Session,
        query: str,
        trace_id: str = None,
    ):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        snapshot = session.clone()
        session.add_user_message(query)
        try:
            with self.database.transaction() as conn:
                turn = self.store.create_turn(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    query=query,
                    trace_id=trace_id,
                    conn=conn,
                )
                self.sessions.save(session, conn=conn)
            return turn
        except Exception:
            session.restore_from(snapshot)
            raise

    def complete_turn(
        self,
        session: Session,
        turn_id: str,
        *,
        answer: str,
        agent_id: str,
        status: str,
        tags: list = None,
        result: dict = None,
    ):
        snapshot = session.clone()
        session.add_agent_trace(agent_id, status, answer[:300])
        session.set_last_result(result or {"summary": answer[:300]})
        session.add_assistant_message(answer)
        try:
            with self.database.transaction() as conn:
                completed = self.store.complete_turn(
                    turn_id,
                    answer=answer,
                    agent_id=agent_id,
                    status=status,
                    tags=tags,
                    conn=conn,
                )
                self.sessions.save(session, conn=conn)
            return completed
        except Exception:
            session.restore_from(snapshot)
            raise

    def build_context(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        capabilities: dict,
        current_query: str,
        previous_result: dict = None,
        workflow_inputs: dict = None,
    ):
        return self.assembler.build(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            capabilities=capabilities,
            current_query=current_query,
            previous_result=previous_result,
            workflow_inputs=workflow_inputs,
        )

    def promote_memory(self, **kwargs):
        return self.store.promote_memory(**kwargs)

    def get_session_state(self, session_id: str, state_key: str, default=None):
        return self.store.get_session_state(session_id, state_key, default)

    def set_session_state(self, session_id: str, state_key: str, value):
        return self.store.set_session_state(session_id, state_key, value)


_service = None
_service_lock = threading.Lock()


def get_memory_service() -> MemoryService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                data_root = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data",
                )
                db_path = os.environ.get(
                    "SHUSHU_MEMORY_DB",
                    os.path.join(data_root, "agent_memory.db"),
                )
                _service = MemoryService.for_path(
                    db_path,
                    data_root=data_root,
                    migrate_legacy=True,
                )
    return _service


def reset_memory_service_for_tests():
    global _service
    with _service_lock:
        _service = None


__all__ = [
    "MemoryService",
    "get_memory_service",
    "reset_memory_service_for_tests",
]
