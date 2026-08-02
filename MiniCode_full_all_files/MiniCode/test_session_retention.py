import os
import tempfile
import time
import unittest

from memory_db import MemoryDatabase
from memory_store import MemoryStore
from session_manager import SessionManager


class SessionRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.db = MemoryDatabase(self.db_path)
        self.db.initialize()
        self.sessions = SessionManager(database=self.db)
        self.store = MemoryStore(
            data_root=self.temp_dir.name,
            database=self.db,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_turn(self, session_id: str):
        turn = self.store.create_turn(
            session_id=session_id,
            user_id="alice",
            query="test",
            trace_id="trace-1",
        )
        return self.store.complete_turn(
            turn.turn_id,
            answer="ok",
            agent_id="code",
            status="success",
            tags=["test"],
        )

    def test_purge_expired_sessions_removes_old_but_keeps_recent(self):
        self.sessions.resolve("alice", "old-session")
        self.sessions.resolve("alice", "new-session")
        self._make_turn("old-session")
        self._make_turn("new-session")

        now = time.time()
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now - 31 * 86400, "old-session"),
            )

        removed = self.store.purge_expired_sessions(now - 30 * 86400)

        self.assertEqual(removed, 1)
        self.assertEqual(self.store.list_turns("old-session"), [])
        self.assertEqual(len(self.store.list_turns("new-session")), 1)

    def test_cleanup_session_removes_sqlite_turns_and_session(self):
        self.sessions.resolve("alice", "clean-me")
        self._make_turn("clean-me")

        self.store.cleanup_session("clean-me")

        self.assertEqual(self.store.list_turns("clean-me"), [])
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?",
                ("clean-me",),
            ).fetchone()
        self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()
