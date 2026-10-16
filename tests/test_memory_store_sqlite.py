import os
import tempfile
import unittest

from mini_thalamus.memory_db import MemoryDatabase
from mini_thalamus.memory_store import MemoryStore, TurnStateError
from mini_thalamus.session_manager import SessionManager


class MemoryStoreSQLiteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.db = MemoryDatabase(self.db_path)
        self.db.initialize()
        self.sessions = SessionManager(database=self.db)
        self.sessions.resolve("alice", "s1")
        self.sessions.resolve("bob", "s2")
        self.store = MemoryStore(
            data_root=self.temp_dir.name,
            database=self.db,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _turn(self, user_id="alice", session_id="s1", status="success"):
        turn = self.store.create_turn(
            session_id=session_id,
            user_id=user_id,
            query="以后报告使用中文",
            trace_id="trace-1",
        )
        return self.store.complete_turn(
            turn.turn_id,
            answer="已记录",
            agent_id="report_writer",
            status=status,
            tags=["报告", "中文"],
        )

    def test_turn_sequence_and_status_are_persisted(self):
        first = self._turn()
        second = self._turn()

        restored = MemoryStore(
            data_root=self.temp_dir.name,
            database=self.db,
        ).list_turns("s1")

        self.assertEqual([turn.sequence_no for turn in restored], [1, 2])
        self.assertEqual(first.status, "success")
        self.assertEqual(second.tags, ("报告", "中文"))

    def test_terminal_turn_cannot_be_rewritten_after_promotion(self):
        turn = self._turn()
        self.store.promote_memory(
            user_id="alice",
            source_turn_id=turn.turn_id,
            namespace="preference",
            memory_key="language",
            gist="Chinese",
            source_agent="report_writer",
            activate=True,
        )

        with self.assertRaises(TurnStateError):
            self.store.complete_turn(
                turn.turn_id,
                answer="late failure",
                agent_id="report_writer",
                status="error",
            )

        self.assertEqual(self.store.get_turn(turn.turn_id).status, "success")
        self.assertEqual(
            self.store.lookup_long_term("alice")[0].gist,
            "Chinese",
        )

    def test_active_memory_is_user_isolated_and_versioned(self):
        turn = self._turn()
        first = self.store.promote_memory(
            user_id="alice",
            source_turn_id=turn.turn_id,
            namespace="preference",
            memory_key="language",
            gist="中文",
            source_agent="report_writer",
            activate=True,
        )
        second = self.store.promote_memory(
            user_id="alice",
            source_turn_id=turn.turn_id,
            namespace="preference",
            memory_key="language",
            gist="英文",
            source_agent="report_writer",
            activate=True,
        )

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(
            self.store.lookup_long_term("alice")[0].gist,
            "英文",
        )
        self.assertEqual(self.store.lookup_long_term("bob"), [])

    def test_non_success_turn_cannot_be_promoted(self):
        turn = self._turn(status="error")

        with self.assertRaisesRegex(ValueError, "successful"):
            self.store.promote_memory(
                user_id="alice",
                source_turn_id=turn.turn_id,
                namespace="business_fact",
                memory_key="revenue",
                gist="100",
                source_agent="nl2sql",
                activate=True,
            )

    def test_turn_cannot_be_promoted_for_another_user(self):
        turn = self._turn()

        with self.assertRaisesRegex(PermissionError, "owner"):
            self.store.promote_memory(
                user_id="bob",
                source_turn_id=turn.turn_id,
                namespace="preference",
                memory_key="language",
                gist="中文",
                source_agent="report_writer",
                activate=True,
            )

    def test_dataset_reference_is_session_state_not_long_term(self):
        self.store.set_session_state(
            "s1",
            "dataset",
            {"path": "safe.csv", "name": "safe"},
        )

        self.assertEqual(
            self.store.get_session_state("s1", "dataset"),
            {"path": "safe.csv", "name": "safe"},
        )
        self.assertEqual(self.store.lookup_long_term("alice"), [])


if __name__ == "__main__":
    unittest.main()
