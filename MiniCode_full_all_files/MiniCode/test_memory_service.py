import os
import tempfile
import unittest

from memory_service import MemoryService


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.service = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_turn_atomically_updates_turn_and_session(self):
        session = self.service.resolve_session("alice", "s1")
        turn = self.service.start_turn(session, "第一问", "trace-1")

        self.service.complete_turn(
            session,
            turn.turn_id,
            answer="第一答",
            agent_id="rag",
            status="success",
            tags=["第一"],
            result={"summary": "第一答"},
        )

        restarted = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )
        restored = restarted.resolve_session("alice", "s1")
        saved = restarted.store.get_turn(turn.turn_id)
        self.assertEqual(saved.status, "success")
        self.assertEqual(restored.last_result["summary"], "第一答")
        self.assertEqual(restored.messages[0]["content"], "第一问")
        self.assertEqual(restored.messages[-1]["content"], "第一答")

    def test_failed_serialization_rolls_back_turn_and_in_memory_session(self):
        session = self.service.resolve_session("alice", "s1")
        turn = self.service.start_turn(session, "问题")
        before = session.to_dict()

        with self.assertRaises(TypeError):
            self.service.complete_turn(
                session,
                turn.turn_id,
                answer="回答",
                agent_id="rag",
                status="success",
                result={"bad": object()},
            )

        self.assertEqual(self.service.store.get_turn(turn.turn_id).status, "running")
        self.assertEqual(session.to_dict(), before)

    def test_start_turn_rolls_back_session_when_query_is_not_serializable(self):
        session = self.service.resolve_session("alice", "s1")
        before = session.to_dict()

        with self.assertRaises(TypeError):
            self.service.start_turn(session, object())

        self.assertEqual(session.to_dict(), before)
        self.assertEqual(self.service.store.list_turns("s1"), [])

    def test_build_context_delegates_to_capability_aware_assembler(self):
        session = self.service.resolve_session("alice", "s1")
        turn = self.service.start_turn(session, "记住这轮")
        self.service.complete_turn(
            session,
            turn.turn_id,
            answer="完成",
            agent_id="rag",
            status="success",
        )

        context = self.service.build_context(
            user_id="alice",
            session_id="s1",
            agent_id="rag",
            capabilities={
                "needs_history": True,
                "needs_long_term": False,
                "needs_files": False,
                "max_context_chars": 1000,
            },
            current_query="继续",
        )

        self.assertEqual(context.session_id, "s1")
        self.assertIn("记住这轮", context.conversation_history)


if __name__ == "__main__":
    unittest.main()
