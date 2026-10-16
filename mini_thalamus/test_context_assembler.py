import json
import os
import tempfile
import unittest

from base_agent import AgentContext
from context_assembler import ContextAssembler
from memory_db import MemoryDatabase
from memory_store import MemoryStore
from session_manager import SessionManager


class AgentContextTransportTests(unittest.TestCase):
    def test_round_trip_keeps_all_memory_fields(self):
        original = AgentContext(
            current_query="继续",
            relevant_history=[
                {
                    "id": "t1",
                    "q": "前一问",
                    "a": "前一答",
                    "status": "success",
                }
            ],
            tags=["测试"],
            long_term={"language": "中文"},
            files=["a.csv"],
            session_id="s1",
            previous_result={"answer": "done"},
            conversation_history="完整历史",
            workflow_inputs={"step_1.data": {"rows": [1]}},
        )

        restored = AgentContext.from_dict(original.to_dict())

        self.assertEqual(restored, original)


class ContextAssemblerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.db = MemoryDatabase(os.path.join(self.root, "memory.db"))
        self.db.initialize()
        self.sessions = SessionManager(database=self.db)
        session = self.sessions.resolve("alice", "s1")
        session.summary = "用户正在制作销售报告"
        self.sessions.save(session)
        self.store = MemoryStore(data_root=self.root, database=self.db)

        turn = self.store.create_turn(
            session_id="s1",
            user_id="alice",
            query="请制作销售报告",
        )
        turn = self.store.complete_turn(
            turn.turn_id,
            answer="销售额为 100",
            agent_id="nl2sql",
            status="success",
            tags=["销售报告"],
        )
        self.store.promote_memory(
            user_id="alice",
            source_turn_id=turn.turn_id,
            namespace="preference",
            memory_key="language",
            gist="中文",
            source_agent="report_writer",
            activate=True,
        )
        self.assembler = ContextAssembler(self.store, self.sessions)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_capabilities_control_history_and_long_term_injection(self):
        no_memory = self.assembler.build(
            user_id="alice",
            session_id="s1",
            agent_id="websearch",
            capabilities={
                "needs_history": False,
                "needs_long_term": False,
                "needs_files": False,
                "max_context_chars": 1000,
            },
            current_query="继续制作销售报告",
        )
        with_memory = self.assembler.build(
            user_id="alice",
            session_id="s1",
            agent_id="report_writer",
            capabilities={
                "needs_history": True,
                "needs_long_term": True,
                "needs_files": False,
                "max_context_chars": 1000,
            },
            current_query="继续制作销售报告",
        )

        self.assertEqual(no_memory.relevant_history, [])
        self.assertEqual(no_memory.long_term, {})
        self.assertTrue(with_memory.relevant_history)
        self.assertEqual(with_memory.long_term["language"], "中文")
        self.assertIn("用户正在制作销售报告", with_memory.conversation_history)

    def test_recalled_memory_stays_within_agent_budget(self):
        context = self.assembler.build(
            user_id="alice",
            session_id="s1",
            agent_id="report_writer",
            capabilities={
                "needs_history": True,
                "needs_long_term": True,
                "needs_files": False,
                "max_context_chars": 180,
            },
            current_query="继续制作销售报告",
        )
        recalled = (
            context.conversation_history
            + json.dumps(context.relevant_history, ensure_ascii=False)
            + json.dumps(context.long_term, ensure_ascii=False)
        )

        self.assertLessEqual(len(recalled), 180)

    def test_previous_result_and_workflow_inputs_remain_structured(self):
        context = self.assembler.build(
            user_id="alice",
            session_id="s1",
            agent_id="report_writer",
            capabilities={
                "needs_history": False,
                "needs_long_term": False,
                "needs_files": False,
                "max_context_chars": 100,
            },
            current_query="写报告",
            previous_result={"answer": "查询完成"},
            workflow_inputs={"step_1.data": {"revenue": 100}},
        )

        self.assertEqual(context.previous_result["answer"], "查询完成")
        self.assertEqual(
            context.workflow_inputs["step_1.data"]["revenue"],
            100,
        )


if __name__ == "__main__":
    unittest.main()
