import os
import tempfile
import unittest

from memory_service import MemoryService


class MemoryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.service = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evidenced_preference_survives_restart_and_is_recalled(self):
        session = self.service.resolve_session("alice", "stable")
        turn = self.service.start_turn(session, "以后报告使用中文")
        self.service.complete_turn(
            session,
            turn.turn_id,
            answer="已记录",
            agent_id="report_writer",
            status="success",
            result={
                "summary": "已记录",
                "metadata": {
                    "memory_candidates": [
                        {
                            "namespace": "preference",
                            "memory_key": "language",
                            "gist": "中文",
                            "evidence": "使用中文",
                        }
                    ]
                },
            },
        )

        restarted = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )
        context = restarted.build_context(
            user_id="alice",
            session_id="stable",
            agent_id="report_writer",
            capabilities={
                "needs_history": True,
                "needs_long_term": True,
                "needs_files": False,
                "max_context_chars": 1000,
            },
            current_query="生成报告",
        )

        self.assertEqual(context.long_term["language"], "中文")

    def test_unproven_or_checkpoint_candidate_is_not_active(self):
        session = self.service.resolve_session("alice", "stable")
        unproven = self.service.start_turn(session, "生成报告")
        self.service.complete_turn(
            session,
            unproven.turn_id,
            answer="完成",
            agent_id="report_writer",
            status="success",
            result={
                "metadata": {
                    "memory_candidates": [
                        {
                            "namespace": "preference",
                            "memory_key": "language",
                            "gist": "英文",
                            "evidence": "用户明确要求英文",
                        }
                    ]
                }
            },
        )
        checkpoint = self.service.start_turn(session, "以后报告使用中文")
        self.service.complete_turn(
            session,
            checkpoint.turn_id,
            answer="等待审批",
            agent_id="report_writer",
            status="checkpoint",
            result={
                "metadata": {
                    "memory_candidates": [
                        {
                            "namespace": "preference",
                            "memory_key": "language",
                            "gist": "中文",
                            "evidence": "使用中文",
                        }
                    ]
                }
            },
        )

        self.assertEqual(self.service.store.lookup_long_term("alice"), [])


if __name__ == "__main__":
    unittest.main()
