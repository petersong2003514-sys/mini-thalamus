import os
import tempfile
import unittest
from unittest.mock import patch

from agents import data_analyst_agent as module
from memory_service import MemoryService


class DataAnalystSessionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.temp_dir.name, "data"), exist_ok=True)
        self.service = MemoryService.for_path(
            os.path.join(self.temp_dir.name, "memory.db"),
            data_root=self.temp_dir.name,
        )
        self.service.resolve_session("alice", "s1")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_path_is_recovered_from_session_state(self):
        agent = module.DataAnalystAgent()
        agent._generate_analysis_code = lambda query, path, profile: "print('ok')"
        agent._sandbox_execute = lambda code: {
            "stdout": "ok",
            "stderr": "",
            "work_dir": "",
        }
        agent._interpret = lambda *args: "分析完成"
        upload = {
            "name": "sales.csv",
            "content": b"region,revenue\nbeijing,100\n",
        }

        state_view = self.service.session_state("s1")
        with patch(
            "memory_service.get_memory_service",
            side_effect=AssertionError("global service must not be used"),
        ), patch.object(module, "_MINICODE_ROOT", self.temp_dir.name):
            first = agent.run(
                "统计销售额",
                context={
                    "session_id": "s1",
                    "files": [upload],
                    "session_state": state_view,
                },
            )
            second = agent.run(
                "继续分析",
                context={
                    "session_id": "s1",
                    "files": [],
                    "session_state": state_view,
                },
            )

        state = self.service.get_session_state("s1", "dataset")
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        self.assertTrue(state["path"].endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
