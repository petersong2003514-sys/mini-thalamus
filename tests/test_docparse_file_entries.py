import unittest
from unittest.mock import patch

from mini_thalamus.agents.docparse_agent import DocParseAgent


class DocParseFileEntryTests(unittest.TestCase):
    @patch("mini_thalamus.agents.docparse_agent.get_store")
    def test_string_file_names_do_not_crash(self, mock_get_store):
        store = mock_get_store.return_value
        store.load_all_parsed_files.return_value = []
        agent = DocParseAgent()
        agent.store = store

        result = agent.run(
            "这个文档的核心内容是什么？",
            context={
                "files": ["季度报告.pdf"],
                "session_id": "session-file-test",
            },
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("未获取到文档内容", result["answer"])


if __name__ == "__main__":
    unittest.main()
