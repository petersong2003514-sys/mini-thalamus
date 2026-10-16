import json
import os
import tempfile
import unittest

from mini_thalamus.memory_db import MemoryDatabase
from mini_thalamus.memory_store import MemoryStore


class MemoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.legacy_root = os.path.join(self.temp_dir.name, "legacy")
        os.makedirs(self.legacy_root)
        self.db = MemoryDatabase(
            os.path.join(self.temp_dir.name, "memory.db")
        )
        self.db.initialize()
        self.store = MemoryStore(
            data_root=self.legacy_root,
            database=self.db,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path, value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)

    def _write_legacy_fixture(self):
        session_dir = os.path.join(
            self.legacy_root,
            "sessions",
            "legacy-session",
        )
        self._write_json(
            os.path.join(session_dir, "session.json"),
            {
                "session_id": "legacy-session",
                "user_id": "alice",
                "created_at": 10,
                "messages": [{"role": "user", "content": "旧问题"}],
                "agent_history": ["rag"],
                "summary": "旧摘要",
            },
        )
        self._write_json(
            os.path.join(session_dir, "history.json"),
            {
                "session_id": "legacy-session",
                "turns": [
                    {
                        "id": "turn_1",
                        "q": "旧问题",
                        "a": "旧回答",
                        "agent": "rag",
                        "tags": ["旧数据"],
                        "ts": 11,
                    }
                ],
            },
        )
        self._write_json(
            os.path.join(self.legacy_root, "memory.json"),
            {
                "entries": {
                    "topic:旧事实": {
                        "value": {
                            "gist": "未经用户归属验证的旧事实",
                            "agent": "rag",
                            "ts": 12,
                        },
                        "updated_at": 12,
                    }
                }
            },
        )

    def test_legacy_import_is_idempotent_and_quarantines_global_memory(self):
        self._write_legacy_fixture()

        first = self.store.migrate_legacy_json(self.legacy_root)
        second = self.store.migrate_legacy_json(self.legacy_root)

        self.assertEqual(first["sessions"], 1)
        self.assertEqual(first["turns"], 1)
        self.assertEqual(first["memories"], 1)
        self.assertEqual(
            second,
            {"sessions": 0, "turns": 0, "memories": 0, "files": 0},
        )
        self.assertEqual(
            self.store.list_turns("legacy-session")[0].answer,
            "旧回答",
        )
        self.assertEqual(self.store.lookup_long_term("alice"), [])
        pending = self.store.lookup_long_term(
            "legacy_import",
            statuses=("pending",),
        )
        self.assertEqual(pending[0].namespace, "legacy")
        self.assertTrue(
            os.path.exists(os.path.join(self.legacy_root, "memory.json"))
        )


if __name__ == "__main__":
    unittest.main()
