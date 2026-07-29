import os
import tempfile
import unittest

from memory_db import MemoryDatabase
from memory_store import MemoryStore


class MemorySafetyTests(unittest.TestCase):
    def test_legacy_record_turn_does_not_auto_promote_free_text(self):
        with tempfile.TemporaryDirectory() as root:
            db = MemoryDatabase(os.path.join(root, "memory.db"))
            db.initialize()
            store = MemoryStore(data_root=root, database=db)
            promoted = []
            store.set_long_term = lambda key, value: promoted.append((key, value))

            store.record_turn(
                "legacy-session",
                "问题",
                "这是一段超过五十个字符但没有经过验证的普通回答。" * 5,
                "rag",
                tags=["普通主题"],
            )

            self.assertEqual(promoted, [])


if __name__ == "__main__":
    unittest.main()
