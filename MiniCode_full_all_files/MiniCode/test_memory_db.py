import os
import tempfile
import unittest

from memory_db import MemoryDatabase


class MemoryDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialize_creates_versioned_schema(self):
        db = MemoryDatabase(self.db_path)
        db.initialize()

        with db.connection() as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            version = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]

        expected = {
            "sessions",
            "turns",
            "long_term_memories",
            "session_state",
            "schema_meta",
        }
        self.assertEqual(expected - names, set())
        self.assertEqual(version, "1")

    def test_connections_enable_foreign_keys_and_wal(self):
        db = MemoryDatabase(self.db_path)
        db.initialize()

        with db.connection() as conn:
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        self.assertEqual(foreign_keys, 1)
        self.assertEqual(journal_mode.lower(), "wal")

    def test_transaction_rolls_back_all_writes(self):
        db = MemoryDatabase(self.db_path)
        db.initialize()

        with self.assertRaisesRegex(RuntimeError, "abort"):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, user_id, summary, recent_messages_json,
                        agent_history_json, max_context_tokens,
                        created_at, updated_at
                    ) VALUES ('s1', 'u1', '', '[]', '[]', 8000, 1, 1)
                    """
                )
                raise RuntimeError("abort")

        with db.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        self.assertEqual(count, 0)

    def test_newer_schema_is_rejected(self):
        db = MemoryDatabase(self.db_path)
        db.initialize()
        with db.transaction() as conn:
            conn.execute(
                "UPDATE schema_meta SET value='999' WHERE key='schema_version'"
            )

        with self.assertRaisesRegex(RuntimeError, "newer"):
            db.initialize()


if __name__ == "__main__":
    unittest.main()
