import os
import tempfile
import unittest
from contextlib import contextmanager

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

    def test_schema_initialization_is_atomic_on_failure(self):
        class FailingConnection:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, *args):
                if "CREATE TABLE IF NOT EXISTS turns" in sql:
                    raise RuntimeError("injected schema failure")
                return self.connection.execute(sql, *args)

            def executescript(self, script):
                return self.connection.executescript(
                    "CREATE TABLE leaked(value TEXT); INVALID SQL;"
                )

        class FailingDatabase(MemoryDatabase):
            @contextmanager
            def transaction(self):
                with super().transaction() as conn:
                    yield FailingConnection(conn)

        db = FailingDatabase(self.db_path)
        with self.assertRaises(Exception):
            db.initialize()

        with MemoryDatabase(self.db_path).connection() as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertEqual(names, set())

    def test_stale_running_turn_is_recovered_as_interrupted(self):
        db = MemoryDatabase(self.db_path)
        db.initialize()
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, created_at, updated_at
                ) VALUES ('s1', 'alice', 1, 1)
                """
            )
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, user_id, sequence_no,
                    query, status, created_at
                ) VALUES ('t1', 's1', 'alice', 1, 'query', 'running', 1)
                """
            )

        recovered = db.recover_interrupted_turns(2)

        with db.connection() as conn:
            status = conn.execute(
                "SELECT status FROM turns WHERE turn_id='t1'"
            ).fetchone()[0]
        self.assertEqual(recovered, 1)
        self.assertEqual(status, "interrupted")

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
