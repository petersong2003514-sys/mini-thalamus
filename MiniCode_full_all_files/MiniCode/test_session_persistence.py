import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from memory_db import MemoryDatabase
from session_manager import SessionManager, SessionOwnershipError


class SessionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.db = MemoryDatabase(self.db_path)
        self.db.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_requested_session_id_is_preserved_and_restored(self):
        first = SessionManager(database=self.db)
        created = first.resolve("alice", "stable-session")
        created.add_user_message("第一轮")
        created.set_last_result({"summary": "第一轮结果"})
        first.save(created)

        restored = SessionManager(database=self.db).resolve(
            "alice", "stable-session"
        )

        self.assertEqual(restored.session_id, "stable-session")
        self.assertEqual(restored.messages[-1]["content"], "第一轮")
        self.assertEqual(restored.last_result["summary"], "第一轮结果")

    def test_session_cannot_be_read_by_another_user(self):
        manager = SessionManager(database=self.db)
        manager.resolve("alice", "private-session")

        with self.assertRaises(SessionOwnershipError):
            SessionManager(database=self.db).resolve("bob", "private-session")

    def test_concurrent_first_claim_has_exactly_one_owner(self):
        managers = [
            SessionManager(database=self.db),
            SessionManager(database=self.db),
        ]
        barrier = threading.Barrier(2)
        for manager in managers:
            original = manager._load_row
            first_read = {"pending": True}

            def synchronized_load(session_id, original=original,
                                  first_read=first_read):
                row = original(session_id)
                if first_read["pending"]:
                    first_read["pending"] = False
                    barrier.wait(timeout=5)
                return row

            manager._load_row = synchronized_load

        def claim(index):
            try:
                return managers[index].resolve(
                    ("alice", "bob")[index],
                    "contested-session",
                ).user_id
            except SessionOwnershipError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(claim, (0, 1)))

        self.assertEqual(outcomes.count("rejected"), 1)
        durable = SessionManager(database=self.db)._load_row(
            "contested-session"
        )
        self.assertIn(durable["user_id"], {"alice", "bob"})

    def test_save_cannot_change_session_owner(self):
        manager = SessionManager(database=self.db)
        session = manager.resolve("alice", "stable-owner")
        session.user_id = "bob"

        with self.assertRaises(SessionOwnershipError):
            manager.save(session)

        restored = SessionManager(database=self.db).resolve(
            "alice", "stable-owner"
        )
        self.assertEqual(restored.user_id, "alice")

    def test_get_or_create_keeps_caller_supplied_id(self):
        session = SessionManager(database=self.db).get_or_create(
            "alice", "client-id"
        )
        self.assertEqual(session.session_id, "client-id")

    def test_save_inside_caller_transaction_rolls_back(self):
        manager = SessionManager(database=self.db)
        session = manager.resolve("alice", "s1")
        session.add_user_message("not durable")

        with self.assertRaisesRegex(RuntimeError, "abort"):
            with self.db.transaction() as conn:
                manager.save(session, conn=conn)
                raise RuntimeError("abort")

        restored = SessionManager(database=self.db).resolve("alice", "s1")
        self.assertEqual(restored.messages, [])


if __name__ == "__main__":
    unittest.main()
