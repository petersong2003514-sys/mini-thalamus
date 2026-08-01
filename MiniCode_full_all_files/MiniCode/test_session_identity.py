import os
import tempfile
import unittest

from memory_store import MemoryStore
from session_identity import InvalidSessionId, contained_session_dir


class SessionIdentityTests(unittest.TestCase):
    def test_valid_session_path_is_contained(self):
        with tempfile.TemporaryDirectory() as root:
            target = contained_session_dir(root, "user-session_1")
            expected_root = os.path.abspath(os.path.join(root, "sessions"))
            self.assertEqual(os.path.commonpath((expected_root, target)), expected_root)

    def test_path_traversal_is_rejected_before_create_or_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            store = MemoryStore(data_root=root)
            outside = os.path.abspath(os.path.join(root, "outside"))

            with self.assertRaises(InvalidSessionId):
                store._session_dir("../outside")
            with self.assertRaises(InvalidSessionId):
                store.cleanup_session("../outside")

            self.assertFalse(os.path.exists(outside))


if __name__ == "__main__":
    unittest.main()
