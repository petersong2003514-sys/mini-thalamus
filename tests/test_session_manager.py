import unittest

from mini_thalamus.session_manager import Session


class SessionCompactionTests(unittest.TestCase):
    def test_old_messages_are_compacted_into_summary(self):
        session = Session(
            session_id="test",
            user_id="user",
            max_context_tokens=24,
        )
        for index in range(10):
            session.add_user_message(f"message-{index}-" + "x" * 24)

        self.assertTrue(session.summary)
        self.assertLess(len(session.messages), 10)
        total_chars = len(session.summary) + sum(
            len(message["content"]) for message in session.messages
        )
        self.assertLessEqual(total_chars, session.max_context_tokens * 4)
        self.assertIn("message-9", session.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
