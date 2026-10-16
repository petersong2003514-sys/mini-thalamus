import unittest
from unittest.mock import patch

import main


class FakeSession:
    session_id = "interactive-session"


class FakeSessionManager:
    def create(self, user_id):
        self.user_id = user_id
        return FakeSession()


class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def process(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return {
            "status": "success",
            "answer": "ok",
            "agent_used": "report_writer",
            "trace": [{"confidence": 1.0}],
        }


class MainMemoryIdentityTests(unittest.TestCase):
    def test_interactive_cli_reuses_session_owner(self):
        orchestrator = FakeOrchestrator()
        sessions = FakeSessionManager()

        with patch("builtins.input", side_effect=["hello", "exit"]):
            main.interactive_mode(orchestrator, sessions)

        self.assertEqual(sessions.user_id, "interactive_user")
        self.assertEqual(
            orchestrator.calls[0][1],
            {
                "user_id": "interactive_user",
                "session_id": "interactive-session",
            },
        )


if __name__ == "__main__":
    unittest.main()
