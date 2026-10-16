import unittest
from unittest.mock import MagicMock

from orchestrator import Orchestrator
from route_schema import RouteDecision


SPECS = {
    "nl2sql": {
        "output_schema": {
            "exports": ["answer", "summary", "data.structured"],
        },
    },
}


class FakeTurn:
    turn_id = "turn-1"


class FakeSession:
    session_id = "session-1"


class FakeMemory:
    def __init__(self):
        self.sessions = MagicMock()
        self.store = self

    def resolve_session(self, user_id, session_id=None):
        return FakeSession()

    def start_turn(self, session, message):
        return FakeTurn()

    def complete_turn(self, *args, **kwargs):
        return None

    def build_context(self, *args, **kwargs):
        return MagicMock(to_dict=lambda: {})

    def session_state(self, session_id):
        return {}

    def get_files(self, session_id):
        return []

    def get_turn(self, turn_id):
        return MagicMock(status="running")


def route_decision():
    payload = {
        "mode": "single",
        "calls": [
            {
                "call_id": "step_1",
                "intent": "database_query",
                "function": {
                    "name": "nl2sql",
                    "arguments": {
                        "instruction": "query data",
                        "input_refs": [],
                    },
                },
                "depends_on": [],
                "confidence": 0.9,
            }
        ],
        "candidates": [],
        "overall_confidence": 0.9,
        "method": "test",
        "reason": "test",
        "trace_id": "trace-fallback",
    }
    return RouteDecision.from_dict(payload, agent_specs=SPECS)


class OrchestratorGracefulFallbackTests(unittest.TestCase):
    def _orchestrator(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.agents = {}
        orchestrator.registry = None
        orchestrator.runtime = MagicMock()
        orchestrator.memory_service = FakeMemory()
        orchestrator.memory = orchestrator.memory_service.store
        orchestrator.session_mgr = MagicMock()
        orchestrator.execution_policy = MagicMock()
        orchestrator.execution_policy.evaluate_call.return_value = {
            "action": "allow"
        }
        orchestrator.workflow_executor = MagicMock()
        orchestrator.workflow_executor.execute.side_effect = RuntimeError(
            "simulated executor failure"
        )
        orchestrator._general_reply = lambda message, session: "fallback answer"
        return orchestrator

    def test_executor_exception_falls_back_to_general_reply(self):
        orchestrator = self._orchestrator()
        result = orchestrator.process(
            "query data",
            decision_override=route_decision(),
        )
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["answer"], "fallback answer")
        self.assertIn("fallback", result["trace"][-1]["step"])


if __name__ == "__main__":
    unittest.main()
