import unittest

from replanner import Replanner
from route_schema import RouteDecision


SPECS = {
    "nl2sql": {
        "output_schema": {"exports": ["answer"]},
        "intents": ["database_query"],
    },
    "report_writer": {
        "output_schema": {"exports": ["answer", "data.report"]},
        "intents": ["report_generation"],
    },
}


def make_decision():
    payload = {
        "mode": "workflow",
        "calls": [
            {
                "call_id": "step_1",
                "intent": "database_query",
                "function": {
                    "name": "nl2sql",
                    "arguments": {"instruction": "查询", "input_refs": []},
                },
                "depends_on": [],
                "confidence": 1.0,
            },
            {
                "call_id": "step_2",
                "intent": "report_generation",
                "function": {
                    "name": "report_writer",
                    "arguments": {
                        "instruction": "写报告",
                        "input_refs": ["step_1.answer"],
                    },
                },
                "depends_on": ["step_1"],
                "confidence": 1.0,
            },
        ],
        "candidates": [],
        "overall_confidence": 1.0,
        "method": "test",
        "reason": "t",
    }
    return RouteDecision.from_dict(payload, agent_specs=SPECS)


class ReplannerTests(unittest.TestCase):
    def test_namespace_renames_call_ids_deps_and_refs(self):
        decision = make_decision()
        results = {"step_1": {"status": "error", "answer": "db down"}}
        Replanner()._namespace(decision, results, 0)
        ids = [call.call_id for call in decision.calls]
        self.assertEqual(ids, ["rp0_step_1", "rp0_step_2"])
        self.assertEqual(decision.calls[1].depends_on, ["rp0_step_1"])
        self.assertEqual(
            decision.calls[1].function.arguments["input_refs"],
            ["rp0_step_1.answer"],
        )
        # 不与已完成结果冲突
        self.assertNotIn("step_1", ids)

    def test_namespace_second_attempt_uses_new_prefix(self):
        decision = make_decision()
        Replanner()._namespace(decision, {}, 1)
        self.assertEqual(
            [call.call_id for call in decision.calls],
            ["rp1_step_1", "rp1_step_2"],
        )

    def test_build_context_includes_completed_and_failures(self):
        results = {
            "step_1": {
                "status": "success",
                "answer": "42",
                "metadata": {"agent_id": "nl2sql"},
            }
        }
        failures = {
            "step_2": {
                "status": "error",
                "error": "db down",
                "suggested_alternative": "rag",
            }
        }
        ctx = Replanner()._build_context(results, failures)
        self.assertIn("step_1", ctx)
        self.assertIn("42", ctx)
        self.assertIn("db down", ctx)
        self.assertIn("rag", ctx)

    def test_replan_without_router_returns_none(self):
        self.assertIsNone(
            Replanner().replan(
                original_query="q", results={}, failures={}
            )
        )


if __name__ == "__main__":
    unittest.main()
