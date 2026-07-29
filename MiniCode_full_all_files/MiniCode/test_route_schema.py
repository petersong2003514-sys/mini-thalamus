import unittest

from base_agent import AgentResult
from route_schema import RouteDecision, RouteValidationError


AGENT_SPECS = {
    "nl2sql": {
        "output_schema": {
            "exports": ["answer", "summary", "data.structured"],
        },
    },
    "code": {
        "output_schema": {
            "exports": ["answer", "summary", "data.artifacts"],
        },
    },
    "report_writer": {
        "output_schema": {
            "exports": ["answer", "summary", "data.report"],
        },
    },
}


def single_call_payload():
    return {
        "mode": "single",
        "calls": [
            {
                "call_id": "step_1",
                "intent": "database_query",
                "function": {
                    "name": "nl2sql",
                    "arguments": {
                        "instruction": "查询销售额",
                        "input_refs": [],
                    },
                },
                "depends_on": [],
                "confidence": 0.95,
            }
        ],
        "candidates": [
            {"agent_id": "nl2sql", "score": 0.95},
        ],
        "overall_confidence": 0.95,
        "method": "semantic",
        "reason": "数据库查询意图明确",
    }


class AgentResultContractTests(unittest.TestCase):
    def test_metadata_round_trip(self):
        result = AgentResult.from_dict({
            "status": "success",
            "answer": "ok",
            "data": {"rows": 1},
            "summary": "done",
            "metadata": {"agent_id": "nl2sql", "trace_id": "trace-1"},
        })

        self.assertEqual(result.metadata["agent_id"], "nl2sql")
        self.assertEqual(result.to_dict()["metadata"]["trace_id"], "trace-1")


class RouteDecisionContractTests(unittest.TestCase):
    def test_valid_single_call(self):
        decision = RouteDecision.from_dict(
            single_call_payload(),
            agent_specs=AGENT_SPECS,
        )

        self.assertEqual(decision.calls[0].function.name, "nl2sql")
        self.assertEqual(decision.to_legacy_dict()["agent"], "nl2sql")

    def test_unknown_agent_is_rejected(self):
        payload = single_call_payload()
        payload["calls"][0]["function"]["name"] = "invented_agent"

        with self.assertRaisesRegex(RouteValidationError, "not registered"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_additional_properties_are_rejected(self):
        payload = single_call_payload()
        payload["calls"][0]["unexpected"] = True

        with self.assertRaisesRegex(RouteValidationError, "unexpected fields"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_duplicate_call_ids_are_rejected(self):
        payload = single_call_payload()
        second = {
            "call_id": "step_1",
            "intent": "data_visualization",
            "function": {
                "name": "code",
                "arguments": {
                    "instruction": "绘图",
                    "input_refs": ["step_1.data.structured"],
                },
            },
            "depends_on": ["step_1"],
            "confidence": 0.9,
        }
        payload["mode"] = "workflow"
        payload["calls"].append(second)

        with self.assertRaisesRegex(RouteValidationError, "duplicate call_id"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_missing_dependency_is_rejected(self):
        payload = single_call_payload()
        payload["calls"][0]["depends_on"] = ["missing"]

        with self.assertRaisesRegex(RouteValidationError, "unknown dependency"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_cycle_is_rejected(self):
        payload = single_call_payload()
        payload["mode"] = "workflow"
        payload["calls"][0]["depends_on"] = ["step_2"]
        payload["calls"].append({
            "call_id": "step_2",
            "intent": "data_visualization",
            "function": {
                "name": "code",
                "arguments": {
                    "instruction": "绘图",
                    "input_refs": ["step_1.data.structured"],
                },
            },
            "depends_on": ["step_1"],
            "confidence": 0.9,
        })

        with self.assertRaisesRegex(RouteValidationError, "cycle"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_more_than_six_calls_are_rejected(self):
        payload = single_call_payload()
        payload["mode"] = "workflow"
        payload["calls"] = [
            {
                "call_id": f"step_{index}",
                "intent": "database_query",
                "function": {
                    "name": "nl2sql",
                    "arguments": {
                        "instruction": f"查询 {index}",
                        "input_refs": [],
                    },
                },
                "depends_on": [],
                "confidence": 0.9,
            }
            for index in range(1, 8)
        ]

        with self.assertRaisesRegex(RouteValidationError, "at most 6"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_input_ref_requires_declared_dependency(self):
        payload = single_call_payload()
        payload["mode"] = "workflow"
        payload["calls"].append({
            "call_id": "step_2",
            "intent": "data_visualization",
            "function": {
                "name": "code",
                "arguments": {
                    "instruction": "绘图",
                    "input_refs": ["step_1.data.structured"],
                },
            },
            "depends_on": [],
            "confidence": 0.9,
        })

        with self.assertRaisesRegex(RouteValidationError, "must be a dependency"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_input_ref_must_be_exported(self):
        payload = single_call_payload()
        payload["mode"] = "workflow"
        payload["calls"].append({
            "call_id": "step_2",
            "intent": "data_visualization",
            "function": {
                "name": "code",
                "arguments": {
                    "instruction": "绘图",
                    "input_refs": ["step_1.data.secret"],
                },
            },
            "depends_on": ["step_1"],
            "confidence": 0.9,
        })

        with self.assertRaisesRegex(RouteValidationError, "not exported"):
            RouteDecision.from_dict(payload, agent_specs=AGENT_SPECS)

    def test_platform_policy_overrides_model_policy(self):
        payload = single_call_payload()
        payload["calls"][0]["policy"] = {
            "risk_level": "none",
            "requires_approval": False,
        }
        decision = RouteDecision.from_dict(
            payload,
            agent_specs=AGENT_SPECS,
        )

        decision.apply_policy(lambda call: {
            "risk_level": "database_read",
            "requires_approval": True,
            "reason": "platform policy",
        })

        self.assertEqual(
            decision.calls[0].policy,
            {
                "risk_level": "database_read",
                "requires_approval": True,
                "reason": "platform policy",
            },
        )


if __name__ == "__main__":
    unittest.main()
