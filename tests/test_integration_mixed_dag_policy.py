import unittest

from mini_thalamus.route_schema import RouteDecision
from mini_thalamus.workflow_executor import WorkflowExecutor


SPECS = {
    "code": {"output_schema": {"exports": ["answer", "summary", "data.artifacts"]}},
    "websearch": {"output_schema": {"exports": ["answer", "summary", "data.structured"]}},
    "report_writer": {"output_schema": {"exports": ["answer", "summary", "data.report"]}},
    "rag": {"output_schema": {"exports": ["answer", "summary", "data.structured"]}},
}


class RecordingAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, query, context=None):
        self.calls.append(query)
        return self.result


class FakeRuntime:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, agent_id):
        return self.agents[agent_id]


def decision(payload):
    return RouteDecision.from_dict(payload, agent_specs=SPECS)


def call(call_id, intent, agent, instruction, depends_on=None, input_refs=None, policy=None):
    value = {
        "call_id": call_id,
        "intent": intent,
        "function": {
            "name": agent,
            "arguments": {
                "instruction": instruction,
                "input_refs": input_refs or [],
            },
        },
        "depends_on": depends_on or [],
        "confidence": 0.9,
    }
    if policy is not None:
        value["policy"] = policy
    return value


class IntegrationMixedDagPolicyTests(unittest.TestCase):
    def test_parallel_branches_with_blocked_and_skipped_steps_emit_events(self):
        agents = {
            "code": RecordingAgent({
                "status": "success",
                "answer": "chart",
                "data": {"artifacts": []},
            }),
            "websearch": RecordingAgent({
                "status": "success",
                "answer": "news",
                "summary": "searched",
                "data": {"structured": {"items": ["n1"]}},
            }),
            "report_writer": RecordingAgent({
                "status": "success",
                "answer": "report",
                "summary": "report done",
                "data": {"report": "r.md"},
            }),
            "rag": RecordingAgent({
                "status": "success",
                "answer": "docs",
                "data": {"structured": {"items": ["d1"]}},
            }),
        }
        events = []
        route = decision({
            "mode": "workflow",
            "calls": [
                call(
                    "step_1",
                    "code_execution",
                    "code",
                    "run code",
                    policy={"action": "deny", "reason": "blocked by policy"},
                ),
                call("step_2", "web_search", "websearch", "search news"),
                call(
                    "step_3",
                    "report_generation",
                    "report_writer",
                    "write report",
                    depends_on=["step_2"],
                    input_refs=["step_2.data.structured"],
                ),
                call(
                    "step_4",
                    "knowledge_retrieval",
                    "rag",
                    "retrieve docs",
                    depends_on=["step_1"],
                ),
            ],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "test",
            "reason": "mixed dag",
        })

        executor = WorkflowExecutor(FakeRuntime(agents))
        result = executor.execute(
            route,
            original_query="mixed",
            on_stage=lambda kind, payload: events.append((kind, payload)),
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["step_1"]["status"], "blocked")
        self.assertEqual(result["results"]["step_2"]["status"], "success")
        self.assertEqual(result["results"]["step_3"]["status"], "success")
        self.assertEqual(result["results"]["step_4"]["status"], "skipped")
        self.assertEqual(
            [entry["status"] for entry in result["trace"]],
            ["blocked", "success", "skipped", "success"],
        )
        end_statuses = [
            payload["status"] for kind, payload in events if kind == "step_end"
        ]
        self.assertEqual(end_statuses, ["blocked", "success", "skipped", "success"])


if __name__ == "__main__":
    unittest.main()
