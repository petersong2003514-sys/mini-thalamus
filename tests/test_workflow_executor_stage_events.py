import unittest

from mini_thalamus.route_schema import RouteDecision
from mini_thalamus.workflow_executor import WorkflowExecutor


SPECS = {
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


def call(call_id, intent, agent, instruction, depends_on=None, input_refs=None):
    return {
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


def route_payload(calls):
    return {
        "mode": "single" if len(calls) == 1 else "workflow",
        "calls": calls,
        "candidates": [],
        "overall_confidence": 0.9,
        "method": "test",
        "reason": "test workflow",
        "trace_id": "trace-test",
    }


class WorkflowExecutorStageEventsTests(unittest.TestCase):
    def test_execute_emits_step_start_and_step_end_stage_events(self):
        agents = {
            "nl2sql": RecordingAgent({
                "status": "success",
                "answer": "data",
                "summary": "query done",
                "data": {"structured": {"rows": [["x"]]}},
            }),
            "code": RecordingAgent({
                "status": "success",
                "answer": "chart",
                "summary": "chart done",
                "data": {"artifacts": ["chart.png"]},
            }),
            "report_writer": RecordingAgent({
                "status": "success",
                "answer": "report",
                "summary": "report done",
                "data": {"report": "final.md"},
            }),
        }
        events = []
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query"),
            call(
                "step_2",
                "data_visualization",
                "code",
                "draw chart",
                depends_on=["step_1"],
                input_refs=["step_1.data.structured"],
            ),
            call(
                "step_3",
                "report_generation",
                "report_writer",
                "write report",
                depends_on=["step_2"],
                input_refs=["step_2.data.artifacts"],
            ),
        ]))

        executor = WorkflowExecutor(FakeRuntime(agents))
        result = executor.execute(
            route,
            original_query="compound",
            on_stage=lambda kind, payload: events.append((kind, payload)),
        )

        self.assertEqual(result["status"], "success")
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds, [
            "step_start",
            "step_end",
            "step_start",
            "step_end",
            "step_start",
            "step_end",
        ])
        self.assertEqual(events[0][1]["agent_id"], "nl2sql")
        self.assertEqual(events[1][1]["agent_id"], "nl2sql")
        self.assertEqual(events[2][1]["agent_id"], "code")
        self.assertEqual(events[3][1]["agent_id"], "code")
        self.assertEqual(events[4][1]["agent_id"], "report_writer")
        self.assertEqual(events[5][1]["agent_id"], "report_writer")
        self.assertEqual(
            [payload["status"] for kind, payload in events if kind == "step_end"],
            ["success", "success", "success"],
        )


if __name__ == "__main__":
    unittest.main()
