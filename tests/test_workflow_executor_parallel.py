import threading
import time
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
        self.calls.append({"query": query, "context": context or {}})
        return self.result


class ConcurrencyProbe:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return {"status": "success", "answer": "done", "data": {}}


class CoordinatedAgent:
    def __init__(self, probe, result=None):
        self.probe = probe
        self.result = result or {
            "status": "success",
            "answer": "done",
            "data": {},
        }
        self.calls = []

    def run(self, query, context=None):
        self.calls.append({"query": query, "context": context or {}})
        self.probe.enter()
        return self.result


class FakeRuntime:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, agent_id):
        return self.agents[agent_id]


def decision(payload):
    return RouteDecision.from_dict(payload, agent_specs=SPECS)


def call(call_id, intent, agent, instruction, depends_on=None, input_refs=None,
         policy=None):
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


class WorkflowExecutorParallelTests(unittest.TestCase):
    def test_independent_calls_run_concurrently(self):
        probe = ConcurrencyProbe()
        nl2sql = CoordinatedAgent(probe)
        code = CoordinatedAgent(probe)
        executor = WorkflowExecutor(
            runtime=FakeRuntime({"nl2sql": nl2sql, "code": code})
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query a"),
            call("step_2", "data_visualization", "code", "query b"),
        ]))

        result = executor.execute(route, original_query="parallel")

        self.assertEqual(result["status"], "success")
        self.assertEqual(probe.max_active, 2)
        self.assertEqual(
            [entry["call_id"] for entry in result["trace"]],
            ["step_1", "step_2"],
        )

    def test_parallel_producers_feed_dependent_consumer(self):
        probe = ConcurrencyProbe()
        nl2sql = CoordinatedAgent(
            probe,
            {
                "status": "success",
                "answer": "data",
                "summary": "query done",
                "data": {"structured": {"source": "db"}},
            },
        )
        code = CoordinatedAgent(
            probe,
            {
                "status": "success",
                "answer": "chart",
                "summary": "chart done",
                "data": {"artifacts": ["chart.png"]},
            },
        )
        report = RecordingAgent({
            "status": "success",
            "answer": "report",
            "summary": "report done",
            "data": {"report": "final.md"},
        })
        executor = WorkflowExecutor(
            runtime=FakeRuntime({
                "nl2sql": nl2sql,
                "code": code,
                "report_writer": report,
            })
        )
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query db"),
            call("step_2", "data_visualization", "code", "draw chart"),
            call(
                "step_3",
                "report_generation",
                "report_writer",
                "write report",
                depends_on=["step_1", "step_2"],
                input_refs=[
                    "step_1.data.structured",
                    "step_2.data.artifacts",
                ],
            ),
        ]))

        result = executor.execute(route, original_query="parallel pipeline")

        self.assertEqual(result["status"], "success")
        self.assertEqual(probe.max_active, 2)
        self.assertEqual(
            report.calls[0]["context"]["workflow_inputs"],
            {
                "step_1.data.structured": {"source": "db"},
                "step_2.data.artifacts": ["chart.png"],
            },
        )
        self.assertEqual(
            [entry["call_id"] for entry in result["trace"]],
            ["step_1", "step_2", "step_3"],
        )

    def test_same_agent_independent_calls_stay_serialized(self):
        probe = ConcurrencyProbe()
        nl2sql = CoordinatedAgent(probe)
        executor = WorkflowExecutor(FakeRuntime({"nl2sql": nl2sql}))
        route = decision(route_payload([
            call("step_1", "database_query", "nl2sql", "query 1"),
            call("step_2", "sql_generation", "nl2sql", "query 2"),
        ]))

        result = executor.execute(route, original_query="same agent")

        self.assertEqual(result["status"], "success")
        self.assertEqual(probe.max_active, 1)
        self.assertEqual(len(nl2sql.calls), 2)


if __name__ == "__main__":
    unittest.main()
