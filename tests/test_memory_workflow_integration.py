import os
import tempfile
import unittest

from mini_thalamus.memory_service import MemoryService
from mini_thalamus.orchestrator import MultiStepOrchestrator, Orchestrator
from mini_thalamus.route_schema import RouteDecision


SPECS = {
    "websearch": {
        "capabilities": {
            "needs_history": False,
            "needs_long_term": False,
            "needs_files": False,
            "max_context_chars": 300,
        },
        "output_schema": {"exports": ["answer", "summary"]},
    },
    "report_writer": {
        "capabilities": {
            "needs_history": True,
            "needs_long_term": True,
            "needs_files": False,
            "max_context_chars": 1000,
        },
        "output_schema": {"exports": ["answer", "summary"]},
    },
}


class RecordingAgent:
    def __init__(self, answer):
        self.answer = answer
        self.contexts = []

    def run(self, query, context=None):
        self.contexts.append(context or {})
        return {
            "status": "success",
            "answer": self.answer,
            "summary": self.answer,
            "data": {},
        }


class FakeRuntime:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, agent_id):
        return self.agents[agent_id]


class FakeRegistry:
    def get(self, agent_id):
        spec = SPECS.get(agent_id)
        return {"meta": spec} if spec else None

    def get_capabilities(self, agent_id):
        return dict(SPECS[agent_id]["capabilities"])

    def list_all(self):
        return {key: dict(value) for key, value in SPECS.items()}


class FakeRouter:
    def route_decision(self, message, session_context="", file_context=None):
        if message == "第一轮销售报告":
            calls = [
                {
                    "call_id": "step_1",
                    "intent": "report_generation",
                    "function": {
                        "name": "report_writer",
                        "arguments": {
                            "instruction": "记录第一轮销售报告",
                            "input_refs": [],
                        },
                    },
                    "depends_on": [],
                    "confidence": 1.0,
                }
            ]
        else:
            calls = [
                {
                    "call_id": "step_1",
                    "intent": "web_search",
                    "function": {
                        "name": "websearch",
                        "arguments": {
                            "instruction": "搜索最新资料",
                            "input_refs": [],
                        },
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
                            "instruction": "继续销售报告",
                            "input_refs": [],
                        },
                    },
                    "depends_on": ["step_1"],
                    "confidence": 1.0,
                },
            ]
        return RouteDecision.from_dict(
            {
                "mode": "single" if len(calls) == 1 else "workflow",
                "calls": calls,
                "candidates": [],
                "overall_confidence": 1.0,
                "method": "test",
                "reason": "test",
                "trace_id": "trace-memory",
            },
            agent_specs=SPECS,
        )


class MemoryWorkflowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "memory.db")
        self.registry = FakeRegistry()
        self.websearch = RecordingAgent("搜索完成")
        self.report = RecordingAgent("报告完成")
        self.runtime = FakeRuntime(
            {
                "websearch": self.websearch,
                "report_writer": self.report,
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _orchestrator(self, service):
        return Orchestrator(
            runtime=self.runtime,
            registry=self.registry,
            router=FakeRouter(),
            execution_policy=lambda call, spec: {
                "action": "allow",
                "risk_level": "none",
                "requires_approval": False,
                "sandboxed": False,
                "reason": "test",
            },
            memory_service=service,
        )

    def test_cross_instance_session_and_per_step_memory_policy(self):
        first_service = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )
        first_result = self._orchestrator(first_service).process(
            "第一轮销售报告",
            "alice",
            "stable-session",
        )
        source = first_service.store.list_turns("stable-session")[0]
        first_service.promote_memory(
            user_id="alice",
            source_turn_id=source.turn_id,
            namespace="preference",
            memory_key="language",
            gist="中文",
            source_agent="report_writer",
            activate=True,
        )

        second_service = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )
        second_result = self._orchestrator(second_service).process(
            "搜索后继续销售报告",
            "alice",
            "stable-session",
        )

        self.assertEqual(first_result["session_id"], "stable-session")
        self.assertEqual(second_result["session_id"], "stable-session")
        web_memory = self.websearch.contexts[-1]["memory"]
        report_memory = self.report.contexts[-1]["memory"]
        self.assertEqual(web_memory["relevant_history"], [])
        self.assertEqual(web_memory["long_term"], {})
        self.assertIn("第一轮销售报告", report_memory["conversation_history"])
        self.assertEqual(report_memory["long_term"]["language"], "中文")


    def test_legacy_process_chain_uses_durable_memory_lifecycle(self):
        service = MemoryService.for_path(
            self.db_path,
            data_root=self.temp_dir.name,
        )
        orchestrator = MultiStepOrchestrator(
            runtime=self.runtime,
            registry=self.registry,
            router=FakeRouter(),
            execution_policy=lambda route_call, spec: {
                "action": "allow",
                "risk_level": "none",
                "requires_approval": False,
                "sandboxed": False,
                "reason": "test",
            },
            memory_service=service,
        )

        result = orchestrator.process_chain(
            "Write a durable report",
            ["report_writer"],
            user_id="alice",
        )

        turns = service.store.list_turns(result["session_id"])
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].status, "success")
        self.assertIn("memory", self.report.contexts[-1])


if __name__ == "__main__":
    unittest.main()
