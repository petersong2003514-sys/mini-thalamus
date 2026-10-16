import unittest

from router import Router
from route_schema import RouteDecision
from workflow_executor import WorkflowExecutor


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "name": "NL2SQL",
        "description": "查询数据库",
        "keywords": ["数据库", "SQL", "查询"],
        "intents": ["database_query"],
        "examples": ["查询销售额"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {"side_effects": "database_read", "auto_chainable": True},
    },
    "code": {
        "id": "code",
        "name": "Code",
        "description": "编写和执行代码，生成图表",
        "keywords": ["代码", "画图", "图表"],
        "intents": ["code_execution", "data_visualization"],
        "examples": ["画柱状图"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.artifacts"]},
        "capabilities": {"side_effects": "workspace_write", "auto_chainable": True},
    },
    "report_writer": {
        "id": "report_writer",
        "name": "Report",
        "description": "生成报告",
        "keywords": ["报告", "总结"],
        "intents": ["report_generation"],
        "examples": ["写报告"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.report"]},
        "capabilities": {"side_effects": "none", "auto_chainable": True},
    },
}


class FakeRegistry:
    def list_all(self):
        return dict(AGENT_SPECS)

    def get_router_config(self):
        return dict(AGENT_SPECS)

    def on_register(self, callback, owner=None):
        return None


class EmptySemanticMatcher:
    def rank(self, query, agent_specs, top_k=3):
        return []


class FakeLLMRouter:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return {
            "mode": "workflow",
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
                },
                {
                    "call_id": "step_2",
                    "intent": "data_visualization",
                    "function": {
                        "name": "code",
                        "arguments": {
                            "instruction": "画柱状图",
                            "input_refs": ["step_1.data.structured"],
                        },
                    },
                    "depends_on": ["step_1"],
                    "confidence": 0.9,
                },
                {
                    "call_id": "step_3",
                    "intent": "report_generation",
                    "function": {
                        "name": "report_writer",
                        "arguments": {
                            "instruction": "写报告",
                            "input_refs": ["step_2.data.artifacts"],
                        },
                    },
                    "depends_on": ["step_2"],
                    "confidence": 0.9,
                },
            ],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "llm",
            "reason": "compound",
        }


class RecordingAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, query, context=None):
        self.calls.append({"query": query, "context": context or {}})
        return self.result


class FakeRuntime:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, agent_id):
        return self.agents[agent_id]


class IntegrationRouterWorkflowMemoryTests(unittest.TestCase):
    def test_compound_route_executes_with_per_step_memory_and_stage_events(self):
        llm = FakeLLMRouter()
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=llm,
        )
        agents = {
            "nl2sql": RecordingAgent({
                "status": "success",
                "answer": "sales",
                "summary": "query done",
                "data": {"structured": {"rows": [["x", 1]]}},
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

        def context_provider(route_call, workflow_results):
            return {
                "memory": {
                    "current_query": route_call.function.arguments["instruction"],
                    "agent_id": route_call.function.name,
                },
                "session_state": {"session_id": "s1"},
            }

        events = []
        route = router.route_decision("查询销售额，然后画柱状图，最后写报告")
        executor = WorkflowExecutor(FakeRuntime(agents))
        result = executor.execute(
            route,
            original_query="查询销售额，然后画柱状图，最后写报告",
            context_provider=context_provider,
            on_stage=lambda kind, payload: events.append((kind, payload)),
        )

        self.assertEqual(route.mode, "workflow")
        self.assertEqual(
            [call.function.name for call in route.calls],
            ["nl2sql", "code", "report_writer"],
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            agents["report_writer"].calls[0]["context"]["memory"]["agent_id"],
            "report_writer",
        )
        self.assertEqual(
            agents["code"].calls[0]["context"]["workflow_inputs"][
                "step_1.data.structured"
            ],
            {"rows": [["x", 1]]},
        )
        self.assertEqual([kind for kind, _ in events].count("step_start"), 3)
        self.assertEqual([kind for kind, _ in events].count("step_end"), 3)
        self.assertTrue(llm.calls == 1)


if __name__ == "__main__":
    unittest.main()
