import unittest

from router import Router
from workflow_executor import WorkflowExecutor


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "name": "NL2SQL",
        "description": "查询数据库并生成 SQL",
        "keywords": ["数据库", "SQL", "查询", "销售额"],
        "intents": ["database_query", "sql_generation"],
        "examples": ["查询各地区销售额"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {"side_effects": "database_read", "auto_chainable": True},
    },
    "code": {
        "id": "code",
        "name": "Code",
        "description": "编写和执行代码，生成图表",
        "keywords": ["代码", "绘图", "图表"],
        "intents": ["code_execution", "data_visualization"],
        "examples": ["把数据画成柱状图"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.artifacts"]},
        "capabilities": {"side_effects": "workspace_write", "auto_chainable": True},
    },
    "report_writer": {
        "id": "report_writer",
        "name": "Report",
        "description": "生成结构化分析报告",
        "keywords": ["报告", "总结"],
        "intents": ["report_generation"],
        "examples": ["生成分析报告"],
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
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "mode": "workflow",
            "calls": [
                {
                    "call_id": "step_1",
                    "intent": "database_query",
                    "function": {
                        "name": "nl2sql",
                        "arguments": {
                            "instruction": "查询各地区销售额",
                            "input_refs": [],
                        },
                    },
                    "depends_on": [],
                    "confidence": 0.94,
                },
                {
                    "call_id": "step_2",
                    "intent": "data_visualization",
                    "function": {
                        "name": "code",
                        "arguments": {
                            "instruction": "将销售额绘制为柱状图",
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
                            "instruction": "结合数据和图表生成分析报告",
                            "input_refs": [
                                "step_1.data.structured",
                                "step_2.data.artifacts",
                            ],
                        },
                    },
                    "depends_on": ["step_1", "step_2"],
                    "confidence": 0.9,
                },
            ],
            "candidates": [],
            "overall_confidence": 0.91,
            "method": "llm",
            "reason": "复合任务拆分为查询、可视化、报告三步",
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


class RouterDAGIntegrationTests(unittest.TestCase):
    def test_compound_request_is_routed_to_atomic_intents_and_executed(self):
        llm = FakeLLMRouter()
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=llm,
        )
        nl2sql = RecordingAgent({
            "status": "success",
            "answer": "sales",
            "summary": "query done",
            "data": {"structured": {"rows": [["华东", 100]]}},
        })
        code = RecordingAgent({
            "status": "success",
            "answer": "chart",
            "summary": "chart done",
            "data": {"artifacts": ["chart.png"]},
        })
        report = RecordingAgent({
            "status": "success",
            "answer": "report",
            "summary": "report done",
            "data": {"report": "final.md"},
        })

        route = router.route_decision("查询销售额，然后画图，最后写报告")

        self.assertEqual(route.mode, "workflow")
        self.assertEqual(
            [call.intent for call in route.calls],
            ["database_query", "data_visualization", "report_generation"],
        )
        self.assertEqual(
            [call.function.name for call in route.calls],
            ["nl2sql", "code", "report_writer"],
        )
        self.assertEqual(
            [call.function.arguments["instruction"] for call in route.calls],
            [
                "查询各地区销售额",
                "将销售额绘制为柱状图",
                "结合数据和图表生成分析报告",
            ],
        )
        self.assertEqual(route.reason, "复合任务拆分为查询、可视化、报告三步")

        executor = WorkflowExecutor(
            FakeRuntime({"nl2sql": nl2sql, "code": code, "report_writer": report})
        )
        result = executor.execute(
            route,
            original_query="查询销售额，然后画图，最后写报告",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            report.calls[0]["context"]["workflow_inputs"][
                "step_2.data.artifacts"
            ],
            ["chart.png"],
        )
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
