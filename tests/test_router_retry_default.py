import unittest

from mini_thalamus.router import Router


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
        "capabilities": {},
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
        "capabilities": {},
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
        "capabilities": {},
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


class FlakyLLMRouter:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
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
                                "input_refs": ["step_1.answer"],
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
                                "input_refs": ["step_1.answer"],
                            },
                        },
                        "depends_on": ["step_2"],
                        "confidence": 0.9,
                    },
                ],
                "candidates": [],
                "overall_confidence": 0.9,
                "method": "llm",
                "reason": "workflow",
            }
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
                            "input_refs": ["step_1.answer"],
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
                            "input_refs": [
                                "step_1.answer",
                                "step_2.data.artifacts",
                            ],
                        },
                    },
                    "depends_on": ["step_1", "step_2"],
                    "confidence": 0.9,
                },
            ],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "llm",
            "reason": "workflow",
        }


class RouterDefaultRetryTests(unittest.TestCase):
    def test_non_strict_route_retries_invalid_dependency_output(self):
        llm = FlakyLLMRouter()
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=llm,
        )

        decision = router.route_decision("查询销售额然后画图最后写报告")

        self.assertEqual(decision.mode, "workflow")
        self.assertEqual(decision.method, "llm")
        self.assertEqual(
            [call.function.name for call in decision.calls],
            ["nl2sql", "code", "report_writer"],
        )
        self.assertEqual(llm.calls, 2)


if __name__ == "__main__":
    unittest.main()
