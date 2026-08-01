import unittest

from router import Router


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "name": "NL2SQL",
        "description": "查询数据库",
        "keywords": ["数据库", "SQL", "查询"],
        "intents": ["database_query"],
        "examples": ["查询各地区销售额"],
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
        "examples": ["把数据画成柱状图"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.artifacts"]},
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


class FakeLLMRouter:
    def __call__(self, **kwargs):
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
                            "instruction": "绘制柱状图",
                            "input_refs": ["step_1"],
                        },
                    },
                    "depends_on": ["step_1"],
                    "confidence": 0.9,
                },
            ],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "llm",
            "reason": "workflow",
        }


class RouterLLMInputRefNormalizationTests(unittest.TestCase):
    def test_bare_call_id_input_refs_are_expanded_to_answer_export(self):
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=FakeLLMRouter(),
        )

        decision = router.route_decision("查询销售额然后画图")

        self.assertEqual(decision.mode, "workflow")
        self.assertEqual(
            decision.calls[1].function.arguments["input_refs"],
            ["step_1.answer"],
        )


if __name__ == "__main__":
    unittest.main()
