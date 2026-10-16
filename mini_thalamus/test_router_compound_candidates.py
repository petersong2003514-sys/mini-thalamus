import unittest

from router import Router


AGENT_SPECS = {
    "websearch": {
        "id": "websearch",
        "name": "WebSearch",
        "description": "联网搜索实时信息",
        "keywords": ["搜索", "天气", "新闻"],
        "intents": ["web_search"],
        "examples": ["查天气"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {},
    },
    "data_analyst": {
        "id": "data_analyst",
        "name": "DataAnalyst",
        "description": "分析已上传表格数据",
        "keywords": ["分析", "趋势", "表格"],
        "intents": ["data_analysis"],
        "examples": ["分析CSV"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {},
    },
    "code": {
        "id": "code",
        "name": "Code",
        "description": "编写和执行代码，生成图表",
        "keywords": ["画图", "图表", "代码"],
        "intents": ["code_execution", "data_visualization"],
        "examples": ["画趋势图"],
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


class FakeSemanticMatcher:
    def __init__(self, matches):
        self.matches = matches

    def rank(self, query, agent_specs, top_k=3):
        return list(self.matches)[:top_k]


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
                    "intent": "web_search",
                    "function": {
                        "name": "websearch",
                        "arguments": {
                            "instruction": "搜索天气数据",
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
                            "instruction": "画趋势图",
                            "input_refs": ["step_1.answer"],
                        },
                    },
                    "depends_on": ["step_1"],
                    "confidence": 0.9,
                },
            ],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "llm",
            "reason": "compound",
        }


class RouterCompoundCandidatesTests(unittest.TestCase):
    def test_compound_query_keeps_code_in_llm_candidates(self):
        llm = FakeLLMRouter()
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=FakeSemanticMatcher([
                {"agent_id": "websearch", "score": 0.9, "evidence": "semantic"},
                {"agent_id": "data_analyst", "score": 0.8, "evidence": "semantic"},
            ]),
            llm_router=llm,
            top_k=2,
        )

        decision = router.route_decision("搜索天气数据，并且画出趋势图")

        candidate_ids = [
            item["agent_id"] for item in llm.calls[0]["candidates"]
        ]
        self.assertIn("code", candidate_ids)
        self.assertEqual(decision.calls[1].function.name, "code")


if __name__ == "__main__":
    unittest.main()
