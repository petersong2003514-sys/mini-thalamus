import unittest

from mini_thalamus.router import Router


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "name": "NL2SQL",
        "description": "查询数据库并生成 SQL",
        "keywords": ["数据库", "SQL", "查询", "销售额"],
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
        "keywords": ["代码", "绘图", "图表"],
        "intents": ["code_execution", "data_visualization"],
        "examples": ["把数据画成柱状图"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.artifacts"]},
        "capabilities": {"side_effects": "workspace_write"},
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
        self.calls = []

    def rank(self, query, agent_specs, top_k=3):
        self.calls.append({
            "query": query,
            "agent_ids": sorted(agent_specs),
            "top_k": top_k,
        })
        return list(self.matches)[:top_k]


def single_payload(agent_id="nl2sql"):
    return {
        "mode": "single",
        "calls": [
            {
                "call_id": "step_1",
                "intent": "database_query",
                "function": {
                    "name": agent_id,
                    "arguments": {
                        "instruction": "查询销售额",
                        "input_refs": [],
                    },
                },
                "depends_on": [],
                "confidence": 0.8,
            }
        ],
        "candidates": [],
        "overall_confidence": 0.8,
        "method": "llm",
        "reason": "LLM selected",
    }


def workflow_payload():
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
                "confidence": 0.94,
            },
            {
                "call_id": "step_2",
                "intent": "data_visualization",
                "function": {
                    "name": "code",
                    "arguments": {
                        "instruction": "绘制图表",
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
                        "instruction": "生成报告",
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
        "reason": "复合任务拆分",
    }


class CascadedRouterTests(unittest.TestCase):
    def test_explicit_agent_rule_skips_semantic_and_llm(self):
        semantic = FakeSemanticMatcher([])
        llm_calls = []
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=lambda **kwargs: llm_calls.append(kwargs),
        )

        decision = router.route_decision("@nl2sql 查询销售额")

        self.assertEqual(decision.calls[0].function.name, "nl2sql")
        self.assertEqual(decision.method, "rule")
        self.assertEqual(semantic.calls, [])
        self.assertEqual(llm_calls, [])

    def test_high_confidence_semantic_match_skips_llm(self):
        semantic = FakeSemanticMatcher([
            {"agent_id": "nl2sql", "score": 0.91, "evidence": "database"},
            {"agent_id": "code", "score": 0.52, "evidence": "weak"},
        ])
        llm_calls = []
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=lambda **kwargs: llm_calls.append(kwargs),
        )

        decision = router.route_decision("统计各地区销售额")

        self.assertEqual(decision.calls[0].function.name, "nl2sql")
        self.assertEqual(decision.method, "semantic")
        self.assertEqual(llm_calls, [])

    def test_ambiguous_semantic_match_sends_only_top_k_to_llm(self):
        semantic = FakeSemanticMatcher([
            {"agent_id": "nl2sql", "score": 0.72},
            {"agent_id": "code", "score": 0.69},
            {"agent_id": "report_writer", "score": 0.4},
        ])
        received = []

        def llm_router(**kwargs):
            received.append(kwargs)
            return single_payload("nl2sql")

        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=llm_router,
            top_k=2,
        )

        decision = router.route_decision("分析销售数据")

        self.assertEqual(decision.method, "llm")
        self.assertEqual(
            [item["agent_id"] for item in received[0]["candidates"]],
            ["nl2sql", "code"],
        )

    def test_compound_request_is_decomposed_even_with_strong_first_intent(self):
        semantic = FakeSemanticMatcher([
            {"agent_id": "nl2sql", "score": 0.95},
            {"agent_id": "code", "score": 0.9},
            {"agent_id": "report_writer", "score": 0.88},
        ])
        llm_calls = []

        def llm_router(**kwargs):
            llm_calls.append(kwargs)
            return workflow_payload()

        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=llm_router,
        )

        decision = router.route_decision("查询销售额，然后画图，最后写报告")

        self.assertEqual(decision.mode, "workflow")
        self.assertEqual(
            [call.function.name for call in decision.calls],
            ["nl2sql", "code", "report_writer"],
        )
        self.assertEqual(len(llm_calls), 1)

    def test_invalid_llm_agent_falls_back_to_general(self):
        semantic = FakeSemanticMatcher([])
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=lambda **kwargs: single_payload("invented"),
        )

        decision = router.route_decision("帮我处理这个不明确的问题")

        self.assertEqual(decision.mode, "fallback")
        self.assertEqual(decision.method, "fallback")
        self.assertEqual(decision.to_legacy_dict()["agent"], "general")

    def test_legacy_route_shape_contains_structured_decision(self):
        semantic = FakeSemanticMatcher([
            {"agent_id": "nl2sql", "score": 0.91},
            {"agent_id": "code", "score": 0.2},
        ])
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=semantic,
            llm_router=lambda **kwargs: None,
        )

        route = router.route("查询数据库销售额")

        self.assertEqual(route["agent"], "nl2sql")
        self.assertEqual(route["agents"], ["nl2sql"])
        self.assertIn("route_decision", route)


if __name__ == "__main__":
    unittest.main()
