import unittest

from router import Router


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
            "mode": "single",
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
                    "confidence": 0.95,
                }
            ],
            "candidates": ["nl2sql"],
            "overall_confidence": 0.95,
            "method": "llm",
            "reason": "single database query",
        }


class RouterLLMCandidateNormalizationTests(unittest.TestCase):
    def test_string_candidates_from_llm_do_not_discard_valid_route(self):
        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=FakeLLMRouter(),
        )

        decision = router.route_decision("查询各地区销售额")

        self.assertEqual(decision.method, "llm")
        self.assertEqual(decision.calls[0].function.name, "nl2sql")
        self.assertEqual(
            [candidate.agent_id for candidate in decision.candidates],
            ["nl2sql"],
        )


if __name__ == "__main__":
    unittest.main()
