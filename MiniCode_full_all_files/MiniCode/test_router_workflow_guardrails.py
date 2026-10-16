import unittest

from router import Router


AGENT_SPECS = {
    "websearch": {
        "id": "websearch",
        "name": "WebSearch Agent",
        "description": "联网搜索实时信息",
        "keywords": ["搜索", "最新", "新闻"],
        "intents": ["web_search"],
        "examples": ["搜索北京天气"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary"]},
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


class InvalidWorkflowLLMRouter:
    def __init__(self, call_name, instruction):
        self.call_name = call_name
        self.instruction = instruction

    def __call__(self, **kwargs):
        return {
            "mode": "workflow",
            "calls": [
                {
                    "call_id": "step_1",
                    "intent": "web_search",
                    "function": {
                        "name": self.call_name,
                        "arguments": {
                            "instruction": self.instruction,
                            "input_refs": [],
                        },
                    },
                    "depends_on": [],
                    "confidence": 1.0,
                }
            ],
            "candidates": ["websearch"],
            "overall_confidence": 1.0,
            "method": "llm",
            "reason": "guardrail test",
        }


class RouterWorkflowGuardrailTests(unittest.TestCase):
    def _router(self, call_name, instruction):
        return Router(
            registry=FakeRegistry(),
            semantic_matcher=EmptySemanticMatcher(),
            llm_router=InvalidWorkflowLLMRouter(call_name, instruction),
        )

    def test_workflow_rejects_fallback_agent(self):
        router = self._router("general", "answer generically")
        result = router._route_with_llm_retries(
            query="test",
            context="",
            file_context={},
            candidates=[],
            specs=router._agent_specs(),
            trace_id="trace-1",
            max_attempts=2,
        )
        self.assertIsNone(result)

    def test_workflow_rejects_placeholder_instruction(self):
        router = self._router(
            "websearch",
            "How many games did the NFL team from [birthplace] win?",
        )
        result = router._route_with_llm_retries(
            query="test",
            context="",
            file_context={},
            candidates=[],
            specs=router._agent_specs(),
            trace_id="trace-2",
            max_attempts=2,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
