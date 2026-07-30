import unittest

from router import Router, _is_compound_intent


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "name": "NL2SQL",
        "description": "Query databases and generate SQL.",
        "keywords": ["data", "sql", "sales"],
        "intents": ["database_query"],
        "examples": ["query sales amount"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {},
    },
    "docparse": {
        "id": "docparse",
        "name": "DocParse",
        "description": "Parse uploaded documents.",
        "keywords": ["file", "pdf", "parse"],
        "intents": ["document_parse"],
        "examples": ["parse uploaded PDF"],
        "input_schema": {"required": ["instruction"], "properties": {}},
        "output_schema": {"exports": ["answer", "summary", "data.structured"]},
        "capabilities": {},
    },
    "report_writer": {
        "id": "report_writer",
        "name": "Report",
        "description": "Write structured reports.",
        "keywords": ["report", "summary"],
        "intents": ["report_generation"],
        "examples": ["write a report"],
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


def single_payload(agent_id):
    return {
        "mode": "single",
        "calls": [{
            "call_id": "step_1",
            "intent": "database_query",
            "function": {
                "name": agent_id,
                "arguments": {
                    "instruction": "query sales amount",
                    "input_refs": [],
                },
            },
            "depends_on": [],
            "confidence": 0.8,
        }],
        "candidates": [],
        "overall_confidence": 0.8,
        "method": "llm",
        "reason": "LLM selected",
    }


class RouterStrictCascadeTests(unittest.TestCase):
    def test_strict_cascade_retries_invalid_llm_output_then_succeeds(self):
        llm_calls = []

        def llm_router(**kwargs):
            llm_calls.append(kwargs)
            if len(llm_calls) == 1:
                return single_payload("invented")
            return single_payload("nl2sql")

        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=FakeSemanticMatcher([]),
            llm_router=llm_router,
        )

        decision = router.route_decision("query sales amount", strict_cascade=True)

        self.assertEqual(decision.calls[0].function.name, "nl2sql")
        self.assertEqual(decision.method, "llm")
        self.assertEqual(decision.metadata["retry_count"], 1)
        self.assertEqual(len(decision.metadata["attempts"]), 2)
        self.assertIn("validation_errors", llm_calls[1])


    def test_strict_cascade_combines_semantic_and_lexical_candidates(self):
        llm_calls = []

        def llm_router(**kwargs):
            llm_calls.append(kwargs)
            return single_payload("report_writer")

        router = Router(
            registry=FakeRegistry(),
            semantic_matcher=FakeSemanticMatcher([
                {"agent_id": "nl2sql", "score": 0.9, "evidence": "semantic"},
            ]),
            llm_router=llm_router,
            top_k=1,
        )

        router.route_decision(
            "query sales orders then write report",
            strict_cascade=True,
        )

        candidate_ids = [item["agent_id"] for item in llm_calls[0]["candidates"]]
        self.assertIn("nl2sql", candidate_ids)
        self.assertIn("report_writer", candidate_ids)

    def test_single_file_action_with_and_is_not_compound(self):
        self.assertFalse(
            _is_compound_intent(
                "\u8bf7\u89e3\u6790\u4e0a\u4f20\u7684 PDF "
                "\u6587\u4ef6\u5e76\u63d0\u53d6\u5173\u952e\u4fe1\u606f"
            )
        )

    def test_sequential_agent_workflow_is_compound(self):
        self.assertTrue(
            _is_compound_intent(
                "\u5148\u641c\u7d22\u6700\u65b0\u8d44\u6599\uff0c"
                "\u7136\u540e\u751f\u6210\u62a5\u544a"
            )
        )


if __name__ == "__main__":
    unittest.main()
