import json
import os
import tempfile
import unittest

from router_eval import (
    UnavailableSemanticMatcher,
    _rule_predictor,
    evaluate_cases,
    normalize_decision,
    write_report,
)
from router_eval_dataset import ExpectedCall, RouterEvalCase
from router import Router


def sample_case():
    return RouterEvalCase(
        case_id="r001",
        query="查询订单",
        session_context="",
        file_context={},
        expected_mode="single",
        expected_calls=(ExpectedCall("nl2sql"),),
        acceptable_agents=(),
        expected_stage="semantic",
        difficulty="easy",
        tags=("ordinary_single",),
    )


class FakeDecision:
    def to_dict(self):
        return {
            "mode": "single",
            "calls": [{
                "call_id": "step_1",
                "intent": "query_data",
                "function": {
                    "name": "nl2sql",
                    "arguments": {"instruction": "查询订单", "input_refs": []},
                },
                "depends_on": [],
                "confidence": 0.9,
            }],
            "candidates": [{"agent_id": "nl2sql", "score": 0.9}],
            "overall_confidence": 0.9,
            "method": "semantic",
            "reason": "match",
        }


class RouterEvalRunnerTests(unittest.TestCase):
    def test_rule_layer_respects_compound_intent_gate(self):
        router = Router(
            dynamic_agents={"nl2sql": {"intents": ["query_data"]}},
            semantic_matcher=UnavailableSemanticMatcher("disabled"),
            llm_router=lambda **_: {},
        )
        case = sample_case()
        case = RouterEvalCase(
            **{**case.__dict__, "query": "@nl2sql query -> analyze"}
        )
        self.assertEqual(_rule_predictor(router)(case)["method"], "no_rule")

    def test_unavailable_matcher_never_loads_native_model(self):
        matcher = UnavailableSemanticMatcher("native_runtime_crash")
        self.assertEqual(matcher.rank("query", {"nl2sql": {}}, top_k=3), [])
        self.assertEqual(matcher.last_error, "native_runtime_crash")

    def test_normalize_decision_preserves_contract_fields(self):
        prediction = normalize_decision(FakeDecision(), latency_ms=12.5)
        self.assertEqual(prediction["calls"][0]["agent_id"], "nl2sql")
        self.assertEqual(prediction["calls"][0]["input_refs"], [])
        self.assertTrue(prediction["schema_valid"])
        self.assertEqual(prediction["latency_ms"], 12.5)

    def test_evaluate_cases_isolates_case_errors(self):
        def predict(case):
            raise RuntimeError("secret token must not leak")

        report = evaluate_cases(
            [sample_case()],
            layer="semantic",
            predictor=predict,
            dataset_hash="abc",
        )
        self.assertEqual(report["metrics"]["error_count"], 1)
        self.assertEqual(report["predictions"][0]["error"], "RuntimeError")
        self.assertNotIn("secret token", json.dumps(report))

    def test_write_report_outputs_json_and_markdown(self):
        report = evaluate_cases(
            [sample_case()],
            layer="semantic",
            predictor=lambda case: normalize_decision(FakeDecision(), 1.0),
            dataset_hash="abc",
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_report(report, directory)
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(markdown_path))
            with open(markdown_path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("Router Evaluation", text)
            self.assertIn("single_intent_accuracy", text)


if __name__ == "__main__":
    unittest.main()
