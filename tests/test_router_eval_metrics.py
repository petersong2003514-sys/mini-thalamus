import unittest

from mini_thalamus.router_eval_dataset import ExpectedCall, RouterEvalCase
from mini_thalamus.router_eval_metrics import (
    aggregate_scores,
    evaluate_quality_gates,
    score_case,
)


class RouterEvalMetricTests(unittest.TestCase):
    def test_single_prediction_rejects_workflow_with_correct_first_agent(self):
        case = self.single_case
        score = score_case(case, {
            "mode": "workflow",
            "calls": [
                {"call_id": "step_1", "agent_id": "data_analyst", "depends_on": [], "input_refs": []},
                {"call_id": "step_2", "agent_id": "code", "depends_on": [], "input_refs": []},
            ],
            "method": "llm", "schema_valid": True,
        })
        self.assertFalse(score["single_correct"])

    def test_call_free_fallback_is_scored_by_mode_and_method(self):
        case = RouterEvalCase(
            case_id="fallback-1",
            query="你好",
            session_context="",
            file_context={},
            expected_mode="fallback",
            expected_calls=(),
            acceptable_agents=(),
            expected_stage="fallback",
            difficulty="easy",
            tags=("fallback",),
        )
        score = score_case(case, {
            "mode": "fallback", "calls": [], "candidates": [],
            "method": "fallback", "schema_valid": True,
        })
        metrics = aggregate_scores([score])
        self.assertTrue(score["fallback_correct"])
        self.assertEqual(metrics["fallback_accuracy"], 1.0)

    def setUp(self):
        self.single_case = RouterEvalCase(
            case_id="single",
            query="make a chart",
            session_context="",
            file_context={},
            expected_mode="single",
            expected_calls=(ExpectedCall("data_analyst"),),
            acceptable_agents=("code",),
            expected_stage="semantic",
            difficulty="medium",
            tags=("confusable", "code_vs_data_analyst"),
        )
        self.workflow_case = RouterEvalCase(
            case_id="workflow",
            query="query, chart and report",
            session_context="",
            file_context={},
            expected_mode="workflow",
            expected_calls=(
                ExpectedCall("nl2sql"),
                ExpectedCall(
                    "code",
                    depends_on=("step_1",),
                    input_refs=("step_1.data.structured",),
                ),
                ExpectedCall(
                    "report_writer",
                    depends_on=("step_2",),
                    input_refs=("step_2.data.structured",),
                ),
            ),
            acceptable_agents=(),
            expected_stage="llm",
            difficulty="hard",
            tags=("multi_intent",),
        )

    @staticmethod
    def _single_prediction(agent_id, method="semantic", candidates=None):
        return {
            "mode": "single",
            "calls": [
                {
                    "call_id": "step_1",
                    "agent_id": agent_id,
                    "depends_on": [],
                    "input_refs": [],
                }
            ],
            "method": method,
            "candidates": candidates or [agent_id],
            "schema_valid": True,
            "latency_ms": 1.0,
        }

    def test_acceptable_agent_counts_as_correct(self):
        score = score_case(
            self.single_case,
            self._single_prediction("code"),
        )
        self.assertTrue(score["single_correct"])
        self.assertTrue(score["mode_correct"])

    def test_workflow_metrics_separate_steps_order_edges_and_refs(self):
        score = score_case(self.workflow_case, {
            "mode": "workflow",
            "calls": [
                {
                    "call_id": "step_1",
                    "agent_id": "nl2sql",
                    "depends_on": [],
                    "input_refs": [],
                },
                {
                    "call_id": "step_2",
                    "agent_id": "report_writer",
                    "depends_on": ["step_1"],
                    "input_refs": [],
                },
            ],
            "method": "llm",
            "candidates": [],
            "schema_valid": True,
            "latency_ms": 10.0,
        })

        self.assertAlmostEqual(score["step_recall"], 2 / 3)
        self.assertFalse(score["sequence_exact"])
        self.assertFalse(score["workflow_exact"])
        self.assertEqual(score["edge_recall"], 0.5)
        self.assertFalse(score["input_refs_exact"])

    def test_workflow_graph_scoring_is_independent_of_call_id_names(self):
        prediction = {
            "mode": "workflow",
            "method": "llm",
            "schema_valid": True,
            "calls": [
                {"call_id": "fetch", "agent_id": "nl2sql", "depends_on": [], "input_refs": []},
                {"call_id": "chart", "agent_id": "code", "depends_on": ["fetch"], "input_refs": ["fetch.data.structured"]},
                {"call_id": "write", "agent_id": "report_writer", "depends_on": ["chart"], "input_refs": ["chart.data.structured"]},
            ],
        }
        score = score_case(self.workflow_case, prediction)
        self.assertEqual(score["edge_recall"], 1.0)
        self.assertTrue(score["input_refs_exact"])
        self.assertTrue(score["workflow_exact"])

    def test_aggregate_scores_known_rates_and_slices(self):
        correct = score_case(
            self.single_case,
            self._single_prediction(
                "data_analyst",
                candidates=["data_analyst", "code"],
            ),
        )
        wrong_case = RouterEvalCase(
            **{
                **self.single_case.__dict__,
                "case_id": "wrong",
                "expected_calls": (ExpectedCall("nl2sql"),),
                "acceptable_agents": (),
                "expected_stage": "rule",
                "difficulty": "easy",
                "tags": ("ordinary_single",),
            }
        )
        wrong = score_case(
            wrong_case,
            self._single_prediction("code", method="rule"),
        )

        metrics = aggregate_scores([correct, wrong])

        self.assertEqual(metrics["case_count"], 2)
        self.assertEqual(metrics["single_intent_accuracy"], 0.5)
        self.assertEqual(metrics["semantic_top3_recall"], 1.0)
        self.assertIn("medium", metrics["slices"]["difficulty"])
        self.assertEqual(metrics["method_distribution"], {"semantic": 1, "rule": 1})

    def test_unavailable_metrics_skip_quality_gates(self):
        gates = evaluate_quality_gates({
            "rule_accuracy": None,
            "false_rule_capture_rate": 0.0,
            "semantic_top3_recall": None,
            "single_intent_accuracy": 0.9,
            "multi_step_recall": None,
            "schema_valid_rate": 1.0,
            "fallback_accuracy": 1.0,
        })

        self.assertEqual(gates["rule_accuracy"]["status"], "skipped")
        self.assertEqual(gates["single_intent_accuracy"]["status"], "passed")
        self.assertTrue(gates["all_available_passed"])


if __name__ == "__main__":
    unittest.main()
