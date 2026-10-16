import unittest

from dag_validator import DAGRepairError, DAGValidationError, DAGValidator
from workflow_planner import WorkflowPlan, WorkflowPlanStep


SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "intents": ["database_query", "data_retrieval"],
        "exports": ["query_result", "sql"],
        "output_schema": {"exports": ["answer", "summary"]},
        "input_schema": {"required": ["instruction"], "properties": {}},
    },
    "data_analyst": {
        "id": "data_analyst",
        "intents": ["data_analysis", "trend_analysis"],
        "exports": ["summary", "data.structured"],
        "output_schema": {"exports": ["answer"]},
        "input_schema": {"required": ["instruction"], "properties": {}},
    },
    "code": {
        "id": "code",
        "intents": ["data_visualization", "code_execution"],
        "exports": ["data.artifacts"],
        "output_schema": {"exports": ["answer"]},
        "input_schema": {"required": ["instruction"], "properties": {}},
    },
    "report_writer": {
        "id": "report_writer",
        "intents": ["report_generation"],
        "exports": ["data.report"],
        "output_schema": {"exports": ["answer"]},
        "input_schema": {"required": ["instruction"], "properties": {}},
    },
}


def valid_plan():
    steps = [
        WorkflowPlanStep("step_1", "nl2sql", "database_query", [], [], "query data"),
        WorkflowPlanStep(
            "step_2",
            "data_analyst",
            "trend_analysis",
            ["step_1.query_result"],
            ["step_1"],
            "analyze trend",
        ),
        WorkflowPlanStep(
            "step_3",
            "code",
            "data_visualization",
            ["step_2.summary"],
            ["step_2"],
            "generate chart",
        ),
        WorkflowPlanStep(
            "step_4",
            "report_writer",
            "report_generation",
            ["step_1.query_result", "step_2.summary", "step_3.data.artifacts"],
            ["step_1", "step_2", "step_3"],
            "write report",
        ),
    ]
    return WorkflowPlan(
        atomic_intents=[step.intent for step in steps],
        candidate_agents=["nl2sql", "data_analyst", "code", "report_writer"],
        dag=steps,
    )


class DAGValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = DAGValidator(SPECS)

    def test_valid_plan_becomes_executor_ready_route_decision(self):
        result = self.validator.validate(
            valid_plan(),
            original_query="query data, analyze trend, generate chart and write report",
        )

        self.assertEqual(result.route_decision.mode, "workflow")
        self.assertEqual(
            [call.function.name for call in result.route_decision.calls],
            ["nl2sql", "data_analyst", "code", "report_writer"],
        )
        self.assertEqual(
            result.route_decision.calls[3].function.arguments["input_refs"][-1],
            "step_3.data.artifacts",
        )

    def test_rejects_unknown_agent_and_unsupported_intent(self):
        plan = valid_plan()
        plan.dag[1] = WorkflowPlanStep(
            "step_2", "unknown", "not_supported", [], [], "bad"
        )
        with self.assertRaises(DAGValidationError) as ctx:
            self.validator.validate(plan)
        self.assertIn("not registered", str(ctx.exception))

    def test_rejects_bad_export_and_cycle(self):
        plan = valid_plan()
        plan.dag[1] = WorkflowPlanStep(
            "step_2",
            "data_analyst",
            "trend_analysis",
            ["step_1.missing"],
            ["step_1", "step_4"],
            "analyze trend",
        )
        with self.assertRaises(DAGValidationError) as ctx:
            self.validator.validate(plan)
        message = str(ctx.exception)
        self.assertIn("not exported", message)
        self.assertIn("cycle", message)

    def test_missing_visualization_intermediate_is_deterministically_rejected(self):
        steps = [
            WorkflowPlanStep("step_1", "nl2sql", "database_query", [], [], "query data"),
            WorkflowPlanStep(
                "step_2",
                "report_writer",
                "report_generation",
                ["step_1.query_result"],
                ["step_1"],
                "write report",
            ),
        ]
        plan = WorkflowPlan(
            atomic_intents=[step.intent for step in steps],
            candidate_agents=["nl2sql", "report_writer"],
            dag=steps,
        )
        with self.assertRaises(DAGValidationError) as ctx:
            self.validator.validate(
                plan,
                original_query="query data, generate chart and write report",
            )
        self.assertIn("missing required intermediate", str(ctx.exception))

    def test_repair_callback_receives_explicit_errors_and_is_bounded(self):
        broken = {
            "atomic_intents": ["database_query", "report_generation"],
            "candidate_agents": ["nl2sql", "report_writer"],
            "dag": [
                {
                    "call_id": "step_1",
                    "agent": "nl2sql",
                    "intent": "database_query",
                    "input_refs": [],
                    "depends_on": [],
                    "instruction": "query data",
                },
                {
                    "call_id": "step_2",
                    "agent": "report_writer",
                    "intent": "report_generation",
                    "input_refs": ["step_1.query_result"],
                    "depends_on": ["step_1"],
                    "instruction": "write report",
                },
            ],
        }
        seen = []

        def repair(**kwargs):
            seen.append(kwargs)
            return valid_plan().to_dict()

        result = self.validator.validate_or_repair(
            broken,
            original_query="query data, generate chart and write report",
            repair_llm=repair,
            max_repair_attempts=3,
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(any("missing required intermediate" in error for error in seen[0]["errors"]))
        self.assertEqual(len(result.repair_attempts), 1)

    def test_repair_stops_after_three_failed_attempts(self):
        seen = []

        def bad_repair(**kwargs):
            seen.append(kwargs)
            return valid_plan().to_dict() | {"candidate_agents": ["invented"]}

        with self.assertRaises(DAGRepairError) as ctx:
            self.validator.validate_or_repair(
                valid_plan().to_dict() | {"candidate_agents": ["invented"]},
                repair_llm=bad_repair,
                max_repair_attempts=3,
            )
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(ctx.exception.attempts), 3)



    def test_rejects_unresolved_placeholder_in_instruction(self):
        plan = valid_plan()
        plan.dag[0] = WorkflowPlanStep(
            "step_1",
            "nl2sql",
            "database_query",
            [],
            [],
            "find tallest building in {city}",
        )
        with self.assertRaises(DAGValidationError) as ctx:
            self.validator.validate(plan)
        self.assertIn("unresolved placeholder", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
