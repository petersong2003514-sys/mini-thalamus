import unittest

from mini_thalamus.workflow_planner import WorkflowPlanner, WorkflowPlanningError


AGENT_SPECS = {
    "nl2sql": {
        "id": "nl2sql",
        "description": "Query a database and return structured sales data",
        "keywords": ["query", "sales", "database", "SQL"],
        "intents": ["database_query", "data_retrieval"],
        "few_shots": [{"query": "query monthly sales", "expected_intent": "database_query"}],
        "requires": ["user_question"],
        "exports": ["sql", "query_result", "dataframe"],
    },
    "data_analyst": {
        "id": "data_analyst",
        "description": "Analyze trends and anomalies in structured data",
        "keywords": ["analyze", "trend", "anomaly"],
        "intents": ["data_analysis", "trend_analysis"],
        "few_shots": [{"query": "analyze the trend", "expected_intent": "trend_analysis"}],
        "requires": ["data"],
        "exports": ["summary", "data.structured"],
    },
    "code": {
        "id": "code",
        "description": "Run code and generate charts",
        "keywords": ["chart", "plot", "python"],
        "intents": ["code_execution", "data_visualization"],
        "few_shots": [{"query": "generate a chart", "expected_intent": "data_visualization"}],
        "requires": ["data"],
        "exports": ["data.artifacts", "summary"],
    },
    "report_writer": {
        "id": "report_writer",
        "description": "Write a structured report from prior results",
        "keywords": ["report", "write", "summary"],
        "intents": ["report_generation", "summary_writing"],
        "few_shots": [{"query": "write a report", "expected_intent": "report_generation"}],
        "requires": ["source_data"],
        "exports": ["data.report", "summary"],
    },
}


class WorkflowPlannerTests(unittest.TestCase):
    def test_contract_fallback_builds_linear_draft_and_report_fan_in(self):
        planner = WorkflowPlanner(agent_specs=AGENT_SPECS)

        plan = planner.plan(
            "query sales data, analyze trend, generate chart, write report"
        )

        self.assertEqual(
            [step.agent for step in plan.dag],
            ["nl2sql", "data_analyst", "code", "report_writer"],
        )
        self.assertEqual(
            [step.intent for step in plan.dag],
            [
                "database_query",
                "trend_analysis",
                "data_visualization",
                "report_generation",
            ],
        )
        self.assertEqual(plan.dag[1].input_refs, ["step_1.query_result"])
        self.assertEqual(plan.dag[2].input_refs, ["step_2.summary"])
        self.assertEqual(
            plan.dag[3].depends_on,
            ["step_1", "step_2", "step_3"],
        )
        self.assertEqual(
            plan.dag[3].input_refs,
            [
                "step_1.query_result",
                "step_2.summary",
                "step_3.data.artifacts",
            ],
        )

    def test_router_style_llm_override_is_normalized(self):
        payload = {
            "atomic_intents": ["retrieve_sales_data", "write_report"],
            "candidate_agents": ["nl2sql", "report_writer"],
            "dag": [
                {
                    "call_id": "step_1",
                    "agent": "nl2sql",
                    "intent": "retrieve_sales_data",
                    "input_refs": [],
                },
                {
                    "call_id": "step_2",
                    "agent": "report_writer",
                    "intent": "write_report",
                    "input_refs": ["step_1.query_result"],
                    "depends_on": ["step_1"],
                },
            ],
        }
        received = []

        def fake_planner(**kwargs):
            received.append(kwargs)
            return payload

        planner = WorkflowPlanner(agent_specs=AGENT_SPECS, llm_planner=fake_planner)
        plan = planner.plan("query sales and write report")

        self.assertEqual(plan.method, "llm_planner")
        self.assertEqual(plan.to_dict()["dag"][1]["agent"], "report_writer")
        self.assertEqual(received[0]["schema"]["required"], ["atomic_intents", "candidate_agents", "dag"])
        self.assertIn("few_shots", received[0]["candidate_specs"]["nl2sql"])

    def test_planner_rejects_unregistered_agent(self):
        planner = WorkflowPlanner(
            agent_specs=AGENT_SPECS,
            llm_planner=lambda **_: {
                "atomic_intents": ["bad"],
                "candidate_agents": ["not_registered"],
                "dag": [
                    {
                        "call_id": "step_1",
                        "agent": "not_registered",
                        "intent": "bad",
                        "input_refs": [],
                    }
                ],
            },
        )

        with self.assertRaises(WorkflowPlanningError):
            planner.plan("do something")

    def test_schema_has_only_draft_fields(self):
        schema = WorkflowPlanner.json_schema()
        self.assertEqual(
            schema["required"], ["atomic_intents", "candidate_agents", "dag"]
        )
        self.assertIn("input_refs", schema["properties"]["dag"]["items"]["properties"])


if __name__ == "__main__":
    unittest.main()
