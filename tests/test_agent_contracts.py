import unittest

from mini_thalamus.agent_contract import AgentContractError, normalize_agent_contract
from mini_thalamus.agent_registry import AgentRegistry


class AgentContractTests(unittest.TestCase):
    def test_normalizes_required_router_contract_fields(self):
        contract = normalize_agent_contract(
            {
                "id": "nl2sql",
                "name": "NL2SQL Agent",
                "description": "将自然语言问题转成 SQL，并执行或返回结构化查询结果",
                "intents": ["database_query"],
                "keywords": ["数据库", "SQL"],
                "few_shots": [
                    {
                        "query": "查询上个月各地区销售额",
                        "expected_intent": "database_query",
                    }
                ],
                "requires": ["database_schema", "user_question"],
                "exports": ["sql", "query_result", "dataframe"],
                "factory": "mini_thalamus.agents.nl2sql_agent:create_agent",
            }
        )

        self.assertEqual(contract["id"], "nl2sql")
        self.assertEqual(contract["few_shots"][0]["expected_intent"], "database_query")
        self.assertEqual(contract["requires"], ["database_schema", "user_question"])
        self.assertEqual(contract["exports"], ["sql", "query_result", "dataframe"])

    def test_rejects_unknown_few_shot_intent(self):
        with self.assertRaises(AgentContractError):
            normalize_agent_contract(
                {
                    "id": "bad",
                    "name": "Bad Agent",
                    "description": "bad",
                    "intents": ["known_intent"],
                    "few_shots": [
                        {
                            "query": "do something",
                            "expected_intent": "unknown_intent",
                        }
                    ],
                    "factory": "mini_thalamus.agents.bad_agent:create_agent",
                }
            )

    def test_converts_legacy_examples_to_few_shots_and_schema_exports(self):
        contract = normalize_agent_contract(
            {
                "id": "legacy_rag",
                "name": "RAG Agent",
                "description": "检索知识库",
                "intents": ["knowledge_retrieval"],
                "examples": ["从知识库查找部署说明"],
                "input_schema": {"required": ["instruction"]},
                "output_schema": {"exports": ["answer", "summary"]},
                "factory": "mini_thalamus.agents.rag_agent:create_agent",
            }
        )

        self.assertEqual(
            contract["few_shots"],
            [
                {
                    "query": "从知识库查找部署说明",
                    "expected_intent": "knowledge_retrieval",
                }
            ],
        )
        self.assertEqual(contract["requires"], ["instruction"])
        self.assertEqual(contract["exports"], ["answer", "summary"])

    def test_registry_exposes_standard_contract_fields_for_discovered_agents(self):
        registry = AgentRegistry()
        registry.discover()
        router_config = registry.get_router_config()

        self.assertIn("nl2sql", router_config)
        for agent_id, contract in router_config.items():
            with self.subTest(agent_id=agent_id):
                self.assertEqual(contract.get("id"), agent_id)
                self.assertIsInstance(contract.get("description"), str)
                self.assertIsInstance(contract.get("keywords"), list)
                self.assertIsInstance(contract.get("intents"), list)
                self.assertIsInstance(contract.get("few_shots"), list)
                self.assertIsInstance(contract.get("requires"), list)
                self.assertIsInstance(contract.get("exports"), list)

    def test_core_agents_have_planner_hints(self):
        registry = AgentRegistry()
        registry.discover()
        router_config = registry.get_router_config()
        core_agents = {
            "nl2sql",
            "rag",
            "code",
            "docparse",
            "websearch",
            "vision",
            "data_analyst",
            "deep_research",
            "report_writer",
            "browser",
            "url_fetch",
            "api_caller",
            "shell",
        }

        self.assertTrue(core_agents.issubset(router_config))
        for agent_id in sorted(core_agents):
            with self.subTest(agent_id=agent_id):
                self.assertGreater(len(router_config[agent_id]["intents"]), 0)
                self.assertGreater(len(router_config[agent_id]["few_shots"]), 0)

    def test_planner_critical_exports_are_declared(self):
        registry = AgentRegistry()
        registry.discover()
        router_config = registry.get_router_config()

        self.assertIn("data.artifacts", router_config["code"]["exports"])
        self.assertIn("data.artifacts", router_config["data_analyst"]["exports"])
        self.assertIn("data.report", router_config["report_writer"]["exports"])
        self.assertIn("sources", router_config["websearch"]["exports"])
        self.assertIn("query_result", router_config["nl2sql"]["exports"])


if __name__ == "__main__":
    unittest.main()
