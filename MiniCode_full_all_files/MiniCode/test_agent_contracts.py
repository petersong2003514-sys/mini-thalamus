import unittest

from agent_contract import AgentContractError, normalize_agent_contract
from agent_registry import AgentRegistry


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
                "factory": "agents.nl2sql_agent:create_agent",
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
                    "factory": "agents.bad_agent:create_agent",
                }
            )

    def test_converts_legacy_examples_to_few_shots_and_schema_exports(self):
        contract = normalize_agent_contract(
            {
                "id": "rag",
                "name": "RAG Agent",
                "description": "检索知识库",
                "intents": ["knowledge_retrieval"],
                "examples": ["从知识库查找部署说明"],
                "input_schema": {"required": ["instruction"]},
                "output_schema": {"exports": ["answer", "summary"]},
                "factory": "agents.rag_agent:create_agent",
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
                self.assertIsInstance(contract.get("description"), str)
                self.assertIsInstance(contract.get("keywords"), list)
                self.assertIsInstance(contract.get("intents"), list)
                self.assertIsInstance(contract.get("few_shots"), list)
                self.assertIsInstance(contract.get("requires"), list)
                self.assertIsInstance(contract.get("exports"), list)


if __name__ == "__main__":
    unittest.main()
