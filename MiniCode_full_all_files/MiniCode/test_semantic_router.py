import sys
import unittest
from unittest.mock import patch

from semantic_router import SemanticMatcher


AGENTS = {
    "nl2sql": {
        "name": "NL2SQL",
        "description": "Query databases and generate SQL for sales orders.",
        "keywords": ["database", "sql", "sales", "orders"],
        "intents": ["database_query"],
        "examples": ["query total sales orders"],
    },
    "report_writer": {
        "name": "Report Writer",
        "description": "Write structured summaries and reports.",
        "keywords": ["report", "summary", "write"],
        "intents": ["report_generation"],
        "examples": ["write a weekly report"],
    },
    "vision": {
        "name": "Vision",
        "description": "Analyze images, screenshots, and photos.",
        "keywords": ["image", "screenshot", "photo"],
        "intents": ["image_understanding"],
        "examples": ["explain the screenshot"],
    },
}


class SemanticMatcherTests(unittest.TestCase):
    def test_default_rank_does_not_import_sentence_transformers(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            matcher = SemanticMatcher()
            ranked = matcher.rank("query sales orders with SQL", AGENTS, top_k=2)

        self.assertEqual(ranked[0]["agent_id"], "nl2sql")
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["evidence"], "safe_semantic_similarity")
        self.assertEqual(matcher.last_error, "")

    def test_safe_rank_returns_empty_for_unrelated_query(self):
        matcher = SemanticMatcher()

        ranked = matcher.rank("tell me a joke about lunch", AGENTS, top_k=3)

        self.assertEqual(ranked, [])


if __name__ == "__main__":
    unittest.main()
