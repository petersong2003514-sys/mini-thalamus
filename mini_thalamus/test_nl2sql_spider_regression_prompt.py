import os
import sys
import unittest
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PKG_ROOT.parent


class SpiderRegressionPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._added_bundle_root = str(BUNDLE_ROOT) not in sys.path
        if cls._added_bundle_root:
            sys.path.insert(0, str(BUNDLE_ROOT))

    @classmethod
    def tearDownClass(cls):
        if cls._added_bundle_root:
            sys.path.remove(str(BUNDLE_ROOT))

    def test_system_prompt_guides_minimal_column_selection(self):
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        self.assertIn("只返回用户问题明确要求输出的列", NL2SQL_SYSTEM_PROMPT)

    def test_system_prompt_preserves_string_case(self):
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        self.assertIn("保持数据库中的原始大小写", NL2SQL_SYSTEM_PROMPT)

    def test_system_prompt_uses_in_subquery_for_existence(self):
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        self.assertIn("IN (SELECT", NL2SQL_SYSTEM_PROMPT)

    def test_system_prompt_counts_rows_with_count_star(self):
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        self.assertIn("COUNT(*)", NL2SQL_SYSTEM_PROMPT)

    def test_system_prompt_allows_lower_for_case_insensitive_match(self):
        from nl2sql_agent.agent.prompts import NL2SQL_SYSTEM_PROMPT

        self.assertIn("LOWER(column) = LOWER('value')", NL2SQL_SYSTEM_PROMPT)

    def test_fewshot_pool_has_case_insensitive_example(self):
        from nl2sql_agent.agent.fewshot_examples import FEWSHOT_POOL

        self.assertTrue(any("LOWER(" in ex.sql for ex in FEWSHOT_POOL))

    def test_fewshot_pool_has_minimal_column_example(self):
        from nl2sql_agent.agent.fewshot_examples import FEWSHOT_POOL

        self.assertTrue(
            any(
                ex.sql.strip().startswith("SELECT name, date FROM")
                for ex in FEWSHOT_POOL
            )
        )


if __name__ == "__main__":
    unittest.main()
