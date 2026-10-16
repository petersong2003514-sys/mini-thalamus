import json
import os
import sys
import unittest

from mini_thalamus import PACKAGE_ROOT


EVALS_DIR = os.path.join(PACKAGE_ROOT, "evals")


class FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class EmptyModelStep:
    def step(self, messages, tools):
        return FakeMessage("", [])


class MultiTurnRunnerTests(unittest.TestCase):
    def _load_first_entry(self):
        base = os.path.join(EVALS_DIR, "external", "bfcl_data")
        with open(
            os.path.join(base, "BFCL_v4_multi_turn_base.json"),
            encoding="utf-8",
        ) as handle:
            entry = json.loads(handle.readline())
        with open(
            os.path.join(
                base,
                "possible_answer",
                "BFCL_v4_multi_turn_base.json",
            ),
            encoding="utf-8",
        ) as handle:
            answer = json.loads(handle.readline())
        return entry, answer["ground_truth"]

    def test_empty_model_step_produces_turn_shaped_empty_results(self):
        from mini_thalamus.evals.multi_turn_runner import run_multi_turn_entry

        entry, ground_truth = self._load_first_entry()
        checker, model_results, _ = run_multi_turn_entry(
            entry,
            EmptyModelStep(),
            ground_truth,
        )
        self.assertEqual(len(model_results), len(ground_truth))
        self.assertTrue(
            all(len(turn) == 1 and turn[0] == [] for turn in model_results)
        )
        self.assertFalse(checker["valid"])
        self.assertIn("empty_turn_model_response", checker["error_type"])

    def test_add_missed_functions_injects_tools_at_holdout_turn(self):
        from mini_thalamus.evals.multi_turn_runner import add_missed_functions

        tools = [{"type": "function", "function": {"name": "cd"}}]
        docs_by_name = {
            "sort": {
                "name": "sort",
                "description": "Sort a file.",
                "parameters": {"type": "dict", "properties": {}},
            }
        }
        updated = add_missed_functions(
            tools,
            {"3": ["sort"]},
            3,
            docs_by_name,
        )
        names = [tool["function"]["name"] for tool in updated]
        self.assertIn("sort", names)

    def test_add_missed_functions_ignores_other_turns(self):
        from mini_thalamus.evals.multi_turn_runner import add_missed_functions

        tools = [{"type": "function", "function": {"name": "cd"}}]
        updated = add_missed_functions(
            tools,
            {"3": ["sort"]},
            2,
            {},
        )
        self.assertEqual([tool["function"]["name"] for tool in updated], ["cd"])


if __name__ == "__main__":
    unittest.main()
