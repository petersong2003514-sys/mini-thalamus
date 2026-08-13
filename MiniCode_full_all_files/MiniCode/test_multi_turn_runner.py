import json
import os
import sys
import unittest


EVALS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evals",
)
sys.path.insert(0, EVALS_DIR)


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
        from multi_turn_runner import run_multi_turn_entry

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


if __name__ == "__main__":
    unittest.main()
