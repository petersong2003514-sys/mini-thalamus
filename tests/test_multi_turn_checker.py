import json
import os
import sys
import unittest

from mini_thalamus import PACKAGE_ROOT


EVALS_DIR = os.path.join(PACKAGE_ROOT, "evals")


class MultiTurnCheckerTests(unittest.TestCase):
    def test_parse_tool_path(self):
        from mini_thalamus.evals.multi_turn_checker import parse_tool_path

        answer = "GorillaFileSystem.find()\nGorillaFileSystem.mv()"
        self.assertEqual(
            parse_tool_path(answer),
            ["GorillaFileSystem.find", "GorillaFileSystem.mv"],
        )

    def test_score_multi_turn_matches_path(self):
        from mini_thalamus.evals.multi_turn_checker import score_multi_turn

        entry = {"path": ["GorillaFileSystem.find", "GorillaFileSystem.mv"]}
        result = score_multi_turn(
            entry,
            "GorillaFileSystem.find()\nGorillaFileSystem.mv()",
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["expected"], result["actual"])

    def test_score_multi_turn_reports_mismatch(self):
        from mini_thalamus.evals.multi_turn_checker import score_multi_turn

        entry = {"path": ["GorillaFileSystem.find", "GorillaFileSystem.mv"]}
        result = score_multi_turn(entry, "GorillaFileSystem.find()")
        self.assertFalse(result["valid"])
        self.assertEqual(result["error_type"], "multi_turn:path_mismatch")


class OfficialMultiTurnCheckerTests(unittest.TestCase):
    def setUp(self):
        self.entry_path = os.path.join(
            EVALS_DIR,
            "external",
            "bfcl_data",
            "BFCL_v4_multi_turn_base.json",
        )
        self.answer_path = os.path.join(
            EVALS_DIR,
            "external",
            "bfcl_data",
            "possible_answer",
            "BFCL_v4_multi_turn_base.json",
        )

    def _load_first(self):
        from mini_thalamus.evals.multi_turn_checker import score_official_multi_turn

        with open(self.entry_path, encoding="utf-8") as handle:
            entry = json.loads(handle.readline())
        with open(self.answer_path, encoding="utf-8") as handle:
            answer = json.loads(handle.readline())
        ground_truth = answer["ground_truth"]
        model_results = [[list(calls)] for calls in ground_truth]
        return score_official_multi_turn(
            entry,
            ground_truth,
            model_results,
            "multi_turn_base",
            "mini-thalamus-test",
        ), entry

    def test_official_checker_passes_ground_truth_replay(self):
        result, _ = self._load_first()
        self.assertTrue(result["valid"], result)

    def test_official_checker_reports_empty_turn(self):
        from mini_thalamus.evals.multi_turn_checker import score_official_multi_turn

        with open(self.entry_path, encoding="utf-8") as handle:
            entry = json.loads(handle.readline())
        with open(self.answer_path, encoding="utf-8") as handle:
            answer = json.loads(handle.readline())
        ground_truth = answer["ground_truth"]
        empty_results = [[] for _ in ground_truth]
        result = score_official_multi_turn(
            entry,
            ground_truth,
            empty_results,
            "multi_turn_base",
            "mini-thalamus-test",
        )
        self.assertFalse(result["valid"])
        self.assertIn("empty_turn_model_response", result["error_type"])


if __name__ == "__main__":
    unittest.main()
