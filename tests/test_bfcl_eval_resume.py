import json
import os
import tempfile
import unittest

import mini_thalamus.evals.bfcl_eval as bfcl_eval


class FakeOrchestrator:
    def __init__(self, answer):
        self.answer = answer

    def process(self, message, **kwargs):
        return {"status": "success", "answer": self.answer}


class BfclEvalResumeTests(unittest.TestCase):
    def _write_checkpoint(self, directory, stamp, case_count, base_offset=0):
        path = os.path.join(
            directory,
            f"bfcl-mini-thalamus-checkpoint-web_search-{stamp}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "benchmark": "BFCL_v4",
                    "category": "web_search",
                    "mode": "mini-thalamus-checkpoint",
                    "metrics": {"case_count": case_count},
                    "records": [],
                    "base_offset": base_offset,
                },
                handle,
            )
        return path

    def test_no_checkpoint_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                bfcl_eval.latest_checkpoint_offset(tmp, "web_search", "mini-thalamus"),
                0,
            )

    def test_newest_checkpoint_wins_and_includes_base_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_checkpoint(tmp, "20260803-100000", 25)
            self._write_checkpoint(tmp, "20260803-110000", 10, base_offset=25)
            self.assertEqual(
                bfcl_eval.latest_checkpoint_offset(tmp, "web_search", "mini-thalamus"),
                35,
            )

    def test_corrupt_checkpoint_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(
                tmp,
                "bfcl-mini-thalamus-checkpoint-web_search-20260803-100000.json",
            )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            self.assertEqual(
                bfcl_eval.latest_checkpoint_offset(tmp, "web_search", "mini-thalamus"),
                0,
            )

    def test_offline_categories_are_discovered(self):
        categories = bfcl_eval.available_categories()
        self.assertIn("simple_python", categories)
        self.assertIn("memory", categories)
        self.assertIn("multi_turn_base", categories)

    def test_irrelevance_refusal_passes(self):
        entry = {
            "id": "irrelevance_0",
            "question": [[{"role": "user", "content": "calculate triangle area"}]],
        }
        orchestrator = FakeOrchestrator(
            "I cannot use the provided BMI tool for this request."
        )
        checker, answer, latency, status = bfcl_eval.run_mt_irrelevance(
            entry,
            orchestrator,
        )
        self.assertTrue(checker["valid"])
        self.assertIn("cannot", answer.lower())


if __name__ == "__main__":
    unittest.main()
