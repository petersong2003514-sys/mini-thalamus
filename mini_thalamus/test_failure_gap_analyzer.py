import json
import os
import sys
import tempfile
import unittest


EVALS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evals",
)
sys.path.insert(0, os.path.join(EVALS_DIR, "explore"))


class FailureGapAnalyzerTests(unittest.TestCase):
    def test_summarize_failures_groups_by_category_and_error_type(self):
        from failure_gap_analyzer import summarize_failures

        with tempfile.TemporaryDirectory() as tmp:
            reports = {
                "parallel.json": {
                    "category": "parallel",
                    "records": [
                        {"passed": False, "error_type": "wrong_count"},
                        {"passed": False, "error_type": "wrong_count"},
                        {"passed": True, "error_type": ""},
                    ],
                },
                "spider.json": {
                    "benchmark": "Spider",
                    "records": [
                        {"passed": False, "error_type": "execution_error"},
                    ],
                },
                "empty.json": {"records": []},
            }
            for name, payload in reports.items():
                with open(
                    os.path.join(tmp, name), "w", encoding="utf-8"
                ) as handle:
                    json.dump(payload, handle)

            result = summarize_failures(tmp)
            self.assertEqual(
                result,
                [
                    {"category": "parallel", "error_type": "wrong_count", "count": 2},
                    {
                        "category": "Spider",
                        "error_type": "execution_error",
                        "count": 1,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
