import os
import sys
import unittest


EVALS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evals",
)
sys.path.insert(0, EVALS_DIR)

import bfcl_web_search_adapter as adapter  # noqa: E402


class WebSearchAdapterTests(unittest.TestCase):
    def test_official_tool_names_and_schema(self):
        names = [tool["function"]["name"] for tool in adapter.OFFICIAL_TOOLS_NORMALIZED]
        self.assertEqual(names, ["search_engine_query", "fetch_url_content"])
        self.assertTrue(
            all(
                tool["function"]["parameters"]["type"] == "object"
                for tool in adapter.OFFICIAL_TOOLS_NORMALIZED
            )
        )

    def test_unknown_tool_returns_error(self):
        self.assertEqual(
            adapter._run_tool("not_a_tool", {}),
            {"error": "unknown tool not_a_tool"},
        )

    def test_aggregate_metrics(self):
        records = [
            {"passed": True, "latency_s": 1.0, "tool_calls": 2},
            {"passed": False, "latency_s": 3.0, "tool_calls": 4},
        ]
        metrics = adapter.aggregate(records)
        self.assertEqual(metrics["case_count"], 2)
        self.assertEqual(metrics["pass_rate"], 0.5)
        self.assertEqual(metrics["avg_latency_ms"], 2000.0)
        self.assertEqual(metrics["avg_tool_calls"], 3.0)


if __name__ == "__main__":
    unittest.main()
