import json
import os
import sys
import tempfile
import unittest


EVALS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evals",
)
SELF_EVOLUTION_DIR = os.path.join(EVALS_DIR, "self_evolution")
sys.path.insert(0, SELF_EVOLUTION_DIR)
sys.path.insert(0, EVALS_DIR)

import apply as self_apply  # noqa: E402
import collect as self_collect  # noqa: E402
import feedback as self_feedback  # noqa: E402
import guard as self_guard  # noqa: E402


class SelfEvolutionTests(unittest.TestCase):
    def _write_report(self, directory, name, data):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return path

    def test_collect_feedback_apply_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = os.path.join(tmp, "results")
            os.makedirs(results)
            report = self._write_report(
                results,
                "bfcl-shushu-web_search-20260813-100000.json",
                {
                    "benchmark": "BFCL_v4",
                    "category": "web_search",
                    "mode": "shushu",
                    "metrics": {"case_count": 2, "pass_rate": 0.5},
                    "records": [
                        {"id": "web_search_1", "passed": True},
                        {
                            "id": "web_search_2",
                            "passed": False,
                            "error_type": "answer_not_found",
                            "answer": "latest news about Shanghai",
                        },
                    ],
                },
            )
            output_dir = os.path.join(tmp, "collected")
            collected_path = self_collect.collect(
                results_dir=results,
                output_dir=output_dir,
                benchmark="bfcl",
            )
            with open(collected_path, encoding="utf-8") as handle:
                collected = json.load(handle)
            self.assertEqual(len(collected["failures"]), 1)

            patches_dir = os.path.join(tmp, "patches")
            patches_path = self_feedback.generate_patches(
                collected,
                output_dir=patches_dir,
            )
            with open(patches_path, encoding="utf-8") as handle:
                patches = json.load(handle)
            self.assertGreaterEqual(len(patches["patches"]), 1)

            agents_dir = os.path.join(tmp, "agents")
            os.makedirs(agents_dir)
            agent_path = os.path.join(agents_dir, "websearch_agent.py")
            with open(agent_path, "w", encoding="utf-8") as handle:
                handle.write(
                    'AGENT_SPEC = {"keywords": ["search", "news"]}\n'
                )
            for patch in patches["patches"]:
                patch["file"] = os.path.join("agents", "websearch_agent.py")
            dry_run = self_apply.apply_patch_file(
                patches["patches"][0],
                tmp,
                apply_write=False,
            )
            self.assertTrue(dry_run["ok"])
            self.assertTrue(dry_run["changed"])
            with open(agent_path, encoding="utf-8") as handle:
                self.assertNotIn("Shanghai", handle.read())

            written = self_apply.apply_patch_file(
                patches["patches"][0],
                tmp,
                apply_write=True,
            )
            self.assertTrue(written["ok"])
            with open(agent_path, encoding="utf-8") as handle:
                self.assertIn("Shanghai", handle.read())

    def test_guard_accepts_non_negative_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = self._write_report(
                tmp,
                "before.json",
                {
                    "benchmark": "BFCL_v4",
                    "metrics": {"pass_rate": 0.2},
                },
            )
            after = self._write_report(
                tmp,
                "after.json",
                {
                    "benchmark": "BFCL_v4",
                    "metrics": {"pass_rate": 0.3},
                },
            )
            outcome = self_guard.guard(before, after, threshold=0.0)
            self.assertTrue(outcome["passed"])
            self.assertAlmostEqual(outcome["delta"], 0.1)

    def test_apply_supports_agent_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents_dir = os.path.join(tmp, "agents")
            os.makedirs(agents_dir)
            agent_path = os.path.join(agents_dir, "websearch_agent.py")
            with open(agent_path, "w", encoding="utf-8") as handle:
                handle.write('AGENT_META = {"keywords": ["search", "新闻"]}\n')
            patch = {
                "id": "meta-patch",
                "file": os.path.join("agents", "websearch_agent.py"),
                "field": "keywords",
                "values": ["Shanghai"],
            }
            result = self_apply.apply_patch_file(
                patch,
                tmp,
                apply_write=True,
            )
            self.assertTrue(result["ok"])
            with open(agent_path, encoding="utf-8") as handle:
                self.assertIn("Shanghai", handle.read())


if __name__ == "__main__":
    unittest.main()
