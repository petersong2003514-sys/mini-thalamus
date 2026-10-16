import json
import os
import tempfile
import unittest
from collections import Counter
from difflib import SequenceMatcher

from mini_thalamus import PACKAGE_ROOT
from mini_thalamus.router_eval_dataset import (
    RouterEvalDatasetError,
    dataset_fingerprint,
    load_cases,
)

PKG_ROOT = PACKAGE_ROOT


class RouterEvalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "cases.jsonl")
        self.agents = {"nl2sql", "code", "general"}

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _case(case_id="case-1", agent_id="nl2sql"):
        return {
            "id": case_id,
            "query": "query sales",
            "session_context": "",
            "file_context": {},
            "expected": {
                "mode": "single",
                "calls": [
                    {
                        "agent_id": agent_id,
                        "depends_on": [],
                        "input_refs": [],
                    }
                ],
                "acceptable_agents": [],
            },
            "expected_stage": "semantic",
            "difficulty": "easy",
            "tags": ["ordinary_single"],
        }

    @classmethod
    def _workflow_case(cls):
        value = cls._case("workflow-1", "nl2sql")
        value["query"] = "query and chart"
        value["expected"]["mode"] = "workflow"
        value["expected"]["calls"].append(
            {
                "agent_id": "code",
                "depends_on": ["step_1"],
                "input_refs": ["step_1.data.structured"],
            }
        )
        value["expected_stage"] = "llm"
        value["difficulty"] = "hard"
        value["tags"] = ["multi_intent"]
        return value

    def _write(self, values):
        with open(self.path, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def test_loads_valid_cases_and_normalizes_immutable_records(self):
        self._write([self._case(), self._workflow_case()])

        cases = load_cases(self.path, self.agents, expected_count=2)

        self.assertEqual(cases[0].case_id, "case-1")
        self.assertEqual(cases[1].expected_calls[1].agent_id, "code")
        self.assertEqual(cases[1].expected_calls[1].depends_on, ("step_1",))

    def test_loader_rejects_duplicate_ids_and_unknown_agents(self):
        self._write([
            self._case("duplicate", "nl2sql"),
            self._case("duplicate", "ghost"),
        ])

        with self.assertRaises(RouterEvalDatasetError):
            load_cases(self.path, self.agents, expected_count=2)

    def test_loader_requires_acyclic_workflow_dependencies(self):
        value = self._workflow_case()
        value["expected"]["calls"][0]["depends_on"] = ["step_2"]
        value["expected"]["calls"][1]["depends_on"] = ["step_1"]
        self._write([value])

        with self.assertRaisesRegex(RouterEvalDatasetError, "cycle"):
            load_cases(self.path, self.agents, expected_count=1)

    def test_loader_rejects_extra_fields_and_wrong_count(self):
        value = self._case()
        value["unexpected"] = True
        self._write([value])
        with self.assertRaisesRegex(RouterEvalDatasetError, "fields"):
            load_cases(self.path, self.agents, expected_count=1)

        self._write([self._case()])
        with self.assertRaisesRegex(RouterEvalDatasetError, "expected 2"):
            load_cases(self.path, self.agents, expected_count=2)

    def test_loader_accepts_call_free_fallback_contract(self):
        value = self._case("fallback-1", "general")
        value["expected"] = {
            "mode": "fallback",
            "calls": [],
            "acceptable_agents": [],
        }
        value["expected_stage"] = "fallback"
        self._write([value])

        cases = load_cases(self.path, self.agents, expected_count=1)

        self.assertEqual(cases[0].expected_mode, "fallback")
        self.assertEqual(cases[0].expected_calls, ())

    def test_fingerprint_is_order_stable_but_content_sensitive(self):
        first = self._case("first")
        second = self._case("second", "code")
        self._write([first, second])
        forward = load_cases(self.path, self.agents, expected_count=2)
        self._write([second, first])
        reverse = load_cases(self.path, self.agents, expected_count=2)

        self.assertEqual(
            dataset_fingerprint(forward),
            dataset_fingerprint(reverse),
        )

        changed = self._case("second", "general")
        self._write([first, changed])
        changed_cases = load_cases(self.path, self.agents, expected_count=2)
        self.assertNotEqual(
            dataset_fingerprint(forward),
            dataset_fingerprint(changed_cases),
        )

    def test_committed_benchmark_has_approved_distribution(self):
        agents = {
            "api_caller", "browser", "code", "data_analyst", "deep_research",
            "docparse", "general", "nl2sql", "rag", "report_writer", "shell",
            "url_fetch", "vision", "websearch",
        }
        path = os.path.join(PKG_ROOT, "evals", "router_cases.jsonl")
        with open(path, encoding="utf-8") as handle:
            self.assertNotIn("\ufffd", handle.read())
        cases = load_cases(path, agents)
        self.assertEqual(len({case.query for case in cases}), 200)
        distribution = Counter(case.tags[0] for case in cases)
        self.assertEqual(distribution, {
            "ordinary_single": 90,
            "ambiguous": 30,
            "confusable": 30,
            "multi_intent": 30,
            "file_context": 10,
            "fallback": 10,
        })
        covered = {call.agent_id for case in cases for call in case.expected_calls}
        self.assertEqual(covered, agents - {"general"})
        self.assertEqual(
            sum(case.expected_mode == "fallback" for case in cases),
            10,
        )
        self.assertTrue(all(
            not case.acceptable_agents
            for case in cases
            if "confusable" in case.tags
        ))
        substantive = [
            case for case in cases
            if not ({"file_context", "fallback"} & set(case.tags))
        ]
        near_duplicates = [
            (left.case_id, right.case_id)
            for index, left in enumerate(substantive)
            for right in substantive[index + 1:]
            if SequenceMatcher(None, left.query, right.query).ratio() >= 0.97
        ]
        self.assertEqual(near_duplicates, [])


if __name__ == "__main__":
    unittest.main()
