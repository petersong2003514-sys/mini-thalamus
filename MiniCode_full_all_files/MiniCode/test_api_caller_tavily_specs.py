import json
import os
import tempfile
import unittest

from agents.api_caller_agent import ApiCallerAgent


class ApiCallerParamSpecTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_dir = os.environ.get("SHUSHU_API_SPECS_DIR")
        os.environ["SHUSHU_API_SPECS_DIR"] = self._tmp.name
        self._old_key = os.environ.get("SHUSHU_SEARCH_API_KEY")
        os.environ["SHUSHU_SEARCH_API_KEY"] = "test-tavily-key"

        extract = {
            "name": "Tavily Extract",
            "aliases": ["tavily extract", "tavily 提取"],
            "method": "POST",
            "url_template": "https://api.tavily.com/extract",
            "headers": {
                "Authorization": "Bearer ${SHUSHU_SEARCH_API_KEY}",
            },
            "params_schema": {
                "urls": {"type": "array", "max_items": 3},
            },
        }
        with open(os.path.join(self._tmp.name, "extract.json"), "w", encoding="utf-8") as handle:
            json.dump(extract, handle, ensure_ascii=False)

        github = {
            "name": "GitHub User",
            "aliases": ["github user"],
            "method": "GET",
            "url_template": "https://api.github.com/users/{username}",
            "headers": {},
            "params_schema": {
                "username": {"type": "string", "default": "octocat"},
            },
        }
        with open(os.path.join(self._tmp.name, "github.json"), "w", encoding="utf-8") as handle:
            json.dump(github, handle, ensure_ascii=False)

    def tearDown(self):
        if self._old_dir is None:
            os.environ.pop("SHUSHU_API_SPECS_DIR", None)
        else:
            os.environ["SHUSHU_API_SPECS_DIR"] = self._old_dir
        if self._old_key is None:
            os.environ.pop("SHUSHU_SEARCH_API_KEY", None)
        else:
            os.environ["SHUSHU_SEARCH_API_KEY"] = self._old_key
        self._tmp.cleanup()

    def test_env_header_and_urls_are_resolved(self):
        agent = ApiCallerAgent.__new__(ApiCallerAgent)

        op = agent._spec_from_query(
            "调用 Tavily Extract 提取 https://example.com 的内容"
        )

        self.assertIsNotNone(op)
        self.assertEqual(op["url"], "https://api.tavily.com/extract")
        self.assertEqual(op["params"]["urls"], ["https://example.com"])
        self.assertEqual(
            op["headers"]["Authorization"],
            "Bearer test-tavily-key",
        )

    def test_url_placeholder_is_not_duplicated_in_get_params(self):
        agent = ApiCallerAgent.__new__(ApiCallerAgent)

        op = agent._spec_from_query("查询 github user octocat")

        self.assertEqual(op["url"], "https://api.github.com/users/octocat")
        self.assertNotIn("username", op["params"])


if __name__ == "__main__":
    unittest.main()
