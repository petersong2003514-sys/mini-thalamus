import json
import os
import tempfile
import unittest

from agents.api_caller_agent import ApiCallerAgent


class ApiCallerSpecTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        spec = {
            "name": "GitHub User",
            "aliases": ["github user", "github 用户"],
            "method": "GET",
            "url_template": "https://api.github.com/users/octocat",
            "headers": {},
        }
        with open(os.path.join(self._tmp.name, "github.json"), "w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False)
        self._old = os.environ.get("MT_API_SPECS_DIR")
        os.environ["MT_API_SPECS_DIR"] = self._tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("MT_API_SPECS_DIR", None)
        else:
            os.environ["MT_API_SPECS_DIR"] = self._old
        self._tmp.cleanup()

    def test_spec_matched_by_alias(self):
        agent = ApiCallerAgent.__new__(ApiCallerAgent)

        op = agent._spec_from_query("帮我调用 github 用户接口")

        self.assertIsNotNone(op)
        self.assertEqual(op["url"], "https://api.github.com/users/octocat")
        self.assertEqual(op["method"], "GET")


if __name__ == "__main__":
    unittest.main()
