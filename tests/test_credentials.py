import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_thalamus.credentials import CredentialStore


class CredentialAuthorizationTests(unittest.TestCase):
    def test_environment_fallback_cannot_bypass_agent_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            path.write_text(json.dumps({
                "version": 1,
                "credentials": {
                    "deepseek": {
                        "value": {"api_key": "stored"},
                        "agents": ["rag"],
                    },
                },
            }), encoding="utf-8")
            store = CredentialStore(str(path))
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "environment"}):
                self.assertIsNone(store.get("deepseek", agent_id="shell"))
                self.assertEqual(
                    store.get("deepseek", agent_id=None),
                    {"api_key": "stored"},
                )


if __name__ == "__main__":
    unittest.main()
