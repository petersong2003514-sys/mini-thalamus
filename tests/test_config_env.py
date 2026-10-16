import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mini_thalamus.config as config


class ConfigEnvTests(unittest.TestCase):
    def test_load_env_file_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "DEEPSEEK_API_KEY=file-key\n"
                "DEEPSEEK_MODEL=file-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "system-key"}, clear=True):
                config._load_env_file(str(env_file))

                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "system-key")
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "file-model")

    def test_get_llm_config_uses_unified_deepseek_environment(self):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
            "DEEPSEEK_MODEL": "deepseek-chat",
        }, clear=True):
            cfg = config.get_llm_config()

        self.assertEqual(cfg["api_key"], "test-key")
        self.assertEqual(cfg["base_url"], "https://api.deepseek.com/v1")
        self.assertEqual(cfg["model"], "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
