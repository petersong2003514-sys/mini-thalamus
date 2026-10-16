import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PKG_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PKG_ROOT.parent


class NL2SQLPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._added_bundle_root = str(BUNDLE_ROOT) not in sys.path
        if cls._added_bundle_root:
            sys.path.insert(0, str(BUNDLE_ROOT))

    @classmethod
    def tearDownClass(cls):
        if cls._added_bundle_root:
            sys.path.remove(str(BUNDLE_ROOT))

    def test_qualified_package_import_does_not_mutate_import_state(self):
        mt_config = importlib.import_module("config")
        mt_tools = importlib.import_module("tools")
        config_origin = Path(mt_config.__file__).resolve()
        tools_origin = Path(mt_tools.__file__).resolve()
        path_before = list(sys.path)

        factory = importlib.import_module("nl2sql_agent.agent_factory")

        self.assertTrue(callable(factory.build_agent))
        self.assertEqual(sys.path, path_before)
        self.assertIs(sys.modules["config"], mt_config)
        self.assertIs(sys.modules["tools"], mt_tools)
        self.assertEqual(Path(sys.modules["config"].__file__).resolve(), config_origin)
        self.assertEqual(Path(sys.modules["tools"].__file__).resolve(), tools_origin)

    def test_package_entrypoint_exposes_help(self):
        completed = subprocess.run(
            [sys.executable, "-m", "nl2sql_agent", "--help"],
            cwd=BUNDLE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("usage", completed.stdout.lower())

    def test_dotenv_values_are_scoped_to_nl2sql_namespace(self):
        factory = importlib.import_module("nl2sql_agent.agent_factory")
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=fake-test-key\n"
                "LLM_MODEL=fake-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                factory.load_dotenv(str(env_file))

                self.assertNotIn("OPENAI_API_KEY", os.environ)
                self.assertNotIn("LLM_MODEL", os.environ)
                self.assertEqual(os.environ["NL2SQL_API_KEY"], "fake-test-key")
                self.assertEqual(os.environ["NL2SQL_MODEL"], "fake-model")

    def test_missing_key_raises_normal_configuration_error(self):
        factory = importlib.import_module("nl2sql_agent.agent_factory")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                factory.create_llm(api_key="", model="test", base_url="")


if __name__ == "__main__":
    unittest.main()
