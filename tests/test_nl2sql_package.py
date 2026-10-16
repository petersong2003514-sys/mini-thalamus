import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# nl2sql_agent ships from src/ alongside the host package.
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


class NL2SQLPackageTests(unittest.TestCase):
    def test_qualified_package_import_does_not_mutate_import_state(self):
        mt_config = importlib.import_module("mini_thalamus.config")
        mt_tools = importlib.import_module("mini_thalamus.tools")
        config_origin = Path(mt_config.__file__).resolve()
        tools_origin = Path(mt_tools.__file__).resolve()
        path_before = list(sys.path)

        factory = importlib.import_module("nl2sql_agent.agent_factory")

        self.assertTrue(callable(factory.build_agent))
        self.assertEqual(sys.path, path_before)
        self.assertIs(sys.modules["mini_thalamus.config"], mt_config)
        self.assertIs(sys.modules["mini_thalamus.tools"], mt_tools)
        self.assertEqual(
            Path(sys.modules["mini_thalamus.config"].__file__).resolve(),
            config_origin,
        )
        self.assertEqual(
            Path(sys.modules["mini_thalamus.tools"].__file__).resolve(),
            tools_origin,
        )

    def test_package_entrypoint_exposes_help(self):
        completed = subprocess.run(
            [sys.executable, "-m", "nl2sql_agent", "--help"],
            cwd=SRC_ROOT,
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

    def test_default_dotenv_prefers_the_repository_root(self):
        """One .env at the repo root serves both the host and this sub-package."""
        factory = importlib.import_module("nl2sql_agent.agent_factory")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src" / "nl2sql_agent"
            package.mkdir(parents=True)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            (root / ".env").write_text("NL2SQL_MODEL=from-root\n", encoding="utf-8")
            (package / ".env").write_text("NL2SQL_MODEL=from-package\n", encoding="utf-8")

            with patch.object(factory, "PROJECT_ROOT", str(package)):
                candidates = factory._dotenv_candidates()

            self.assertEqual(candidates[0], str(root / ".env"))

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(factory, "PROJECT_ROOT", str(package)):
                    factory.load_dotenv()
                self.assertEqual(os.environ["NL2SQL_MODEL"], "from-root")

    def test_default_dotenv_ignores_the_parent_of_site_packages(self):
        """Under a real install the grandparent is unrelated; never read it."""
        factory = importlib.import_module("nl2sql_agent.agent_factory")
        with tempfile.TemporaryDirectory() as directory:
            site_packages = Path(directory) / "lib" / "site-packages"
            package = site_packages / "nl2sql_agent"
            package.mkdir(parents=True)
            (Path(directory) / "lib" / ".env").write_text(
                "NL2SQL_MODEL=should-not-be-read\n",
                encoding="utf-8",
            )
            (package / ".env").write_text("NL2SQL_MODEL=from-package\n", encoding="utf-8")

            with patch.object(factory, "PROJECT_ROOT", str(package)):
                candidates = factory._dotenv_candidates()

            self.assertEqual(candidates[0], str(package / ".env"))

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(factory, "PROJECT_ROOT", str(package)):
                    factory.load_dotenv()
                self.assertEqual(os.environ["NL2SQL_MODEL"], "from-package")


if __name__ == "__main__":
    unittest.main()
