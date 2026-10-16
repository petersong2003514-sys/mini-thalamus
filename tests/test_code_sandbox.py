import os
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mini_thalamus.agents.code_agent import CodeAgent


class CodeSandboxTests(unittest.TestCase):
    @patch("mini_thalamus.agents.code_agent.subprocess.run")
    def test_docker_execution_uses_isolation_flags_and_scrubbed_env(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        agent = CodeAgent.__new__(CodeAgent)

        result = agent._docker_execute("print('ok')")
        self.addCleanup(shutil.rmtree, result["work_dir"], True)

        command = run.call_args.args[0]
        env = run.call_args.kwargs["env"]
        self.assertEqual(result["status"], "success")
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        self.assertIn("--pids-limit", command)
        self.assertNotIn("DEEPSEEK_API_KEY", env)
        self.assertTrue(os.path.isfile(os.path.join(result["work_dir"], "script.py")))


if __name__ == "__main__":
    unittest.main()
