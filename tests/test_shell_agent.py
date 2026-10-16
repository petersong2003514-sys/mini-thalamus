"""Shell Agent 纯 stdlib 单测（无 openai / fastapi 依赖）"""
import sys
import os
import types
import unittest

# ─── stub openai ───
_openai = types.ModuleType("openai")
_openai.OpenAI = object
sys.modules.setdefault("openai", _openai)

# ─── 加入项目根 ───
from mini_thalamus.agents.shell_agent import (
    _classify_risk,
    _scrub_output,
    HARD_BLOCK_PATTERNS,
    CONFIRM_PATTERNS,
    SECRET_PATTERNS,
    ShellAgent,
    _cwd_allowed,
    SHELL_CWD_ALLOW,
)


class TestRiskClassification(unittest.TestCase):
    def test_hard_block_rm_rf_root(self):
        self.assertEqual(_classify_risk("rm -rf /"), "blocked")

    def test_hard_block_fork_bomb(self):
        self.assertEqual(_classify_risk(":(){:|:&};:"), "blocked")

    def test_hard_block_dd(self):
        self.assertEqual(_classify_risk("dd if=/dev/zero of=/dev/sda"), "blocked")

    def test_hard_block_mkfs(self):
        self.assertEqual(_classify_risk("sudo mkfs.ext4 /dev/sdb"), "blocked")

    def test_high_rm_rf_subdir(self):
        self.assertEqual(_classify_risk("rm -rf /tmp/foo"), "high")

    def test_high_git_force_push(self):
        self.assertEqual(_classify_risk("git push --force origin main"), "high")

    def test_high_sudo(self):
        self.assertEqual(_classify_risk("sudo apt update"), "high")

    def test_high_npm_publish(self):
        self.assertEqual(_classify_risk("npm publish --access public"), "high")

    def test_high_pip_uninstall(self):
        self.assertEqual(_classify_risk("pip uninstall requests"), "high")

    def test_medium_pip_install(self):
        self.assertEqual(_classify_risk("pip install flask"), "medium")

    def test_medium_curl(self):
        self.assertEqual(_classify_risk("curl https://example.com"), "medium")

    def test_low_ls(self):
        self.assertEqual(_classify_risk("ls /tmp"), "low")

    def test_low_git_status(self):
        self.assertEqual(_classify_risk("git status"), "low")

    def test_low_echo(self):
        self.assertEqual(_classify_risk("echo hello"), "low")

    def test_low_python_version(self):
        self.assertEqual(_classify_risk("python3 --version"), "low")


class TestScrubOutput(unittest.TestCase):
    def test_aws_key(self):
        fake_key = "AKIA" + "I" * 16
        out = _scrub_output("Key: " + fake_key)
        self.assertNotIn(fake_key, out)
        self.assertIn("AWS_KEY_REDACTED", out)

    def test_sk_key(self):
        fake_key = "sk-" + "a" * 24
        out = _scrub_output("OPENAI_API_KEY=" + fake_key)
        self.assertNotIn(fake_key, out)

    def test_gh_token(self):
        out = _scrub_output("token: ghp_" + "A" * 36)
        self.assertNotIn("ghp_" + "A" * 36, out)

    def test_bearer_token(self):
        out = _scrub_output("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload")
        self.assertIn("TOKEN_REDACTED", out)

    def test_no_false_positive(self):
        out = _scrub_output("ls /tmp\nhello world")
        self.assertEqual(out, "ls /tmp\nhello world")


class TestScrubEnv(unittest.TestCase):
    def test_api_key_stripped(self):
        """DEEPSEEK_API_KEY 进入 _scrub_env_for_agent 后不应存在"""
        env_before = dict(os.environ)
        os.environ["DEEPSEEK_API_KEY"] = "test_secret_12345"
        try:
            agent = ShellAgent.__new__(ShellAgent)
            agent._client = None
            env = agent._scrub_env_for_agent(os.getcwd())
            self.assertNotIn("DEEPSEEK_API_KEY", env)
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)
            # 恢复
            for k in list(os.environ.keys()):
                if k not in env_before:
                    os.environ.pop(k, None)

    def test_path_kept(self):
        agent = ShellAgent.__new__(ShellAgent)
        agent._client = None
        env = agent._scrub_env_for_agent(os.getcwd())
        self.assertIn("PATH", env)


class TestCwdAllowed(unittest.TestCase):
    def test_allowed_cwd(self):
        self.assertTrue(_cwd_allowed(os.getcwd()))

    def test_allowed_tmp(self):
        import tempfile
        self.assertTrue(_cwd_allowed(tempfile.gettempdir()))

    def test_denied_etc(self):
        self.assertFalse(_cwd_allowed("/etc"))

    def test_denied_root(self):
        self.assertFalse(_cwd_allowed("/"))


class TestShellAgentBlocked(unittest.TestCase):
    def _make_agent(self):
        a = ShellAgent.__new__(ShellAgent)
        a._client = None
        return a

    def test_hard_blocked_returns_blocked(self):
        a = self._make_agent()
        result = a.run("rm -rf /")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["code"], "hard_blocked")

    def test_high_risk_returns_checkpoint(self):
        a = self._make_agent()
        result = a.run("git push --force origin main")
        self.assertEqual(result["status"], "checkpoint")
        self.assertEqual(result["code"], "shell_confirm")
        self.assertIn("cmd", result["data"])

    def test_cwd_denied(self):
        a = self._make_agent()
        result = a._exec_single("ls", cwd="/etc/ssl")
        self.assertEqual(result.get("error_code"), "cwd_denied")


if __name__ == "__main__":
    unittest.main(verbosity=2)
