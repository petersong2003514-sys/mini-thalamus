import unittest
from types import SimpleNamespace

from sandbox import ExecutionPolicy


def code_call():
    return SimpleNamespace(function=SimpleNamespace(name="code"))


class ExecutionPolicyAutoApproveTests(unittest.TestCase):
    def test_auto_approve_allows_workspace_write_with_local_backend(self):
        policy = ExecutionPolicy(
            workspace_root=r"D:\实习",
            local_execution_enabled=True,
            auto_approve=True,
        )

        result = policy.evaluate_call(
            code_call(),
            {"capabilities": {"side_effects": "workspace_write"}},
        )

        self.assertEqual(result["action"], "allow")
        self.assertFalse(result["requires_approval"])

    def test_auto_approve_still_denies_without_execution_backend(self):
        policy = ExecutionPolicy(
            workspace_root=r"D:\实习",
            local_execution_enabled=False,
            auto_approve=True,
        )

        result = policy.evaluate_call(
            code_call(),
            {"capabilities": {"side_effects": "workspace_write"}},
        )

        self.assertEqual(result["action"], "deny")
        self.assertFalse(result["requires_approval"])


if __name__ == "__main__":
    unittest.main()
