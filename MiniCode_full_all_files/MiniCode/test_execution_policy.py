import tempfile
import unittest

from route_schema import RouteDecision
from sandbox import ExecutionPolicy


SPECS = {
    "nl2sql": {
        "capabilities": {"side_effects": "database_read"},
        "output_schema": {"exports": ["answer", "data.structured"]},
    },
    "code": {
        "capabilities": {"side_effects": "workspace_write"},
        "output_schema": {"exports": ["answer", "data.artifacts"]},
    },
    "shell": {
        "capabilities": {"side_effects": "external"},
        "output_schema": {"exports": ["answer", "data.structured"]},
    },
}


def make_decision(agent_id, policy=None):
    call = {
        "call_id": "step_1",
        "intent": "test",
        "function": {
            "name": agent_id,
            "arguments": {
                "instruction": "执行任务",
                "input_refs": [],
            },
        },
        "depends_on": [],
        "confidence": 0.9,
    }
    if policy is not None:
        call["policy"] = policy
    return RouteDecision.from_dict(
        {
            "mode": "single",
            "calls": [call],
            "candidates": [],
            "overall_confidence": 0.9,
            "method": "test",
            "reason": "test",
        },
        agent_specs=SPECS,
    )


class ExecutionPolicyTests(unittest.TestCase):
    def test_platform_overrides_model_risk_for_external_agent(self):
        decision = make_decision(
            "shell",
            policy={"risk_level": "none", "requires_approval": False},
        )
        policy = ExecutionPolicy(
            workspace_root=tempfile.gettempdir(),
            local_execution_enabled=True,
            docker_backend_enabled=False,
        )

        decision.apply_policy(
            lambda call: policy.evaluate_call(call, SPECS["shell"])
        )

        resolved = decision.calls[0].policy
        self.assertEqual(resolved["action"], "checkpoint")
        self.assertEqual(resolved["risk_level"], "external")
        self.assertTrue(resolved["requires_approval"])
        self.assertFalse(resolved["sandboxed"])

    def test_local_code_execution_is_denied_unless_explicitly_enabled(self):
        decision = make_decision("code")
        policy = ExecutionPolicy(
            workspace_root=tempfile.gettempdir(),
            local_execution_enabled=False,
            docker_backend_enabled=False,
        )

        resolved = policy.evaluate_call(decision.calls[0], SPECS["code"])

        self.assertEqual(resolved["action"], "deny")
        self.assertFalse(resolved["sandboxed"])
        self.assertFalse(resolved["requires_approval"])

    def test_configured_and_available_docker_backend_is_sandboxed(self):
        decision = make_decision("code")
        policy = ExecutionPolicy(
            workspace_root=tempfile.gettempdir(),
            local_execution_enabled=False,
            docker_backend_enabled=True,
            docker_cli_available=True,
        )

        resolved = policy.evaluate_call(decision.calls[0], SPECS["code"])

        self.assertEqual(resolved["action"], "checkpoint")
        self.assertTrue(resolved["sandboxed"])

    def test_database_read_is_allowed_without_approval(self):
        decision = make_decision("nl2sql")
        policy = ExecutionPolicy(
            workspace_root=tempfile.gettempdir(),
            local_execution_enabled=False,
            docker_backend_enabled=False,
        )

        resolved = policy.evaluate_call(decision.calls[0], SPECS["nl2sql"])

        self.assertEqual(resolved["action"], "allow")
        self.assertFalse(resolved["requires_approval"])

    def test_shell_is_not_reported_as_docker_sandboxed(self):
        decision = make_decision("shell")
        policy = ExecutionPolicy(
            workspace_root=tempfile.gettempdir(),
            local_execution_enabled=True,
            docker_backend_enabled=True,
            docker_cli_available=True,
        )

        resolved = policy.evaluate_call(decision.calls[0], SPECS["shell"])

        self.assertEqual(resolved["action"], "checkpoint")
        self.assertFalse(resolved["sandboxed"])

    def test_workspace_path_boundary(self):
        with tempfile.TemporaryDirectory() as workspace:
            policy = ExecutionPolicy(workspace_root=workspace)

            self.assertTrue(policy.path_allowed(workspace))
            self.assertFalse(policy.path_allowed(tempfile.gettempdir()))


if __name__ == "__main__":
    unittest.main()
