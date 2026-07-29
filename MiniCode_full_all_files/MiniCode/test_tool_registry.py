import time
import unittest

from tool_registry import ToolDef, ToolRegistry


def schema():
    return {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }


class ToolRegistryTests(unittest.TestCase):
    def test_validation_rejects_missing_and_unknown_arguments(self):
        registry = ToolRegistry()
        registry.register(ToolDef("echo", "echo", schema(), lambda value: value))

        self.assertEqual(registry.execute("echo", {})["code"], "invalid_arguments")
        self.assertEqual(
            registry.execute("echo", {"value": 1, "extra": 2})["code"],
            "invalid_arguments",
        )

    def test_high_risk_tool_requires_platform_approval(self):
        registry = ToolRegistry()
        registry.register(ToolDef(
            "deploy", "deploy", schema(), lambda value: value, risk="high"
        ))

        result = registry.execute("deploy", {"value": 1})

        self.assertEqual(result["status"], "checkpoint")
        self.assertTrue(result["requires_approval"])

    def test_transient_failure_gets_bounded_retry(self):
        attempts = []

        def flaky(value):
            attempts.append(value)
            if len(attempts) == 1:
                raise ConnectionError("temporary")
            return value

        registry = ToolRegistry()
        registry.register(ToolDef(
            "flaky", "flaky", schema(), flaky, max_retries=1, idempotent=True
        ))

        result = registry.execute("flaky", {"value": 7})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempts"], 2)

    def test_timeout_is_classified(self):
        registry = ToolRegistry()
        registry.register(ToolDef(
            "slow",
            "slow",
            schema(),
            lambda value: (time.sleep(0.03), value)[1],
            timeout=0.001,
        ))

        result = registry.execute("slow", {"value": 1})

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
