import unittest

from agent_registry import AgentLoadError, AgentRegistry, AgentRegistryError
from agent_runtime import AgentRuntime


EXPECTED_BUILTINS = {
    "api_caller",
    "browser",
    "code",
    "data_analyst",
    "deep_research",
    "docparse",
    "nl2sql",
    "rag",
    "report_writer",
    "shell",
    "url_fetch",
    "vision",
    "websearch",
}


class AgentRegistryTests(unittest.TestCase):
    def test_discovers_all_peer_agents_without_instantiating(self):
        registry = AgentRegistry()

        discovered = registry.discover()
        ids = set(registry.list_all())

        self.assertGreaterEqual(discovered, len(EXPECTED_BUILTINS))
        self.assertTrue(EXPECTED_BUILTINS.issubset(ids), ids)
        for agent_id in EXPECTED_BUILTINS:
            entry = registry.get(agent_id)
            self.assertEqual(entry["status"], "discovered")
            self.assertIsNone(entry.get("instance"))
            self.assertTrue(entry.get("factory"))

        nl2sql = registry.get("nl2sql")
        code = registry.get("code")
        self.assertEqual(nl2sql["source"], code["source"])
        self.assertIn("capabilities", nl2sql["meta"])

    def test_factory_is_lazy_and_instance_is_cached(self):
        calls = []
        instance = object()
        registry = AgentRegistry(agents_dir="Z:\\does-not-exist")
        registry.register(
            "fake",
            {
                "id": "fake",
                "name": "Fake",
                "description": "test",
                "capabilities": {},
            },
            factory=lambda: calls.append("created") or instance,
        )

        self.assertEqual(calls, [])
        self.assertIs(registry.get_or_create("fake"), instance)
        self.assertIs(registry.get_or_create("fake"), instance)
        self.assertEqual(calls, ["created"])
        self.assertEqual(registry.get("fake")["status"], "ready")

    def test_duplicate_agent_id_is_rejected(self):
        registry = AgentRegistry(agents_dir="Z:\\does-not-exist")
        meta = {
            "id": "fake",
            "name": "Fake",
            "description": "test",
            "capabilities": {},
        }
        registry.register("fake", meta, factory=lambda: object())

        with self.assertRaisesRegex(AgentRegistryError, "already registered"):
            registry.register("fake", meta, factory=lambda: object())

    def test_factory_failure_marks_only_that_agent_unavailable(self):
        registry = AgentRegistry(agents_dir="Z:\\does-not-exist")
        meta = {
            "id": "broken",
            "name": "Broken",
            "description": "test",
            "capabilities": {},
        }

        def fail():
            raise RuntimeError("boom")

        registry.register("broken", meta, factory=fail)
        registry.register("healthy", {**meta, "id": "healthy"}, factory=object)

        with self.assertRaisesRegex(AgentLoadError, "boom"):
            registry.get_or_create("broken")

        self.assertEqual(registry.get("broken")["status"], "unavailable")
        self.assertNotIn("broken", registry.list_all())
        self.assertEqual(registry.get("broken")["error"], "boom")
        self.assertIsNotNone(registry.get_or_create("healthy"))


class AgentRuntimeTests(unittest.TestCase):
    def test_runtime_and_loader_view_share_one_registry(self):
        registry = AgentRegistry()
        runtime = AgentRuntime(registry=registry)
        runtime.discover()

        ids = set(runtime.list_agent_ids())
        loader_ids = set(runtime.loader_registry())

        self.assertTrue(EXPECTED_BUILTINS.issubset(ids))
        self.assertEqual(ids, loader_ids)
        self.assertIs(runtime.registry, registry)


if __name__ == "__main__":
    unittest.main()
