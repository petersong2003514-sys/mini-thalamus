"""Prove an Agent can be added, replaced and removed on a live registry.

These tests are the evidence behind the "hot-plug" claim: nothing here restarts
a process or re-imports the registry. The same AgentRegistry object is mutated
while holding live instances, and the previously handed-out instance must keep
working against the code it started with.
"""

import os
import sys
import tempfile
import unittest



from mini_thalamus.agent_registry import AgentLoadError, AgentRegistry
from mini_thalamus.agent_runtime import AgentRuntime


AGENT_TEMPLATE = '''\
"""Probe Agent used to prove runtime replacement."""

AGENT_SPEC = {{
    "id": "{agent_id}",
    "name": "{agent_id} Agent",
    "description": "probe at {version}",
    "factory": "agents.{agent_id}_agent:create_agent",
}}

VERSION = "{version}"


class ProbeAgent:
    def __init__(self):
        self.capabilities = {{}}

    def run(self, query, context=None):
        return {{"answer": VERSION, "status": "success", "data": {{}}}}


def create_agent():
    return ProbeAgent()
'''


class HotReloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.agents_dir = self._tmp.name
        self.registry = AgentRegistry(agents_dir=self.agents_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(
        self, agent_id="probe", version="v1", broken=False, factory_raises=False
    ):
        source = AGENT_TEMPLATE.format(agent_id=agent_id, version=version)
        if factory_raises:
            # Compiles fine, but blows up when the factory is invoked.
            source = source.replace(
                "def create_agent():\n    return ProbeAgent()",
                "def create_agent():\n    raise RuntimeError('boom')",
            )
        if broken:
            source += "\nthis line is deliberately ( not valid python\n"
        path = os.path.join(self.agents_dir, f"{agent_id}_agent.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        return path

    @staticmethod
    def _hot_modules():
        return {name for name in sys.modules if "__hot" in name}

    # ── runtime ADD ────────────────────────────────────────────────────
    def test_new_agent_becomes_usable_without_restart(self):
        self._write("probe", "v1")
        report = self.registry.rescan()

        self.assertEqual(report["added"], ["probe"])
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v1")

        # Drop a second Agent next to the first on the running registry.
        self._write("extra", "v1")
        report = self.registry.rescan()

        self.assertEqual(report["added"], ["extra"])
        self.assertEqual(self.registry.get_or_create("extra").run("q")["answer"], "v1")

    # ── runtime REPLACE ────────────────────────────────────────────────
    def test_reload_swaps_implementation_on_a_live_registry(self):
        self._write("probe", "v1")
        self.registry.rescan()
        stale = self.registry.get_or_create("probe")
        self.assertEqual(stale.run("q")["answer"], "v1")

        self._write("probe", "v2")
        result = self.registry.reload_agent("probe")

        fresh = self.registry.get_or_create("probe")
        self.assertEqual(result["status"], "reloaded")
        self.assertEqual(fresh.run("q")["answer"], "v2")
        self.assertIsNot(stale, fresh)

    def test_instance_handed_out_before_swap_keeps_running_old_code(self):
        """In-flight safety: an instance already executing is never yanked."""
        self._write("probe", "v1")
        self.registry.rescan()
        in_flight = self.registry.get_or_create("probe")

        self._write("probe", "v2")
        self.registry.reload_agent("probe")

        self.assertEqual(in_flight.run("q")["answer"], "v1")
        self.assertEqual(
            self.registry.get_or_create("probe").run("q")["answer"], "v2"
        )

    def test_failed_reload_keeps_the_running_version(self):
        """Build-then-swap: a broken edit degrades nothing."""
        self._write("probe", "v2")
        self.registry.rescan()
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v2")

        self._write("probe", "v3", broken=True)
        with self.assertRaisesRegex(AgentLoadError, "keeping the running version"):
            self.registry.reload_agent("probe")

        # Registry untouched, Agent still serving the last good version.
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v2")
        self.assertEqual(self.registry.get("probe")["status"], "ready")

    # ── runtime REMOVE ─────────────────────────────────────────────────
    def test_deleted_source_is_unregistered_by_rescan(self):
        self._write("probe", "v1")
        self.registry.rescan()
        self.assertIsNotNone(self.registry.get("probe"))

        os.remove(os.path.join(self.agents_dir, "probe_agent.py"))
        report = self.registry.rescan()

        self.assertEqual(report["removed"], ["probe"])
        self.assertIsNone(self.registry.get("probe"))
        self.assertNotIn("probe", self.registry.list_all())

    def test_source_that_stops_parsing_is_not_treated_as_a_deletion(self):
        """A bad edit must not unregister a healthy, still-serving Agent."""
        self._write("probe", "v1")
        self.registry.rescan()

        self._write("probe", "v2", broken=True)
        report = self.registry.rescan()

        self.assertEqual(report["removed"], [])
        self.assertIn("probe", report["errors"])
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v1")

    def test_unregister_notifies_subscribers_with_removal_event(self):
        events = []
        self._write("probe", "v1")
        self.registry.rescan()
        self.registry.on_register(lambda agent_id, meta: events.append((agent_id, meta)))

        self.assertTrue(self.registry.unregister("probe"))

        self.assertEqual(events, [("probe", None)])
        # Idempotent: removing twice reports no work.
        self.assertFalse(self.registry.unregister("probe"))

    # ── change detection ───────────────────────────────────────────────
    def test_rescan_is_a_no_op_when_nothing_changed(self):
        """Untouched Agents keep their warm instance instead of being rebuilt."""
        self._write("probe", "v1")
        self.registry.rescan()
        warm = self.registry.get_or_create("probe")

        report = self.registry.rescan()

        self.assertEqual(report["unchanged"], ["probe"])
        self.assertEqual(report["reloaded"], [])
        self.assertIs(self.registry.get_or_create("probe"), warm)

    def test_one_broken_manifest_does_not_block_the_others(self):
        self._write("probe", "v1")
        self._write("extra", "v1", broken=True)
        report = self.registry.rescan()

        self.assertEqual(report["added"], ["probe"])
        # A file that cannot even be parsed has no readable id, so it is reported
        # by filename; the healthy Agent next to it still registers.
        self.assertIn("extra_agent.py", report["errors"])
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v1")

    # ── 资源回收：反复热替换不能把旧模块堆在内存里 ────────────────────
    def test_repeated_reload_does_not_accumulate_modules(self):
        """每次替换应净增 0 个模块条目，而不是留一份旧代码。"""
        self._write("probe", "v1")
        self.registry.rescan()
        before = len(self._hot_modules())

        for index in range(20):
            self._write("probe", f"v{index + 2}")
            self.registry.reload_agent("probe")

        growth = len(self._hot_modules()) - before
        self.assertEqual(growth, 0, f"热重载 20 次后净增 {growth} 个模块条目")
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v21")

    def test_old_module_is_collected_once_nothing_references_it(self):
        """旧版本的内存由引用决定：在途实例一放手，旧模块就该被回收。"""
        import gc
        import weakref

        self._write("probe", "v1")
        self.registry.rescan()
        stale = self.registry.get_or_create("probe")
        old_module = sys.modules[stale.__class__.__module__]
        ref = weakref.ref(old_module)

        self._write("probe", "v2")
        self.registry.reload_agent("probe")
        del stale, old_module  # 在途请求结束，引用释放
        gc.collect()

        self.assertIsNone(ref(), "旧模块对象没有被回收，说明还有地方在引用它")

    def test_failed_factory_does_not_leak_a_module(self):
        """模块编译成功但工厂抛异常时，也不能把这份新代码留在 sys.modules。"""
        self._write("probe", "v1")
        self.registry.rescan()
        before = self._hot_modules()

        self._write("probe", "v2", factory_raises=True)
        for _ in range(5):
            with self.assertRaisesRegex(AgentLoadError, "keeping the running version"):
                self.registry.reload_agent("probe")

        leaked = self._hot_modules() - before
        self.assertEqual(leaked, set(), f"失败的 reload 泄漏了模块: {leaked}")

    def test_two_registries_do_not_stomp_on_each_others_modules(self):
        """sys.modules 是进程全局的：两个 registry 不能生成相同的模块名。"""
        self._write("probe", "v1")
        first = AgentRegistry(agents_dir=self.agents_dir)
        second = AgentRegistry(agents_dir=self.agents_dir)
        first.rescan()
        second.rescan()

        name_a = first.get("probe")["hot_module"]
        name_b = second.get("probe")["hot_module"]

        self.assertNotEqual(name_a, name_b)
        self.assertIn(name_a, sys.modules)
        self.assertIn(name_b, sys.modules)

    # ── discover() 不能摧毁热加载的成果 ──────────────────────────────
    def test_discover_does_not_downgrade_hot_loaded_agents(self):
        """discover() 是增量式的：不能把路径加载过的 Agent 打回字符串工厂。"""
        self._write("probe", "v1")
        self.registry.rescan()
        self.assertTrue(callable(self.registry.get("probe")["factory"]))

        self.registry.discover()  # 老的全量重扫路径

        entry = self.registry.get("probe")
        self.assertIsNotNone(entry, "discover() 把热加载的 Agent 弄丢了")
        self.assertTrue(callable(entry["factory"]), "工厂被降级回字符串了")
        self.assertEqual(self.registry.get_or_create("probe").run("q")["answer"], "v1")

    def test_agent_runtime_does_not_downgrade_hot_loaded_agents(self):
        """AgentRuntime 首次取值会触发 discover()，不能因此弄坏热加载的 Agent。"""
        self._write("probe", "v1")
        self.registry.rescan()

        runtime = AgentRuntime(registry=self.registry)
        runtime.discover()

        self.assertEqual(runtime.get_agent("probe").run("q")["answer"], "v1")

    def test_router_config_reflects_a_runtime_addition(self):
        """The Router reads live state, so it sees the swap with no restart."""
        self._write("probe", "v1")
        self.registry.rescan()
        self.assertIn("probe", self.registry.get_router_config())

        self._write("probe", "v2")
        self.registry.reload_agent("probe")

        self.assertEqual(
            self.registry.get_router_config()["probe"]["description"],
            "probe at v2",
        )


if __name__ == "__main__":
    unittest.main()
