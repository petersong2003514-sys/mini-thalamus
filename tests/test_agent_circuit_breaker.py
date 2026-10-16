"""Running-time fault isolation: failures trip a breaker, and recovery is a path.

The story these tests lock down:
  detect -> trip -> leave the candidate queue -> cooldown -> probe -> recover
Nothing here restarts a process, and nothing here is a one-way switch.
"""

import os
import sys
import unittest



import mini_thalamus.agent_registry as registry_module
from mini_thalamus.agent_adapter import adapt_agent
from mini_thalamus.agent_registry import (
    AgentLoadError,
    AgentRegistry,
    CIRCUIT_FAILURE_THRESHOLD,
    HARD_FAILURE,
    SOFT_FAILURE,
)


META = {
    "id": "probe",
    "name": "Probe Agent",
    "description": "circuit breaker probe",
    "factory": "mini_thalamus.agents.probe_agent:create_agent",
}


class _Probe:
    """An Agent whose behaviour the test drives directly."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.capabilities = {}

    def run(self, query, context=None):
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "empty":
            return {"answer": "", "status": "success", "data": {}}
        if self.mode == "error":
            return {"answer": "查询失败", "status": "error", "data": {}}
        return {"answer": "ok", "status": "success", "data": {}}


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry(agents_dir="Z:\\does-not-exist")
        self._cooldown = registry_module.CIRCUIT_COOLDOWN_SECONDS

    def tearDown(self):
        registry_module.CIRCUIT_COOLDOWN_SECONDS = self._cooldown

    def _register(self, mode="ok"):
        self.registry.register(
            "probe", dict(META), factory=lambda: _Probe(mode)
        )
        return self.registry.get_or_create("probe")

    def _fail(self, times=None):
        for _ in range(times or CIRCUIT_FAILURE_THRESHOLD):
            self.registry.record_failure("probe", "boom")

    # ── 检测 ──────────────────────────────────────────────────────────
    def test_hard_failures_trip_the_breaker(self):
        self._register()
        self.assertEqual(self.registry.health_of("probe")["state"], "closed")

        self._fail(CIRCUIT_FAILURE_THRESHOLD - 1)
        self.assertEqual(self.registry.health_of("probe")["state"], "closed")

        self._fail(1)
        health = self.registry.health_of("probe")
        self.assertEqual(health["state"], "open")
        self.assertEqual(health["consecutive_failures"], CIRCUIT_FAILURE_THRESHOLD)
        self.assertIn("boom", health["last_error"])

    def test_soft_failures_never_trip_on_their_own(self):
        """A model judged these; a wrong judgement must not remove an Agent."""
        self._register()
        for _ in range(CIRCUIT_FAILURE_THRESHOLD * 5):
            self.registry.record_failure("probe", "数值可能对不上", severity=SOFT_FAILURE)

        health = self.registry.health_of("probe")
        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["consecutive_failures"], 0)
        self.assertIn("数值", health["last_error"])  # 仍然记下来了

    def test_success_closes_the_breaker_and_clears_the_count(self):
        self._register()
        self._fail()
        self.assertEqual(self.registry.health_of("probe")["state"], "open")

        self.registry.record_success("probe")

        health = self.registry.health_of("probe")
        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["consecutive_failures"], 0)

    # ── 排除出候选队列 ────────────────────────────────────────────────
    def test_tripped_agent_leaves_every_candidate_view(self):
        self._register()
        self.assertIn("probe", self.registry.list_all())
        self.assertIn("probe", self.registry.get_router_config())
        self.assertIn("probe", self.registry.loader_registry())

        self._fail()

        self.assertNotIn("probe", self.registry.list_all())
        self.assertNotIn("probe", self.registry.get_router_config())
        self.assertNotIn("probe", self.registry.loader_registry())
        # 隐藏而不是删除：条目还在，身份还留着，所以还能恢复
        self.assertIsNotNone(self.registry.get("probe"))
        self.assertIn("probe", self.registry.list_all(include_unavailable=True))

    def test_tripped_agent_refuses_to_serve(self):
        self._register()
        self._fail()
        with self.assertRaisesRegex(AgentLoadError, "tripped open"):
            self.registry.get_or_create("probe")

    def test_one_tripped_agent_does_not_affect_the_others(self):
        self._register()
        self.registry.register("other", {**META, "id": "other"}, factory=lambda: _Probe())
        self.registry.get_or_create("other")

        self._fail()

        self.assertNotIn("probe", self.registry.list_all())
        self.assertIn("other", self.registry.list_all())
        self.assertEqual(self.registry.get_or_create("other").run("q")["answer"], "ok")

    # ── 换代记账 ──────────────────────────────────────────────────────
    def test_failures_from_a_superseded_generation_are_discarded(self):
        """老实例失败不能记到刚换上来的新版本头上。"""
        self._register()
        stale_epoch = self.registry.epoch_of("probe")

        # 换代：重新注册同一个 id，epoch 必然不同
        self.registry.unregister("probe")
        self._register()
        self.assertNotEqual(self.registry.epoch_of("probe"), stale_epoch)

        for _ in range(CIRCUIT_FAILURE_THRESHOLD * 3):
            self.registry.record_failure("probe", "旧版本挂了", epoch=stale_epoch)

        health = self.registry.health_of("probe")
        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["consecutive_failures"], 0)

    def test_reload_resets_health_and_moves_the_epoch(self):
        """修好之后换上去，健康度清零，旧世代的失败不再上账。"""
        self.registry.agents_dir = None  # 用 register 手工建，reload 另测
        self._register()
        old_epoch = self.registry.epoch_of("probe")
        self._fail()
        self.assertEqual(self.registry.health_of("probe")["state"], "open")

        self.registry.unregister("probe")
        self._register()
        health = self.registry.health_of("probe")

        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["consecutive_failures"], 0)
        self.assertNotEqual(health["epoch"], old_epoch)

    # ── 恢复：冷却 + 半开探测 ─────────────────────────────────────────
    def test_cooldown_lets_a_probe_through_and_success_recovers(self):
        self._register()
        self._fail()
        registry_module.CIRCUIT_COOLDOWN_SECONDS = 0.0  # 冷却结束

        self.assertIn("probe", self.registry.list_all())  # 重新进候选
        instance = self.registry.get_or_create("probe")   # 放一个探测请求
        self.assertEqual(self.registry.health_of("probe")["state"], "half_open")

        self.registry.record_success("probe")
        self.assertEqual(self.registry.health_of("probe")["state"], "closed")
        self.assertEqual(instance.run("q")["answer"], "ok")

    def test_failed_probe_goes_straight_back_out_of_service(self):
        self._register()
        self._fail()
        registry_module.CIRCUIT_COOLDOWN_SECONDS = 0.0

        self.registry.get_or_create("probe")               # half_open 探测
        self.registry.record_failure("probe", "还是不行")   # 探测失败

        self.assertEqual(self.registry.health_of("probe")["state"], "open")

        registry_module.CIRCUIT_COOLDOWN_SECONDS = 1e9      # 冷却重新开始计时
        self.assertNotIn("probe", self.registry.list_all())
        with self.assertRaisesRegex(AgentLoadError, "tripped open"):
            self.registry.get_or_create("probe")

    def test_unavailable_construction_is_also_recoverable(self):
        """构造失败过去是死状态；现在冷却后可以重试。"""
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("第一次构造失败")
            return _Probe()

        self.registry.register("probe", dict(META), factory=flaky)
        with self.assertRaises(AgentLoadError):
            self.registry.get_or_create("probe")
        self.assertNotIn("probe", self.registry.list_all())
        with self.assertRaises(AgentLoadError):
            self.registry.get_or_create("probe")  # 冷却期内不重试

        registry_module.CIRCUIT_COOLDOWN_SECONDS = 0.0

        self.assertIsNotNone(self.registry.get_or_create("probe"))
        self.assertIn("probe", self.registry.list_all())

    # ── 接线：所有请求都从 adapter 过 ─────────────────────────────────
    def test_adapter_reports_outcomes_to_the_registry(self):
        adapter = adapt_agent(_Probe("ok"), agent_id="probe", registry=self.registry)
        self.registry.register("probe", dict(META), factory=lambda: _Probe("ok"))

        adapter.run("q")
        self.assertEqual(self.registry.health_of("probe")["consecutive_failures"], 0)

        failing = adapt_agent(_Probe("raise"), agent_id="probe", registry=self.registry)
        for _ in range(CIRCUIT_FAILURE_THRESHOLD):
            with self.assertRaises(RuntimeError):
                failing.run("q")

        self.assertEqual(self.registry.health_of("probe")["state"], "open")

    def test_adapter_treats_status_error_and_empty_output_as_hard_failures(self):
        self.registry.register("probe", dict(META), factory=lambda: _Probe("error"))
        adapt_agent(_Probe("error"), agent_id="probe", registry=self.registry).run("q")
        self.assertEqual(self.registry.health_of("probe")["consecutive_failures"], 1)

        self.registry.unregister("probe")
        self.registry.register("probe", dict(META), factory=lambda: _Probe("empty"))
        adapt_agent(_Probe("empty"), agent_id="probe", registry=self.registry).run("q")
        self.assertEqual(self.registry.health_of("probe")["consecutive_failures"], 1)

    def test_adapter_without_a_registry_still_works(self):
        """老调用方（只传实例和 id）不受影响。"""
        result = adapt_agent(_Probe("ok"), agent_id="probe").run("q")
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
