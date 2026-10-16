"""语义后置条件检查器单元测试 —— 验证"能跑 ≠ 正确"的规则层与 agent 接线。"""

import sys
import unittest
from pathlib import Path


MINICODE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = MINICODE_ROOT.parent


class SemanticPostconditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._added_bundle_root = str(BUNDLE_ROOT) not in sys.path
        if cls._added_bundle_root:
            sys.path.insert(0, str(BUNDLE_ROOT))

    @classmethod
    def tearDownClass(cls):
        if cls._added_bundle_root:
            sys.path.remove(str(BUNDLE_ROOT))

    def setUp(self):
        from nl2sql_agent.core.semantic_postcondition import (
            SemanticPostconditionChecker,
        )

        self.checker = SemanticPostconditionChecker()

    def _result(self, columns=("sales",)):
        from types import SimpleNamespace

        return SimpleNamespace(columns=list(columns), row_count=5)

    # ── R1: 聚合意图必须聚合 ──
    def test_r1_aggregation_required_fires(self):
        res = self.checker.check(
            "统计各区域销售额", "SELECT * FROM orders", self._result()
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.rule, "R1_aggregation_required")
        self.assertIn("聚合", res.reason)

    def test_r1_aggregation_group_by_passes(self):
        res = self.checker.check(
            "统计各区域销售额",
            "SELECT region, SUM(total_amount) FROM orders GROUP BY region",
            self._result(columns=("region", "sum")),
        )
        self.assertTrue(res.ok)

    def test_r1_aggregation_sum_passes(self):
        res = self.checker.check(
            "各区域销售额是多少",
            "SELECT region, SUM(total_amount) FROM orders GROUP BY region",
            self._result(),
        )
        self.assertTrue(res.ok)

    # ── R2: 排序意图必须 ORDER BY ──
    def test_r2_ranking_required_fires(self):
        res = self.checker.check(
            "销售额最高的商品是什么", "SELECT name FROM products", self._result()
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.rule, "R2_ranking_required")

    def test_r2_ranking_order_by_passes(self):
        res = self.checker.check(
            "销售额最高的商品是什么",
            "SELECT name FROM products ORDER BY total_amount DESC LIMIT 1",
            self._result(),
        )
        self.assertTrue(res.ok)

    def test_r2_ranking_topn_pattern_fires(self):
        res = self.checker.check(
            "销量前10的商品", "SELECT name FROM products", self._result()
        )
        self.assertFalse(res.ok)

    # ── R3: 计数意图必须聚合计数 ──
    # 注意：用纯计数词"几个"（"总数"会同时命中聚合意图，R1 先触发）
    def test_r3_count_select_star_fires(self):
        res = self.checker.check(
            "订单有几个", "SELECT * FROM orders", self._result()
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.rule, "R3_count_required")

    def test_r3_count_count_star_passes(self):
        res = self.checker.check(
            "订单有几个", "SELECT COUNT(*) FROM orders", self._result()
        )
        self.assertTrue(res.ok)

    # ── 无意图关键词 → 放行 ──
    def test_plain_query_passes(self):
        res = self.checker.check(
            "查询上个月的订单明细",
            "SELECT order_id, total_amount FROM orders WHERE order_date >= '2026-07-01'",
            self._result(),
        )
        self.assertTrue(res.ok)

    # ── agent 接线 ──
    def test_agent_default_checker_enabled(self):
        from nl2sql_agent.agent.nl2sql_agent import NL2SQLAgent

        agent = NL2SQLAgent(None, None, None)  # 默认开启规则版
        res = agent._check_semantic_postcondition(
            "统计各区域销售额", "SELECT * FROM orders", self._result()
        )
        self.assertIsNotNone(res)
        self.assertFalse(res.ok)

    def test_agent_disabled_when_false(self):
        from nl2sql_agent.agent.nl2sql_agent import NL2SQLAgent

        agent = NL2SQLAgent(None, None, None, semantic_postcondition=False)
        res = agent._check_semantic_postcondition(
            "统计各区域销售额", "SELECT * FROM orders", self._result()
        )
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
