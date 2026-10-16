"""语义后置条件检查器 —— 机器可检查的"结果符合查询语义意图"断言。

对应 harness 的"能跑 ≠ 正确"：SQL 执行成功、语法正确、质量检查通过，
都不等于结果满足用户意图。这个检查器在 SQL 执行成功后、接受结果前运行，
用确定性规则（无 LLM、无随机）把"语义意图"变成可检查的断言：

  R1 问题含聚合意图（统计/各/数量/汇总）→ SQL 必须有 GROUP BY 或聚合函数
  R2 问题含排序意图（最/前N/Top）      → SQL 必须有 ORDER BY
  R3 问题在问计数（多少/总数）          → SQL 必须是聚合，而不是 SELECT * 明细

设计原则与 harness 一致：
- 机器可检查：规则确定，同一输入必得同一判定，可测试、可审计；
- 与"预校验"分层：预校验管"能不能跑"（执行前），这里管"跑出来对不对"（执行后）；
- 失败是"可恢复"的：违规信息回喂模型修正，在固定轮次预算内收敛。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── 问题侧意图关键词（中文）──
_AGGREGATION_TERMS = (
    "统计", "汇总", "总额", "总销", "总量", "数量", "销量",
    "每个", "各", "按月", "按区域", "按部门", "按年份", "按月份",
    "平均", "总数", "合计", "共",
)
_RANKING_TERMS = (
    "最高", "最低", "最多", "最少", "最大", "最小",
    "最贵", "最便宜", "最畅销", "最热", "最新",
    "排名", "排序", "排行",
    "top", "top5", "top10", "top20", "top50",
    "前5", "前10", "前20", "前50",
)
_COUNT_TERMS = ("多少", "几个", "总数", "个数", "数量", "共计")

# ── SQL 侧特征（正则）──
_AGGREGATE_FUNC = re.compile(r"\b(SUM|COUNT|AVG|MAX|MIN)\s*\(", re.IGNORECASE)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_SELECT_ALL = re.compile(r"SELECT\s+\*", re.IGNORECASE)
_RANKING_PATTERN = re.compile(r"(?:前|top)\s*\d+", re.IGNORECASE)


@dataclass(frozen=True)
class PostconditionResult:
    """一次语义后置条件的判定结果。"""

    ok: bool
    reason: str = ""
    rule: str = ""


class SemanticPostconditionChecker:
    """确定性、机器可检查的语义后置条件检查器。

    check(question, sql, result) -> PostconditionResult
      - question: 原始自然语言问题（消歧后）
      - sql:      最终执行的 SQL 文本
      - result:   执行结果（SQLResult，至少用到 columns / row_count）
    """

    def check(self, question: str, sql: str, result) -> PostconditionResult:
        question = question or ""
        sql = sql or ""

        # R1: 聚合意图 → 必须聚合
        if self._has_aggregation_intent(question):
            if not (_AGGREGATE_FUNC.search(sql) or _GROUP_BY.search(sql)):
                return PostconditionResult(
                    ok=False,
                    rule="R1_aggregation_required",
                    reason=(
                        "语义后置条件 R1：问题包含聚合意图（统计/各/数量/汇总），"
                        "但生成的 SQL 没有 GROUP BY 或聚合函数（SUM/COUNT/AVG/MAX/MIN）。"
                        "请改为按目标维度分组的聚合查询，例如 GROUP BY <维度列> + SUM(<数值列>)。"
                    ),
                )

        # R2: 排序意图 → 必须 ORDER BY
        if self._has_ranking_intent(question):
            if not _ORDER_BY.search(sql):
                return PostconditionResult(
                    ok=False,
                    rule="R2_ranking_required",
                    reason=(
                        "语义后置条件 R2：问题包含排序意图（最/前N/Top），"
                        "但生成的 SQL 没有 ORDER BY。请添加排序，"
                        "例如 ORDER BY <数值列> DESC 并配合 LIMIT。"
                    ),
                )

        # R3: 计数意图 → 必须聚合计数，而非明细
        if self._has_count_intent(question):
            if _SELECT_ALL.search(sql) and not _AGGREGATE_FUNC.search(sql):
                return PostconditionResult(
                    ok=False,
                    rule="R3_count_required",
                    reason=(
                        "语义后置条件 R3：问题在问数量/总数，"
                        "但生成的 SQL 是 SELECT * 明细查询。请改为 COUNT(*) 或按维度分组后计数。"
                    ),
                )

        return PostconditionResult(ok=True)

    # ── 意图检测（确定性关键词/正则）──

    @staticmethod
    def _has_aggregation_intent(question: str) -> bool:
        return any(term in question for term in _AGGREGATION_TERMS)

    @staticmethod
    def _has_ranking_intent(question: str) -> bool:
        if any(term in question for term in _RANKING_TERMS):
            return True
        return bool(_RANKING_PATTERN.search(question))

    @staticmethod
    def _has_count_intent(question: str) -> bool:
        return any(term in question for term in _COUNT_TERMS)


__all__ = [
    "PostconditionResult",
    "SemanticPostconditionChecker",
]
