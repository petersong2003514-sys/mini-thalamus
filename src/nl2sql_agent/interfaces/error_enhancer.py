"""
接口 6: IErrorEnhancer —— 错误增强器（难点E）

SQL 执行失败后，对原始报错进行"翻译增强"，帮助 LLM 更有效地修正 SQL。

设计依据（你的决策）:
  主策略（方法A）：原始错误直传 + 预定义错误映射表补全
    - 大部分错误 DeepSeek 能自己理解并修正
    - 映射表补全 5-8 种常见错误的修复提示

  兜底策略（方法C）：多轮修正失败的顽固错误 → 切换 CoT 两步推理
    - Round 1 失败 → 方法A（错误直传 + 映射表）
    - Round 2 同类型失败 → 继续方法A
    - Round 3 同类型失败 → 切换方法C（让 LLM 先解释错误再修正）
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import SQLResult, ErrorType


class IErrorEnhancer(ABC):
    """
    错误增强器抽象接口。

    输入: 原始 SQLResult（含 error / error_type）
    输出: 增强后的错误描述字符串（直接注入纠错 Prompt）
    """

    @abstractmethod
    def enhance(self, sql: str, result: SQLResult,
                round_idx: int, last_error_type: ErrorType | None = None) -> str:
        """
        根据当前轮次和错误历史，选择合适的增强策略。

        Args:
            sql: 当前 SQL
            result: 执行结果（含错误信息）
            round_idx: 当前轮次（1/2/3）
            last_error_type: 上一轮的 ErrorType（用于判断是否顽固错误）

        Returns:
            增强后的错误描述字符串
        """
        ...

    @abstractmethod
    def should_use_cot(self, round_idx: int,
                       current_error: ErrorType,
                       last_error: ErrorType | None) -> bool:
        """
        判断是否应切换到 CoT 两步推理模式（方法C）。

        触发条件: 同一 ErrorType 在连续轮次中重复出现 ≥ 2 次
        """
        ...


# ═══════════════════════════════════════════════════════════════
#  HybridErrorEnhancer — 方法A + 方法C 混合策略
# ═══════════════════════════════════════════════════════════════

class HybridErrorEnhancer(IErrorEnhancer):
    """
    混合错误增强器。

    方法A（默认）: 原始错误 + 映射表提示
      - 映射表预定义 5-8 种常见错误的修复提示
      - 对 DeepSeek 等强模型，大部分错误只需原始错误信息

    方法C（兜底）: CoT 两步推理
      - 第一步: "请分析这条 SQL 为什么报错？"
      - 第二步: "基于你的分析，请输出修正后的 SQL"
      - Token 消耗翻倍但在顽固错误上有明显收益
    """

    # ── 错误映射表（可扩展） ──
    ERROR_HINTS = {
        ErrorType.COLUMN_NOT_FOUND: (
            "可能原因: 列名拼写错误或列名使用了 LLM 训练数据中的常见名而非 Schema 中的实际名。"
            "修正建议: 检查 SQL 中的列名是否与 Schema DDL 中的列名严格一致。"
        ),
        ErrorType.SYNTAX_ERROR: (
            "可能原因: SQL 关键字顺序错误(如 LIMIT 放在 ORDER BY 前)、缺少逗号、"
            "多了一个逗号、引号不匹配。"
            "修正建议: 检查 SQLite 语法——ORDER BY 在 LIMIT 之前，"
            "GROUP BY 在 HAVING 之前。"
        ),
        ErrorType.AMBIGUOUS_COLUMN: (
            "可能原因: 多表 JOIN 时列名相同但未加表前缀。"
            "修正建议: 为每个列加上表别名前缀，如 orders.user_id 而非 user_id。"
        ),
        ErrorType.TABLE_NOT_FOUND: (
            "可能原因: 表名拼写错误或表不存在。"
            "修正建议: 检查表名是否与 Schema 中定义的一致，注意大小写。"
        ),
        ErrorType.TYPE_MISMATCH: (
            "可能原因: 在数值列上使用了字符串函数，或比较运算符两侧类型不一致。"
            "修正建议: 检查 WHERE 条件中比较的类型是否一致，"
            "必要时使用 CAST(column AS type)。"
        ),
        ErrorType.OTHER: (
            "通用修正建议: 逐子句检查 SQL 语法，确保每个关键字、括号、引号正确。"
        ),
    }

    def __init__(self):
        # NOTE: _same_error_count and _last_error_type are instance state that is
        # not thread-safe. They should only be used in single-threaded agent loops.
        # For multi-threaded use, wrap this instance per-thread or add locking.
        self._last_error_type: ErrorType | None = None
        self._same_error_count: int = 0

    def enhance(self, sql: str, result: SQLResult,
                round_idx: int, last_error_type: ErrorType | None = None) -> str:
        error_type = result.error_type or ErrorType.OTHER

        # 追踪连续相同错误
        if error_type == last_error_type:
            self._same_error_count += 1
        else:
            self._same_error_count = 0
        self._last_error_type = error_type

        # ── 判断是否切换 CoT ──
        if self.should_use_cot(round_idx, error_type, last_error_type):
            return self._enhance_cot(sql, result)

        # ── 默认: 方法A（原始错误 + 映射表提示）──
        return self._enhance_direct(result)

    def _enhance_direct(self, result: SQLResult) -> str:
        """方法A: 原始错误 + 映射表补充提示"""
        error_type = result.error_type or ErrorType.OTHER
        hint = self.ERROR_HINTS.get(error_type, self.ERROR_HINTS[ErrorType.OTHER])

        return f"""SQL 执行报错: {result.error}

{hint}

请修正 SQL。只输出修正后的 SQL，不要解释。"""

    def _enhance_cot(self, sql: str, result: SQLResult) -> str:
        """方法C: CoT 两步推理模式"""
        return f"""你的 SQL 在连续多轮修正后仍然失败。请仔细分析错误原因。

## 当前 SQL
{sql}

## 执行报错
{result.error}

## 请按以下步骤输出:
Step 1 — 分析: 这条 SQL 为什么会报错？根源问题是什么？
Step 2 — 修正: 基于你的分析，输出修正后的完整 SQL。

格式:
[ANALYSIS]
你的分析...
[/ANALYSIS]
[SQL]
修正后的完整 SQL
[/SQL]"""

    def should_use_cot(self, round_idx: int,
                       current_error: ErrorType,
                       last_error: ErrorType | None) -> bool:
        """同一 ErrorType 连续出现 ≥1 次（即第2轮开始）→ 切换 CoT"""
        if round_idx < 2:
            return False
        if current_error is None:
            return False
        return current_error == last_error and self._same_error_count >= 1

    def reset(self):
        """重置错误追踪状态（每次新查询时调用）"""
        self._last_error_type = None
        self._same_error_count = 0
