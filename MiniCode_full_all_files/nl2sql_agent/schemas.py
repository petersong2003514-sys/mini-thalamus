"""
核心数据结构 —— 系统中所有模块使用的 dataclass 定义。
这些是和 Phase 5 接口配套的数据载体。
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from nl2sql_agent.config import Difficulty, StopReason, ErrorType, SQLSkill


# ═══════════════════════════════════════════════════════════════
#  数据库相关
# ═══════════════════════════════════════════════════════════════

@dataclass
class TableMeta:
    """单张表的元数据"""
    name: str                                    # 表名，如 "products"
    description: str = ""                        # 中文注释，如 "商品信息表"
    columns: list["ColumnMeta"] = field(default_factory=list)
    row_count: int = 0                           # 行数估计


@dataclass
class ColumnMeta:
    """单列的元数据"""
    name: str                                    # 列名，如 "product_name"
    data_type: str = ""                          # 类型，如 "TEXT" / "INTEGER"
    description: str = ""                        # 中文注释，如 "商品名称"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_ref: str = ""                        # 外键引用，如 "products.product_id"
    sample_values: list[str] = field(default_factory=list)  # 示例值，如 ["iPhone 15", "MacBook"]
    embedding: Optional[list[float]] = None      # 列注释的向量（Embedding 模式用，当前预留）


@dataclass
class SchemaInfo:
    """完整 Schema 信息"""
    db_type: str = "sqlite"
    tables: list[TableMeta] = field(default_factory=list)

    def to_ddl(self) -> str:
        """将 SchemaInfo 转化为 DDL 文本（含列注释），供 Prompt 注入"""
        lines = [f"-- 数据库类型: {self.db_type}", ""]

        # 表关系说明
        fk_lines = []
        for table in self.tables:
            for col in table.columns:
                if col.is_foreign_key and col.foreign_ref:
                    fk_lines.append(
                        f"--   {table.name}.{col.name} → {col.foreign_ref}"
                    )
        if fk_lines:
            lines.append("-- 表关系（外键）:")
            lines.extend(fk_lines)
            lines.append("")

        # 每张表的 DDL
        for table in self.tables:
            lines.append(self._table_ddl(table))
            lines.append("")

        return "\n".join(lines)

    def get_column_whitelist(self) -> dict[str, set[str]]:
        """返回 {tablename: {col1, col2, ...}} 的白名单，供预校验使用"""
        whitelist: dict[str, set[str]] = {}
        for table in self.tables:
            whitelist[table.name] = {col.name for col in table.columns}
        return whitelist

    def _table_ddl(self, table: TableMeta) -> str:
        """将单张 TableMeta 转为含注释的 DDL"""
        parts = []

        # 表注释
        if table.description:
            parts.append(f"-- {table.description}")
        parts.append(f"-- 行数: {table.row_count}")

        col_defs = []
        for c in table.columns:
            col_str = f"  {c.name} {c.data_type}"
            extras = []
            if c.is_primary_key:
                extras.append("PRIMARY KEY")
            if c.is_foreign_key and c.foreign_ref:
                extras.append(f"REFERENCES {c.foreign_ref}")

            if extras:
                col_str += " " + " ".join(extras)

            # 列注释 + 示例值
            comment_parts = []
            if c.description:
                comment_parts.append(c.description)
            if c.sample_values:
                comment_parts.append(f"示例: {', '.join(c.sample_values[:3])}")

            if comment_parts:
                col_str += f"  -- {'; '.join(comment_parts)}"

            col_defs.append(col_str)

        parts.append(f"CREATE TABLE {table.name} (")
        parts.append(",\n".join(col_defs))
        parts.append(");")

        # 跨表提示
        hints = self._usage_hints(table.name)
        if hints:
            parts.append(f"-- [提示] {hints}")

        return "\n".join(parts)

    def _usage_hints(self, table_name: str) -> str:
        """返回表的跨表使用提示"""
        hints = {
            "products": "商品销量/热度需通过 order_items 统计; 用户行为通过 user_behaviors 查询",
            "orders": "订单金额统计用 total_amount; 关联商品通过 order_items; 关联用户通过 users",
            "order_items": "商品销量 = COUNT(*)/SUM(quantity); 销售额 = SUM(subtotal); 需 JOIN orders 获取用户和日期",
            "users": "用户购买行为通过 orders + order_items 关联查询",
            "user_behaviors": "behavior_type: view=浏览, cart=加购, fav=收藏, search=搜索; 关联商品通过 products",
        }
        return hints.get(table_name, "")


# ═══════════════════════════════════════════════════════════════
#  SQL 执行相关
# ═══════════════════════════════════════════════════════════════

@dataclass
class SQLResult:
    """SQL 执行结果"""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: str = ""
    error_type: Optional[ErrorType] = None


@dataclass
class ValidationResult:
    """SQL 预校验结果（难点C —— LLM 智能校验）"""
    is_valid: bool
    error: str = ""
    error_type: Optional[ErrorType] = None
    suggestion: str = ""                          # 修复建议，如 "列 '商业品' 不存在，是否指 products 表？"
    corrected_columns: dict[str, str] = field(default_factory=dict)  # {"商业品": "products"}


# ═══════════════════════════════════════════════════════════════
#  Schema 检索相关
# ═══════════════════════════════════════════════════════════════

@dataclass
class SchemaRetrievalResult:
    """Schema 检索结果"""
    ddl_text: str                                # 拼接好的 DDL 字符串
    matched_tables: list[str] = field(default_factory=list)
    matched_columns: list[str] = field(default_factory=list)
    retrieval_method: str = "full"               # "full" | "keyword" | "hybrid"
    retrieval_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Few-shot 相关
# ═══════════════════════════════════════════════════════════════

@dataclass
class FewShotExample:
    """单条 Few-shot 示例"""
    question: str
    sql: str
    difficulty: Difficulty = Difficulty.MEDIUM
    skills: list[SQLSkill] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)   # 涉及的表名
    description: str = ""                              # 为什么选这条示例


# ═══════════════════════════════════════════════════════════════
#  难度分级相关（难点B）
# ═══════════════════════════════════════════════════════════════

@dataclass
class DifficultyClassification:
    """轻量 LLM 对问题的难度分级结果（难点B）"""
    difficulty: Difficulty
    skills: list[SQLSkill] = field(default_factory=list)
    tables_needed: list[str] = field(default_factory=list)
    reasoning: str = ""                            # LLM 的推理过程（可解释性）


# ═══════════════════════════════════════════════════════════════
#  停止判断相关（难点F）
# ═══════════════════════════════════════════════════════════════

@dataclass
class StopJudgment:
    """轻量 LLM 对是否应停止的判断（难点F）"""
    should_stop: bool
    stop_reason: StopReason
    reasoning: str = ""                            # 判断依据
    is_recoverable: bool = True                    # 错误是否可恢复


# ═══════════════════════════════════════════════════════════════
#  Agent 循环级别
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoundTrace:
    """单轮 Agent 执行的追踪记录"""
    round_idx: int
    sql: str = ""
    result: Optional[SQLResult] = None
    validation: Optional[ValidationResult] = None
    error_enhanced: str = ""                       # 增强后的错误信息（难点E）
    stop_judgment: Optional[StopJudgment] = None


@dataclass
class AgentResult:
    """Agent 最终返回结果"""
    success: bool
    question: str
    final_sql: str = ""
    result: Optional[SQLResult] = None
    stop_reason: StopReason = StopReason.SUCCESS
    rounds_attempted: int = 0
    difficulty: Optional[Difficulty] = None        # 难度分级结果
    trace: list[RoundTrace] = field(default_factory=list)
    total_time_ms: float = 0.0
    total_tokens: int = 0                          # 消耗的总 Token 数
    nl_summary: str = ""                           # 自然语言结果总结


# ═══════════════════════════════════════════════════════════════
#  批量评测相关
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    """单条测试用例"""
    id: str
    question: str
    expected_sql_pattern: str = ""                 # 期望的正则模式
    expected_result_hint: dict = field(default_factory=dict)  # {"category": "电子产品"}
    difficulty: Difficulty = Difficulty.MEDIUM
    tables: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """批量评测报告"""
    total: int
    execution_success: int
    result_match: int
    avg_rounds: float
    avg_tokens: int
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_stop_reason: dict[str, int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
