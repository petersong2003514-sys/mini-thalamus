"""
全局配置 —— API Key、模型选择、路径、阈值等。
"""

import os
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════
#  数据库配置
# ═══════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ecommerce.db")
DB_TYPE = "sqlite"  # "sqlite" | "mysql" | "postgresql"
DB_READONLY = True  # 只读模式打开

# ═══════════════════════════════════════════════════════════════
#  LLM 配置 ── 支持两档模型：主模型(大)+ 辅助模型(轻量)
# ═══════════════════════════════════════════════════════════════

# 主模型 —— SQL 生成 + 纠错（需要强推理能力）
PRIMARY_MODEL = os.environ.get("NL2SQL_MODEL", "deepseek-chat")
PRIMARY_BASE_URL = os.environ.get(
    "NL2SQL_BASE_URL", "https://api.deepseek.com/v1"
)
PRIMARY_API_KEY = os.environ.get("NL2SQL_API_KEY", "")

# 辅助轻量模型 —— 难度分级 / 预校验 / 停止判断（快速、便宜）
LIGHT_MODEL = os.environ.get("NL2SQL_LIGHT_MODEL", PRIMARY_MODEL)  # 默认复用主模型
LIGHT_BASE_URL = os.environ.get("NL2SQL_LIGHT_BASE_URL", PRIMARY_BASE_URL)
LIGHT_API_KEY = os.environ.get("NL2SQL_LIGHT_API_KEY", PRIMARY_API_KEY)

# ═══════════════════════════════════════════════════════════════
#  Agent 循环配置
# ═══════════════════════════════════════════════════════════════

MAX_ROUNDS = 3                # 最大重试轮次
TEMPERATURE = 0.0             # SQL 生成用 0 温度（确定性输出）
MAX_RESULT_ROWS = 50          # 单次返回最大行数
ROW_COUNT_WARN_THRESHOLD = 1000  # 结果行数超过此值触发质量警告

# ═══════════════════════════════════════════════════════════════
#  Schema Provider 配置
# ═══════════════════════════════════════════════════════════════

SCHEMA_STRATEGY = "full"      # "full" | "keyword" | "hybrid"
SCHEMA_TOP_K_TABLES = 5       # 检索模式下的最大返回表数
SCHEMA_TOP_K_COLUMNS = 15     # 检索模式下的最大返回列数

# ═══════════════════════════════════════════════════════════════
#  Few-shot 配置
# ═══════════════════════════════════════════════════════════════

FEWSHOT_MAX_EXAMPLES = 3      # 每次注入的示例数
FEWSHOT_POOL_SIZE = 15        # 示例池总量
DIFFICULTY_CLASSIFIER_ENABLED = True  # 是否启用难度分级（难点B）

# ═══════════════════════════════════════════════════════════════
#  枚举定义
# ═══════════════════════════════════════════════════════════════

class Difficulty(Enum):
    """问题难度分级 —— 由轻量 LLM 判定（难点B）"""
    EASY = "easy"          # 单表 + 简单条件
    MEDIUM = "medium"      # 单表聚合 or 2表JOIN
    HARD = "hard"          # 3表+ JOIN or 嵌套子查询


class StopReason(Enum):
    """Agent 停止原因"""
    SUCCESS = "success"                     # ✅ 执行成功 + 质量通过
    SUCCESS_WITH_WARNING = "success_warn"   # ⚠️  执行成功但有质量问题
    ERROR_UNRECOVERABLE = "error_unrecov"   # ❌ 不可恢复错误
    MAX_ROUNDS = "max_rounds"               # ⏱️  达到最大轮次
    STAGNATION = "stagnation"               # 🔄 相同错误重复
    NO_PROGRESS = "no_progress"             # 🛑 SQL 完全没变


class ErrorType(Enum):
    """SQL 错误分类 —— 用于 STAGNATION 判断"""
    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    SYNTAX_ERROR = "syntax_error"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    TYPE_MISMATCH = "type_mismatch"
    PERMISSION_DENIED = "permission_denied"
    OTHER = "other"


class SQLSkill(Enum):
    """SQL 技能标签 —— 用于 Few-shot 匹配"""
    SINGLE_TABLE = "single_table"
    JOIN = "join"
    AGGREGATION = "aggregation"
    GROUP_BY = "group_by"
    ORDER_BY = "order_by"
    SUBQUERY = "subquery"
    HAVING = "having"
    DATE_FILTER = "date_filter"
    LIKE = "like"
    LIMIT = "limit"


# ═══════════════════════════════════════════════════════════════
#  安全配置
# ═══════════════════════════════════════════════════════════════

# SQL 黑名单关键字 —— 只拦截真正危险的操作
# 原则: 防注入/泄露/破坏，不禁用标准只读功能
# WITH (CTE) 不在黑名单中，它是标准只读语法
SQL_BLACKLIST = (
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER",
    "CREATE", "TRUNCATE", "REPLACE", "GRANT", "REVOKE",
    "ATTACH", "DETACH",
)

# 只允许的 SQL 操作
ALLOWED_OPERATIONS = ("SELECT", "WITH")
