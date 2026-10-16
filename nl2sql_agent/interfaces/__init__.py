"""
NL2SQL Agent 接口层 —— 所有抽象接口和关键实现。

8 个核心接口:
  1. ISchemaProvider      — Schema 提供 (Full/Keyword/Hybrid 三策略)
  2. IDatabaseConnector   — 数据库连接 (安全只读 + 多库兼容)
  3. IDifficultyClassifier — 难度分级 (LLM主分类 + 规则兜底)
  4. IFewShotSelector     — Few-shot 选择 (标签匹配 + 难度自适应)
  5. ISQLValidator        — SQL 预校验 (三层防御: 硬规则→白名单→LLM)
  6. IErrorEnhancer       — 错误增强 (方法A映射表 + 方法C CoT)
  7. IStopJudge           — 停止判断 (硬规则预判 + LLM语义兜底)
  8. ITaskDecomposer      — 任务拆解 (Plan-Execute 复杂查询兜底)
"""

from nl2sql_agent.interfaces.schema_provider import (
    ISchemaProvider,
    FullSchemaProvider,
    KeywordSchemaProvider,
    HybridSchemaProvider,
)
from nl2sql_agent.interfaces.database_connector import (
    IDatabaseConnector,
    SQLiteConnector,
)
from nl2sql_agent.interfaces.difficulty_classifier import (
    IDifficultyClassifier,
    LLMDifficultyClassifier,
    RuleBasedDifficultyClassifier,
)
from nl2sql_agent.interfaces.fewshot_selector import (
    IFewShotSelector,
    TagBasedFewShotSelector,
)
from nl2sql_agent.interfaces.sql_validator import (
    ISQLValidator,
    LayeredSQLValidator,
)
from nl2sql_agent.interfaces.error_enhancer import (
    IErrorEnhancer,
    HybridErrorEnhancer,
)
from nl2sql_agent.interfaces.stop_judge import (
    IStopJudge,
    HybridStopJudge,
)
from nl2sql_agent.interfaces.task_decomposer import (
    ITaskDecomposer,
    LLMTaskDecomposer,
    DecomposedPlan,
    SubTask,
)

__all__ = [
    # Schema (3策略)
    "ISchemaProvider",
    "FullSchemaProvider",
    "KeywordSchemaProvider",
    "HybridSchemaProvider",
    # Database
    "IDatabaseConnector",
    "SQLiteConnector",
    # Difficulty
    "IDifficultyClassifier",
    "LLMDifficultyClassifier",
    "RuleBasedDifficultyClassifier",
    # Few-shot
    "IFewShotSelector",
    "TagBasedFewShotSelector",
    # Validation
    "ISQLValidator",
    "LayeredSQLValidator",
    # Error
    "IErrorEnhancer",
    "HybridErrorEnhancer",
    # Stop
    "IStopJudge",
    "HybridStopJudge",
    # Decompose
    "ITaskDecomposer",
    "LLMTaskDecomposer",
    "DecomposedPlan",
    "SubTask",
]
