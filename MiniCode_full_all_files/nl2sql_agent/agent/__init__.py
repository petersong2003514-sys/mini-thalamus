"""
Agent 模块 —— NL2SQL Agent 核心循环 + Prompt 模板 + Few-shot 示例池。

组件:
  - nl2sql_agent.py: 主 Agent 循环（Phase 6 实现）
  - prompts.py: System Prompt 模板 + Few-shot 格式化
  - fewshot_examples.py: 15 条带标签的示例池
"""
from nl2sql_agent.agent.nl2sql_agent import NL2SQLAgent, run_nl2sql
from nl2sql_agent.agent.prompts import (
    NL2SQL_SYSTEM_PROMPT,
    CORRECTION_PROMPT,
    format_prompt,
)
from nl2sql_agent.agent.fewshot_examples import (
    FEWSHOT_POOL,
    create_default_pool,
)

__all__ = [
    "NL2SQLAgent", "run_nl2sql",
    "NL2SQL_SYSTEM_PROMPT", "CORRECTION_PROMPT", "format_prompt",
    "FEWSHOT_POOL", "create_default_pool",
]
