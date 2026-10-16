"""
工具层 —— NL2SQL Agent 可调用的工具。

工具清单:
  - get_table_schema  : 获取表结构（全量 or 单表）
  - execute_sql       : 执行 SQL 查询（只读）
  - search_column     : 在 Schema 中搜索列名
  - list_tables       : 列出所有表名

所有工具通过 core/tools.py 的 TOOL_REGISTRY 注册，
通过 OpenAI Function Calling 格式暴露给 LLM。
"""
from nl2sql_agent.tools.schema_tools import (
    register_schema_tools,
    get_table_schema,
    search_column,
    list_tables,
)
from nl2sql_agent.tools.sql_tools import (
    register_sql_tools,
    execute_sql,
)

__all__ = [
    "register_schema_tools", "register_sql_tools",
    "get_table_schema", "search_column", "list_tables",
    "execute_sql",
]
