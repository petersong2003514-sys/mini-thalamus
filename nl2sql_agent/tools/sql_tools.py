"""
SQL 执行工具 —— execute_sql

注册到 TOOL_REGISTRY，供 Agent 通过 Function Calling 调用。
包含安全检查（只允许 SELECT + 黑名单过滤）。
"""

from nl2sql_agent.core.tools import _register

# Database Connector 引用（由 register_sql_tools() 注入）
# NOTE: 模块级单例 _db_connector 非线程安全——多线程并发执行 SQL 时共享同一连接
# 状态。SQLite 连接本身非线程安全，生产环境建议使用连接池或 threading.local()。
_db_connector = None


def register_sql_tools(db_connector):
    """
    将 SQL 工具注册到全局 TOOL_REGISTRY。

    Args:
        db_connector: IDatabaseConnector 实例
    """
    global _db_connector
    _db_connector = db_connector

    _register(
        name="execute_sql",
        description=(
            "在数据库上执行 SQL 查询。只支持 SELECT 语句。"
            "返回查询结果的列名和数据行（最多 50 行）。"
            "如果 SQL 有语法错误，返回错误信息。"
            "⚠️ 只能执行 SELECT 查询，不支持 INSERT/UPDATE/DELETE/DROP 等修改操作。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "要执行的 SQL SELECT 查询语句",
                }
            },
            "required": ["sql"],
        },
        fn=execute_sql,
    )


def execute_sql(sql: str) -> str:
    """
    执行 SQL 查询 —— 供 Function Calling 调用。

    返回格式化后的结果字符串，方便 LLM 理解和呈现。
    """
    if _db_connector is None:
        return "错误: 数据库连接未初始化"

    from nl2sql_agent.schemas import SQLResult
    result = _db_connector.execute(sql)

    if not result.success:
        return f"SQL 执行失败 ❌\n错误: {result.error}"

    if result.row_count == 0:
        return f"SQL 执行成功 ✅\n查询返回 0 行结果。\n字段: {', '.join(result.columns)}"

    # 格式化结果
    lines = [
        f"SQL 执行成功 ✅",
        f"返回 {result.row_count} 行, {len(result.columns)} 列",
        f"耗时: {result.execution_time_ms:.1f}ms",
        "",
        f"字段: {', '.join(result.columns)}",
        "数据:",
    ]

    for i, row in enumerate(result.rows):
        lines.append(f"  [{i+1}] " + " | ".join(str(v) for v in row))

    if result.row_count > 50:
        lines.append(f"  ... (仅显示前 50 行)")

    return "\n".join(lines)
