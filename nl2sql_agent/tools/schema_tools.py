"""
Schema 相关工具 —— get_table_schema / search_column / list_tables

注册到 TOOL_REGISTRY，供 Agent 通过 Function Calling 调用。
"""

from nl2sql_agent.core.tools import _register

# Schema Provider 引用（由 register_schema_tools() 注入）
# NOTE: 模块级单例 _schema_provider 非线程安全——若多个 Agent 并发初始化
# 或运行时切换 SchemaProvider，会出现竞态。生产环境建议用 threading.local() 或
# 将 provider 作为闭包捕获在注册的 fn 中。
_schema_provider = None


def register_schema_tools(schema_provider):
    """
    将 Schema 工具注册到全局 TOOL_REGISTRY。

    Args:
        schema_provider: ISchemaProvider 实例
    """
    global _schema_provider
    _schema_provider = schema_provider

    # 清理 core/tools.py 自带的旧 playground 工具
    from nl2sql_agent.core.tools import TOOL_REGISTRY
    for old_tool in ["calculator", "knowledge_base", "search"]:
        TOOL_REGISTRY.pop(old_tool, None)

    _register(
        name="get_table_schema",
        description=(
            "获取数据库表结构信息。返回指定表的 CREATE TABLE DDL 语句，"
            "包括列名、数据类型、约束等。如果不指定表名则返回所有表的概览。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "表名。填 'all' 获取所有表概览，或指定具体表名如 'products'。",
                }
            },
            "required": ["table_name"],
        },
        fn=get_table_schema,
    )

    _register(
        name="search_column",
        description=(
            "按关键词搜索数据库中的列名。当你需要找到存储特定信息的列时使用。"
            "例如搜索 '价格' 可能返回 products.price / order_items.unit_price 等。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，如 '价格'、'用户'、'日期'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回前 K 个结果，默认 10",
                    "default": 10,
                },
            },
            "required": ["keyword"],
        },
        fn=search_column,
    )

    _register(
        name="list_tables",
        description=(
            "列出数据库中所有表的名称和简要描述。"
            "当不确定有哪些可用表时使用此工具。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        fn=list_tables,
    )


def get_table_schema(table_name: str) -> str:
    """获取表结构 —— 供 Function Calling 调用"""
    if _schema_provider is None:
        return "错误: Schema Provider 未初始化"

    if table_name.lower() == "all":
        tables = _schema_provider.get_all_tables()
        lines = [f"数据库共 {len(tables)} 张表:\n"]
        for t in tables:
            ddl = _schema_provider.get_table_ddl(t)
            lines.append(ddl)
            lines.append("")
        return "\n".join(lines)
    else:
        return _schema_provider.get_table_ddl(table_name)


def search_column(keyword: str, top_k: int = 10) -> str:
    """搜索列名 —— 供 Function Calling 调用"""
    if _schema_provider is None:
        return "错误: Schema Provider 未初始化"

    results = _schema_provider.search_columns(keyword, top_k)
    if not results:
        return f"未找到与 '{keyword}' 相关的列"

    lines = [f"找到 {len(results)} 个与 '{keyword}' 相关的列:\n"]
    for col in results:
        lines.append(f"  • {col.name} ({col.data_type}) — {col.description}")
    return "\n".join(lines)


def list_tables() -> str:
    """列出所有表 —— 供 Function Calling 调用"""
    if _schema_provider is None:
        return "错误: Schema Provider 未初始化"

    tables = _schema_provider.get_all_tables()
    lines = [f"数据库共 {len(tables)} 张表:\n"]
    for t in tables:
        lines.append(f"  • {t}")
    return "\n".join(lines)
