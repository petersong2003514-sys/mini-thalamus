"""Peer adapter for the sibling :mod:`nl2sql_agent` package."""


AGENT_SPEC = {
    "id": "nl2sql",
    "name": "NL2SQL Agent",
    "icon": "🗄️",
    "description": "数据库查询、SQL 生成、执行校验、消歧和自修正",
        "keywords": ["数据库", "查询", "SQL", "统计", "销售额", "订单", "自然语言", "自然语言问题", "转换", "转成查询", "数据库查询", "查一下"],
    "intents": ["database_query", "sql_generation"],
    "examples": ["统计各区域销售额", "查询本月新增用户", "把自然语言问题转换为数据库查询", "请把自然语言问题转换为数据库查询"],
    "intents": ["database_query", "sql_generation"],
    "examples": ["统计各地区销售额", "查询本月新增用户"],
    "factory": "agents.nl2sql_agent:create_agent",
    "config_namespace": "nl2sql",
    "input_schema": {
        "required": ["instruction"],
        "properties": {},
    },
    "output_schema": {
        "exports": ["answer", "summary", "data.structured"],
    },
    "capabilities": {
        "needs_history": True,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        "side_effects": "database_read",
        "auto_chainable": True,
        "needs_credentials": [],
    },
}

# Compatibility with existing metadata consumers.
AGENT_META = AGENT_SPEC


class NL2SQLPeerAgent:
    agent_id = "nl2sql"
    capabilities = AGENT_SPEC["capabilities"]

    def __init__(self, schema_strategy: str = "full"):
        from nl2sql_agent.agent_factory import build_agent, load_dotenv
        from query_disambiguator import SmartDisambiguator

        load_dotenv()
        self.agent, self.db, self.schema_provider = build_agent(
            schema_strategy=schema_strategy
        )
        self.disambiguator = SmartDisambiguator(self.db)

    def run(self, query: str, context: dict = None, mode: str = "auto") -> dict:
        original_query = query
        disambiguation = self.disambiguator.disambiguate(original_query)
        if disambiguation.get("action") == "enhance":
            query = disambiguation.get("enhanced_query", original_query)

        try:
            result = self.agent.run(query, mode=mode)
        except Exception as exc:
            return {
                "status": "error",
                "answer": str(exc),
                "summary": f"查询失败: {exc}",
                "data": {},
                "metadata": {"agent_id": self.agent_id},
            }

        rows = result.result.rows if result.result else []
        columns = result.result.columns if result.result else []
        plan_steps = [
            {
                "step": index + 1,
                "question": getattr(step, "question", str(step)[:100]),
                "sql": getattr(step, "sql", ""),
            }
            for index, step in enumerate(getattr(result, "plan_steps", []) or [])
        ]
        summary = result.nl_summary or (
            f"查询到 {len(rows)} 条记录。"
            if rows
            else "查询完成。"
        )
        answer_parts = []
        if result.final_sql:
            answer_parts.append(f"SQL: `{result.final_sql}`")
        answer_parts.append(summary)

        return {
            "status": "success" if result.success else "error",
            "answer": "\n".join(answer_parts),
            "summary": summary,
            "data": {
                "sql": result.final_sql,
                "rows": rows,
                "columns": columns,
                "rounds": result.rounds_attempted,
                "plan_steps": plan_steps,
                "structured": {
                    "format": "table",
                    "data": {
                        "headers": columns,
                        "rows": rows[:20],
                    },
                    "sources": [],
                    "confidence": "high" if result.success else "low",
                },
            },
            "metadata": {
                "agent_id": self.agent_id,
                "mode_used": "plan" if plan_steps else "simple",
                "disambiguation": disambiguation,
            },
        }


def create_agent():
    return NL2SQLPeerAgent()


__all__ = ["AGENT_META", "AGENT_SPEC", "NL2SQLPeerAgent", "create_agent"]

