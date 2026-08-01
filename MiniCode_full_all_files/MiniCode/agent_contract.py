"""Standard Agent capability contract used by Router and Workflow Planner."""


class AgentContractError(ValueError):
    """Raised when an Agent profile/contract is invalid."""


STANDARD_AGENT_HINTS = {
    "api_caller": {
        "intents": ["api_call", "http_request", "service_integration"],
        "examples": ["调用接口获取订单状态", "向外部服务发送 POST 请求"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "browser": {
        "intents": ["browser_automation", "web_interaction", "form_operation"],
        "examples": ["打开网页并填写注册表单", "登录网站后提取页面信息"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "code": {
        "intents": ["code_execution", "data_visualization", "file_processing"],
        "examples": ["用 Python 计算这组数据", "把查询结果画成折线图"],
        "exports": ["answer", "summary", "data.structured", "data.artifacts"],
    },
    "data_analyst": {
        "intents": ["data_analysis", "trend_analysis", "anomaly_detection"],
        "examples": ["分析 Excel 中的销售趋势", "找出 CSV 数据里的异常点"],
        "exports": ["answer", "summary", "data.structured", "data.artifacts"],
    },
    "deep_research": {
        "intents": ["deep_research", "source_synthesis", "comparative_analysis"],
        "examples": ["调研最近三年相关论文并对比", "多来源交叉验证一个行业问题"],
        "exports": ["answer", "summary", "data.structured", "data.report"],
    },
    "docparse": {
        "intents": ["document_parsing", "file_extraction", "document_summary"],
        "examples": ["解析上传 PDF 的主要内容", "提取 Word 文档里的表格信息"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "nl2sql": {
        "intents": ["database_query", "sql_generation", "data_retrieval"],
        "examples": ["统计各地区销售额", "查询本月新增用户"],
        "exports": ["answer", "summary", "sql", "query_result", "dataframe", "data.structured"],
    },
    "rag": {
        "intents": ["knowledge_retrieval", "document_question"],
        "examples": ["从知识库查找部署说明", "根据上传文档回答问题"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "report_writer": {
        "intents": ["report_generation", "summary_writing", "markdown_report"],
        "examples": ["生成结构化分析报告", "把前面结果整理成 Markdown 周报"],
        "exports": ["answer", "summary", "data.report", "data.structured"],
    },
    "shell": {
        "intents": ["shell_command", "project_maintenance", "script_execution"],
        "examples": ["查看 git 状态", "运行项目测试脚本"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "url_fetch": {
        "intents": ["url_fetch", "webpage_reading", "content_extraction"],
        "examples": ["读取这个链接的正文", "抓取网页并总结要点"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "vision": {
        "intents": ["image_understanding", "ocr", "chart_analysis"],
        "examples": ["识别截图里的文字", "分析图片中的图表内容"],
        "exports": ["answer", "summary", "data.structured"],
    },
    "websearch": {
        "intents": ["web_search", "current_information_lookup", "news_search"],
        "examples": ["搜索最新政策信息", "查询网上关于某公司的新闻"],
        "exports": ["answer", "summary", "data.structured", "sources"],
    },
}


def _string_list(value, *, field: str, agent_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AgentContractError(f"agent '{agent_id}' {field} must be a list of strings")
    return list(value)


def _normalize_few_shots(meta: dict, intents: list[str], agent_id: str) -> list[dict]:
    raw = meta.get("few_shots")
    if raw is None:
        default_intent = intents[0] if intents else "general_task"
        raw = [
            {"query": example, "expected_intent": default_intent}
            for example in (
                meta.get("examples")
                or STANDARD_AGENT_HINTS.get(agent_id, {}).get("examples", [])
            )
        ]

    if not isinstance(raw, list):
        raise AgentContractError(f"agent '{agent_id}' few_shots must be a list")

    few_shots = []
    allowed_intents = set(intents)
    for item in raw:
        if isinstance(item, str):
            expected_intent = intents[0] if intents else "general_task"
            item = {"query": item, "expected_intent": expected_intent}
        if not isinstance(item, dict):
            raise AgentContractError(
                f"agent '{agent_id}' few_shots entries must be objects"
            )
        query = item.get("query")
        expected_intent = item.get("expected_intent")
        if not isinstance(query, str) or not query:
            raise AgentContractError(
                f"agent '{agent_id}' few_shots.query must be a non-empty string"
            )
        if not isinstance(expected_intent, str) or not expected_intent:
            raise AgentContractError(
                f"agent '{agent_id}' few_shots.expected_intent must be a non-empty string"
            )
        if allowed_intents and expected_intent not in allowed_intents:
            raise AgentContractError(
                f"agent '{agent_id}' few_shot intent '{expected_intent}' is not registered"
            )
        few_shots.append({
            "query": query,
            "expected_intent": expected_intent,
        })
    return few_shots


def normalize_agent_contract(meta: dict, default_id: str = "") -> dict:
    """Return a normalized Agent contract.

    The contract keeps legacy fields (`examples`, `input_schema`, `output_schema`)
    while adding the standard fields consumed by the next DAG planning stage:
    `few_shots`, `requires`, and `exports`.
    """
    if not isinstance(meta, dict):
        raise AgentContractError("Agent contract must be an object")

    agent_id = meta.get("id") or default_id
    if not isinstance(agent_id, str) or not agent_id:
        raise AgentContractError("Agent id must be a non-empty string")

    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise AgentContractError(f"agent '{agent_id}' requires a name")

    description = meta.get("description", "")
    if not isinstance(description, str):
        raise AgentContractError(f"agent '{agent_id}' description must be a string")

    hints = STANDARD_AGENT_HINTS.get(agent_id, {})

    normalized = dict(meta)
    normalized["id"] = agent_id
    normalized["description"] = description
    normalized["keywords"] = _string_list(meta.get("keywords", []), field="keywords", agent_id=agent_id)
    normalized["intents"] = _string_list(meta.get("intents") or hints.get("intents", []), field="intents", agent_id=agent_id)
    normalized["examples"] = _string_list(meta.get("examples") or hints.get("examples", []), field="examples", agent_id=agent_id)
    normalized["few_shots"] = _normalize_few_shots(
        {
            **meta,
            "examples": normalized["examples"],
        },
        normalized["intents"],
        agent_id,
    )
    normalized.setdefault("config_namespace", agent_id)
    normalized.setdefault(
        "input_schema",
        {"required": ["instruction"], "properties": {}},
    )
    normalized.setdefault(
        "output_schema",
        {
            "exports": [
                "answer",
                "summary",
                "data.structured",
            ]
        },
    )
    normalized["requires"] = _string_list(
        meta.get("requires", normalized["input_schema"].get("required", [])),
        field="requires",
        agent_id=agent_id,
    )
    if "exports" in meta:
        exports_value = meta["exports"]
    else:
        exports_value = list(normalized["output_schema"].get("exports", []))
        for export_name in hints.get("exports", []):
            if export_name not in exports_value:
                exports_value.append(export_name)
    normalized["exports"] = _string_list(
        exports_value,
        field="exports",
        agent_id=agent_id,
    )
    normalized.setdefault("capabilities", {})
    return normalized


__all__ = ["AgentContractError", "normalize_agent_contract"]
