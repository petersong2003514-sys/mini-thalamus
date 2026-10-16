"""Agent 模板 —— 复制此文件创建新 Agent

1. 复制: cp _template.py my_agent.py
2. 改 AGENT_SPEC: id/name/description/intents/examples
3. 实现 run(): 接收 query + context，返回标准 dict
4. 重启或刷新 AgentRegistry，系统自动发现

run() 必须返回:
{
    "status": "success" | "error",
    "answer": "给用户看的文本",
    "data": {
        "structured": {           # 可选: 结构化输出
            "format": "text|table|list|comparison|candidates|weather|code",
            "data": {...},        # 依 format 不同
            "sources": [...],
            "confidence": "high|medium|low"
        },
        ...                       # 其他自定义字段
    },
    "summary": "answer 的摘要，用于记忆系统"
}

context 参数（推荐用法）:
    - 旧路径: context 是 dict（如 {"files": [...], "session_id": "..."}）
    - 新路径: context 是 base_agent.AgentContext —— 可结构化拿历史/标签/长期记忆
        from mini_thalamus.base_agent import AgentContext
        if isinstance(context, AgentContext):
            for h in context.relevant_history:
                ...  # 取 q/a/agent
            files = context.files
            ltm = context.long_term  # {tag: gist}
"""
from mini_thalamus.config import get_llm_config
LLM_CONFIG = get_llm_config()

from openai import OpenAI
from mini_thalamus.token_budget import max_tokens_for
from mini_thalamus.structured_output import build_prompt_instruction, parse_llm_output

# ═══════════════════════════════════════════
# Agent 元信息（必填）
# ═══════════════════════════════════════════
AGENT_SPEC = {
    "id": "my",
    "name": "新 Agent",
    "description": "这个 Agent 做什么——会显示在侧边栏和 Router 描述中",
    "keywords": ["关键词1", "关键词2"],
    "intents": ["custom_task"],
    "examples": [],
    "icon": "🔧",
    "factory": "mini_thalamus.agents.my_agent:create_agent",
    "config_namespace": "my",
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
    "capabilities": {
        "needs_history": True,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        "auto_chainable": False,
        "side_effects": "external",
        "needs_credentials": [],
    },
}

AGENT_META = AGENT_SPEC


class MyAgent:
    """新 Agent 实现"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

    def run(self, query: str, context: dict = None) -> dict:
        # ── TODO: 实现你的 Agent 逻辑 ──
        # 1. 用 self.client 调 LLM
        # 2. 调用外部 API / 数据库
        # 3. 返回标准格式

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{
                    "role": "system",
                    "content": "你是一个有帮助的助手。"
                }, {
                    "role": "user",
                    "content": query + "\n\n" + build_prompt_instruction(),
                }],
                temperature=0.1,
                max_tokens=max_tokens_for(query, "answer"),
            )
            structured = parse_llm_output(resp.choices[0].message.content.strip())
        except Exception as e:
            return {
                "status": "error",
                "answer": f"处理失败: {e}",
                "data": {},
                "summary": str(e)[:200],
            }

        return {
            "status": "success",
            "answer": structured.get("answer", ""),
            "data": {"structured": structured},
            "summary": structured.get("answer", "")[:300],
        }


def create_agent():
    return MyAgent()
