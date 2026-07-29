"""MCP Agent —— 通用工具代理，通过 LLM Function Calling 调用 MCP 工具"""
import sys, os, json

_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MINICODE_ROOT)

import importlib.util
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("mcp_agent_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for
from structured_output import build_prompt_instruction, parse_llm_output


class MCPAgent:
    """通用 MCP Agent：LLM ReAct 循环 + MCP 工具调用"""

    # MCP 工具有外部副作用，永不自动链式执行
    capabilities = {
        "auto_chainable": False,
        "side_effects": "external",
        "needs_credentials": [],
    }

    def __init__(self, tool_registry, mcp_manager):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        self.registry = tool_registry
        self.mcp = mcp_manager
        self.max_steps = 5

    def run(self, query: str, context: dict = None) -> dict:
        tools_schema = self.registry.to_openai_schema("mcp")
        if not tools_schema:
            return {"status": "error",
                    "answer": "MCP 工具代理未发现可用工具。请检查 MCP 服务器配置。",
                    "data": {}, "summary": "无工具"}

        messages = [
            {"role": "system", "content": (
                "你是 ShuShu Agent 的 MCP 工具代理。你可以调用外部 MCP 工具完成任务。"
                "分析用户需求，选择合适的工具调用。如果工具返回错误，尝试其他方式。"
                "最终用中文给出清晰回答。"
            )},
            {"role": "user", "content": query},
        ]

        for step in range(self.max_steps):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_CONFIG["model"],
                    messages=messages,
                    tools=tools_schema,
                    temperature=0.1,
                    max_tokens=max_tokens_for(query, "answer"),
                )
            except Exception as e:
                return {"status": "error", "answer": f"LLM 调用失败: {e}",
                        "data": {}, "summary": str(e)}

            msg = resp.choices[0].message

            # LLM 给出了最终答案
            if msg.content and not msg.tool_calls:
                structured = parse_llm_output(msg.content)
                return {
                    "status": "success",
                    "answer": structured.get("answer", msg.content),
                    "data": {"structured": structured},
                    "summary": structured.get("answer", "")[:300],
                }

            # LLM 要求调用工具
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [
                                     {"id": tc.id, "type": "function",
                                      "function": {"name": tc.function.name,
                                                   "arguments": tc.function.arguments}}
                                     for tc in msg.tool_calls
                                 ]})

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = self.registry.execute(tool_name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    print(f"  [MCP Agent] 调用 {tool_name} -> {result.get('status','?')}")
            else:
                # 没有 tool_calls 也没有 content → 异常
                break

        return {"status": "partial",
                "answer": "处理超时。请尝试简化问题，或检查 MCP 工具是否正常。",
                "data": {}, "summary": "超时"}
