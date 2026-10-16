"""
Core 模块 —— LLM 抽象层 + 工具注册框架。

复用自 multi_agent_playground，提供:
  - LLMBase / MockLLM / OpenAILLM / create_llm
  - TOOL_REGISTRY / get_tool / get_tool_schema / list_tools / get_openai_tools
"""
from nl2sql_agent.core.llm import LLMBase, MockLLM, OpenAILLM, create_llm
from nl2sql_agent.core.tools import (
    TOOL_REGISTRY,
    get_tool,
    get_tool_schema,
    list_tools,
    get_openai_tools,
    get_tools_prompt,
)

__all__ = [
    "LLMBase", "MockLLM", "OpenAILLM", "create_llm",
    "TOOL_REGISTRY",
    "get_tool", "get_tool_schema", "list_tools",
    "get_openai_tools", "get_tools_prompt",
]
