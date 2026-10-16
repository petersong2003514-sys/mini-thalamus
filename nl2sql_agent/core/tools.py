"""
工具集模块 —— Agent 可以调用的所有工具。

包含:
- Calculator: 安全的算术计算器
- KnowledgeBase: 模拟的结构化知识库（数学公式、常识等）
- SearchEngine: 模拟的搜索引擎
"""

import math
import re
from typing import Any


# ─── 工具注册表 ─────────────────────────────────────────────
# 每个工具都是一个 dict: {name, description, parameters, fn}
# 这个 schema 结构模仿了 OpenAI function calling 的 JSON Schema 格式

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register(name: str, description: str, parameters: dict, fn: callable):
    """注册一个工具到全局注册表"""
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "fn": fn,
    }


# ─── 工具实现 ────────────────────────────────────────────────

def _safe_eval(expr: str) -> str:
    """安全地求值数学表达式，只允许数字、运算符和 math 函数"""
    allowed = set("0123456789.+-*/()%^, eEpPiI")
    allowed |= set("sqrt sincostanlogexpab")
    # 白名单检查
    cleaned = expr.replace("math.", "")
    for ch in cleaned:
        if ch not in allowed and ch not in "abcdefghijklmnopqrstuvwxyz":
            raise ValueError(f"不允许的字符: {ch}")
    # 替换 ^ 为 **
    expr_safe = expr.replace("^", "**")
    # 限制命名空间
    namespace = {
        "math": math,
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "abs": abs,  # Python builtin, math 模块没有 abs
        "pow": math.pow,
    }
    result = eval(expr_safe, {"__builtins__": {}}, namespace)
    return str(result)


_register(
    name="calculator",
    description="执行数学计算。输入一个数学表达式字符串，返回计算结果。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 '(120 - 45) / 3' 或 'sqrt(144)'",
            }
        },
        "required": ["expression"],
    },
    fn=_safe_eval,
)


# ─── 知识库 ──────────────────────────────────────────────────

KNOWLEDGE = {
    "圆的面积公式": "S = π × r²，其中 r 为半径",
    "勾股定理": "直角三角形两直角边 a, b，斜边 c，有 a² + b² = c²",
    "速度公式": "速度 = 路程 / 时间，即 v = s / t",
    "地球到太阳距离": "约 1.5 亿公里（1 AU）",
    "光速": "约 3×10⁸ m/s",
    "太阳光到地球时间": "约 8 分 20 秒（500 秒）",
    "水的沸点": "标准大气压下为 100°C",
    "重力加速度": "g ≈ 9.8 m/s²",
    "牛顿第二定律": "F = m × a，力 = 质量 × 加速度",
    "欧姆定律": "V = I × R，电压 = 电流 × 电阻",
    "功率公式": "P = W / t = F × v，功率 = 功 / 时间",
    "动能公式": "E_k = ½ × m × v²",
    "圆的周长": "C = 2πr",
    "三角形面积": "S = ½ × 底 × 高",
    "复利公式": "A = P × (1 + r/n)^(n×t)",
}


def _knowledge_lookup(query: str) -> str:
    """在知识库中搜索相关信息"""
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE.items():
        # 简单关键词匹配
        key_words = set(key)
        query_words = set(query)
        if len(key_words & query_words) >= 2 or any(
            w in key for w in query_lower.split()
        ):
            results.append(f"【{key}】: {value}")
    if not results:
        return f"未找到与 '{query}' 相关的知识。"
    return "\n".join(results[:3])


_register(
    name="knowledge_base",
    description="查询内置知识库，获取公式、常数、常识等信息。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询关键词，例如 '圆的面积' 或 '光速'",
            }
        },
        "required": ["query"],
    },
    fn=_knowledge_lookup,
)


# ─── 模拟搜索 ────────────────────────────────────────────────

SIMULATED_WEB: dict[str, str] = {
    "地球 半径": "地球平均半径约为 6371 公里",
    "太阳 质量": "太阳质量约为 1.989×10³⁰ kg",
    "地球 质量": "地球质量约为 5.972×10²⁴ kg",
    "月球 距离": "月球与地球平均距离约为 38.4 万公里",
    "马拉松 距离": "马拉松全程为 42.195 公里",
    "光年": "1 光年 = 9.461×10¹² 公里",
    "珠穆朗玛峰 高度": "珠穆朗玛峰海拔 8848.86 米",
    "太平洋 面积": "太平洋面积约 1.65 亿平方公里",
    "中国 人口 2024": "中国人口约 14.1 亿（2024年）",
    "GDP 中国 2024": "2024年中国GDP约 134.9 万亿元人民币",
    "GDP 美国 2024": "2024年美国GDP约 29.2 万亿美元",
    "美元 人民币 汇率": "1 美元 ≈ 7.2 人民币（2025年5月）",
}


def _search(query: str) -> str:
    """模拟搜索引擎，返回预存数据"""
    query_lower = query.lower()
    for key, value in SIMULATED_WEB.items():
        if all(kw.lower() in query_lower for kw in key.split()):
            return value
    # 模糊匹配
    best = None
    best_score = 0
    for key, value in SIMULATED_WEB.items():
        score = sum(1 for kw in key.split() if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best = value
    if best and best_score >= 1:
        return best
    return f"[模拟搜索] 未找到关于 '{query}' 的精确信息，请尝试其他关键词。"


_register(
    name="search",
    description="在互联网上搜索信息。输入查询关键词，返回搜索结果摘要。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，例如 '地球半径 多少公里'",
            }
        },
        "required": ["query"],
    },
    fn=_search,
)


# ─── 公共接口 ────────────────────────────────────────────────

def get_tool(name: str):
    """根据名称获取工具函数。不存在返回 None。"""
    tool = TOOL_REGISTRY.get(name)
    return tool["fn"] if tool else None


def get_tool_schema(name: str) -> dict | None:
    """获取工具的完整 schema（不含 fn）"""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return None
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
    }


def list_tools() -> list[dict]:
    """列出所有可用工具的 schema"""
    return [get_tool_schema(name) for name in TOOL_REGISTRY]


def get_tools_prompt() -> str:
    """生成用于 prompt 的工具描述文本（阶段 1 正则解析用）"""
    lines = []
    for name, tool in TOOL_REGISTRY.items():
        props = tool["parameters"].get("properties", {})
        args_desc = ", ".join(
            f"{k}: {v.get('description', '')}" for k, v in props.items()
        )
        lines.append(f"- {name}({args_desc}): {tool['description']}")
    return "\n".join(lines)


def get_openai_tools() -> list[dict]:
    """
    导出 OpenAI Function Calling 格式的工具列表。

    阶段 1 的做法: 把工具描述写成文本注入 prompt，靠正则解析 LLM 输出
    阶段 2 的做法: 把工具 schema 作为 API 参数 tools=[...] 传给 LLM，
                 LLM 原生返回结构化的 tool_calls，不需要正则解析

    格式:
    [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行数学计算...",
                "parameters": {"type": "object", "properties": {...}, "required": [...]}
            }
        },
        ...
    ]
    """
    openai_tools = []
    for name, tool in TOOL_REGISTRY.items():
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return openai_tools
