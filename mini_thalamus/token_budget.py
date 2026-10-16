"""统一 Token 预算分配 —— 根据任务复杂度自适应调整（纯启发式 + LRU 缓存）"""
import os as _os, importlib.util as _util
from functools import lru_cache

# 显式加载 Mini Thalamus 的 config，避免被 nl2sql_agent 的 config.py 覆盖
_cfg_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.py")
_spec = _util.spec_from_file_location("mt_config", _cfg_path)
_cfg = _util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

# ─── 预算档位 ───
BUDGET = {
    "simple":   {"answer": 600,  "thought": 300},
    "medium":   {"answer": 1500, "thought": 600},
    "complex":  {"answer": 3000, "thought": 1200},
}


@lru_cache(maxsize=512)
def _assess_cached(query_key: str, hint: str) -> str:
    """缓存命中以 (query_前80字符, hint) 为键，避免相似 query 重复算（REPAIR_PLAN #16）"""
    if "多份文档" in hint or "多个文件" in hint:
        return "complex" if ("对比" in query_key or "差异" in query_key) else "medium"
    # 长度启发式（用原始长度近似 —— 截断后总长仍单调反映原长度档位）
    if len(query_key) < 15:
        return "simple"
    if len(query_key) > 200:
        return "complex"
    # 关键词启发式
    complex_kw = ("分析", "对比", "报告", "总结全部", "深度", "多步", "详细解释", "研究")
    medium_kw = ("解释", "为什么", "怎么", "如何", "区别", "优缺点", "列举", "步骤")
    q = query_key.lower()
    if any(k in q for k in complex_kw):
        return "complex"
    if any(k in q for k in medium_kw):
        return "medium"
    return "simple"


def assess_complexity(query: str, context_hint: str = "") -> str:
    """评估任务复杂度 → simple/medium/complex（纯启发式，无 LLM；带 LRU 缓存）"""
    # 用 query 前 200 字 + hint 作缓存键。原始 query 可能很长，但启发式只看长度档位+关键词。
    # 截断到 200 字保留：(a) <15 / 15-200 / >200 三档判断的近似；(b) 几乎所有关键词。
    key = query[:200] if len(query) <= 200 else query[:200] + "...long"
    return _assess_cached(key, context_hint)


def get_budget(query: str, context: dict = None) -> dict:
    """获取当前任务的 token 预算"""
    hint = ""
    if context:
        docs = context.get("files", [])
        if len(docs) > 1:
            hint = "多个文件"
    level = assess_complexity(query, hint)
    return BUDGET[level]


def max_tokens_for(query: str, role: str = "answer", context: dict = None) -> int:
    """快捷接口：获取某个角色的 max_tokens"""
    budget = get_budget(query, context)
    return budget.get(role, budget["answer"])
