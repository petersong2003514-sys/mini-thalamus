"""DeepResearch Agent —— 多轮迭代搜索 + 交叉验证 + 结构化研究报告

使用:
  提出研究问题 → 自动拆解子主题 → 多轮搜索 → 交叉验证 → 输出含引用的报告
  适合: 竞品分析、市场调研、技术深挖、多源事实核查
"""
import sys, os, json, re, time as _time

_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MINICODE_ROOT)

import importlib.util
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("dr_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for
from structured_output import build_prompt_instruction, parse_llm_output

try:
    from logging_config import get_logger
    _log = get_logger("deep_research")
except Exception:
    import logging as _logging
    _log = _logging.getLogger("deep_research")

MAX_ROUNDS = 3
MAX_SOURCES_PER_ROUND = 3
TOTAL_TIMEOUT = 90  # 总超时秒数

AGENT_META = {
    "name": "DeepResearch Agent",
    "id": "deep_research",
    "description": "多源深度研究：搜索、交叉验证、生成含引用的研究报告",
    "keywords": ["研究", "对比", "调研", "vs"],
    "icon": "🔬",
    "capabilities": {
        "needs_history": False, "needs_files": False,
        "needs_long_term": True, "max_context_chars": 3000,
        "auto_chainable": True, "side_effects": "none", "needs_credentials": [],
        "actions": ["http_get", "web_search"],
        "can_handle_errors": ["content_not_found"],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class DeepResearchAgent:
    """多轮深度研究 Agent"""

    capabilities = {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": True,
        "max_context_chars": 3000,
        "auto_chainable": True,
        "side_effects": "none",
        "needs_credentials": [],
    }

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        # 复用 WebSearch 的搜索能力
        from agents.websearch_agent import WebSearchAgent
        self._searcher = WebSearchAgent()

    def run(self, query: str, context: dict = None) -> dict:
        start_time = _time.time()
        ctx = context or {}
        all_sources = []  # [{url, title, snippet, content, round}]
        findings = []     # [{claim, support_count, sources}]

        # ── 第 1 轮: 主搜索 ──
        _log.info("DeepResearch 开始: %s", query[:80])
        try:
            raw = self._searcher.search(self._search_query(query, 1))
            sources = self._process_sources(raw, 1, start_time)
            all_sources.extend(sources)
        except Exception as e:
            _log.warning("第 1 轮搜索失败: %s", e)
            sources = []

        # ── 第 2-3 轮: 细化搜索 ──
        for round_num in range(2, MAX_ROUNDS + 1):
            if _time.time() - start_time > TOTAL_TIMEOUT:
                _log.info("超时，停止搜索")
                break

            # LLM 判断是否需要继续 + 生成细化查询
            decision = self._llm_decide_next(query, all_sources, findings, round_num)
            if not decision.get("continue"):
                break

            sub_queries = decision.get("queries", [])
            if not sub_queries:
                break

            for sq in sub_queries[:2]:  # 每轮最多 2 个子查询
                try:
                    raw = self._searcher.search(sq)
                    new_sources = self._process_sources(raw, round_num, start_time)
                    all_sources.extend(new_sources)
                except Exception as e:
                    _log.warning("子查询失败 '%s': %s", sq[:40], e)

        # ── 交叉验证 ──
        verified = self._cross_verify(findings, all_sources)

        # ── 合成报告 ──
        report = self._synthesize_report(query, verified, all_sources)

        return {
            "status": "success",
            "answer": report.get("answer", ""),
            "data": {
                "structured": {
                    "format": "list",
                    "data": {
                        "items": verified,
                    },
                    "sources": [{"title": s["title"], "url": s.get("url", "")}
                                for s in all_sources[:10]],
                    "confidence": "high" if len(verified) >= 3 else "medium",
                },
                "source_count": len(all_sources),
                "rounds": len(set(s.get("round", 1) for s in all_sources)),
            },
            "summary": report.get("answer", "")[:300],
        }

    def _search_query(self, question: str, round_num: int) -> str:
        """LLM 生成高质量搜索关键词"""
        if round_num == 1:
            prompt = f"""你是搜索专家。将用户问题转化为 2-3 个搜索引擎关键词（用空格分隔，中英文皆可）。

用户问题: {question}

要求: 关键词精准、覆盖问题核心维度。只输出关键词，不要解释。

示例:
- "对比Python和Rust优缺点" → Python Rust 对比 优缺点 性能
- "新能源汽车市场趋势" → 新能源汽车 市场趋势 2025 2026"""

            try:
                resp = self.client.chat.completions.create(
                    model=LLM_CONFIG["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=80,
                )
                keywords = resp.choices[0].message.content.strip()
                if keywords:
                    return keywords
            except Exception:
                pass
        return f"{question} 对比 分析"

    def _process_sources(self, raw_results: list, round_num: int, start_time: float) -> list:
        """抓取搜索结果页面内容"""
        sources = []
        for r in raw_results[:MAX_SOURCES_PER_ROUND]:
            if _time.time() - start_time > TOTAL_TIMEOUT:
                break
            url = r.get("url", "")
            if not url:
                continue
            try:
                from tools.url_fetcher import fetch_url
                fetched = fetch_url(url, mode="readability", max_chars=3000, timeout=10)
                content = fetched.get("text", "") if isinstance(fetched, dict) else ""
                sources.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "content": content[:3000],
                    "round": round_num,
                })
            except Exception:
                # 至少保留 snippet
                sources.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "content": r.get("snippet", ""),
                    "round": round_num,
                })
        return sources

    def _llm_decide_next(self, question: str, sources: list, findings: list, round_num: int) -> dict:
        """LLM 判断是否需要继续搜索并生成子查询"""
        summary = "\n".join(
            f"- [{s['title']}]({s.get('url','')}): {s['content'][:200]}"
            for s in sources[-6:]
        )
        prompt = f"""你是研究助手。你正在进行多轮研究，当前是第 {round_num} 轮。

研究问题: {question}

已收集的来源:
{summary if summary else "(暂无)"}

当前发现:
{json.dumps(findings, ensure_ascii=False)[:1000] if findings else "(暂无)"}

请判断:
1. 信息是否足够回答研究问题？(continue: true/false)
2. 如果不够，需要搜索什么子问题来补充？(queries: 1-2 个具体查询)

返回 JSON: {{"continue": true/false, "reason": "...", "queries": ["..."]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {"continue": False, "reason": "parse error"}
        except Exception:
            return {"continue": False, "reason": "LLM error"}

    def _cross_verify(self, findings: list, sources: list) -> list:
        """从所有来源中提取关键发现 + 置信度评估"""
        # 合并所有来源内容
        combined = "\n\n".join(
            f"[{s['title']}]\n{s['content'][:1500]}"
            for s in sources[:10]
        )
        if not combined or len(combined) < 200:
            return []  # 返回空，触发知识兜底

        prompt = f"""根据以下研究资料，提取 3-8 个关键发现。每个发现的置信度取决于多少来源支持它（high=3+, medium=2, low=1）。

资料:
{combined[:8000]}

返回格式:
{{
  "findings": [
    {{"title": "关键发现", "detail": "详细说明和证据", "confidence": "high/medium/low", "source_count": 3}}
  ]
}}

{build_prompt_instruction()}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1500,
            )
            parsed = parse_llm_output(resp.choices[0].message.content.strip())
            data = parsed.get("data", {})
            items = data.get("findings", data.get("items", []))
            return items if isinstance(items, list) else []
        except Exception:
            return [{"title": "分析失败", "detail": "请重试"}]

    def _synthesize_report(self, question: str, findings: list, sources: list) -> dict:
        """生成最终研究报告（无搜索结果时用 LLM 知识兜底）"""
        findings_text = "\n".join(
            f"- [{f.get('confidence','?')}] {f.get('title','?')}: {f.get('detail','?')}"
            for f in findings
        ) if findings else "(搜索结果不足，请基于你的知识回答)"

        sources_text = "\n".join(
            f"[{i+1}] {s['title']} - {s.get('url','')}"
            for i, s in enumerate(sources[:10])
        ) if sources else "(无有效来源)"

        # 根据来源质量调整指令
        if not findings:
            knowledge_hint = (
                "搜索结果不足以覆盖该问题，请基于你的训练知识直接回答。"
                "在回答中诚实标注'基于模型知识，非实时搜索'。"
            )
        else:
            knowledge_hint = "基于以上搜索来源回答。"

        prompt = f"""回答关于「{question}」的问题。

{knowledge_hint}

来源资料:
{findings_text}
{sources_text}

{build_prompt_instruction()}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=max_tokens_for(question, "answer"),
            )
            return parse_llm_output(resp.choices[0].message.content.strip())
        except Exception as e:
            return {"answer": f"研究完成但报告生成失败: {e}", "format": "text"}
