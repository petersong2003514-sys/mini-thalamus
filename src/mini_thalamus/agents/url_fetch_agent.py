"""URL Fetch Agent —— 用户给定 URL 的内容抓取与摘要

FEATURE_PLAN.md §3。薄壳：抽取 query 中（或上一步 chain 结果中）的 URL，
调用 tools.url_fetcher.fetch_url 拿到正文/JSON/二进制，按需走 LLM 摘要。

设计要点：
- 不做搜索，只抓取用户给的 URL（搜索是 WebSearchAgent 的活）。
- 单 URL + 非摘要意图 → 直接返回结构化结果（format='url_content'，前端可特化渲染；
  老前端 fallback 到 list 也兼容，见 structured_output.py 注释）。
- 多 URL 或含「总结/摘要/要点/分析/tldr/summarize」关键词 → 拼上下文 + LLM 摘要。
- 前一步 chain 结果（context.previous_result）里如果带 url/urls/links/sources，
  query 里没 URL 时也能继续工作（websearch → url_fetch → general 链路友好）。
- capabilities：保守 = 无副作用（read-only HTTP）+ 不需要凭证（用户必要时通过
  query/headers 自己带），auto_chainable=True 让 Auto-Chain 后能跳过 checkpoint。
"""
import os
import re
import sys
import time
from typing import Optional

# ─── Config 显式路径加载（沿用项目惯例，避免 sys.path 污染）───
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mini_thalamus import config as _cfg
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from mini_thalamus.token_budget import max_tokens_for
from mini_thalamus.structured_output import build_prompt_instruction, parse_llm_output
from mini_thalamus.tools.url_fetcher import fetch_url

# ─── URL 提取（含中文标点尾部裁剪）───
URL_REGEX = re.compile(
    r'https?://[^\s<>"\']+',
    re.IGNORECASE,
)
# 中文标点 + 常见尾部杂字符；从 URL 末尾剥掉
_URL_TRAIL_STRIP = '.,;:!?)]}\'"，。；：！？）】」』、　'

SUMMARY_KEYWORDS = ("总结", "摘要", "要点", "分析", "tldr", "summarize", "summary",
                    "概括", "提炼", "归纳")
MAX_URLS_PER_RUN = 3            # 单次请求最多抓几个 URL
PER_URL_MAX_CHARS = 6000
LLM_INPUT_MAX_CHARS = 12000     # 拼到 LLM prompt 里的总上限


def _extract_urls(text: str) -> list:
    """从文本中抽 URL，剥末尾标点，去重保序。"""
    if not text:
        return []
    urls: list = []
    seen: set = set()
    for m in URL_REGEX.finditer(text):
        u = m.group(0).rstrip(_URL_TRAIL_STRIP)
        if u and u not in seen:
            urls.append(u)
            seen.add(u)
    return urls


def _urls_from_previous(prev: Optional[dict]) -> list:
    """从上一步 chain 结果里挖 URL（websearch sources / 任意 data.urls 等）。"""
    if not isinstance(prev, dict):
        return []
    data = prev.get("data") or {}
    if not isinstance(data, dict):
        return []
    out: list = []
    seen: set = set()

    def _add(u):
        if isinstance(u, str) and u.startswith(("http://", "https://")) and u not in seen:
            out.append(u.rstrip(_URL_TRAIL_STRIP))
            seen.add(u)

    # 直接字段
    for key in ("url", "urls", "links"):
        v = data.get(key)
        if isinstance(v, str):
            _add(v)
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, str):
                    _add(it)
                elif isinstance(it, dict):
                    _add(it.get("url") or it.get("href"))

    # sources：[{title,url}, ...]
    sources = data.get("sources")
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                _add(s.get("url") or s.get("href"))

    # structured.sources（websearch 的实际形态）
    structured = data.get("structured")
    if isinstance(structured, dict):
        for s in structured.get("sources", []) or []:
            if isinstance(s, dict):
                _add(s.get("url") or s.get("href"))

    return out


def _wants_summary(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    return any(k in q for k in SUMMARY_KEYWORDS)


AGENT_META = {
    "name": "URL 抓取",
    "id": "url_fetch",
    "description": "抓取并阅读用户给定的 URL（HTML/JSON/纯文本），返回正文或结构化数据。"
                   "支持多 URL；含『总结/要点』时叠加 LLM 摘要。私有 IP / file:// 已被拒绝。",
        "keywords": ["抓取", "读一下", "fetch", "打开链接", "网址内容", "url", "网页内容", "看看这个链接", "总结这个网址", "公开链接", "链接内容", "原始内容", "读取链接", "网页正文", "获取链接", "提取网页内容"],
    "intents": ["url_fetch", "page_extract"],
    "examples": ["请直接获取公开链接的原始内容", "读取这个链接的网页内容", "抓取这个网页并总结要点"],
    "icon": "🔗",
    "capabilities": {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 2000,
        "side_effects": "read_only_http",
        "auto_chainable": True,
        "needs_credentials": [],
        "actions": ["http_get"],
        "can_handle_errors": ["file_not_found", "page_not_found", "network_timeout"],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class UrlFetchAgent:
    """直接抓取用户提供的 URL —— 不搜索，不发现。"""

    capabilities = AGENT_META["capabilities"]

    def __init__(self):
        # OpenAI client 仅在摘要分支用；构造时不强制活性，方便单测
        self._client: Optional[OpenAI] = None

    def _llm(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        return self._client

    # ─── 入口 ────────────────────────────────────────────
    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        previous_result = None
        if isinstance(ctx, dict):
            previous_result = ctx.get("previous_result")
        else:
            # AgentContext dataclass
            previous_result = getattr(ctx, "previous_result", None)

        # 1. 找 URL：先 query，再 previous_result 兜底
        urls = _extract_urls(query)
        if not urls:
            urls = _urls_from_previous(previous_result)
        if not urls:
            return {
                "status": "error",
                "answer": "未在你的请求或上一步结果中找到可抓取的 http(s) URL。",
                "data": {"query": query, "urls": []},
                "summary": "无可抓取 URL",
            }
        urls = urls[:MAX_URLS_PER_RUN]

        # 2. 抓取（顺序，单次最多 3 个；fetch_url 自带 1s/host 限速）
        fetched: list = []
        errors: list = []
        for u in urls:
            r = fetch_url(u, mode="auto", max_chars=PER_URL_MAX_CHARS, timeout=15)
            if r.get("status") == "success":
                fetched.append(r)
            else:
                errors.append({"url": u, "error": r.get("error", "unknown"),
                               "http_status": r.get("http_status")})

        # 3. 全部失败 → 直接返回错误（保留每个 URL 的详细错因）
        if not fetched:
            err_lines = [f"- {e['url']}: {e['error']}" for e in errors]
            return {
                "status": "error",
                "answer": "无法抓取任何 URL：\n" + "\n".join(err_lines),
                "data": {"query": query, "urls": urls, "errors": errors},
                "summary": f"抓取 {len(urls)} 个 URL 全部失败",
            }

        # 4. 决定要不要跑 LLM 摘要
        want_summary = _wants_summary(query)
        single = len(fetched) == 1

        # 4a. 单 URL + 不要摘要 → 直接返回结构化内容（绕过 LLM）
        if single and not want_summary:
            r = fetched[0]
            mode_used = r.get("mode_used", "raw")
            content = r.get("content", "")
            preview = self._preview(content, mode_used)
            answer = f"已抓取 {r.get('url')}\n\n{preview}"
            structured = {
                "answer": answer,
                "format": "url_content",     # 新 format（前端 fallback to text 也安全）
                "data": {
                    "url": r.get("url"),
                    "content_type": r.get("content_type", ""),
                    "mode_used": mode_used,
                    "content": content,
                    "warnings": r.get("warnings", []),
                },
                "sources": [{"title": r.get("url"), "url": r.get("url")}],
                "confidence": "high",
            }
            return {
                "status": "success",
                "answer": answer,
                "data": {
                    "structured": structured,
                    "fetched": [self._slim(r) for r in fetched],
                    "urls": [r.get("url") for r in fetched],
                    "errors": errors,
                    "query": query,
                    "operations": self._extract_operations(content, r.get("url", "")),
                },
                "summary": f"抓取 {r.get('url')} 成功（{mode_used}）",
            }

        # 4b. 多 URL 或要摘要 → LLM 综合
        return self._summarize(query, fetched, errors)

    # ─── LLM 摘要分支 ─────────────────────────────────────
    def _summarize(self, query: str, fetched: list, errors: list) -> dict:
        # 拼上下文，防总长爆炸
        budget = LLM_INPUT_MAX_CHARS
        per = max(500, budget // max(1, len(fetched)))
        parts = []
        for i, r in enumerate(fetched, 1):
            body = self._stringify(r.get("content"), r.get("mode_used", "raw"))
            parts.append(f"[{i}] {r.get('url')}\n"
                         f"Content-Type: {r.get('content_type','')}\n"
                         f"内容:\n{body[:per]}")
        ctx_text = "\n\n---\n\n".join(parts)

        prompt = (
            f"用户请求：{query}\n\n"
            f"以下是已抓取到的 {len(fetched)} 个网页/接口内容。"
            f"请基于这些内容回答用户请求；如要总结/摘要，提炼要点并给出每条要点的来源编号。"
            f"\n\n抓取内容：\n{ctx_text}\n\n"
            f"{build_prompt_instruction()}"
        )

        try:
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system",
                     "content": "你是网页内容阅读助手。基于给定网页正文回答，输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens_for(query, "answer"),
            )
            raw = resp.choices[0].message.content.strip()
            structured = parse_llm_output(raw)
        except Exception as e:
            structured = {
                "answer": f"已抓取 {len(fetched)} 个网页，但生成摘要时出错：{e}",
                "format": "text", "data": {},
                "sources": [{"title": r.get("url"), "url": r.get("url")} for r in fetched],
                "confidence": "low",
            }

        # 回填来源（即便 LLM 没填）
        if not structured.get("sources"):
            structured["sources"] = [
                {"title": r.get("url"), "url": r.get("url")} for r in fetched
            ]

        answer = structured.get("answer", "")
        return {
            "status": "success",
            "answer": answer,
            "data": {
                "structured": structured,
                "fetched": [self._slim(r) for r in fetched],
                "urls": [r.get("url") for r in fetched],
                "errors": errors,
                "query": query,
                "operations": self._extract_operations(
                    fetched[0].get("content", "") if fetched else "",
                    fetched[0].get("url", "") if fetched else "",
                ),
            },
            "summary": (answer or f"抓取 {len(fetched)} 个 URL 并总结")[:300],
        }

    # ─── 工具 ─────────────────────────────────────────────
    @staticmethod
    def _stringify(content, mode_used: str) -> str:
        """LLM prompt 里只放字符串。json/binary 转字符串处理。"""
        if isinstance(content, str):
            return content
        if mode_used == "json":
            try:
                import json as _j
                return _j.dumps(content, ensure_ascii=False)[:LLM_INPUT_MAX_CHARS]
            except Exception:
                return str(content)
        if mode_used == "binary":
            return "[binary content omitted; use DocParse/Vision Agent]"
        return str(content)

    @staticmethod
    def _preview(content, mode_used: str, n: int = 1500) -> str:
        """给单 URL 直接回路径用的短预览。"""
        if mode_used == "json":
            try:
                import json as _j
                s = _j.dumps(content, ensure_ascii=False, indent=2)
            except Exception:
                s = str(content)
        elif mode_used == "binary":
            return "[binary content (base64) omitted; use DocParse/Vision Agent]"
        else:
            s = content if isinstance(content, str) else str(content)
        return s[:n] + ("..." if len(s) > n else "")

    def _extract_operations(self, content: str, url: str) -> list:
        """从文档内容提取可执行操作列表（API端点、CLI命令、Web表单）。"""
        if not content or len(content) < 100:
            return []
        try:
            text = content[:3000] if isinstance(content, str) else str(content)[:3000]
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system",
                     "content": "你是操作提取助手。从文档中提取可执行操作，输出JSON数组。"},
                    {"role": "user", "content": (
                        f"从以下文档提取可执行操作（API端点、CLI命令、Web表单），最多5个。\n\n"
                        f"{text}\n\n"
                        "只输出JSON数组，每项格式：{\"type\": \"api_call\"|\"cli\"|\"web_form\", "
                        "\"method\"(仅api_call时), \"url\"(仅api_call/web_form时), "
                        "\"command\"(仅cli时), \"note\"}"
                    )},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```\w*\n?", "", raw).replace("```", "")
            import json as _json
            ops = _json.loads(raw)
            return ops if isinstance(ops, list) else []
        except Exception:
            return []

    @staticmethod
    def _slim(r: dict) -> dict:
        """回 data.fetched 的瘦身版（不带 base64 大块）。"""
        out = {
            "url": r.get("url"),
            "content_type": r.get("content_type", ""),
            "mode_used": r.get("mode_used", ""),
            "warnings": r.get("warnings", []),
            "cached": r.get("cached", False),
        }
        c = r.get("content")
        if r.get("mode_used") == "binary":
            out["content"] = "[base64 omitted]"
        elif isinstance(c, str):
            out["content"] = c[:PER_URL_MAX_CHARS]
        else:
            out["content"] = c  # JSON 对象，原样保留（已在 fetch_url 控量）
        return out


__all__ = ["UrlFetchAgent", "AGENT_META"]
