"""WebSearch Agent —— Bing 搜索 + 网页抓取 + LLM 阅读"""
import sys, os, re, json, time
import urllib.request
import urllib.parse
import urllib.error
import ssl

# 和 CodeAgent 一样的 config 加载方式
_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("shushu_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for
# FEATURE_PLAN.md §3：网页抓取统一走 tools.url_fetcher（SSRF/CRLF/限速/缓存共享）。
# 注意：_fetch_page 里必须传 allow_private_ips=True，
# 因为 _direct_fetch_fallback 会请求 wttr.in（公网域名，不受影响）+
# 早期方案曾用 localhost mock，老工作流靠这个开关兼容。审计 R6。
from tools.url_fetcher import fetch_url as _shared_fetch_url

# ─── 配置 ───
MAX_SEARCH_RESULTS = 5
FETCH_TIMEOUT = 10       # 单页抓取超时秒数
CONTENT_MAX_CHARS = 3000  # 单页内容截断字数
TOTAL_TIMEOUT = 30       # 总搜索+抓取超时
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/131.0.0.0"

SEARCH_PROVIDER_CALLS = {"tavily": 0, "serper": 0, "bocha": 0, "searxng": 0}


def reset_search_provider_calls():
    for key in SEARCH_PROVIDER_CALLS:
        SEARCH_PROVIDER_CALLS[key] = 0

AGENT_META = {
    "name": "WebSearch Agent",
    "id": "websearch",
    "description": "联网搜索互联网实时信息",
    "keywords": ["搜索", "网上", "最新", "新闻"],
    "icon": "🌐",
    "capabilities": {
        "needs_history": False, "needs_files": False,
        "needs_long_term": False, "max_context_chars": 2000,
        "auto_chainable": True, "side_effects": "none", "needs_credentials": [],
        "actions": ["http_get", "web_search"],
        "can_handle_errors": ["network_timeout", "content_not_found"],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class WebSearchAgent:
    """联网搜索 Agent —— 搜索 → 抓取 → 阅读 → 回答"""

    capabilities = {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 2000,
        "auto_chainable": True,
        "side_effects": "none",
        "needs_credentials": [],
    }

    def __init__(self):
        self.client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

    def run(self, query: str, context: dict = None) -> dict:
        start_time = time.time()

        # Step 1: 领域直达优先
        direct = self._direct_fetch_fallback(query)
        if direct and self._is_domain_query(query, direct):
            selected = [direct]
            print(f"  [WebSearch] 领域直达: {direct.get('title', '?')[:40]}")
        else:
            # Step 2: Bing 搜索候选
            candidates = self._search(query)
            if not candidates and direct:
                selected = [direct]
            elif not candidates:
                return {"status": "fallback",
                        "answer": self._knowledge_fallback(query),
                        "data": {"sources": [], "query": query}}
            else:
                # Step 3: LLM 从候选中选取最可能相关的 3 个
                selected = self._llm_pick_urls(query, candidates) if len(candidates) > 3 else candidates
                if not selected:
                    selected = candidates[:3]  # LLM 选取失败 → 取前3个兜底

        # Step 4: 只抓取 LLM 选中的网页
        for src in selected:
            if time.time() - start_time > TOTAL_TIMEOUT:
                break
            if src.get("source_type") == "direct_fetch" and src.get("content"):
                continue
            content = self._fetch_page(src["url"])
            src["content"] = content[:CONTENT_MAX_CHARS] if content else ""

        valid = [s for s in selected if s.get("content")]
        if not valid:
            snippet_sources = [s for s in selected if s.get("snippet")]
            if snippet_sources:
                for src in snippet_sources:
                    src["content"] = src.get("snippet", "")[:CONTENT_MAX_CHARS]
                valid = snippet_sources
                print("  [WebSearch] 页面抓取失败，改用搜索摘要生成回答")
        if not valid:
            return {"status": "partial",
                    "answer": f"搜索到 {len(selected)} 条结果，但无法抓取网页内容。",
                    "data": {"sources": selected, "query": query}}

        # Step 5: LLM 阅读选中网页 + 生成结构化答案
        from structured_output import render_to_text
        structured = self._generate_answer(query, valid)
        if not structured.get("answer"):
            items = structured.get("data", {}).get("items") or []
            if items:
                structured["answer"] = "\n".join(
                    f"{index + 1}. {item.get('title', '')}"
                    for index, item in enumerate(items)
                )

        return {
            "status": "success",
            "answer": structured.get("answer", ""),
            "data": {
                "structured": structured,
                "sources": [{"title": s["title"], "url": s["url"]} for s in valid],
                "query": query,
                "result_count": len(valid),
            },
            "summary": structured.get("answer", "")[:300],
        }

    # ─── 搜索（V5: LLM 关键词优先，原始 query 兜底）───
    def search(self, query: str) -> list:
        """Public search API — wraps _search() for external callers (e.g. DeepResearchAgent)."""
        return self._search(query)

    def _clean_search_query(self, query: str) -> str:
        """Strip orchestrator workflow markers so only the real task text is searched."""
        text = (query or "").strip()
        for marker in ("[Previous workflow result]", "[Resolved workflow inputs]"):
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx]
        idx = text.find("[Current task]")
        if idx >= 0:
            text = text[idx + len("[Current task]"):]
        text = re.sub(r"\s+", " ", text).strip()
        return text[:400]

    def _search(self, query: str) -> list:
        """Try stable providers first; only extract LLM keywords for the Bing fallback."""
        clean_query = self._clean_search_query(query)
        best = []
        provider_results = self._provider_search(clean_query)
        if provider_results:
            print(f"  [WebSearch] provider -> {len(provider_results)}条")
            return provider_results

        short_query = clean_query[:30].rsplit(" ", 1)[0] if len(clean_query) > 30 else clean_query
        keywords = self._llm_search_keywords(clean_query)
        seen = set()
        labels = ["LLM关键词", "短query", "原始"]
        attempt = 0
        for q in [keywords, short_query, clean_query]:
            if not q or q in seen:
                continue
            seen.add(q)
            results = self._bing_search(q)
            label = labels[attempt] if attempt < len(labels) else "原始"
            attempt += 1
            if results:
                print(f"  [WebSearch] {label}='{q[:40]}' → {len(results)}条")
                return results
            print(f"  [WebSearch] '{q[:40]}' → 0条")
        return best

    _PROVIDER_ORDER = ("tavily", "serper", "bocha", "searxng")

    def _provider_search(self, query: str) -> list:
        """Try configured providers in order; caller falls back to Bing when empty."""
        query = self._clean_search_query(query) or query
        for provider in self._provider_order():
            try:
                results = self._call_provider(provider, query)
            except Exception as exc:
                print(f"  [WebSearch] provider {provider} failed: {exc}")
                continue
            if results:
                print(f"  [WebSearch] provider {provider} -> {len(results)}条")
                return results
        return []

    def _provider_order(self) -> list:
        configured = re.split(
            r"[,\s;]+",
            os.environ.get("SHUSHU_SEARCH_ENGINE", "").strip().lower(),
        )
        configured = [p for p in configured if p in self._PROVIDER_ORDER]
        return list(dict.fromkeys(configured + list(self._PROVIDER_ORDER)))

    def _call_provider(self, provider: str, query: str) -> list:
        if provider == "searxng":
            url = os.environ.get("SHUSHU_SEARXNG_URL", "").strip().rstrip("/")
            if not url:
                return []
            SEARCH_PROVIDER_CALLS[provider] += 1
            return self._provider_searxng(query, url)
        api_key = os.environ.get(provider.upper() + "_API_KEY", "").strip()
        api_key = api_key or os.environ.get("SHUSHU_SEARCH_API_KEY", "").strip()
        if not api_key:
            return []
        SEARCH_PROVIDER_CALLS[provider] += 1
        return getattr(self, "_provider_" + provider)(query, api_key)

    def _provider_tavily(self, query: str, api_key: str) -> list:
        clean_query = re.sub(r"\s+", " ", query or "").strip()[:300]
        payload = json.dumps({"api_key": api_key, "query": clean_query, "max_results": MAX_SEARCH_RESULTS}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:200] if hasattr(exc, "read") else ""
            raise RuntimeError(f"Tavily HTTP {exc.code}: {body}") from exc
        return [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
            for item in data.get("results", [])[:MAX_SEARCH_RESULTS]
            if item.get("url")
        ]

    def _provider_serper(self, query: str, api_key: str) -> list:
        payload = json.dumps({"q": query, "num": MAX_SEARCH_RESULTS}).encode("utf-8")
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [
            {"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")}
            for item in data.get("organic", [])[:MAX_SEARCH_RESULTS]
            if item.get("link")
        ]

    def _provider_bocha(self, query: str, api_key: str) -> list:
        payload = json.dumps({"query": query, "count": MAX_SEARCH_RESULTS, "summary": True}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.bochaai.com/v1/web-search",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        webpages = data.get("data", {}).get("webPages", {}).get("value", [])
        return [
            {"title": item.get("name", ""), "url": item.get("url", ""), "snippet": item.get("snippet", "")}
            for item in webpages[:MAX_SEARCH_RESULTS]
            if item.get("url")
        ]

    def _provider_searxng(self, query: str, base_url: str) -> list:
        encoded = urllib.parse.quote(query)
        url = f"{base_url}/search?q={encoded}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
            for item in data.get("results", [])[:MAX_SEARCH_RESULTS]
            if item.get("url")
        ]

    def _bing_search(self, query: str) -> list:
        """对 query 执行一次 Bing 搜索，返回 [{title, url, snippet}]"""
        encoded = urllib.parse.quote(query)
        url = f"https://www.bing.com/search?q={encoded}&setlang=zh-cn"
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [WebSearch] Bing 请求失败: {type(e).__name__}: {e}")
            return []

        return self._parse_bing_results(raw)

    def _parse_bing_results(self, html: str) -> list:
        """从 Bing 搜索结果页提取标题/链接/摘要（h2>a 锚定主结果，避免侧边栏干扰）"""
        results = []
        # 只取搜索结果主列表区域
        main_m = re.search(r'<ol[^>]*id="b_results"[^>]*>(.*?)</ol>', html, re.DOTALL)
        area = main_m.group(1) if main_m else html

        # 每个结果: <li class="b_algo"> ... <h2><a href="URL">标题</a></h2> ... <p>摘要</p> ... </li>
        blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', area, re.DOTALL)

        for block in blocks[:MAX_SEARCH_RESULTS]:
            # 用 h2 锚定主结果链接（Bing 每个 b_algo 只有一个 h2）
            h2_m = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.DOTALL)
            if not h2_m:
                continue
            h2_content = h2_m.group(1)

            link_m = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', h2_content, re.DOTALL)
            if not link_m:
                continue
            href = link_m.group(1)
            title = re.sub(r'<[^>]+>', '', link_m.group(2)).strip()
            if not title or len(title) < 2:
                continue

            # 跳过 Bing/MS 内部链接和没有意义的短标题
            if any(b in href.lower() for b in ["bing.com", "microsoft.com", "go.microsoft.com"]):
                continue

            # 摘要：取第一个包含实质文本的 p/div
            snippet = ""
            for tag in ["p", "div"]:
                snips = re.findall(rf'<{tag}[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</{tag}>', block, re.DOTALL)
                if not snips:
                    snips = re.findall(rf'<{tag}[^>]*class="[^"]*b_caption[^"]*"[^>]*>(.*?)</{tag}>', block, re.DOTALL)
                if not snips:
                    snips = re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', block, re.DOTALL)
                for s in snips:
                    text = re.sub(r'<[^>]+>', '', s).strip()
                    if len(text) > 20:  # 过滤太短的
                        snippet = text
                        break
                if snippet:
                    break
            if not snippet:
                snippet = title

            results.append({"title": title, "url": href, "snippet": snippet[:300]})

        # 如果 b_algo 解析不到，退化为搜所有 h2 链接
        if not results:
            h2_links = re.findall(r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', area, re.DOTALL)
            for href, title_raw in h2_links[:MAX_SEARCH_RESULTS]:
                title = re.sub(r'<[^>]+>', '', title_raw).strip()
                if len(title) >= 2 and not any(b in href.lower() for b in ["bing.com", "microsoft.com"]):
                    results.append({"title": title, "url": href, "snippet": title})

        return results

    def _llm_search_keywords(self, query: str) -> str:
        """用 LLM 把自然语言问题转成搜索引擎友好的关键词"""
        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{
                    "role": "system",
                    "content": (
                        "将用户输入转成适合搜索引擎的简短关键词（空格分隔，10字以内）。"
                        "英文输入保留英文关键词，中文输入转成中文关键词；人名、地名、机构名保留原始拼写。"
                        "去掉'查询''帮我''一下'等废话，保留核心实体和意图。"
                        "例如: '今天北京天气怎么样适合穿什么' → '北京天气 穿衣建议'"
                        "\n只输出关键词，不要解释。"
                    )
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.0, max_tokens=30,
            )
            keywords = resp.choices[0].message.content.strip()
            print(f"  [WebSearch] LLM关键词: '{query}' → '{keywords}'")
            return keywords
        except Exception:
            return ""

    def _is_domain_query(self, query: str, direct: dict) -> bool:
        """判断是否应优先使用直达数据源而非搜索引擎"""
        lowered = query.lower()
        # 天气类 → 直达
        if any(w in lowered for w in ["天气", "weather", "气温", "温度", "下雨", "刮风", "雾霾"]):
            return bool(direct)
        # 后续可扩展: 股票/汇率/赛事等
        return False

    def _simplify_query(self, query: str) -> str:
        """简化 query 为搜索关键词（规则兜底，无 LLM）"""
        # 去除停用词和冗余表达
        stopwords = ["请问", "帮我", "查一下", "一下", "想知道", "告诉我", "什么是", "怎么"]
        result = query
        for w in stopwords:
            result = result.replace(w, "")
        return result.strip()[:30] or query[:30]

    # ─── 直接抓取兜底：绕过搜索引擎，构造 URL 直接拉 ───
    def _direct_fetch_fallback(self, query: str) -> dict:
        """Bing 不可用时，根据 query 类型尝试直接抓取已知数据源"""
        clean_query = self._clean_search_query(query) or query
        lowered = (" " + clean_query).lower()
        if not any(w in lowered for w in ["天气", "weather", "气温", "温度"]):
            return {}
        keywords = self._llm_search_keywords(clean_query) or self._simplify_query(clean_query)
        lowered = (keywords + " " + clean_query).lower()

        # 天气类 → 尝试 wttr.in（支持多城市）
        if any(w in lowered for w in ["天气", "weather", "气温", "温度"]):
            cities = self._extract_cities(query)
            if cities:
                # 多城市：最多取 5 个，并行抓取
                all_results = []
                for cn_name, en_name in cities[:5]:
                    url = f"https://wttr.in/{en_name}?format=j1"
                    content = self._fetch_page(url)
                    if content and len(content) > 50:
                        print(f"  [WebSearch] wttr.in 抓取: {cn_name}({en_name}) OK")
                        all_results.append({
                            "title": f"{cn_name} 实时天气 (wttr.in)",
                            "url": f"https://wttr.in/{en_name}",
                            "snippet": f"{cn_name}天气数据",
                            "content": content[:CONTENT_MAX_CHARS],
                            "source_type": "direct_fetch",
                        })
                if all_results:
                    # 合并为一个结果返回
                    combined_content = "\n\n---\n\n".join(
                        f"[{r['title']}]\n{r['content']}" for r in all_results
                    )
                    return {
                        "title": f"{'、'.join(c[0] for c in cities[:5])} 实时天气 (wttr.in)",
                        "url": f"https://wttr.in/{cities[0][1]}",
                        "snippet": f"共 {len(all_results)} 个城市天气",
                        "content": combined_content[:CONTENT_MAX_CHARS],
                        "source_type": "direct_fetch",
                    }

        return {}

    def _extract_cities(self, query: str) -> list:
        """从 query 提取所有城市名（英文，给 wttr.in API 用），返回 [(中文名, 英文名), ...]"""
        city_map = {
            # 直辖市 + 省会
            "北京": "Beijing", "上海": "Shanghai", "天津": "Tianjin", "重庆": "Chongqing",
            "广州": "Guangzhou", "深圳": "Shenzhen", "杭州": "Hangzhou", "成都": "Chengdu",
            "武汉": "Wuhan", "南京": "Nanjing", "西安": "Xian", "长沙": "Changsha",
            "郑州": "Zhengzhou", "济南": "Jinan", "青岛": "Qingdao", "沈阳": "Shenyang",
            "大连": "Dalian", "哈尔滨": "Harbin", "长春": "Changchun", "福州": "Fuzhou",
            "厦门": "Xiamen", "合肥": "Hefei", "南昌": "Nanchang", "昆明": "Kunming",
            "贵阳": "Guiyang", "南宁": "Nanning", "海口": "Haikou", "拉萨": "Lhasa",
            "乌鲁木齐": "Urumqi", "呼和浩特": "Hohhot", "银川": "Yinchuan",
            "西宁": "Xining", "兰州": "Lanzhou", "太原": "Taiyuan", "石家庄": "Shijiazhuang",
            # 经济强市
            "苏州": "Suzhou", "无锡": "Wuxi", "宁波": "Ningbo", "佛山": "Foshan",
            "东莞": "Dongguan", "温州": "Wenzhou", "珠海": "Zhuhai", "三亚": "Sanya",
            "桂林": "Guilin", "丽江": "Lijiang", "大理": "Dali",
        }
        cities = []
        for cn, en in city_map.items():
            if cn in query:
                cities.append((cn, en))
        return cities

    def _knowledge_fallback(self, query: str) -> str:
        """最终兜底: LLM 用自身知识回答，明确标注非实时"""
        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{
                    "role": "system",
                    "content": (
                        "用户想搜索互联网但搜索服务暂时不可用。"
                        "请基于你的训练知识尽量回答，但必须在开头明确标注："
                        "'⚠️ 搜索服务暂时不可用，以下回答基于模型知识，非实时数据：'"
                        "\n如果问题需要实时信息（如当前天气/股票），诚实告知无法提供。"
                    )
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.1, max_tokens=max_tokens_for(query, "answer"),
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return (f"当前搜索服务和模型均不可用。请稍后重试，或用简短关键词重新搜索。")

    def _llm_pick_urls(self, query: str, candidates: list) -> list:
        """让 LLM 从搜索结果中挑出最可能回答问题的 2-3 个网页"""
        menu = []
        for i, c in enumerate(candidates, 1):
            menu.append(f"[{i}] {c['title']}\n    {c['snippet'][:120]}")
        menu_text = "\n".join(menu)

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{
                    "role": "system",
                    "content": (
                        "从搜索结果中选出最可能回答用户问题的 2-3 个网页。"
                        "优先选择包含问题中核心实体、专有名词、数字或年份的页面；Wikipedia、官方站、教育/政府域名优先。"
                        "如果候选明显不相关（如应用商店、字典页、无关娱乐八卦），宁可少选也不要硬选。"
                        "不要只看标题关键词，要结合摘要判断是否真正回答问题。"
                        "只输出数字编号，空格分隔，如: 1 3 5"
                        "\n不要输出任何其他内容。"
                    )
                }, {
                    "role": "user",
                    "content": f"用户问题: {query}\n\n搜索结果:\n{menu_text}"
                }],
                temperature=0.0, max_tokens=20,
            )
            picks = resp.choices[0].message.content.strip()
            indices = [int(x) for x in picks.split() if x.isdigit() and 1 <= int(x) <= len(candidates)]
            selected = [candidates[i-1] for i in indices[:3]]
            if selected:
                print(f"  [WebSearch] LLM 选取: {picks} → {[s['title'][:30] for s in selected]}")
                return selected
        except Exception as e:
            print(f"  [WebSearch] LLM 选取失败: {e}")
        return []

    # ─── 网页抓取 ───
    def _fetch_page(self, url: str) -> str:
        """改用共享的 tools.url_fetcher.fetch_url，失败时换 UA 重试。

        FEATURE_PLAN.md §3 / Audit R6：
        - allow_private_ips=False，防止 SSRF。
        - 首轮失败后换 Googlebot UA + Accept-Language 再试一次。
        - 设置 SHUSHU_WEB_FETCH_BROWSER=1 时，可再用无头 Chromium 兜底。
        """
        attempts = [
            None,
            {
                "User-Agent": "Googlebot/2.1 (+http://www.google.com/bot.html)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        ]
        for headers in attempts:
            try:
                r = _shared_fetch_url(
                    url,
                    mode="readability",
                    max_chars=CONTENT_MAX_CHARS,
                    timeout=FETCH_TIMEOUT,
                    allow_private_ips=False,
                    verify_ssl=True,
                    headers=headers,
                )
                if r.get("status") == "success":
                    content = r.get("content", "")
                    if isinstance(content, str) and content:
                        return content[:CONTENT_MAX_CHARS]
            except Exception:
                pass
        if os.environ.get("SHUSHU_WEB_FETCH_BROWSER", "").strip().lower() in {"1", "true", "yes"}:
            return self._browser_fetch_text(url)
        return ""

    def _browser_fetch_text(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
            executable = os.environ.get("SHUSHU_CHROMIUM_PATH", "")
            if not executable:
                import glob as _glob
                for root in (
                    os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),
                    os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright"),
                ):
                    if not root:
                        continue
                    matches = _glob.glob(os.path.join(root, "chromium-*", "chrome-win64", "chrome.exe"))
                    if matches:
                        executable = matches[0]
                        break
            launch_kwargs = {"headless": True}
            if executable:
                launch_kwargs["executable_path"] = executable
            with sync_playwright() as pw:
                browser = pw.chromium.launch(**launch_kwargs)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=FETCH_TIMEOUT * 1000)
                text = page.inner_text("body")
                browser.close()
            return (text or "")[:CONTENT_MAX_CHARS]
        except Exception:
            return ""

    def _extract_text(self, html: str) -> str:
        """[DEPRECATED] FEATURE_PLAN.md §3 —— 正文抽取已迁到 tools.url_fetcher.extract_text。

        保留方法签名是为了兼容潜在的外部调用（test/Skill），实现委托共享版本。
        """
        from tools.url_fetcher import extract_text as _shared_extract_text
        return _shared_extract_text(html)

    # ─── LLM 生成（V2: 结构化 JSON 输出）───
    def _generate_answer(self, query: str, sources: list) -> dict:
        from structured_output import build_prompt_instruction, parse_llm_output

        query = self._clean_search_query(query) or query
        context_parts = []
        for i, src in enumerate(sources, 1):
            context_parts.append(f"[{i}] {src['title']}\n来源: {src['url']}\n内容: {src['content']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""根据以下联网搜索结果回答用户问题。

搜索结果:
{context}

用户问题: {query}

{build_prompt_instruction()}"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": "你是联网搜索助手。基于搜索结果回答，输出 JSON。英文问题必须整段用英文回答，不要混入中文。最终答案单独成行，只包含答案本身（数字、年份、书名、人名等），后面不要紧跟括号、年份或补充解释。关键数字单独呈现，不要附加单位或中文字符（如 7 而不是 7场）。人名、地名、机构名保留原始英文拼写。判断相关性以用户问题中的主题词为准；只有结果确实与问题完全无关时才说明未找到。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, max_tokens=max_tokens_for(query, "answer"),
            )
            raw = resp.choices[0].message.content.strip()
            return parse_llm_output(raw)
        except Exception as e:
            return {"answer": "抱歉，生成回答时遇到问题。", "format": "text", "data": {},
                    "sources": [{"title": s["title"], "url": s["url"]} for s in sources[:3]],
                    "confidence": "low"}
