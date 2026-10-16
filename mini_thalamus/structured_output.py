"""Mini Thalamus 统一结构化输出协议

所有 Agent 最终输出统一为 JSON，前端据此渲染不同组件。

Schema:
{
  "answer": "string",           // 主回答文本（必填）
  "format": "text|list|table|comparison|candidates|weather|url_content",  // 前端渲染提示
  "data": {},                   // 结构化数据（依 format 不同）
  "sources": [{"title":"", "url":""}],  // 参考来源
  "confidence": "high|medium|low"
}

各 format 的 data 格式：
- text:        无额外 data
- list:        {"items": [{"title": "要点1", "detail": "说明"}, ...]}
- table:       {"headers": ["姓名","年龄"], "rows": [["张三",25],...]}
- comparison:  {"subjects": ["A","B"], "criteria": [{"name":"学历","A":"...","B":"..."},...]}
- candidates:  {"candidates": [{"name":"张三","score":"85","pros":["..."],"cons":["..."],"verdict":"推荐"}]}
- weather:     {"city":"北京","temp":27,"condition":"晴","details":[...],"advice":"穿衣建议"}
- url_content: {"url":"https://...","title":"页面标题","content_type":"text/html","text":"正文摘要（前500字）","cached":false}
"""
import json, re


def build_prompt_instruction() -> str:
    """返回追加到 Agent prompt 末尾的结构化输出指令"""
    return """
【核心规则】
1. 只基于提供给你的上下文（文档内容/上一步结果/搜索结果）回答问题
2. 禁止编造不存在的事实、数据、包名、命令、API 或函数
3. 如果上下文信息不足，诚实说你不知道，不要猜测或虚构
4. 如果是执行命令/安装工具，严格使用上下文中提到的命令，不要自己发明

【输出格式——必须严格遵守】
以 JSON 格式输出，不要输出任何其他内容。JSON 结构:
{
  "answer": "主回答文本",
  "format": "text|list|table|comparison|candidates|weather|url_content",
  "data": { 按 format 填写结构 },
  "sources": [{"title": "来源名", "url": "链接"}],
  "confidence": "high|medium|low"
}

format 选择规则和 data 填写:
- 用户问天气/实时信息 → format="weather", data: {"city":"","temp":0,"condition":"","details":[],"advice":""}
- 用户问列表型信息(步骤/要点/原因) → format="list", data: {"items":[{"title":"要点","detail":"说明"}]}
- 用户要求对比/比较 → format="comparison", data: {"subjects":["Python","Rust"],"criteria":[{"name":"性能","Python":"较慢","Rust":"极快"},{"name":"学习曲线","Python":"平缓","Rust":"陡峭"}],"verdict":"总结一句话"}
- 用户问数据表格 → format="table", data: {"headers":["列1"],"rows":[["值1"]]}
- 用户要求推荐候选人 → format="candidates", data: {"candidates":[{"name":"","score":"","pros":[],"cons":[],"verdict":""}]}
- 抓取网页/URL 内容 → format="url_content", data: {"url":"https://...","title":"页面标题","content_type":"text/html","text":"正文摘要（前500字）","cached":false}
- 默认 → format="text", data: {}

sources 数组: 每个来源包含 title 和 url，没有就填空数组。
confidence: 信息充足→"high"，部分可靠→"medium"，推测或不确定→"low"
"""


def parse_llm_output(raw: str) -> dict:
    """解析 LLM 输出的 JSON，带容错"""
    # 1. 直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2. 提取 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 提取 { ... } 块 (balanced-brace, non-greedy)
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. 解析失败 → 退回纯文本模式
    return {
        "answer": raw.strip(),
        "format": "text",
        "data": {},
        "sources": [],
        "confidence": "medium",
    }


def render_to_text(parsed: dict) -> str:
    """将结构化 JSON 展平为纯文本（给前端 bak 或 CLI 模式用）"""
    answer = parsed.get("answer", "")
    fmt = parsed.get("format", "text")
    data = parsed.get("data", {})
    sources = parsed.get("sources", [])

    parts = [answer]

    if fmt == "list" and data.get("items"):
        parts.append("")
        for item in data["items"]:
            parts.append(f"- {item.get('title','')}: {item.get('detail','')}")
    elif fmt == "table" and data.get("headers"):
        parts.append("")
        parts.append(" | ".join(data["headers"]))
        parts.append(" | ".join(["---"] * len(data["headers"])))
        for row in data.get("rows", []):
            parts.append(" | ".join(str(c) for c in row))
    elif fmt == "comparison":
        parts.append("")
        for c in data.get("criteria", []):
            parts.append(f"- {c.get('name','')}: A={c.get('A','')}, B={c.get('B','')}")
        if data.get("verdict"):
            parts.append(f"\n结论: {data['verdict']}")
    elif fmt == "candidates":
        for c in data.get("candidates", []):
            parts.append(f"\n{c.get('name','')} (评分: {c.get('score','')})")
            parts.append(f"  优点: {', '.join(c.get('pros',[]))}")
            parts.append(f"  不足: {', '.join(c.get('cons',[]))}")
            parts.append(f"  结论: {c.get('verdict','')}")

    if sources:
        parts.append("\n---")
        for i, s in enumerate(sources, 1):
            parts.append(f"[{i}] {s.get('title','')} {s.get('url','')}")

    return "\n".join(parts)
