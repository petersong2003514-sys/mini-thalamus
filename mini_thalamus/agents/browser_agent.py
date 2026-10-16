"""Browser Agent —— Playwright 浏览器自动化：导航、填表、点击、内容提取

FEATURE_PLAN.md §N。让 Agent 能像人一样操作网页——注册、登录、表单提交。

架构：LLM 驱动 ReAct 循环
  1. 打开页面 → 提取交互元素列表（按钮/输入框/链接）
  2. LLM 根据元素列表 + 用户目标 → 决定下一步操作（click/fill/navigate/extract）
  3. 执行操作 → 等待页面响应 → 回到步骤 1
  4. 目标达成或达到最大步数 → 返回结果

安全：auto_chainable=False，高风险操作（登录/支付）需 checkpoint 确认。
"""
import os, sys, re, json, time, tempfile
from typing import Optional

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
import importlib.util
_cfg_path = os.path.join(_PKG_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("browser_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI
from token_budget import max_tokens_for

# ─── 日志 ───
import logging
_log = logging.getLogger("browser_agent")

MAX_STEPS = 15           # 最多执行几步操作（OAuth 等流程需要更多步）
PAGE_TIMEOUT = 20000     # 页面加载超时（ms）
CLICK_TIMEOUT = 10000    # 点击/填写超时（ms）
EXTRACT_MAX_ELEMENTS = 50  # 单页最多提取多少交互元素

AGENT_META = {
    "name": "Browser Agent",
    "id": "browser",
    "description": "浏览器自动化：打开网页、填写表单、点击按钮、提取内容。适合注册/登录/表单提交等需要交互的网页操作。",
    "keywords": ["浏览器", "打开网页", "注册", "登录", "填表", "提交", "browser", "表单", "网页操作", "自动填"],
    "icon": "🌐",
    "capabilities": {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 3000,
        "side_effects": "browser_automation",
        "auto_chainable": False,   # 浏览器操作风险较高，不自动流转
        "needs_credentials": [],
        "actions": ["browser_navigate", "browser_form_fill"],
        "can_handle_errors": ["auth_required", "web_form_required"],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}


class BrowserAgent:
    """LLM 驱动的浏览器自动化 Agent"""

    capabilities = AGENT_META["capabilities"]

    def __init__(self):
        self._client: Optional[OpenAI] = None

    def _llm(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])
        return self._client

    def run(self, query: str, context: dict = None) -> dict:
        """主入口：解析用户意图 → ReAct 循环执行浏览器操作"""
        ctx = context or {}
        steps_log = []

        # 1. LLM 解析用户意图 → 初始 URL + 目标
        intent = self._parse_intent(query, ctx)
        url = intent.get("url", "")
        goal = intent.get("goal", query)
        if not url:
            return {"status": "error",
                    "answer": "请提供一个 URL。例如：\"帮我打开 https://example.com 并注册账号\"",
                    "data": {}}

        # 2. 启动浏览器并执行 ReAct 循环
        try:
            from playwright.sync_api import sync_playwright
            executable = os.environ.get("MT_CHROMIUM_PATH", "")
            if not executable:
                import glob as _glob
                search_roots = [
                    os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),
                    os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright"),
                ]
                for root in search_roots:
                    if not root:
                        continue
                    matches = _glob.glob(os.path.join(
                        root,
                        "chromium-*",
                        "chrome-win64",
                        "chrome.exe",
                    ))
                    if matches:
                        executable = matches[0]
                        break
            launch_kwargs = {"headless": True}
            if executable:
                launch_kwargs["executable_path"] = executable
            with sync_playwright() as pw:
                browser = pw.chromium.launch(**launch_kwargs)
                page = browser.new_page()
                page.set_default_timeout(PAGE_TIMEOUT)

                _log.info("BrowserAgent: 导航到 %s", url)
                # SSRF 检查（防止访问内网 IP）
                from urllib.parse import urlparse as _urlparse
                _parsed = _urlparse(url)
                if _parsed.hostname:
                    from tools.url_fetcher import _check_ssrf
                    try:
                        _check_ssrf(_parsed.hostname, allow_private_ips=False)
                    except ValueError as _ssrf_err:
                        return {"status": "error",
                                "answer": f"安全拦截: 不允许访问内网地址 ({_ssrf_err})",
                                "data": {}}
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)

                for step in range(MAX_STEPS):
                    # 提取当前页面状态
                    page_state = self._extract_page_state(page)
                    page_state["step"] = step + 1
                    page_state["goal"] = goal

                    # LLM 决定下一步
                    action = self._llm_decide(page_state, steps_log)
                    _log.info("BrowserAgent step %d: %s", step + 1, action.get("type", "?"))

                    if action.get("type") == "done":
                        result_text = action.get("answer", "操作完成")
                        steps_log.append(f"[完成] {result_text}")
                        break
                    elif action.get("type") == "error":
                        steps_log.append(f"[错误] {action.get('answer', '')}")
                        break

                    # 执行操作
                    success = self._execute_action(page, action, steps_log)
                    if not success:
                        steps_log.append(f"[失败] {action.get('type')} 操作未成功")

                # 截图留存
                screenshot_b64 = ""
                try:
                    screenshot_b64 = page.screenshot(type="png")
                except Exception:
                    pass

                # 最终页面内容
                final_content = page.content()[:2000]
                title = page.title()
                browser.close()

        except Exception as e:
            _log.exception("BrowserAgent 异常: %s", e)
            return {"status": "error",
                    "answer": f"浏览器操作失败: {e}\n\n提示: 请确认 URL 可访问。已执行步骤: {json.dumps(steps_log, ensure_ascii=False)}",
                    "data": {"steps": steps_log}}

        # 3. LLM 总结结果
        answer = self._summarize(goal, steps_log, final_content, title)
        return {
            "status": "success",
            "answer": answer,
            "data": {
                "steps": steps_log,
                "url": url,
                "page_title": title,
                "screenshot_png": bool(screenshot_b64),
            },
            "summary": answer[:300],
        }

    # ── 意图解析 ──
    def _parse_intent(self, query: str, ctx: dict) -> dict:
        """LLM 从用户输入中提取 URL 和目标"""
        urls = re.findall(r'https?://[^\s<>"\']+', query)
        if urls:
            return {"url": urls[0].rstrip(".,;:!?)，。；：！？"), "goal": query}

        prompt = f"""从用户消息中提取 URL 和操作意图。只输出 JSON。

用户消息: {query}

{{"url": "https://...", "goal": "用户想做什么（一句话）"}}
若没有 URL 则 url 为空字符串。"""
        try:
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {"url": "", "goal": query}
        except Exception:
            return {"url": "", "goal": query}

    # ── 页面状态提取 ──
    def _extract_page_state(self, page) -> dict:
        """从页面提取 LLM 可理解的结构化描述"""
        try:
            title = page.title()
            url = page.url

            # 提取交互元素
            elements = []
            # 链接
            links = page.evaluate("""() => {
                const links = document.querySelectorAll('a[href]');
                return Array.from(links).slice(0, 30).map((a, i) => ({
                    id: 'link_' + i,
                    text: (a.textContent || '').trim().slice(0, 80),
                    href: a.href
                }));
            }""")
            for l in links:
                if l["text"]:
                    elements.append(f"[link_{l['id']}] {l['text']} → {l['href'][:100]}")

            # 按钮
            buttons = page.evaluate("""() => {
                const btns = document.querySelectorAll('button, input[type=submit], [role=button]');
                return Array.from(btns).slice(0, 20).map((b, i) => ({
                    id: 'btn_' + i,
                    text: (b.textContent || b.value || b.getAttribute('aria-label') || '').trim().slice(0, 60)
                }));
            }""")
            for b in buttons:
                if b["text"]:
                    elements.append(f"[button_{b['id']}] {b['text']}")

            # 输入框
            inputs = page.evaluate("""() => {
                const inps = document.querySelectorAll('input:not([type=hidden]):not([type=submit]), textarea, select');
                return Array.from(inps).slice(0, 15).map((el, i) => ({
                    id: 'inp_' + i,
                    name: el.name || el.id || '',
                    type: el.type || el.tagName.toLowerCase(),
                    placeholder: (el.placeholder || '').slice(0, 40),
                    value: (el.value || '').slice(0, 30)
                }));
            }""")
            for inp in inputs:
                label = inp["name"] or inp["placeholder"] or inp["type"]
                val_hint = f" (当前值: {inp['value']})" if inp["value"] else ""
                elements.append(f"[input_{inp['id']}] {inp['type']} name={label}{val_hint}")

            # 可见文本（前 1000 字符）
            text = page.evaluate("""() => {
                return (document.body?.innerText || '').slice(0, 1500);
            }""")

            return {
                "title": title,
                "url": url,
                "elements": elements[:EXTRACT_MAX_ELEMENTS],
                "text_preview": text[:1500],
            }
        except Exception as e:
            return {"title": "", "url": "", "elements": [], "text_preview": "", "error": str(e)}

    # ── LLM 决策 ──
    def _llm_decide(self, page_state: dict, steps_log: list) -> dict:
        """LLM 根据当前页面状态决定下一步操作"""
        elements_text = "\n".join(page_state.get("elements", [])[:40])
        steps_text = "\n".join(steps_log[-5:]) if steps_log else "(无)"

        prompt = f"""你是浏览器操作助手。根据页面内容和用户目标，决定下一步操作。

用户目标: {page_state.get('goal', '')}
当前页面: {page_state.get('title', '')} ({page_state.get('url', '')})
页面内容预览: {page_state.get('text_preview', '')[:800]}

可交互元素:
{elements_text}

已执行步骤:
{steps_text}

输出 JSON 选择下一步操作:
{{
  "type": "click|fill|navigate|extract|done|error",
  "target": "元素 id（如 btn_0, inp_1, link_3）或 URL",
  "value": "填写内容（type=fill 时）或提取要求（type=extract 时）",
  "reason": "一句话解释为什么这样做",
  "answer": "type=done 时的最终回答，type=error 时的错误说明"
}}

规则:
- 表单填写：逐个 fill 每个字段，最后 click 提交按钮
- 目标已达成 → type=done + answer 里写最终回答
- 页面打不开/报错 → type=error
- 不要重复已执行过的操作"""

        try:
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {"type": "error", "answer": "无法解析 LLM 响应"}
        except Exception:
            return {"type": "error", "answer": "LLM 调用失败"}

    # ── 执行操作 ──
    def _execute_action(self, page, action: dict, steps_log: list) -> bool:
        """执行 LLM 决定的单个浏览器操作"""
        atype = action.get("type", "")
        target = action.get("target", "")
        value = action.get("value", "")

        try:
            if atype == "navigate" and target:
                page.goto(target, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                steps_log.append(f"[导航] → {target[:80]}")
                return True

            elif atype == "click" and target:
                sel = self._target_to_selector(target)
                if sel:
                    page.click(sel, timeout=CLICK_TIMEOUT)
                    page.wait_for_timeout(1500)  # 等页面响应（OAuth 跳转需要更久）
                    steps_log.append(f"[点击] {target}")
                    return True

            elif atype == "fill" and target:
                sel = self._target_to_selector(target)
                if sel:
                    page.fill(sel, value, timeout=CLICK_TIMEOUT)
                    steps_log.append(f"[填写] {target} ← {value[:50]}")
                    return True

            elif atype == "extract":
                content = page.content()[:3000]
                steps_log.append(f"[提取] 页面内容 {len(content)} 字符")
                return True

            elif atype in ("done", "error"):
                return True  # 不需要执行，由主循环处理

        except Exception as e:
            _log.warning("BrowserAgent 操作失败: %s → %s", atype, str(e)[:100])
            steps_log.append(f"[异常] {atype} {target}: {str(e)[:80]}")
            return False

        return False

    def _target_to_selector(self, target: str) -> Optional[str]:
        """将元素 ID 转为 CSS selector"""
        if target.startswith("btn_"):
            idx = target.replace("btn_", "")
            return f"button:nth-of-type({int(idx)+1})" if idx.isdigit() else f"[id*='{idx}']"
        if target.startswith("inp_"):
            idx = target.replace("inp_", "")
            return f"input:nth-of-type({int(idx)+1})" if idx.isdigit() else f"[name*='{idx}']"
        if target.startswith("link_"):
            idx = target.replace("link_", "")
            return f"a:nth-of-type({int(idx)+1})" if idx.isdigit() else f"a[href*='{idx}']"
        return None

    # ── 结果总结 ──
    def _summarize(self, goal: str, steps_log: list, content: str, title: str) -> str:
        """LLM 将所有步骤总结为自然语言回答"""
        steps_text = "\n".join(steps_log[-10:])
        prompt = f"""根据浏览器操作日志回答用户。

用户目标: {goal}
页面标题: {title}
操作日志:
{steps_text}

用 2-5 句话总结发生了什么，是否达成目标。如果成功，说出关键结果。如果失败，说明原因和建议。"""

        try:
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "\n".join(steps_log[-5:])


__all__ = ["BrowserAgent", "AGENT_META"]
