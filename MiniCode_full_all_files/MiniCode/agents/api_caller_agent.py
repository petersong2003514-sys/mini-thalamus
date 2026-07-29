"""API Caller Agent —— 执行 REST API 调用（GET/POST/PUT/DELETE）

接收来自 url_fetch_agent operations 字段的结构化 API 端点，
或用户直接给出的 API 调用指令，执行 HTTP 请求并返回结果。
"""
from __future__ import annotations
import os, re, sys, json, importlib.util
from typing import Optional

_MINICODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MINICODE_ROOT not in sys.path:
    sys.path.insert(0, _MINICODE_ROOT)
_cfg_path = os.path.join(_MINICODE_ROOT, "config.py")
_spec = importlib.util.spec_from_file_location("apicaller_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
LLM_CONFIG = _cfg.LLM_CONFIG

from openai import OpenAI

AGENT_META = {
    "name": "API 调用",
    "id": "api_caller",
    "description": "执行 REST API 调用（GET/POST/PUT/DELETE JSON），支持从上一步结果中读取 API 端点。",
    "keywords": ["api", "接口", "rest", "http请求", "post", "调用接口", "发请求"],
    "icon": "📡",
    "capabilities": {
        "needs_history": False,
        "needs_files": False,
        "needs_long_term": False,
        "max_context_chars": 4000,
        "side_effects": "external",
        "auto_chainable": True,
        "needs_credentials": [],
        "actions": ["http_post", "http_get", "http_put", "http_delete"],
        "can_handle_errors": ["file_not_found", "auth_required", "api_error"],
    },
    "input_schema": {"required": ["instruction"], "properties": {}},
    "output_schema": {"exports": ["answer", "summary", "data.structured"]},
}

import ipaddress, socket
_PRIVATE_NETS = [ipaddress.ip_network(n) for n in [
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
    "192.168.0.0/16", "::1/128", "fc00::/7",
]]


def _is_private(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return any(addr in net for net in _PRIVATE_NETS)
    except Exception:
        return False


def _ssrf_check(url: str) -> bool:
    """返回 True 表示被 SSRF 防护拦截。"""
    try:
        from urllib.parse import urlparse
        h = urlparse(url).hostname or ""
        if h in ("localhost", "0.0.0.0") or _is_private(h):
            return True
    except Exception:
        pass
    return False


class ApiCallerAgent:
    """执行 REST API 调用 Agent"""

    capabilities = AGENT_META["capabilities"]

    def __init__(self):
        self._client: Optional[OpenAI] = None

    def _llm(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
            )
        return self._client

    def run(self, query: str, context: dict = None) -> dict:
        ctx = context or {}
        previous_result = (
            ctx.get("previous_result") if isinstance(ctx, dict)
            else getattr(ctx, "previous_result", None)
        )
        # 1. 从 previous_result.data.operations 找 api_call 操作
        api_op = self._find_api_op(previous_result)
        # 2. 若无，用 LLM 从 query 解析
        if not api_op:
            api_op = self._parse_from_query(query)
        if not api_op:
            return {
                "status": "error",
                "answer": "无法从请求或上一步结果中解析出 API 调用参数（需要 URL、method）。",
                "data": {},
            }

        url = api_op.get("url", "")
        method = api_op.get("method", "GET").upper()
        params = api_op.get("params") or {}
        headers = api_op.get("headers") or {}

        # 3. SSRF 防护
        if _ssrf_check(url):
            return {
                "status": "error",
                "answer": f"SSRF 防护：拒绝访问私有/本地地址 {url}",
                "data": {"url": url},
            }

        # 4. 注入 credentials（若已配置）
        try:
            from credentials import get_store
            env_creds = get_store().get_env_dict("api_caller")
            for k, v in env_creds.items():
                if k.endswith("_TOKEN") or k.endswith("_KEY"):
                    headers.setdefault("Authorization", f"Bearer {v}")
        except Exception:
            pass

        return self._execute(method, url, params, headers)

    def _find_api_op(self, prev: Optional[dict]) -> Optional[dict]:
        """从 previous_result.data.operations 提取第一个 api_call。"""
        if not isinstance(prev, dict):
            return None
        data = prev.get("data") or {}
        ops = (
            data.get("operations")
            or (data.get("structured") or {}).get("data", {}).get("operations")
            or []
        )
        for op in (ops or []):
            if isinstance(op, dict) and op.get("type") == "api_call" and op.get("url"):
                return op
        return None

    def _parse_from_query(self, query: str) -> Optional[dict]:
        """让 LLM 从 query 中提取 API 调用参数。"""
        try:
            resp = self._llm().chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system",
                     "content": "你是 API 调用解析助手。从用户描述中提取 HTTP API 调用参数，输出 JSON。"},
                    {"role": "user", "content": (
                        f"从以下请求中提取 API 调用参数。\n{query}\n\n"
                        '输出格式：{"url": "...", "method": "POST", "params": {}, "headers": {}}\n'
                        "若无法提取则输出 null。"
                    )},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            if text.lower() in ("null", "none", ""):
                return None
            text = re.sub(r"```\w*\n?", "", text).replace("```", "")
            return json.loads(text)
        except Exception:
            return None

    def _execute(self, method: str, url: str, params: dict, headers: dict) -> dict:
        import urllib.request
        import urllib.error
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MiniCode-ApiCaller/1.0",
            **headers,
        }
        try:
            body = (
                json.dumps(params).encode()
                if params and method in ("POST", "PUT", "PATCH")
                else None
            )
            if params and method == "GET":
                from urllib.parse import urlencode
                sep = "&" if "?" in url else "?"
                url = url + sep + urlencode(params)
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8", errors="replace")[:8000]
                status_code = r.status
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")[:2000]
            return {
                "status": "error",
                "answer": f"API 调用失败（HTTP {e.code}）: {raw[:500]}",
                "data": {"url": url, "method": method, "status_code": e.code, "response": raw},
            }
        except Exception as e:
            return {
                "status": "error",
                "answer": f"API 调用出错: {str(e)[:300]}",
                "data": {"url": url, "method": method, "error": str(e)},
            }

        try:
            resp_data = json.loads(raw) if isinstance(raw, str) else raw

            # 智能结构化：识别 BotLearn 注册等常见 API 响应
            structured = _format_response(resp_data, method, url, status_code)
            answer = structured["answer"]
            fmt = structured.get("format", "text")
            fmt_data = structured.get("data", {})
        except Exception:
            resp_data = raw
            answer = f"API 调用成功（{method} {url}，HTTP {status_code}）\n\n{raw[:1500]}"
            fmt = "text"
            fmt_data = {}

        return {
            "status": "success",
            "answer": answer,
            "data": {
                "url": url, "method": method,
                "status_code": status_code,
                "response": resp_data,
                "structured": {
                    "format": fmt,
                    "data": fmt_data,
                    "confidence": "high" if status_code < 400 else "medium",
                }
            },
            "summary": f"{method} {url} → {status_code}",
        }

# ─── 智能响应格式化 ───
def _format_response(data: dict, method: str, url: str, status_code: int) -> dict:
    """根据 API 响应内容智能选择结构化格式"""
    d = data.get("data", data) if isinstance(data, dict) else {}

    # BotLearn 注册响应
    agent = d.get("agent", {})
    if agent and "claim_url" in agent:
        return {
            "answer": (
                f"🎓 BotLearn 注册成功！\n\n"
                f"Agent API Key: `{agent['api_key']}`\n"
                f"Claim 链接: {agent['claim_url']}\n"
                f"验证码: `{agent.get('verification_code', '')}`\n\n"
                f"⚠️ 请保存 API Key，打开 claim 链接完成验证。"
            ),
            "format": "list",
            "data": {
                "items": [
                    {"title": "API Key", "detail": agent.get("api_key", "")},
                    {"title": "Claim URL", "detail": agent.get("claim_url", "")},
                    {"title": "验证码", "detail": agent.get("verification_code", "")},
                ]
            }
        }

    # 通用成功响应
    if status_code < 400:
        return {
            "answer": f"API 调用成功（{method} {url}，HTTP {status_code}）\n\n{json.dumps(data, ensure_ascii=False, indent=2)[:1200]}",
            "format": "text",
            "data": {}
        }

    # 错误响应
    return {
        "answer": f"API 返回错误（HTTP {status_code}）: {json.dumps(data, ensure_ascii=False)[:800]}",
        "format": "text",
        "data": {}
    }


__all__ = ["ApiCallerAgent", "AGENT_META"]
