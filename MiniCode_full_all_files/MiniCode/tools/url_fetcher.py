"""tools/url_fetcher.py —— 共享 URL 抓取工具（SSRF 防护 + TTL/LRU 缓存）

FEATURE_PLAN.md §3 URL Fetch。同时被 agents/url_fetch_agent.py 和
agents/websearch_agent.py 使用（websearch 路径保留 `allow_private_ips=True`
兼容老 localhost 抓取场景，audit R6）。

设计要点：
- SSRF: DNS 解析 → ipaddress 检查 private/loopback/link_local/reserved/multicast。
- 重定向: 自定义 NoRedirect handler，每跳重新校验目标地址。
- 仅允许 http/https；其他 scheme (file://, ftp://, gopher://) 拒绝。
- 限流: 同主机 1s 间隔。
- header 注入: 拒绝任何含 CR/LF 的 user-supplied header。
- 缓存: OrderedDict TTL+LRU，15min/200 entries；带 Authorization 直接 bypass。
"""
from __future__ import annotations

import base64
import ipaddress
import json
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Optional

# ─── 常量 ───────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 "
    "Safari/537.36 Edg/131.0.0.0"
)
DEFAULT_TIMEOUT = 10
MAX_TIMEOUT = 30
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_REDIRECTS = 5
CACHE_TTL_SEC = 900           # 15 min
CACHE_MAX_ENTRIES = 200
HOST_MIN_INTERVAL = 1.0       # 同主机最小请求间隔
ALLOWED_SCHEMES = {"http", "https"}

# ─── 内部状态 ───────────────────────────────────────────────
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_lock = threading.Lock()
_host_last_request: dict[str, float] = {}
_rate_lock = threading.Lock()


# ─── 安全检查 ───────────────────────────────────────────────
def _validate_headers(headers: Optional[dict]) -> None:
    """拒绝 CR/LF 注入。"""
    if not headers:
        return
    for k, v in headers.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"header 必须是 str: {k!r}={v!r}")
        if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
            raise ValueError(f"header injection detected in {k!r}")


def _check_ssrf(host: str, *, allow_private_ips: bool = False) -> None:
    """DNS 解析 host，拦截私网 / loopback / link-local / reserved / multicast。

    allow_private_ips=True 是给 WebSearchAgent 兼容老 localhost 抓取留的口子。
    新代码（UrlFetchAgent / /api/fetch）应保持默认 False。
    """
    if allow_private_ips:
        return
    if not host:
        raise ValueError("host 为空")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {host}: {e}") from e
    # 遍历 getaddrinfo 返回的所有地址（含 IPv4/IPv6 多条 A/AAAA 记录），逐条做私网拦截
    for info in infos:
        ip = info[4][0]
        try:
            ipo = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (ipo.is_private or ipo.is_loopback or ipo.is_link_local
                or ipo.is_reserved or ipo.is_multicast or ipo.is_unspecified):
            raise ValueError(f"blocked IP {ip} ({host})")


def _rate_limit(host: str) -> None:
    """同主机 1s 间隔，简易 sleep。"""
    if not host:
        return
    with _rate_lock:
        last = _host_last_request.get(host, 0.0)
        wait = HOST_MIN_INTERVAL - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        _host_last_request[host] = time.time()


# ─── HTML 正文抽取 ──────────────────────────────────────────
class _TextExtractor(HTMLParser):
    """简易抽正文：跳过 script/style，保留段落分隔。"""
    SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link", "svg"}
    BLOCK_TAGS = {"p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                  "li", "tr", "pre", "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag.lower() in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag.lower() in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        return raw.strip()


def extract_text(html: str) -> str:
    """从 HTML 抽出纯文本。"""
    if not html:
        return ""
    ex = _TextExtractor()
    try:
        ex.feed(html)
    except Exception:
        # HTMLParser 偶尔抛 AssertionError 等；退化到正则去标签
        return re.sub(r"<[^>]+>", "", html).strip()
    return ex.text()


# ─── 重定向控制 ─────────────────────────────────────────────
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """不在 opener 层自动跳转，由上层循环显式控制 + 每跳重做 SSRF。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 把目标 URL 塞回 HTTPError 让上层接住
        raise urllib.error.HTTPError(req.full_url, code, newurl, headers, fp)


def _build_opener(verify_ssl: bool = True) -> urllib.request.OpenerDirector:
    if verify_ssl:
        ctx = ssl.create_default_context()
    else:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        _NoRedirect(),
    )


# ─── 缓存 ────────────────────────────────────────────────────
def _cache_key(url: str, mode: str, max_chars: int) -> tuple:
    return (url, mode, max_chars)


def _cache_get(key: tuple) -> Optional[dict]:
    with _cache_lock:
        rec = _cache.get(key)
        if not rec:
            return None
        exp, val = rec
        if time.time() >= exp:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return dict(val)  # copy


def _cache_put(key: tuple, val: dict) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + CACHE_TTL_SEC, val)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


def _cache_clear() -> None:
    """供测试用，生产代码不调。"""
    with _cache_lock:
        _cache.clear()


# ─── 单次 HTTP 请求 ─────────────────────────────────────────
def _do_request(url: str, headers: dict, timeout: float,
                allow_private_ips: bool, verify_ssl: bool):
    """执行一次 HTTP（不跟随重定向），返回 response 对象 + 实际 URL。"""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"unsupported scheme {parsed.scheme!r} (允许: {sorted(ALLOWED_SCHEMES)})")
    if not parsed.hostname:
        raise ValueError("URL 缺 host")

    _check_ssrf(parsed.hostname, allow_private_ips=allow_private_ips)
    _rate_limit(parsed.hostname)

    req_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method="GET")
    opener = _build_opener(verify_ssl=verify_ssl)
    return opener.open(req, timeout=timeout)


def _read_capped(resp, max_bytes: int = MAX_BYTES) -> bytes:
    """读取响应，超过 max_bytes 直接报错。"""
    cl_hdr = resp.headers.get("Content-Length")
    if cl_hdr and cl_hdr.isdigit() and int(cl_hdr) > max_bytes:
        raise ValueError(f"content too large ({int(cl_hdr)} > {max_bytes})")
    data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"content too large (> {max_bytes} bytes)")
    return data


# ─── 主入口 ─────────────────────────────────────────────────
def fetch_url(url: str, *, mode: str = "auto",
              max_chars: int = 8000,
              headers: Optional[dict] = None,
              timeout: float = DEFAULT_TIMEOUT,
              allow_private_ips: bool = False,
              verify_ssl: bool = True) -> dict:
    """抓取 URL，返回标准 dict。

    返回结构：
      {"status": "success"|"error",
       "url": <final url>,           # 跟随重定向后的最终 url
       "content_type": "...",        # raw header
       "mode_used": "readability"|"json"|"raw"|"binary",
       "content": str | dict | list, # binary 时 content 是 base64
       "warnings": [...],            # redirected / truncated 等
       "http_status": int,           # 错误时也尽量带上
       "cached": True/False,
       "error": "..."}               # 仅 error 时
    """
    # 参数校验
    if mode not in ("auto", "readability", "raw", "json"):
        return {"status": "error", "error": f"invalid mode {mode!r}"}
    try:
        _validate_headers(headers)
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    timeout = max(1.0, min(float(timeout), MAX_TIMEOUT))
    max_chars = max(100, min(int(max_chars), 50000))

    bypass_cache = bool(headers and any(k.lower() == "authorization" for k in headers))
    cache_key = _cache_key(url, mode, max_chars)
    if not bypass_cache:
        cached = _cache_get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

    warnings: list[str] = []
    current_url = url
    hop = 0
    try:
        while True:
            try:
                resp = _do_request(current_url, headers or {}, timeout,
                                   allow_private_ips, verify_ssl)
            except urllib.error.HTTPError as he:
                # 重定向（NoRedirect 把 newurl 放在 reason 字段）
                if he.code in (301, 302, 303, 307, 308):
                    hop += 1
                    if hop > MAX_REDIRECTS:
                        return {"status": "error", "url": current_url,
                                "http_status": he.code,
                                "error": f"too many redirects (>{MAX_REDIRECTS})"}
                    newurl = he.reason if isinstance(he.reason, str) else \
                             (he.headers.get("Location") if he.headers else "")
                    if not newurl:
                        return {"status": "error", "url": current_url,
                                "http_status": he.code,
                                "error": "redirect 无 Location"}
                    # 同主机继续带 headers；跨主机剥 Authorization
                    new_parsed = urllib.parse.urlsplit(newurl)
                    old_parsed = urllib.parse.urlsplit(current_url)
                    if new_parsed.hostname != old_parsed.hostname and headers:
                        headers = {k: v for k, v in headers.items()
                                   if k.lower() != "authorization"}
                        warnings.append("redirect cross-host: stripped Authorization")
                    warnings.append(f"redirected {hop}: {current_url} -> {newurl}")
                    current_url = newurl
                    continue
                # 非重定向 HTTP 错误
                msg = f"HTTP {he.code}"
                if he.code == 401:
                    msg += "（可加 headers={'Authorization':'Bearer ...'}）"
                elif he.code == 403:
                    msg += "（可改用 WebSearch Agent 寻找替代来源）"
                return {"status": "error", "url": current_url, "http_status": he.code,
                        "error": msg, "warnings": warnings}
            break  # 非重定向，正常响应

        ct = (resp.headers.get("Content-Type") or "").lower()
        data = _read_capped(resp)
        final_url = current_url

        # ─── 根据 mode + Content-Type 决定解析方式 ───
        if mode == "json" or (mode == "auto" and "application/json" in ct):
            try:
                obj = json.loads(data.decode("utf-8", errors="replace"))
            except Exception as e:
                return {"status": "error", "url": final_url,
                        "error": f"JSON 解析失败: {e}", "warnings": warnings}
            result = {"status": "success", "url": final_url, "content_type": ct,
                      "mode_used": "json", "content": obj,
                      "warnings": warnings, "cached": False}

        elif mode == "readability" or (mode == "auto" and "text/html" in ct):
            text = data.decode("utf-8", errors="replace")
            extracted = extract_text(text)
            if len(extracted) > max_chars:
                extracted = extracted[:max_chars] + "..."
                warnings.append(f"content truncated at {max_chars} chars")
            result = {"status": "success", "url": final_url, "content_type": ct,
                      "mode_used": "readability", "content": extracted,
                      "warnings": warnings, "cached": False}

        elif mode == "raw" or (mode == "auto" and ct.startswith("text/")):
            text = data.decode("utf-8", errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                warnings.append(f"content truncated at {max_chars} chars")
            result = {"status": "success", "url": final_url, "content_type": ct,
                      "mode_used": "raw", "content": text,
                      "warnings": warnings, "cached": False}

        elif mode == "auto" and (ct.startswith("application/pdf") or ct.startswith("image/")):
            b64 = base64.b64encode(data).decode("ascii")
            warnings.append("binary content; 建议用 DocParse/Vision Agent 解析")
            result = {"status": "success", "url": final_url, "content_type": ct,
                      "mode_used": "binary", "content": b64,
                      "warnings": warnings, "cached": False}

        else:
            return {"status": "error", "url": final_url, "content_type": ct,
                    "error": f"unsupported content-type {ct!r}",
                    "warnings": warnings}

        if not bypass_cache:
            _cache_put(cache_key, result)
        return result

    except ValueError as ve:
        return {"status": "error", "url": url, "error": str(ve), "warnings": warnings}
    except urllib.error.URLError as ue:
        return {"status": "error", "url": url,
                "error": f"network error: {ue.reason}", "warnings": warnings}
    except socket.timeout:
        return {"status": "error", "url": url, "error": "timeout", "warnings": warnings}
    except Exception as e:
        return {"status": "error", "url": url,
                "error": f"{type(e).__name__}: {e}", "warnings": warnings}


def fetch_url_tool_handler(**kw) -> dict:
    """ToolRegistry 注册的同步入口（agent='any', risk='medium'）。"""
    return fetch_url(
        url=kw.get("url", ""),
        mode=kw.get("mode", "auto"),
        max_chars=int(kw.get("max_chars", 8000)),
        headers=kw.get("headers") or None,
        timeout=float(kw.get("timeout", DEFAULT_TIMEOUT)),
    )


__all__ = [
    "fetch_url", "fetch_url_tool_handler", "extract_text",
    "USER_AGENT", "DEFAULT_TIMEOUT", "MAX_TIMEOUT", "MAX_BYTES",
    "MAX_REDIRECTS", "CACHE_TTL_SEC",
]
