"""
Mini Thalamus Web Server —— 多 Agent 统一入口
FastAPI + SSE 流式输出
"""

import sys
import os
import json
import time
import uuid
import asyncio
import queue
import hashlib

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from mini_thalamus import PACKAGE_ROOT
from mini_thalamus.config import ROUTER_AGENTS, SKILLS_DIR, MCP_CONFIG_PATH, MCP_AUTO_START, SKILL_AUTO_DISCOVER
from mini_thalamus.router import Router  # 必须在 Agent 加载前 import，防止 sys.path 污染后拿错 config
from mini_thalamus.logging_config import setup as _setup_logging, get_logger
from mini_thalamus.agent_runtime import get_runtime
import threading

# 初始化日志（最先执行，后续调试依赖它）
_setup_logging()
_log = get_logger("web_server")

# ─── 动态 Agent 系统（MCP + Skill）───
_dynamic_cache = {}  # {"skill_loader": ..., "mcp_manager": ..., "started": bool}
_startup_lock = threading.Lock()  # 保证 _ensure_startup 只执行一次
_startup_ready = threading.Event()
_agents_cache_lock = threading.Lock()  # 保护 _agents_cache 重建与清除

# ─── 安全措施 ───
import re

# API Key 认证（通过环境变量 MT_API_KEY 启用，留空则不启用）
import secrets as _secrets
_SERVER_API_KEY = os.environ.get("MT_API_KEY", "")

# 简易 IP 限流
import time as _time
from collections import defaultdict as _defaultdict
_rate_limit_store: dict = _defaultdict(lambda: {"tokens": 30, "last": 0.0})
_rate_limit_lock = threading.Lock()
RATE_LIMIT_PER_MINUTE = int(os.environ.get("MT_RATE_LIMIT", "30"))  # 默认30次/分钟

def _rate_limit_check(request: Request) -> bool:
    """简易 token bucket 限流。每 IP 每分钟 RATE_LIMIT_PER_MINUTE 次。"""
    client_ip = request.client.host if request.client else "unknown"
    now = _time.time()
    with _rate_limit_lock:
        bucket = _rate_limit_store[client_ip]
        elapsed = now - bucket["last"]
        bucket["last"] = now
        # 每秒恢复 1/60 的 token
        bucket["tokens"] = min(RATE_LIMIT_PER_MINUTE, bucket["tokens"] + elapsed * RATE_LIMIT_PER_MINUTE / 60.0)
        if bucket["tokens"] < 1:
            return False
        bucket["tokens"] -= 1
    return True

# 导入 nl2sql_agent 的 Prompt Guard（比当前的简单 regex 更全面）
try:
    from nl2sql_agent.interfaces.prompt_guard import _build_rule_guard
    _nl2sql_prompt_guard = _build_rule_guard()
except Exception:
    _nl2sql_prompt_guard = None

# 伪指令标记（防止外部内容被 LLM 当作系统指令）
_INJECTION_PATTERNS = [
    r'<system[^>]*>', r'</system>', r'<user[^>]*>', r'</user>',
    r'<assistant[^>]*>', r'</assistant>', r'<instruction[^>]*>', r'</instruction>',
    r'<system-reminder[^>]*>', r'</system-reminder>',
    r'忽略.*指令', r'ignore.*instruction', r'你之前的指令',
    r'你的新任务是', r'your new task is',
]

def sanitize_input(text: str) -> tuple:
    """清洗外部输入，返回 (清洗后文本, 是否检测到注入企图)"""
    flagged = False
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            text = re.sub(pat, lambda m: f'[{m.group().replace("<","").replace(">","")}]', text)
            flagged = True
    return text[:4000], flagged  # 限制长度

def wrap_boundary(content: str, tag: str) -> str:
    """用 XML 边界标记包裹内容，区分指令和数据"""
    return f"<{tag}>\n{content}\n</{tag}>"

def log_security_event(event_type: str, detail: str):
    """记录安全事件"""
    _log.warning("[SECURITY] %s: %s", event_type, detail[:200])

# ─── 懒加载 Agent ───
_agents_cache = None


def _filter_history_for_query(query: str, conv_history: str) -> str:
    """纯规则版历史过滤（REPAIR_PLAN #12：去掉同步阻塞 LLM 调用）。

    模糊地带（1-4 个重叠词）从 LLM yes/no 判断改为规则兜底：
      - 有任意重叠 → 保留（不漏）
      - 完全无重叠且历史很长 → 截断
    """
    if not conv_history:
        return ""

    if len(conv_history) < 500:
        return conv_history[:2000]

    q_words = set(re.findall(r'[一-鿿]{2,}|[a-zA-Z]{3,}', query.lower()))
    h_words = set(re.findall(r'[一-鿿]{2,}|[a-zA-Z]{3,}', conv_history.lower()))
    overlap = q_words & h_words

    # 明确延续
    if any(w in query for w in ["继续", "再", "还是", "版本", "同样的", "也", "接着"]):
        return conv_history[-2000:]

    # 明确切换
    if not overlap and len(conv_history) > 800:
        return ""

    # 明确相同
    if len(overlap) >= 5:
        return conv_history[:2000]

    # 模糊地带（1-4 个重叠词）：纯规则兜底——有重叠就保留尾部 2000 字
    return conv_history[-2000:] if overlap else ""


def _make_mcp_handler(mm, server, tool_name):
    """创建 MCP 工具的 handler 闭包"""
    def handler(**kwargs):
        return mm.call_tool_sync(server, tool_name, kwargs)
    return handler


def _ensure_startup():
    """懒初始化 MCP + Skill + AgentRegistry。仅在首次请求时调用一次（线程安全）。"""
    global _dynamic_cache
    if _dynamic_cache.get("started"):
        return
    with _startup_lock:
        if _dynamic_cache.get("started"):  # double-checked locking
            return
        if _dynamic_cache.get("initializing"):
            if _startup_ready.wait(timeout=10):
                return
            _dynamic_cache["initializing"] = False
            raise RuntimeError("Web runtime initialization failed or timed out")
        _dynamic_cache["initializing"] = True

        # Web and CLI share the same runtime and registry.
        runtime = get_runtime()
        registry = runtime.registry
        _dynamic_cache["runtime"] = runtime
        _dynamic_cache["registry"] = registry

        # 30 天未活跃的旧会话自动清理
        try:
            from mini_thalamus.memory_service import get_memory_service
            get_memory_service().store.purge_expired_sessions(
                time.time() - 30 * 86400
            )
        except Exception:
            pass

        # ─── Credential Store（FEATURE_PLAN.md §5）───
        try:
            from mini_thalamus.credentials import get_store
            cred_store = get_store()
            _dynamic_cache["credentials"] = cred_store
            _log.info("CredentialStore 已加载（%d 条凭证）", len(cred_store.list()))
        except Exception as e:
            _log.warning("CredentialStore 初始化失败: %s", e)

        # Skill 发现
        if SKILL_AUTO_DISCOVER:
            from mini_thalamus.skill_loader import SkillLoader
            sl = SkillLoader(SKILLS_DIR)
            count = sl.discover()
            _dynamic_cache["skill_loader"] = sl
            if sl.errors:
                for err in sl.errors[:5]:
                    _log.warning("Skill 加载警告: %s", err)
            _log.info("发现 %d 个 Skill", count)
            # 注册到 AGENT_LOADERS
            sl.register_in_loader_registry(AGENT_LOADERS)
            # 同步注册到 AgentRegistry
            registry.register_skill_loader(sl)

    # Vision Agent 预加载 EasyOCR（后台线程，不阻塞启动）
    import threading as _thr, os as _os
    _os.environ["FLAGS_use_onednn"] = "0"
    _os.environ["FLAGS_pir_use_onednn"] = "0"
    _os.environ["FLAGS_enable_pir_api"] = "0"
    _os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    def _preload_vision():
        try:
            import easyocr
            easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            _log.info("Vision: EasyOCR 已预加载")
        except Exception as e:
            _log.warning("Vision: EasyOCR 预加载失败: %s", e)
    _thr.Thread(target=_preload_vision, daemon=True).start()

    # MCP 启动
    if MCP_AUTO_START:
        from mini_thalamus.mcp_client import MCPClientManager
        mm = MCPClientManager(MCP_CONFIG_PATH)
        if mm.load_config():
            mm.start()
            results = mm.connect_all()
            _dynamic_cache["mcp_manager"] = mm
            # 注册 MCP 工具到 ToolRegistry
            from mini_thalamus.tool_registry import registry as tool_registry, ToolDef
            for tool in mm.get_all_tools():
                server = tool.get("server", "unknown")
                tname = f"mcp.{server}.{tool['name']}"
                tool_registry.register(ToolDef(
                    name=tname,
                    description=tool.get("description", f"MCP tool: {tool['name']}"),
                    parameters=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    handler=_make_mcp_handler(mm, server, tool["name"]),
                    agent="mcp",
                    source="mcp",
                    timeout=60,
                ))
            _log.info("MCP %d 个服务器, %d 个工具", len(results), len(mm.get_all_tools()))
            # 注册 MCP agent
            if mm.get_all_tools():
                # AGENT_LOADERS 被 get_agents()/delete_skill()/crystallize() 在
                # _agents_cache_lock 下读写；此处的 MCP loader 写入发生在锁外的
                # post-startup 阶段，必须同样进锁，否则 get_agents 迭代 AGENT_LOADERS
                # 时可能撞上本行的 dict 写入（dictionary changed size during iteration）。
                with _agents_cache_lock:
                    AGENT_LOADERS["mcp"] = lambda mm2=mm: (
                        __import__("mini_thalamus.agents.mcp_agent", fromlist=["MCPAgent"]).MCPAgent(registry, mm2))
                # 同步注册到 AgentRegistry
                registry.register_mcp_tools(mm)
                # FEATURE_PLAN.md §4：MCP agent 强制 auto_chainable=False（R3）
                # MCPAgent.capabilities 已声明，此处 meta 层再兜底一次
                try:
                    from mini_thalamus.agent_registry import AgentRegistry as _AR
                    _mcp_meta = registry.get("mcp")
                    if _mcp_meta and "capabilities" in _mcp_meta:
                        _mcp_meta["capabilities"]["auto_chainable"] = False
                except Exception:
                    pass

    # ─── 注册 fetch_url 工具（FEATURE_PLAN.md §3）───
    # agent="any" 让 Code/DeepResearch/MCP 等都能在工具调用层直接抓取，
    # url_fetch_agent 是它的 Router 直达入口。risk="medium" 因为有网络出口；
    # SSRF/scheme/size/CRLF 防护在 tools.url_fetcher 内部完成。
    try:
        from mini_thalamus.tool_registry import registry as _tool_registry, ToolDef
        from mini_thalamus.tools.url_fetcher import fetch_url_tool_handler
        if _tool_registry.get("fetch_url") is None:
            _tool_registry.register(ToolDef(
                name="fetch_url",
                description=("抓取一个 http/https URL 的内容。HTML→正文文本；"
                             "JSON→结构化字典；二进制 (PDF/image)→base64+content-type 提示。"
                             "私有 IP / file:// 被拒绝。"),
                parameters={
                    "type": "object",
                    "properties": {
                        "url":       {"type": "string", "format": "uri"},
                        "mode":      {"type": "string",
                                      "enum": ["auto", "readability", "raw", "json"],
                                      "default": "auto"},
                        "max_chars": {"type": "integer", "default": 8000,
                                      "minimum": 100, "maximum": 50000},
                        "headers":   {"type": "object",
                                      "additionalProperties": {"type": "string"}},
                        "timeout":   {"type": "number", "default": 10,
                                      "minimum": 1, "maximum": 30},
                    },
                    "required": ["url"],
                },
                handler=fetch_url_tool_handler,
                agent="any",
                risk="medium",
                timeout=35,
                source="builtin",
            ))
            _log.info("已注册工具 fetch_url（agent=any, risk=medium）")
    except Exception as e:
        _log.warning("fetch_url 工具注册失败: %s", e)

    # MCP、工具和 Agent 注册全部完成后再原子发布 ready 状态。
    _dynamic_cache["started"] = True
    _dynamic_cache["initializing"] = False
    _startup_ready.set()


def get_agents():
    global _agents_cache
    _ensure_startup()
    if _agents_cache is not None:
        return _agents_cache
    with _agents_cache_lock:
        if _agents_cache is not None:  # double-checked locking
            return _agents_cache
        runtime = _dynamic_cache.get("runtime") or get_runtime()
        cache = runtime.load_all()

        _log.info("已加载 %d 个 Agent: %s", len(cache), list(cache.keys()))
        _agents_cache = cache
    return _agents_cache


# Compatibility view only. AgentRegistry is the single source of truth.
AGENT_LOADERS = get_runtime().loader_registry()


def _load_agent(name: str):
    # Skill agent: 通过 SkillLoader 获取
    if name.startswith("skill:"):
        sl = _dynamic_cache.get("skill_loader")
        if sl:
            run_func = sl.get_skill(name)
            if run_func:
                from mini_thalamus.skill_loader import _SkillWrapper
                return _SkillWrapper(run_func, name)
        return None

    loader = AGENT_LOADERS.get(name)
    if not loader:
        _log.warning("Agent '%s' 无注册加载器", name)
        return None
    try:
        return loader()
    except Exception as e:
        _log.exception("Agent '%s' 加载失败: %s", name, e)
        return None


# ─── FastAPI App ───
app = FastAPI(title="Mini Thalamus", description="多 Agent 统一协作平台")

# 静态文件
# TODO: TTL cleanup for static/generated/ files — a periodic cron or
# file-age-based purge should clean generated artifacts (charts, temp
# uploads, cached reports) older than e.g. 24 hours to prevent disk bloat.
static_dir = os.path.join(PACKAGE_ROOT, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, encoding="utf-8").read())
    return HTMLResponse("<h1>Mini Thalamus</h1><p>前端页面未找到。</p>")


def _verify_api_key(request: Request):
    """API Key 认证中间件。MT_API_KEY 为空时跳过认证。"""
    if not _SERVER_API_KEY:
        return "local_anonymous"  # local single-user mode
    # 支持 ?api_key=... 和 Authorization: Bearer ... 两种方式
    query_key = request.query_params.get("api_key")
    if query_key and _secrets.compare_digest(query_key, _SERVER_API_KEY):
        return "api_key:" + hashlib.sha256(
            query_key.encode("utf-8")
        ).hexdigest()
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if _secrets.compare_digest(token, _SERVER_API_KEY):
            return "api_key:" + hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _request_user_id(request: Request = None) -> str:
    """Return only a server-authenticated principal, never a body field."""
    if request is None:
        return "local_anonymous"
    principal = request.scope.get("user")
    if principal is not None and getattr(principal, "is_authenticated", False):
        identity = (
            getattr(principal, "subject", None)
            or getattr(principal, "user_id", None)
            or getattr(principal, "id", None)
        )
        if identity is None:
            raise HTTPException(
                status_code=500,
                detail="Authenticated principal lacks a stable subject",
            )
        return "principal:" + hashlib.sha256(
            str(identity).encode("utf-8")
        ).hexdigest()
    return getattr(request.state, "authenticated_user_id", "local_anonymous")


def _resolve_web_session(user_id: str, session_id: str):
    from mini_thalamus.memory_service import get_memory_service
    from mini_thalamus.session_identity import InvalidSessionId
    from mini_thalamus.session_manager import SessionOwnershipError

    try:
        return get_memory_service().resolve_session(user_id, session_id)
    except InvalidSessionId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionOwnershipError as exc:
        raise HTTPException(status_code=403, detail="Session access denied") from exc


@app.middleware("http")
async def _api_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        request.state.authenticated_user_id = _verify_api_key(request)
    return await call_next(request)


# ─── 请求模型 ───
class ChatRequest(BaseModel):
    message: str = ""
    agent: Optional[str] = None
    session_id: Optional[str] = None
    # FEATURE_PLAN.md §1 P0.4 (audit R8) chain_id 预埋：
    # /api/chat/continue 用它校验 sid 上的暂停链是不是本次启动的那一条；
    # 留给 Auto-Chain 上线后真正消费。Optional，老前端不传也兼容。
    chain_id: Optional[str] = None
    # FEATURE_PLAN.md §4 Auto-Chain 预留（不影响本次 P0 行为）
    require_confirm: Optional[bool] = None
    approved_call_ids: Optional[list[str]] = None


# ─── Agent 列表接口 ───
@app.get("/api/agents")
async def list_agents():
    """返回所有可用 Agent（内置 + Skill + MCP + 动态发现）"""
    _ensure_startup()
    agents_list = []

    # FEATURE_PLAN.md §1 P0.4 / Audit R15:
    # 用 get_router_agents() 拿到内置 + AgentRegistry 发现的全部 Agent
    # （url_fetch / shell 这些只在 agents/ 目录里有 AGENT_META 但不在
    # _BUILTIN_ROUTER_AGENTS 静态字典中的，也能在 Sidebar 出现）。
    try:
        from mini_thalamus.config import get_router_agents
        all_agents = get_router_agents()
    except Exception as e:
        _log.warning("get_router_agents 失败，退化为静态 ROUTER_AGENTS: %s", e)
        all_agents = ROUTER_AGENTS

    for name, info in all_agents.items():
        agents_list.append({
            "id": name,
            "name": info.get("name", name),
            "icon": info.get("icon", "🔧"),
            "description": info.get("description", ""),
            "source": info.get("source", "builtin"),
        })

    # Skill Agent
    sl = _dynamic_cache.get("skill_loader")
    if sl:
        existing_ids = {a["id"] for a in agents_list}
        for sid, sm in sl.list_as_agent_configs().items():
            if sid in existing_ids:
                continue
            agents_list.append({
                "id": sid,
                "name": sm["name"],
                "icon": sm.get("icon", "🧩"),
                "description": sm["description"],
                "source": "skill",
            })

    # MCP Agent
    mm = _dynamic_cache.get("mcp_manager")
    if mm and mm.get_all_tools():
        if not any(a["id"] == "mcp" for a in agents_list):
            tool_count = len(mm.get_all_tools())
            servers = list(mm.servers.keys())
            agents_list.append({
                "id": "mcp",
                "name": "MCP 工具代理",
                "icon": "🔌",
                "description": f"调用外部 MCP 工具：{', '.join(servers)}（{tool_count} 个工具）",
                "source": "mcp",
                "servers": servers,
                "tool_count": tool_count,
            })

    return {"agents": agents_list}


@app.post("/api/agents/reload")
async def reload_agents(agent_id: str = None):
    """热插拔 Agent：不重启进程完成 新增 / 替换 / 移除。

    不带 agent_id 时对 agents/ 目录做增量重扫——新文件注册、改动过的热替换、
    已删除的注销，未变动的保留其热实例。带 agent_id 时只热替换那一个。

    任一 Agent 重载失败只会记进 report.errors，不影响其它 Agent，也不影响
    已经拿到旧实例的在途请求。
    """
    global _agents_cache
    from mini_thalamus.agent_registry import AgentLoadError

    _ensure_startup()
    registry = _dynamic_cache.get("registry")
    if registry is None:
        return JSONResponse(
            {"ok": False, "error": "AgentRegistry 未初始化"}, status_code=503
        )

    try:
        if agent_id:
            report = {"reloaded": [registry.reload_agent(agent_id)["agent_id"]],
                      "errors": {}}
        else:
            report = registry.rescan()
            sl = _dynamic_cache.get("skill_loader")
            if sl:
                registry.register_skill_loader(sl)
    except AgentLoadError as e:
        # 新版本构建失败：旧版本原样保留，服务继续可用。
        return JSONResponse({"ok": False, "error": str(e)}, status_code=409)

    with _agents_cache_lock:
        _agents_cache = None
        # AGENT_LOADERS 是 import 时取的快照；就地刷新而不是重新赋值，
        # 这样其它模块已经持有的引用同样能看到新 Agent。
        AGENT_LOADERS.clear()
        AGENT_LOADERS.update(get_runtime().loader_registry())
    _log.info("Agent 热重载完成: %s", report)
    return {"ok": True, "report": report}


@app.get("/api/agents/health")
async def agents_health():
    """每个 Agent 的熔断状态 —— 熔断/不可用的也要列出来。

    /api/agents 走的是路由候选视图，被熔断的 Agent 已经不在里面了，
    所以排查时需要这个接口：能看到「谁被熔断了、为什么、多久之后会重试」。
    """
    from mini_thalamus.agent_registry import (
        CIRCUIT_COOLDOWN_SECONDS,
        CIRCUIT_FAILURE_THRESHOLD,
    )

    _ensure_startup()
    registry = _dynamic_cache.get("registry")
    if registry is None:
        return JSONResponse(
            {"ok": False, "error": "AgentRegistry 未初始化"}, status_code=503
        )

    healthy, quarantined = {}, {}
    for agent_id, entry in registry.entries().items():
        health = dict(entry.get("health") or {})
        health["status"] = entry.get("status", "")
        health["error"] = entry.get("error", "")
        health["serving"] = bool(registry._is_serving(entry))
        (healthy if health["serving"] else quarantined)[agent_id] = health
    return {
        "ok": True,
        "serving": sorted(healthy),
        "quarantined": sorted(quarantined),
        "detail": {**healthy, **quarantined},
        "threshold": CIRCUIT_FAILURE_THRESHOLD,
        "cooldown_seconds": CIRCUIT_COOLDOWN_SECONDS,
    }


@app.post("/api/chat/continue")
async def continue_chain(req: ChatRequest):
    """用户编辑完中间数据后，继续执行 Agent 链的剩余步骤"""
    return JSONResponse(
        {
            "answer": "旧版直调续传已停用；请携带 approved_call_ids 重新请求 /api/chat 或 /api/chat/stream",
            "error": "legacy_continue_disabled",
        },
        status_code=410,
    )


# ─── Skill 管理接口 ───
@app.get("/api/skills")
async def list_skills():
    """返回所有已安装 Skill 的详情"""
    _ensure_startup()
    sl = _dynamic_cache.get("skill_loader")
    if not sl:
        return {"skills": []}
    skills = []
    for sid, sd in sl.skills.items():
        skills.append({
            "id": sid,
            "name": sd.meta.get("name", sid),
            "description": sd.meta.get("description", ""),
            "icon": sd.meta.get("icon", "🧩"),
            "keywords": sd.meta.get("keywords", []),
            "file": sd.source_path,
            "loaded_at": sd.loaded_at,
        })
    return {"skills": skills}


@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str):
    """删除一个 Skill 文件"""
    import os as _os
    sl = _dynamic_cache.get("skill_loader")
    if not sl:
        return {"ok": False, "error": "Skill 系统未初始化"}
    sd = sl.skills.get(skill_id)
    if not sd:
        return {"ok": False, "error": f"Skill '{skill_id}' 不存在"}
    try:
        _os.remove(sd.source_path)
        # 清除缓存，下次重新发现
        global _agents_cache
        with _agents_cache_lock:
            _agents_cache = None
            sl.skills.pop(skill_id, None)
            if skill_id in AGENT_LOADERS:
                del AGENT_LOADERS[skill_id]
            registry = _dynamic_cache.get("registry")
            if registry is not None:
                try:
                    registry.unregister(skill_id)
                except Exception:
                    pass
        _log.info("已删除 Skill: %s", skill_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── MCP 状态接口 ───
@app.get("/api/mcp/status")
async def mcp_status():
    """返回 MCP 服务器连接状态"""
    _ensure_startup()
    mm = _dynamic_cache.get("mcp_manager")
    if not mm:
        return {"available": False, "servers": {}, "tools": [], "config_loaded": False}
    tools = mm.get_all_tools()
    servers = {}
    for name in mm.servers:
        connected = name in mm._connections
        servers[name] = {"connected": connected, "config": {
            "command": mm.servers[name].command,
            "args": mm.servers[name].args,
        }}
    return {
        "available": True,
        "config_loaded": True,
        "servers": servers,
        "tools": [{"name": t["name"], "server": t.get("server", "?"), "description": t.get("description", "")} for t in tools],
        "tool_count": len(tools),
    }


# ─── URL Fetch 接口（FEATURE_PLAN.md §3）───
# 前端 link preview / 浏览插件用。共用 tools.url_fetcher.fetch_url，
# 默认 max_chars/timeout 比 Agent 路径更小（链接预览不需要 8k）。
@app.post("/api/fetch")
async def api_fetch(req: dict):
    """直抓 URL —— 不走 Router/Agent，前端预览用。"""
    from mini_thalamus.tools.url_fetcher import fetch_url
    url = (req or {}).get("url", "")
    if not url:
        return JSONResponse({"status": "error", "error": "缺 url"}, status_code=400)
    # 过滤用户提供的 headers（只允许安全的请求头）
    unsafe_headers = {"host", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                      "x-real-ip", "cookie", "set-cookie", "authorization", "proxy-authorization",
                      "x-request-id", "x-trace-id"}
    user_headers = (req.get("headers") or None)
    if user_headers:
        user_headers = {k: v for k, v in user_headers.items()
                       if k.lower() not in unsafe_headers}
    return await asyncio.to_thread(
        fetch_url,
        url,
        mode=(req.get("mode") or "auto"),
        max_chars=int(req.get("max_chars") or 1500),
        timeout=float(req.get("timeout") or 8),
        headers=user_headers,
    )


# ─── Credential Store 接口（FEATURE_PLAN.md §5）───
class CredentialPayload(BaseModel):
    name: str
    type: str = "api_key"
    value: dict
    scope: str = "custom"
    agents: list = ["*"]
    env_map: Optional[dict] = None
    note: str = ""


def _default_env_map(type_: str, name: str) -> dict:
    """type 默认的 env_map 模板，用户没传时套用。"""
    upper = name.upper().replace("-", "_")
    return {
        "api_key":   {"api_key": f"{upper}_API_KEY"},
        "token":     {"token":   f"{upper}_TOKEN"},
        "aws_keypair": {
            "access_key": "AWS_ACCESS_KEY_ID",
            "secret_key": "AWS_SECRET_ACCESS_KEY",
            "region":     "AWS_DEFAULT_REGION",
        },
        "basic_auth": {"username": f"{upper}_USER", "password": f"{upper}_PASS"},
    }.get(type_, {})


@app.get("/api/credentials")
async def list_credentials():
    _ensure_startup()
    store = _dynamic_cache.get("credentials")
    if not store:
        return []
    return store.list(redact=True)


@app.post("/api/credentials")
async def upsert_credential(p: CredentialPayload):
    _ensure_startup()
    store = _dynamic_cache.get("credentials")
    if not store:
        return JSONResponse({"ok": False, "error": "CredentialStore 服务暂不可用，请稍后重试"}, status_code=503)
    if not p.name or not isinstance(p.value, dict) or not p.value:
        return JSONResponse({"ok": False, "error": "name / value 不能为空"}, status_code=400)
    env_map = p.env_map if p.env_map else _default_env_map(p.type, p.name)
    try:
        store.set(p.name, p.value, scope=p.scope, agents=p.agents,
                  type_=p.type, env_map=env_map, note=p.note)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "name": p.name}


@app.delete("/api/credentials/{name}")
async def delete_credential(name: str):
    _ensure_startup()
    store = _dynamic_cache.get("credentials")
    if not store:
        return {"ok": False, "error": "CredentialStore 未初始化"}
    return {"ok": store.delete(name)}


@app.post("/api/credentials/test")
async def test_credential(body: dict):
    """对凭证做一次最低成本的活性测试。scope 决定测试方式。"""
    _ensure_startup()
    store = _dynamic_cache.get("credentials")
    if not store:
        return {"ok": False, "error": "CredentialStore 未初始化"}
    name = body.get("name")
    if not name:
        return {"ok": False, "error": "缺 name"}
    cred = store.get(name)
    if not cred:
        return {"ok": False, "error": "凭证未找到"}
    # 找到 scope
    entry = next((e for e in store.list(redact=False) if e["name"] == name), None)
    scope = (entry or {}).get("scope", "custom")
    try:
        if scope == "llm":
            from openai import OpenAI
            base = "https://api.deepseek.com/v1" if name == "deepseek" else "https://api.openai.com/v1"
            client = OpenAI(api_key=cred.get("api_key", ""), base_url=base)
            client.models.list()
            return {"ok": True}
        if scope == "url_fetch":
            import urllib.request
            if name == "github":
                req = urllib.request.Request(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {cred.get('token','')}",
                             "User-Agent": "MiniThalamus-CredTest"})
            elif name == "bing":
                req = urllib.request.Request(
                    "https://api.bing.microsoft.com/v7.0/search?q=test",
                    headers={"Ocp-Apim-Subscription-Key": cred.get("api_key", "")})
            else:
                return {"ok": True, "warning": f"未为 {name} 配置 url_fetch 测试，仅基本格式校验通过"}
            urllib.request.urlopen(req, timeout=6).read(100)
            return {"ok": True}
        return {"ok": True, "warning": f"未为 scope={scope} 配置测试"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ─── Skill 固化接口 ───
@app.post("/api/skill/crystallize")
async def crystallize_session(req: ChatRequest, request: Request = None):
    """将当前会话的工作流固化为一键复用的 Skill"""
    sid = req.session_id
    if not sid:
        return {"ok": False, "error": "缺少 session_id"}
    user_id = _request_user_id(request)
    _resolve_web_session(user_id, sid)
    try:
        from mini_thalamus.crystallize import crystallize
        result = crystallize(sid, req.message or None)
        if result.get("ok"):
            # 触发重新加载
            global _agents_cache, _dynamic_cache
            with _agents_cache_lock:
                _agents_cache = None
                sl = _dynamic_cache.get("skill_loader")
                if sl:
                    sl.discover()
                    sl.register_in_loader_registry(AGENT_LOADERS)
                    # 同步进 AgentRegistry：Router 的路由候选来自 registry，
                    # 只写 AGENT_LOADERS 的话新 Skill 能被执行却路由不到。
                    registry = _dynamic_cache.get("registry")
                    if registry is not None:
                        registry.register_skill_loader(sl)
            _log.info("Skill 已固化并重新加载: %s", result.get('skill_id'))
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── 会话清理接口 ───
@app.post("/api/session/cleanup")
async def cleanup_session_endpoint(req: ChatRequest, request: Request = None):
    """对话结束 → 清理临时文件"""
    sid = req.session_id
    if not sid:
        return {"status": "ok", "message": "无 session_id"}
    user_id = _request_user_id(request)
    _resolve_web_session(user_id, sid)
    memory.cleanup_session(sid)
    try:
        from mini_thalamus.memory_service import get_memory_service
        get_memory_service().sessions.delete(sid)
    except Exception:
        pass
    _log.info("会话 %s: 已清理", sid[:8])
    return {"status": "ok", "message": f"会话 {sid[:8]} 已清理"}


@app.get("/api/session/history")
async def session_history_endpoint(session_id: str, request: Request = None):
    """Return persisted turns for a session."""
    user_id = _request_user_id(request)
    _resolve_web_session(user_id, session_id)
    turns = memory.list_turns(session_id)
    return {
        "session_id": session_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "query": turn.query,
                "answer": turn.answer,
                "agent_id": turn.agent_id,
                "status": turn.status,
                "tags": list(turn.tags),
                "created_at": turn.created_at,
                "completed_at": turn.completed_at,
            }
            for turn in turns
        ],
    }


# ─── 普通对话接口 ───
@app.get("/api/sessions")
async def list_sessions_endpoint(request: Request = None):
    """Return the current user's saved sessions, newest first."""
    user_id = _request_user_id(request)
    with memory.database.connection() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.summary, s.updated_at,
                   (SELECT COUNT(*) FROM turns t
                    WHERE t.session_id = s.session_id) AS turn_count,
                   (SELECT query FROM turns t
                    WHERE t.session_id = s.session_id
                    ORDER BY t.sequence_no DESC LIMIT 1) AS last_query
            FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.updated_at DESC
            LIMIT 50
            """,
            (user_id,),
        ).fetchall()
    return {
        "sessions": [
            {
                "session_id": row["session_id"],
                "summary": row["summary"] or row["last_query"] or "",
                "last_query": row["last_query"] or "",
                "turn_count": row["turn_count"] or 0,
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    }


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request = None):
    """对话接口：自动路由或手动指定 Agent"""
    session_id = req.session_id or str(uuid.uuid4())[:8]
    user_id = _request_user_id(request)

    # 手动指定 Agent（兼容内置 + 动态 Agent）
    if req.agent and req.agent in get_agents():
        return await _auto_route(
            f"@{req.agent} {req.message}",
            session_id,
            user_id=user_id,
            approved_call_ids=req.approved_call_ids,
        )

    # 自动路由
    return await _auto_route(
        req.message,
        session_id,
        user_id=user_id,
        approved_call_ids=req.approved_call_ids,
    )


async def _auto_route(
    message: str,
    session_id: str,
    *,
    user_id: str,
    approved_call_ids=None,
    on_stage=None,
    require_confirm: bool = False,
    files: list = None,
):
    """Execute automatic routing through the shared MainAgent orchestrator."""
    from mini_thalamus.orchestrator import Orchestrator
    from mini_thalamus.sandbox import ExecutionPolicy
    from mini_thalamus.config import ALLOW_LOCAL_EXECUTION, SANDBOX_USE_DOCKER

    runtime = _dynamic_cache.get("runtime") or get_runtime()
    registry = _dynamic_cache.get("registry")
    execution_policy = None
    if require_confirm:
        execution_policy = ExecutionPolicy(
            workspace_root=os.path.dirname(os.path.abspath(__file__)),
            local_execution_enabled=ALLOW_LOCAL_EXECUTION,
            docker_backend_enabled=SANDBOX_USE_DOCKER,
            auto_approve=False,
        )
    orchestrator = Orchestrator(
        runtime=runtime,
        registry=registry,
        execution_policy=execution_policy,
    )
    return await asyncio.to_thread(
        orchestrator.process,
        message,
        user_id,
        session_id,
        approved_call_ids,
        on_stage=on_stage,
        files=files,
    )


# 多步规划已合并到 Router.route()（一次 LLM 调用同时返回单/多步 agents 数组），
# 旧的 _plan_agent_chain + MULTI_STEP_PROMPT 已删除（REPAIR_PLAN #17）。


# ─── SSE 流式对话接口 ───
@app.post("/api/chat/stream")
@app.post("/api/chat/stream/json")
async def chat_stream_json(req: ChatRequest, request: Request):
    """JSON 模式（无文件时用）"""
    return await _chat_stream_impl(
        message=req.message, agent=req.agent, session_id=req.session_id,
        files=[], require_confirm=req.require_confirm, request=request,
        approved_call_ids=req.approved_call_ids,
    )


# 文件上传安全限制
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_MIME_TYPES = {
    "application/pdf", "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain", "text/csv", "text/markdown",
    "image/png", "image/jpeg", "image/webp", "image/bmp",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@app.post("/api/chat/stream/form")
async def chat_stream_form(
    message: str = Form(""),
    agent: str = Form(""),
    session_id: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    request: Request = None,
):
    """FormData 模式（有文件时用）"""
    file_data = []
    for f in (files or []):
        if f.filename:
            # 文件类型校验
            content_type = f.content_type or ""
            if content_type and content_type not in ALLOWED_MIME_TYPES \
                    and not content_type.startswith("text/") \
                    and not content_type.startswith("image/"):
                _log.warning("[SECURITY] 拒绝上传: %s (content_type=%s)", f.filename, content_type)
                continue
            content = await f.read()
            # 文件大小校验
            if len(content) > MAX_UPLOAD_SIZE:
                _log.warning("[SECURITY] 文件过大被拒绝: %s (%d bytes)", f.filename, len(content))
                continue
            file_data.append({"name": f.filename, "content": content,
                              "size": len(content)})
    return await _chat_stream_impl(
        message=message, agent=agent or None,
        session_id=session_id or None, files=file_data, request=request
    )


# ─── 统一记忆存储 ───
from mini_thalamus.memory_store import get_store

memory = get_store()


def _stage_sse(kind, payload):
    if kind == "route":
        agents = payload.get("agents") or []
        event = {
            "type": "route",
            "agent": agents[0] if agents else "",
            "multi_step": agents,
            "reason": payload.get("reason", ""),
            "confidence": payload.get("confidence", 0),
            "method": payload.get("method", ""),
            "mode": payload.get("mode", ""),
        }
    elif kind == "step_start":
        event = {
            "type": "status",
            "subtype": "agent_start",
            "agent": payload.get("agent_id", ""),
            "text": "正在执行 [{}] {}".format(
                payload.get("agent_id", ""),
                payload.get("instruction", "")[:60],
            ),
        }
    elif kind == "replan":
        event = {
            "type": "status",
            "subtype": "replan",
            "text": "⚠️ 部分步骤未达预期，正在重新规划剩余任务...",
        }
    elif kind == "step_end":
        status = payload.get("status", "")
        text = "[{}] 完成".format(payload.get("agent_id", ""))
        if status == "blocked":
            reason = payload.get("answer", "") or payload.get("summary", "")
            text += "（被执行策略阻止：" + str(reason)[:80] + "）"
        elif status == "skipped":
            text += "（依赖失败，已跳过）"
        elif status == "error":
            text += "（执行失败）"
        event = {
            "type": "status",
            "subtype": "agent_end",
            "agent": payload.get("agent_id", ""),
            "text": text,
        }
    else:
        event = {"type": "status", "text": str(payload)}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _chat_stream_impl(message: str, agent: str, session_id: str,
                            files: list = None, require_confirm: bool = None,
                            request: Request = None, approved_call_ids=None):
    """流式对话核心实现 —— 支持多步 + 文件 + 跨轮记忆"""
    files = files or []
    sid = session_id or str(uuid.uuid4())[:8]
    user_id = _request_user_id(request)
    _resolve_web_session(user_id, sid)

    # ═══ 安全层：限流 ═══
    if request and not _rate_limit_check(request):
        raise HTTPException(status_code=429, detail="请求频率过高，请稍后再试")

    # ═══ DEBUG ═══
    cached = memory.get_files(sid)
    _log.debug("sid=%s agent=%s files=%d cache=%s manual=%s msg=%.50s",
               sid[:8], agent or 'auto', len(files),
               'hit' if cached else 'miss',
               agent is not None, message)

    # ═══ 安全层 ═══
    raw_message = message
    clean_message, flagged = sanitize_input(raw_message)
    # 叠加 nl2sql_agent 的 RuleBasedPromptGuard（比简单 regex 更全面）
    if _nl2sql_prompt_guard is not None:
        try:
            _nl2sql_prompt_guard(clean_message)
        except Exception as _guard_e:
            flagged = True
            _log.warning("[SECURITY] nl2sql prompt guard 拦截: %s", str(_guard_e)[:200])
    if flagged:
        log_security_event("injection_attempt", f"用户输入含伪指令: {raw_message[:100]}")
    message = wrap_boundary(clean_message, "user_query")
    chosen_agent = agent
    is_manual = agent is not None

    # ═══ 流式执行逻辑 ═══
    async def event_stream():
        agents = get_agents()

        # 文件处理
        file_names = []
        if files:
            from mini_thalamus.agents.docparse_agent import DocParseAgent
            for finfo in files:
                try:
                    parsed = {"name": finfo["name"], "size": finfo["size"], "format": finfo["name"].rsplit(".", 1)[-1]}
                    memory.register_file(sid, parsed)
                    file_names.append(finfo["name"])
                except Exception:
                    pass
            if file_names:
                yield f"data: {json.dumps({'type': 'status', 'text': f'已接收 {len(file_names)} 个文件'}, ensure_ascii=False)}\n\n"

        # SSE is a presentation layer only. Routing, policy and execution all
        # go through the same Main Agent path as /api/chat.
        route_input = (
            f"@{chosen_agent} {clean_message}"
            if is_manual and chosen_agent in agents
            else clean_message
        )
        yield f"data: {json.dumps({'type': 'status', 'text': '正在识别意图并规划任务...'}, ensure_ascii=False)}\n\n"
        stage_queue = queue.Queue()

        def on_stage(kind, payload):
            stage_queue.put((kind, payload))

        route_task = asyncio.create_task(
            _auto_route(
                route_input,
                sid,
                user_id=user_id,
                approved_call_ids=approved_call_ids,
                on_stage=on_stage,
                require_confirm=bool(require_confirm),
                files=files,
            )
        )
        while not route_task.done():
            while True:
                try:
                    kind, payload = stage_queue.get_nowait()
                except queue.Empty:
                    break
                yield _stage_sse(kind, payload)
            await asyncio.sleep(0.05)
        while True:
            try:
                kind, payload = stage_queue.get_nowait()
            except queue.Empty:
                break
            yield _stage_sse(kind, payload)
        result = route_task.result()
        if result.get("status") == "checkpoint":
            pending = result.get("pending_call", {})
            payload = {
                "type": "approval_checkpoint",
                "answer": result.get("answer", "等待用户批准"),
                "pending_call": pending,
                "call_id": pending.get("call_id"),
                "session_id": sid,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        else:
            structured = result.get("data", {}).get("structured")
            if structured:
                yield f"data: {json.dumps({'type': 'structured', 'data': structured}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'text': result.get('answer', ''), 'agent': result.get('agent_used')}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': sid}, ensure_ascii=False)}\n\n"
        return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── 启动入口 ───
def main() -> None:
    """启动 Web 服务。

    三种等价启动方式：
        python -m mini_thalamus.web_server
        mini-thalamus                       # pip install 后的 console script
        uvicorn mini_thalamus.web_server:app --port 8765
    """
    import uvicorn

    host = os.getenv("MT_HOST", "127.0.0.1")
    port = int(os.getenv("MT_PORT", "8765"))
    reload_ = os.getenv("MT_RELOAD", "false").lower() == "true"
    if reload_:
        # reload 需要 import string 而非 app 对象
        uvicorn.run("mini_thalamus.web_server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

