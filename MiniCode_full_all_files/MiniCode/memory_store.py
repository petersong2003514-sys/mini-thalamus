"""ShuShu Agent 统一记忆存储 —— 文件持久化 + 标签映射

架构:
  data/
  ├── memory.json              ← 共享长期记忆（跨 session）
  └── sessions/{sid}/
      ├── history.json         ← 轮次 Q&A + 标签（跨轮短期记忆）
      ├── files/               ← 上传文件（已解析 JSON）
      └── context.json         ← 当前上下文快照（标签列表）

使用:
  store = MemoryStore()
  store.record_turn(sid, query, answer, agent, tags)
  ctx = store.get_context(sid, current_query)  # 给上下文窗口用的紧凑摘要
  full = store.lookup(sid, "天气")             # 按标签查找完整记录
"""
import os, json, time, re, threading
from typing import Optional

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MEMORY_FILE = os.path.join(DATA_ROOT, "memory.json")
MAX_HISTORY_TURNS = 50  # 每 session 最多保留轮次
CONTEXT_MAX_CHARS = 500  # 上下文窗口最大字符


class MemoryStore:
    """统一记忆存储：文件持久化 + 标签索引"""

    def __init__(self, data_root: str = None, clean_sessions_on_start: bool = False):
        self.root = data_root or DATA_ROOT
        os.makedirs(self.root, exist_ok=True)
        self._session_locks: dict = {}   # sid → threading.Lock，保护同 session 的读写竞态
        self._session_locks_lock = threading.Lock()  # 保护 _session_locks dict 本身
        if clean_sessions_on_start:
            self._clean_all_sessions()

    def _get_session_lock(self, sid: str) -> threading.Lock:
        with self._session_locks_lock:
            if sid not in self._session_locks:
                self._session_locks[sid] = threading.Lock()
            return self._session_locks[sid]

    def _clean_all_sessions(self):
        """清理所有旧 session 数据（仅显式调用时执行，重启不自动清理）"""
        import shutil
        sessions_dir = os.path.join(self.root, "sessions")
        if os.path.isdir(sessions_dir):
            shutil.rmtree(sessions_dir, ignore_errors=True)
            if hasattr(self, '_log'):
                self._log.info("已清理所有 session 数据")

    # ═══════════════════════════════════════════════
    # 会话路径
    # ═══════════════════════════════════════════════
    def _session_dir(self, sid: str) -> str:
        d = os.path.join(self.root, "sessions", sid)
        os.makedirs(d, exist_ok=True)
        return d

    def _history_path(self, sid: str) -> str:
        return os.path.join(self._session_dir(sid), "history.json")

    def _context_path(self, sid: str) -> str:
        return os.path.join(self._session_dir(sid), "context.json")

    def _files_dir(self, sid: str) -> str:
        d = os.path.join(self._session_dir(sid), "files")
        os.makedirs(d, exist_ok=True)
        return d

    # ═══════════════════════════════════════════════
    # 跨轮短期记忆：记录 + 标签
    # ═══════════════════════════════════════════════
    def record_turn(self, sid: str, query: str, answer: str,
                    agent: str, tags: list = None):
        """记录一轮对话到 history.json，并尝试沉淀长期记忆"""
        with self._get_session_lock(sid):
            history = self._load_history(sid)
            turn = {
                "id": f"turn_{len(history) + 1}",
                "q": query,
                "a": answer[:2000],
                "agent": agent,
                "tags": tags or self._auto_tag(query),
                "ts": time.time(),
                "ts_readable": time.strftime("%m-%d %H:%M"),
            }
            history.append(turn)
            if len(history) > MAX_HISTORY_TURNS:
                history = history[-MAX_HISTORY_TURNS:]
            self._save_history(sid, history)
            self._update_context(sid, history)
        # 长期记忆沉淀在锁外执行，避免持锁时做 LLM 调用
        self._try_crystallize_ltm(turn)
        return turn["id"]

    def _try_crystallize_ltm(self, turn: dict):
        """当回答包含有价值结论时，写入长期记忆"""
        answer = turn.get("a", "")
        # 简单启发式：回答超过 50 字 且 agent 不是 general
        if len(answer) < 50 or turn.get("agent") in ("general", ""):
            return
        # 提取主题词作为 key
        tags = turn.get("tags", [])
        if not tags:
            return
        key = f"topic:{tags[0]}"
        gist = answer[:150]
        self.set_long_term(key, {"gist": gist, "agent": turn.get("agent"), "ts": turn.get("ts")})

    _tag_client = None
    _tag_cache: dict = {}      # {规则键: tags 列表} 复用相似 query 的 LLM 结果（REPAIR_PLAN #8）
    _TAG_CACHE_MAX = 256       # 简单 LRU 上限，超出时按插入顺序裁剪一半

    def _get_tag_client(self):
        if self._tag_client is None:
            from openai import OpenAI
            import importlib.util, os as _os
            # memory_store.py 在 MiniCode/ 下，config.py 也在同一目录
            _cfg_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.py")
            _spec = importlib.util.spec_from_file_location("shushu_config_tag", _cfg_path)
            _cfg = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_cfg)
            self._tag_client = OpenAI(api_key=_cfg.LLM_CONFIG["api_key"], base_url=_cfg.LLM_CONFIG["base_url"])
            self._tag_model = _cfg.LLM_CONFIG["model"]
        return self._tag_client, self._tag_model

    def _auto_tag(self, query: str) -> list:
        """规则优先 + LLM 兜底 + 标签缓存（REPAIR_PLAN #8）"""
        # 规则提取：中文词 + 英文词
        tags = re.findall(r'[一-鿿]{2,4}', query)
        tags += re.findall(r'[a-zA-Z]{3,}', query.lower())
        # 去重 + 去停用词
        stopwords = {"什么", "怎么", "如何", "可以", "帮我", "查询", "一下", "请问", "告诉"}
        tags = [t for t in dict.fromkeys(tags) if t not in stopwords][:5]
        if len(tags) >= 2:
            return tags

        # 缓存键：用规则提取出的非停用词集合（相同关键词集合的 query 共享 LLM 结果）
        cache_key = "·".join(sorted(tags)) if tags else query[:30]
        cached = MemoryStore._tag_cache.get(cache_key)
        if cached is not None:
            return cached

        # 规则不足时 LLM 兜底
        try:
            client, model = self._get_tag_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": "为用户问题生成 2-4 个简短标签（空格分隔）。只输出标签。"
                }, {
                    "role": "user", "content": query[:100]
                }],
                temperature=0.0, max_tokens=20,
            )
            llm_tags = resp.choices[0].message.content.strip().split()
            result = (tags + llm_tags)[:5] if llm_tags else (tags or ["通用"])
        except Exception:
            result = tags or ["通用"]

        # 写入缓存（简单 LRU：满了时砍一半最早的）
        if len(MemoryStore._tag_cache) >= MemoryStore._TAG_CACHE_MAX:
            half = MemoryStore._TAG_CACHE_MAX // 2
            for k in list(MemoryStore._tag_cache.keys())[:half]:
                MemoryStore._tag_cache.pop(k, None)
        MemoryStore._tag_cache[cache_key] = result
        return result

    # ═══════════════════════════════════════════════
    # 上下文窗口：只放标签 + turn_id（不放内容）
    # ═══════════════════════════════════════════════
    def get_context_for_agent(self, sid: str, agent_name: str,
                              current_query: str = "",
                              tag_turns: int = 20,
                              ref_turns: int = 10,
                              capabilities: dict = None) -> str:
        """构建 Agent 上下文：根据 capabilities 过滤记忆注入。

        capabilities: Agent 的能力声明，控制注入哪些内容
        返回: 拼接后的字符串（向后兼容；新代码请用 build_agent_context_struct 拿 dict）
        """
        # 复用结构化版本，再 to_prompt_string 拼回字符串
        ctx_struct = self._collect_context_struct(
            sid, agent_name, current_query, tag_turns, ref_turns, capabilities)
        # 渲染成字符串（仅 tags / relevant_history / current_query 三段，与旧逻辑一致）
        from base_agent import AgentContext
        sub = AgentContext(
            current_query=ctx_struct["current_query"],
            relevant_history=ctx_struct["relevant_history"],
            tags=ctx_struct["tags"],
        )
        rendered = sub.to_prompt_string()
        return rendered or current_query

    def _collect_context_struct(self, sid: str, agent_name: str,
                                current_query: str = "",
                                tag_turns: int = 20,
                                ref_turns: int = 10,
                                capabilities: dict = None) -> dict:
        """内部：根据 capabilities 收集结构化上下文字段"""
        caps = capabilities if capabilities is not None else self._get_agent_capabilities(agent_name)
        history = self._load_history(sid)
        tags: list = []
        relevant: list = []

        if history:
            # 标签
            tag_set = set()
            for t in history[-tag_turns:]:
                tag_set.update(t.get("tags", []))
            tags = sorted(tag_set)

            # 相关历史（受 needs_history 控制）
            if caps.get("needs_history", True):
                q_words = set(re.findall(r'[一-鿿\w]{2,}', current_query.lower()))
                for t in history[-ref_turns:]:
                    t_words = set(re.findall(r'[一-鿿\w]{2,}', t["q"].lower()))
                    if q_words & t_words:
                        relevant.append({
                            "id": t.get("id", ""),
                            "q": t.get("q", ""),
                            "a": t.get("a", ""),
                            "agent": t.get("agent", ""),
                            "tags": t.get("tags", []),
                        })

        return {
            "current_query": current_query,
            "relevant_history": relevant[-3:],  # 与旧 to_prompt_string 保持一致
            "tags": tags,
        }

    def build_agent_context(self, sid: str, agent_name: str, query: str,
                            capabilities: dict = None,
                            as_struct: bool = False):
        """统一记忆注入入口：根据 Agent capabilities 决定注入哪些记忆。

        as_struct=False（默认，向后兼容）→ 返回拼接好的 prompt 字符串
        as_struct=True → 返回 AgentContext dataclass，供新 Agent 结构化消费

        REPAIR_PLAN #7 #9 #19 #20：统一入口 + 结构化 dict + capabilities 过滤
        """
        caps = capabilities if capabilities is not None else self._get_agent_capabilities(agent_name)
        max_chars = caps.get("max_context_chars", 4000)

        # 1. 标签 + 相关历史（结构化）
        struct = self._collect_context_struct(sid, agent_name, query, capabilities=caps)

        # 2. 完整对话历史（仅 needs_history=True 时）
        conv = ""
        if caps.get("needs_history", True):
            conv = self.get_conversation_history(sid, max_turns=10, max_chars=max_chars)

        # 3. 长期记忆（仅 needs_long_term=True 时）
        ltm: dict = {}
        if caps.get("needs_long_term", False):
            tags_in_query = re.findall(r'[一-鿿\w]{2,}', query)
            for tag in tags_in_query[:5]:
                entry = self.get_long_term(f"topic:{tag}")
                if entry and isinstance(entry, dict) and entry.get("gist"):
                    ltm[tag] = entry["gist"]

        # 4. 文件列表
        files = self.get_files(sid) if caps.get("needs_files", False) else []

        from base_agent import AgentContext
        ctx_obj = AgentContext(
            current_query=query,
            relevant_history=struct["relevant_history"],
            tags=struct["tags"],
            long_term=ltm,
            files=files,
            session_id=sid,
            conversation_history=conv,
        )

        if as_struct:
            return ctx_obj
        return ctx_obj.to_prompt_string()


    def _get_agent_capabilities(self, agent_name: str) -> dict:
        """从已加载的 Agent 获取 capabilities，未找到则返回默认值

        FEATURE_PLAN.md §1 P0.2 (audit R3): defaults 必须与 BaseAgent.capabilities
        字段集严格对齐，否则 wrappers/skill/mcp 这些没 .capabilities 的对象
        会让上层代码（如 auto-chain 判断）KeyError。
        """
        defaults = {
            "needs_history": True,
            "needs_files": False,
            "needs_long_term": False,
            "max_context_chars": 4000,
            "side_effects": "external",      # 保守：未知 = 有外部副作用
            "auto_chainable": False,         # 保守：未知 = 不允许自动续接
            "needs_credentials": [],          # 保守：未知 = 不注入凭证视图
        }
        try:
            from base_agent import BaseAgent
            # 尝试从全局 agents 缓存获取
            import sys
            ws = sys.modules.get("web_server") or sys.modules.get("__main__")
            if ws and hasattr(ws, "get_agents"):
                agents = ws.get_agents()
                agent = agents.get(agent_name)
                if agent and hasattr(agent, "capabilities"):
                    return agent.capabilities
        except Exception:
            pass
        return defaults

    # ═══════════════════════════════════════════════
    # 对话历史注入：从 history.json 直接流向 LLM 上下文
    # ═══════════════════════════════════════════════
    def get_conversation_history(self, sid: str, max_turns: int = 30,
                                 max_chars: int = 12000) -> str:
        """返回最近 N 轮对话的完整 Q&A，直接注入 LLM 上下文。

        max_turns: 最多取最近 N 轮
        max_chars: 总字符上限（超出则截断最早的轮次）
        """
        history = self._load_history(sid)
        if not history:
            return ""

        turns = history[-max_turns:]
        parts = []
        total = 0
        for t in reversed(turns):  # 最新的在前
            text = f"[{t['id']}][{'·'.join(t.get('tags',[]))}] 用户: {t['q']}\n助手: {t['a'][:500]}"
            if total + len(text) > max_chars:
                remain = max_chars - total
                if remain > 50:
                    parts.append(text[:remain] + "...")
                break
            parts.append(text)
            total += len(text)

        if not parts:
            return ""

        return "## 对话历史\n" + "\n---\n".join(reversed(parts)) + "\n---\n"

    # ═══════════════════════════════════════════════
    # 标签 → 完整内容映射
    def resolve(self, sid: str, turn_ids: list = None, tags: list = None,
                max_chars: int = 2000, max_turns: int = 5) -> str:
        """按 turn_id 或 tag 从 history.json 查找完整记录。

        max_chars: 返回文本的最大字符数
        max_turns: 最多返回 N 轮
        """
        history = self._load_history(sid)
        matched = []

        ids = set(turn_ids or [])
        tag_set = set(tags or [])

        for t in history:
            if ids and t["id"] in ids:
                matched.append(t)
            elif tag_set and tag_set & set(t.get("tags", [])):
                matched.append(t)

        if not matched:
            return ""

        parts = []
        total = 0
        for t in matched[-max_turns:]:
            text = f"[{t['id']}][{'·'.join(t.get('tags',[]))}] Q: {t['q']}\nA: {t['a'][:300]}"
            if total + len(text) > max_chars:
                parts.append(text[:max_chars - total] + "...")
                break
            parts.append(text)
            total += len(text)

        return "\n---\n".join(parts)

    def get_last_turn(self, sid: str) -> Optional[dict]:
        history = self._load_history(sid)
        return history[-1] if history else None

    def get_recent_turns(self, sid: str, n: int = 3) -> list:
        return self._load_history(sid)[-n:]

    # ═══════════════════════════════════════════════
    # 文件管理
    # ═══════════════════════════════════════════════
    def register_file(self, sid: str, file_info: dict):
        """注册一个上传文件到 session"""
        files_dir = self._files_dir(sid)
        # 保存解析后的 JSON
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', file_info["name"]) + ".json"
        path = os.path.join(files_dir, safe_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(file_info.get("parsed", file_info), f, ensure_ascii=False, default=str)
        # 更新文件索引
        index_path = os.path.join(files_dir, "_index.json")
        index = self._load_json(index_path) or []
        existing = {f["name"] for f in index}
        if file_info["name"] not in existing:
            index.append({
                "name": file_info["name"],
                "stored_at": path,
                "pages": file_info.get("pages", "?"),
                "paragraphs": file_info.get("paragraph_count", file_info.get("paragraphs", "?")),
                "format": file_info.get("format", "?"),
                "summary": file_info.get("summary", ""),
            })
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False)

    def get_files(self, sid: str) -> list:
        """获取 session 的文件名列表"""
        index_path = os.path.join(self._files_dir(sid), "_index.json")
        index = self._load_json(index_path) or []
        return [f["name"] for f in index]

    def get_file_labels(self, sid: str) -> list:
        """获取 session 的文件元数据列表"""
        index_path = os.path.join(self._files_dir(sid), "_index.json")
        return self._load_json(index_path) or []

    def load_all_parsed_files(self, sid: str) -> list:
        """加载 session 所有已解析文档的完整内容"""
        docs = []
        for label in self.get_file_labels(sid):
            path = label.get("stored_at", "")
            if path and os.path.exists(path):
                data = self._load_json(path)
                if data:
                    docs.append(data)
        return docs

    # ═══════════════════════════════════════════════
    # 长期记忆（跨 session）
    # ═══════════════════════════════════════════════
    def set_long_term(self, key: str, value):
        """写入长期记忆"""
        mem = self._load_json(MEMORY_FILE) or {}
        if "entries" not in mem:
            mem["entries"] = {}
        mem["entries"][key] = {
            "value": value,
            "updated_at": time.time(),
        }
        self._save_json(MEMORY_FILE, mem)

    def get_long_term(self, key: str, default=None):
        """读取长期记忆"""
        mem = self._load_json(MEMORY_FILE) or {}
        entry = mem.get("entries", {}).get(key)
        return entry["value"] if entry else default

    # ═══════════════════════════════════════════════
    # 清理
    # ═══════════════════════════════════════════════
    def cleanup_session(self, sid: str):
        """删除 session 所有数据"""
        import shutil
        d = os.path.join(self.root, "sessions", sid)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    # ═══════════════════════════════════════════════
    # 内部：JSON 读写
    # ═══════════════════════════════════════════════
    def _load_json(self, path: str) -> Optional[dict]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
        except Exception:
            pass
        return None

    def _save_json(self, path: str, data):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _load_history(self, sid: str) -> list:
        data = self._load_json(self._history_path(sid))
        return data.get("turns", []) if data else []

    def _save_history(self, sid: str, history: list):
        self._save_json(self._history_path(sid), {
            "session_id": sid,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turns": history,
        })

    def _update_context(self, sid: str, history: list):
        """更新上下文快照（标签列表）"""
        all_tags = set()
        for t in history[-10:]:
            all_tags.update(t.get("tags", []))
        self._save_json(self._context_path(sid), {
            "active_tags": sorted(all_tags),
            "turn_count": len(history),
            "last_turn_id": history[-1]["id"] if history else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })


# ─── 全局单例 ───
_store: Optional[MemoryStore] = None
_store_lock = threading.Lock()


def get_store(data_root: str = None) -> MemoryStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryStore(data_root)
    return _store
