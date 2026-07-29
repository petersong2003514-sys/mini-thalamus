"""credentials.py —— Credential Store

集中、scoped 的密钥存储。FEATURE_PLAN.md §5。

存储：data/credentials.json（chmod 0600，gitignored）。
schema 见 data/credentials.example.json。

设计契约：
  - get(name, agent_id) 未授权返 None，永不 raise。
  - get_env_dict(agent_id) 是 Agent 把 secret 透给 subprocess 的**唯一**通道，
    自动应用 env_map。
  - list(redact=True) 是 UI / 日志看到的形态；redact=False 仅供进程内 LLM 调用使用。
  - 任何 secret 字符串都不允许出现在 LLM prompt 中。
"""
from __future__ import annotations
import json
import os
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

_log = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "credentials.json")
_ROTATION_WARN_DAYS = 180


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(v: str) -> str:
    """掩码：保留首尾 4 字符，中间换成 ***"""
    if not v or len(v) < 8:
        return "***"
    return f"{v[:4]}***{v[-4:]}"


# 环境变量回退表：当 store 内没有该条目时，从这些 env 变量取（保 DEEPSEEK_API_KEY 等老用法）
_ENV_FALLBACKS = {
    "deepseek": ("DEEPSEEK_API_KEY", "api_key"),
    "openai":   ("OPENAI_API_KEY",   "api_key"),
    "anthropic": ("ANTHROPIC_API_KEY", "api_key"),
    "bing":     ("BING_API_KEY",     "api_key"),
    "github":   ("GITHUB_TOKEN",     "token"),
}


class CredentialStore:
    """线程安全的 JSON-backed 凭证存储。"""

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict = {"version": 1, "updated_at": _now_iso(), "credentials": {}}
        self._load()

    # ── I/O ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            # 权限审计
            try:
                if os.name == "nt":
                    return
                mode = os.stat(self.path).st_mode & 0o777
                if mode & 0o077:
                    _log.warning("credentials.json 权限过宽 %o（建议 0600）", mode)
            except OSError:
                pass
        except Exception as e:
            _log.error("加载 credentials.json 失败: %s", e)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data["updated_at"] = _now_iso()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        finally:
            # 清理残留 tmp 文件（异常 / 跨盘 rename 失败后仍在）
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            # Windows: 尝试 icacls 限制权限
            if os.name == "nt":
                try:
                    import subprocess as _sp
                    _sp.run(
                        ["icacls", self.path, "/inheritance:r", "/grant:r",
                         f"{os.environ.get('USERNAME','Everyone')}:(R,W)"],
                        capture_output=True, timeout=5)
                except Exception:
                    pass

    # ── 授权检查 ────────────────────────────────────────────────────
    def _authorized(self, entry: dict, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return True  # 内部 / admin 调用（如 config.py 解析 LLM key）
        allowed = entry.get("agents", [])
        return "*" in allowed or agent_id in allowed

    # ── 公开 API ────────────────────────────────────────────────────
    def get(self, name: str, agent_id: Optional[str] = None) -> Optional[dict]:
        """返回 value dict 拷贝，未找到 / 未授权返 None。env fallback 兜底。"""
        with self._lock:
            entry = self._data.get("credentials", {}).get(name)
            if entry and self._authorized(entry, agent_id):
                return dict(entry["value"])  # copy 防外部改

        # 环境变量是平台管理面兼容入口，不能绕过具名 Agent 的 allowlist。
        if agent_id is None and name in _ENV_FALLBACKS:
            envvar, field = _ENV_FALLBACKS[name]
            v = os.getenv(envvar)
            if v:
                return {field: v}
        return None

    def get_env_dict(self, agent_id: str, names: list[str] | None = None) -> dict:
        """构造 {ENV_VAR_NAME: value} —— 供 subprocess.Popen(env=...) 用。

        仅返回该 agent_id 有权读的条目。env_map 决定每个 value 字段映射到哪个 env 名。
        """
        out: dict = {}
        allowed_names = set(names) if names is not None else None
        with self._lock:
            for name, entry in self._data.get("credentials", {}).items():
                if allowed_names is not None and name not in allowed_names:
                    continue
                if not self._authorized(entry, agent_id):
                    continue
                env_map = entry.get("env_map", {})
                for field, envvar in env_map.items():
                    val = entry["value"].get(field)
                    if val:
                        out[envvar] = val
        return out

    def set(self, name: str, value: dict, scope: str = "custom",
            agents: list | None = None, type_: str = "api_key",
            env_map: dict | None = None, note: str = "") -> None:
        if not isinstance(value, dict):
            raise ValueError("value 必须是 dict")
        with self._lock:
            existing = self._data.setdefault("credentials", {}).get(name, {})
            entry = {
                "type": type_,
                "value": value,
                "scope": scope,
                "agents": agents if agents is not None else ["*"],
                "env_map": env_map or {},
                "created_at": existing.get("created_at", _now_iso()),
                "updated_at": _now_iso(),
                "note": note,
            }
            self._data["credentials"][name] = entry
            self._save()

    def delete(self, name: str) -> bool:
        with self._lock:
            if name in self._data.get("credentials", {}):
                del self._data["credentials"][name]
                self._save()
                return True
            return False

    def list(self, redact: bool = True) -> list[dict]:
        with self._lock:
            out = []
            for name, e in self._data.get("credentials", {}).items():
                v = e["value"]
                if redact:
                    v = {k: _redact(str(vv)) for k, vv in v.items()}
                stale = False
                try:
                    upd = datetime.fromisoformat(e["updated_at"].replace("Z", "+00:00"))
                    stale = (datetime.now(timezone.utc) - upd).days > _ROTATION_WARN_DAYS
                except Exception:
                    pass
                out.append({
                    "name": name,
                    "type": e.get("type"),
                    "value": v,
                    "scope": e.get("scope"),
                    "agents": e.get("agents", []),
                    "env_map": e.get("env_map", {}),
                    "updated_at": e.get("updated_at"),
                    "stale": stale,
                    "note": e.get("note", ""),
                })
            return out


class ScopedCredView:
    """每次 Agent.run 注入的薄包装，bake 死 agent_id，防止 Agent 自行越权。"""
    __slots__ = ("_store", "_agent_id")

    def __init__(self, store: CredentialStore, agent_id: str):
        self._store = store
        self._agent_id = agent_id

    def get(self, name: str) -> Optional[dict]:
        return self._store.get(name, self._agent_id)

    def get_env_dict(self) -> dict:
        return self._store.get_env_dict(self._agent_id)


# ── 模块级 singleton（懒加载） ──────────────────────────────────────
_store: Optional[CredentialStore] = None
_store_lock = threading.Lock()


def get_store() -> CredentialStore:
    """获取全局 CredentialStore 单例。第一次调用时加载磁盘。"""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = CredentialStore()
    return _store


def reset_store_for_test(path: str | None = None) -> CredentialStore:
    """仅用于单测：重置 singleton 指向另一路径。"""
    global _store
    with _store_lock:
        _store = CredentialStore(path or _DEFAULT_PATH)
    return _store


__all__ = ["CredentialStore", "ScopedCredView", "get_store", "reset_store_for_test"]
