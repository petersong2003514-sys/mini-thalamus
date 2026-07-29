"""Manifest-driven Agent registry with lazy, isolated instantiation."""

import ast
import importlib
import logging
import os
import threading
from typing import Callable, Optional


_log = logging.getLogger(__name__)


class AgentRegistryError(RuntimeError):
    """Base error for invalid registration or registry state."""


class AgentLoadError(AgentRegistryError):
    """Raised when a registered Agent cannot be instantiated."""


class AgentRegistry:
    """Discover Agent manifests without importing their implementation modules."""

    def __init__(self, agents_dir: str = None):
        self._agents: dict[str, dict] = {}
        self._callbacks: list[tuple[Optional[str], Callable]] = []
        self._agents_dir = agents_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "agents"
        )
        self._lock = threading.RLock()
        self._discovery_errors: dict[str, str] = {}

    def discover(self) -> int:
        """Read AGENT_SPEC/AGENT_META using AST and register lazy factories."""
        if not os.path.isdir(self._agents_dir):
            return 0

        with self._lock:
            self._agents = {
                agent_id: entry
                for agent_id, entry in self._agents.items()
                if entry.get("source") != "builtin"
            }
            self._discovery_errors = {}

            count = 0
            for filename in sorted(os.listdir(self._agents_dir)):
                if not filename.endswith("_agent.py") or filename.startswith("_"):
                    continue
                path = os.path.join(self._agents_dir, filename)
                try:
                    meta, factory = self._read_manifest(path)
                    if meta is None:
                        continue
                    agent_id = meta["id"]
                    if agent_id in self._agents:
                        raise AgentRegistryError(
                            f"agent '{agent_id}' is already registered"
                        )
                    self._agents[agent_id] = self._entry(
                        meta=meta,
                        factory=factory,
                        source="builtin",
                        path=path,
                    )
                    count += 1
                    self._notify(agent_id, meta)
                except Exception as exc:
                    self._discovery_errors[filename] = str(exc)
                    _log.warning("Agent manifest %s invalid: %s", filename, exc)
            return count

    def _read_manifest(self, path: str) -> tuple[Optional[dict], Optional[str]]:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        manifests = {}
        class_names = []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "AGENT_SPEC",
                        "AGENT_META",
                    }:
                        try:
                            manifests[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
            elif isinstance(node, ast.ClassDef) and node.name.endswith("Agent"):
                class_names.append(node.name)

        raw = manifests.get("AGENT_SPEC") or manifests.get("AGENT_META")
        if raw is None:
            return None, None
        if not isinstance(raw, dict):
            raise AgentRegistryError("Agent manifest must be an object")

        stem = os.path.splitext(os.path.basename(path))[0]
        default_id = stem.removesuffix("_agent")
        meta = self._validate_meta(dict(raw), default_id=default_id)
        factory = meta.get("factory")
        if factory is None:
            if not class_names:
                raise AgentRegistryError(
                    f"agent '{meta['id']}' has no factory or Agent class"
                )
            factory = f"agents.{stem}:{class_names[0]}"
            meta["factory"] = factory
        if not isinstance(factory, str) or ":" not in factory:
            raise AgentRegistryError(
                f"agent '{meta['id']}' factory must use 'module:attribute'"
            )
        return meta, factory

    @staticmethod
    def _validate_meta(meta: dict, default_id: str = "") -> dict:
        agent_id = meta.get("id") or default_id
        if not isinstance(agent_id, str) or not agent_id:
            raise AgentRegistryError("Agent id must be a non-empty string")
        name = meta.get("name")
        if not isinstance(name, str) or not name:
            raise AgentRegistryError(f"agent '{agent_id}' requires a name")
        description = meta.get("description", "")
        if not isinstance(description, str):
            raise AgentRegistryError(
                f"agent '{agent_id}' description must be a string"
            )

        normalized = dict(meta)
        normalized["id"] = agent_id
        normalized["description"] = description
        normalized.setdefault("keywords", [])
        normalized.setdefault("intents", [])
        normalized.setdefault("examples", [])
        normalized.setdefault("config_namespace", agent_id)
        normalized.setdefault(
            "input_schema",
            {"required": ["instruction"], "properties": {}},
        )
        normalized.setdefault(
            "output_schema",
            {
                "exports": [
                    "answer",
                    "summary",
                    "data.structured",
                ]
            },
        )
        normalized.setdefault("capabilities", {})
        return normalized

    @staticmethod
    def _entry(
        *,
        meta: dict,
        factory,
        source: str,
        path: str = "",
        instance=None,
    ) -> dict:
        return {
            "meta": meta,
            "factory": factory,
            "instance": instance,
            "source": source,
            "path": path,
            "status": "ready" if instance is not None else "discovered",
            "error": "",
        }

    def register(
        self,
        agent_id: str,
        meta: dict,
        instance=None,
        source: str = "dynamic",
        factory=None,
    ) -> None:
        with self._lock:
            if agent_id in self._agents:
                raise AgentRegistryError(f"agent '{agent_id}' is already registered")
            normalized = self._validate_meta(dict(meta), default_id=agent_id)
            if normalized["id"] != agent_id:
                raise AgentRegistryError(
                    f"registration id '{agent_id}' does not match manifest id "
                    f"'{normalized['id']}'"
                )
            resolved_factory = factory or normalized.get("factory")
            if instance is None and resolved_factory is None:
                raise AgentRegistryError(
                    f"agent '{agent_id}' requires a factory or instance"
                )
            self._agents[agent_id] = self._entry(
                meta=normalized,
                factory=resolved_factory,
                source=source,
                instance=instance,
            )
            self._notify(agent_id, normalized)

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[dict]:
        return self._agents.get(agent_id)

    def entries(self) -> dict:
        return dict(self._agents)

    def get_instance(self, agent_id: str):
        entry = self._agents.get(agent_id)
        return entry.get("instance") if entry else None

    def get_or_create(self, agent_id: str):
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                raise AgentLoadError(f"agent '{agent_id}' is not registered")
            if entry.get("instance") is not None:
                return entry["instance"]
            if entry.get("status") == "unavailable":
                raise AgentLoadError(
                    f"agent '{agent_id}' is unavailable: {entry.get('error', '')}"
                )

            try:
                factory = self._resolve_factory(entry.get("factory"))
                instance = factory()
                if instance is None:
                    raise RuntimeError("factory returned None")
            except Exception as exc:
                entry["status"] = "unavailable"
                entry["error"] = str(exc)
                raise AgentLoadError(
                    f"agent '{agent_id}' failed to load: {exc}"
                ) from exc

            entry["instance"] = instance
            entry["status"] = "ready"
            entry["error"] = ""
            return instance

    @staticmethod
    def _resolve_factory(factory):
        if callable(factory):
            return factory
        if not isinstance(factory, str) or ":" not in factory:
            raise AgentLoadError("invalid Agent factory")
        module_name, attribute = factory.split(":", 1)
        module = importlib.import_module(module_name)
        resolved = getattr(module, attribute)
        if not callable(resolved):
            raise AgentLoadError(f"factory '{factory}' is not callable")
        return resolved

    def get_capabilities(self, agent_id: str) -> dict:
        entry = self._agents.get(agent_id)
        if not entry:
            return {}
        instance = entry.get("instance")
        if instance is not None and hasattr(instance, "capabilities"):
            return instance.capabilities
        return entry["meta"].get("capabilities", {})

    def list_all(self, include_unavailable: bool = False) -> dict:
        return {
            agent_id: entry["meta"]
            for agent_id, entry in self._agents.items()
            if include_unavailable or entry.get("status") != "unavailable"
        }

    def get_router_config(self) -> dict:
        config = {}
        for agent_id, entry in self._agents.items():
            if entry.get("status") == "unavailable":
                continue
            meta = entry["meta"]
            config[agent_id] = {
                "name": meta.get("name", agent_id),
                "description": meta.get("description", ""),
                "keywords": meta.get("keywords", []),
                "intents": meta.get("intents", []),
                "examples": meta.get("examples", []),
                "icon": meta.get("icon", "🔧"),
                "source": entry.get("source", "builtin"),
                "status": entry.get("status", "discovered"),
                "capabilities": meta.get("capabilities", {}),
                "input_schema": meta.get("input_schema", {}),
                "output_schema": meta.get("output_schema", {}),
            }
        return config

    def loader_registry(self) -> dict:
        return {
            agent_id: (lambda current=agent_id: self.get_or_create(current))
            for agent_id, entry in self._agents.items()
            if entry.get("status") != "unavailable"
        }

    def on_register(self, callback: Callable, owner: str = None) -> None:
        if owner is not None:
            self._callbacks = [
                (registered_owner, registered_callback)
                for registered_owner, registered_callback in self._callbacks
                if registered_owner != owner
            ]
        self._callbacks.append((owner, callback))

    def _notify(self, agent_id: str, meta: dict) -> None:
        for _, callback in list(self._callbacks):
            try:
                callback(agent_id, meta)
            except Exception:
                _log.debug("Agent registry callback failed", exc_info=True)

    def register_skill_loader(self, skill_loader) -> None:
        for skill_id, skill in skill_loader.skills.items():
            run_func = skill_loader.get_skill(skill_id)
            if run_func is None:
                continue
            from skill_loader import _SkillWrapper

            self.register(
                skill_id,
                skill.meta,
                factory=lambda func=run_func, sid=skill_id: _SkillWrapper(func, sid),
                source="skill",
            )

    def register_mcp_tools(self, mcp_manager) -> None:
        tools = mcp_manager.get_all_tools()
        if not tools or "mcp" in self._agents:
            return
        servers = list(mcp_manager.servers.keys())
        from agents.mcp_agent import MCPAgent
        from tool_registry import registry as tool_registry

        self.register(
            "mcp",
            {
                "id": "mcp",
                "name": "MCP 工具代理",
                "description": f"调用外部 MCP 工具: {', '.join(servers)}",
                "icon": "🔌",
                "keywords": [],
                "capabilities": {
                    "needs_history": False,
                    "needs_files": False,
                    "needs_long_term": False,
                    "max_context_chars": 2000,
                },
            },
            factory=lambda: MCPAgent(tool_registry, mcp_manager),
            source="mcp",
        )

    @property
    def discovery_errors(self) -> dict:
        return dict(self._discovery_errors)


__all__ = ["AgentLoadError", "AgentRegistry", "AgentRegistryError"]
