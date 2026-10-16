"""Manifest-driven Agent registry with lazy, isolated instantiation."""

import ast
import hashlib
import importlib
import importlib.util
import itertools
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

from agent_contract import AgentContractError, normalize_agent_contract


_log = logging.getLogger(__name__)

# Process-wide, monotonic, never reused. A per-instance counter would let two
# AgentRegistry objects mint the SAME versioned module name, and since
# sys.modules is global they would then evict each other's Agents.
_HOT_SEQ = itertools.count(1)

# Every entry incarnation gets its own epoch, so an outcome reported by an
# instance that has since been replaced can never be attributed to its
# successor -- even if the Agent was unregistered and registered again.
_EPOCH_SEQ = itertools.count(1)

# Circuit breaker. Read from the environment rather than config.py, which
# imports this module (a module-level import back would be circular).
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("MT_CIRCUIT_THRESHOLD", "3"))
CIRCUIT_COOLDOWN_SECONDS = float(os.getenv("MT_CIRCUIT_COOLDOWN", "60"))

# Failures that make an Agent's output unusable are "hard" and count toward the
# breaker. Semantic failures -- the output looks fine but may be wrong -- are
# "soft": a model judged those, and a wrong judgement would replace a correct
# answer with nothing, so they are recorded but never trip the breaker on their
# own. See record_failure().
HARD_FAILURE = "hard"
SOFT_FAILURE = "soft"


class AgentRegistryError(RuntimeError):
    """Base error for invalid registration or registry state."""


class AgentLoadError(AgentRegistryError):
    """Raised when a registered Agent cannot be instantiated."""


class AgentRegistry:
    """Discover Agent manifests without importing their implementation modules."""

    def __init__(self, agents_dir: str = None):
        self._agents: dict[str, dict] = {}
        # One subscriber channel for both events; _notify passes meta=None to
        # signal a removal (see unregister).
        self._callbacks: list[tuple[Optional[str], Callable]] = []
        self._agents_dir = agents_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "agents"
        )
        self._lock = threading.RLock()
        self._discovery_errors: dict[str, str] = {}

    @staticmethod
    def _is_hot(entry: dict) -> bool:
        """True when this entry was loaded from a file path, not from a manifest.

        Such an entry carries a live callable factory and the name of the module
        it came from. Re-registering it from the AST would silently downgrade it
        back to a lazy ``module:attr`` string, throwing away a working Agent.
        """
        return bool(entry.get("hot_module")) or callable(entry.get("factory"))

    def discover(self) -> int:
        """Read AGENT_SPEC/AGENT_META using AST and register lazy factories.

        Additive by design: Agents already hot-loaded from a path keep their live
        factory. Use :meth:`rescan` when you want changed sources re-applied.
        """
        if not os.path.isdir(self._agents_dir):
            return 0

        with self._lock:
            self._agents = {
                agent_id: entry
                for agent_id, entry in self._agents.items()
                if entry.get("source") != "builtin" or self._is_hot(entry)
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
                        if self._is_hot(self._agents[agent_id]):
                            # Already live from this path; do not downgrade it.
                            continue
                        raise AgentRegistryError(
                            f"agent '{agent_id}' is already registered"
                        )
                    self._agents[agent_id] = self._entry(
                        meta=meta,
                        factory=factory,
                        source="builtin",
                        path=path,
                        sig=self._source_sig(path),
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
        try:
            return normalize_agent_contract(meta, default_id=default_id)
        except AgentContractError as exc:
            raise AgentRegistryError(str(exc)) from exc

    @staticmethod
    def _new_health() -> dict:
        """Fresh breaker state for one Agent generation.

        ``epoch`` identifies the generation. Callers that report an outcome
        include the epoch they served; a report from a superseded generation is
        discarded, so a stale instance cannot trip the version that replaced it.
        """
        return {
            "state": "closed",  # closed -> open -> half_open -> closed
            "consecutive_failures": 0,
            "last_error": "",
            "epoch": 0,
        }

    @staticmethod
    def _entry(
        *,
        meta: dict,
        factory,
        source: str,
        path: str = "",
        instance=None,
        sig: str = "",
        hot_module: str = "",
        health: dict = None,
        opened_at: float = 0.0,
        epoch: int = None,
    ) -> dict:
        resolved_health = dict(health) if health else AgentRegistry._new_health()
        resolved_health["epoch"] = next(_EPOCH_SEQ) if epoch is None else epoch
        return {
            "meta": meta,
            "factory": factory,
            "instance": instance,
            "source": source,
            "path": path,
            "sig": sig,
            "hot_module": hot_module,
            "status": "ready" if instance is not None else "discovered",
            "error": "",
            "opened_at": opened_at,
            "health": resolved_health,
        }

    @staticmethod
    def _cooldown_elapsed(entry: dict) -> bool:
        """True once a quarantined Agent has cooled down enough to be probed."""
        return (time.monotonic() - float(entry.get("opened_at") or 0.0)) >= (
            CIRCUIT_COOLDOWN_SECONDS
        )

    def _is_serving(self, entry: dict) -> bool:
        """Whether this Agent may still appear in the routing candidate queue.

        Two ways an Agent is held back, both self-healing:
          · construction kept failing          -> status "unavailable"
          · its executions kept failing        -> health "open"
        Either way it returns to the queue once the cooldown elapses and a
        half-open probe succeeds. Nothing here is a one-way switch.
        """
        if entry.get("status") == "unavailable":
            return self._cooldown_elapsed(entry)
        if (entry.get("health") or {}).get("state") == "open":
            return self._cooldown_elapsed(entry)
        return True

    @staticmethod
    def _source_sig(path: str) -> str:
        """Content hash of an Agent source file; used to detect edits."""
        try:
            with open(path, "rb") as handle:
                return hashlib.sha1(handle.read()).hexdigest()
        except OSError:
            return ""

    def register(
        self,
        agent_id: str,
        meta: dict,
        instance=None,
        source: str = "dynamic",
        factory=None,
        path: str = "",
        hot_module: str = "",
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
                path=path,
                sig=self._source_sig(path) if path else "",
                hot_module=hot_module,
            )
            self._notify(agent_id, normalized)

    def unregister(self, agent_id: str) -> bool:
        """Remove an Agent at runtime and wake up every subscriber.

        Returns True when an entry was actually removed. ``discover()`` swaps in
        fresh builtin entries while keeping the registry lock for the whole scan,
        so removal has to happen under the same lock to stay consistent.
        """
        with self._lock:
            entry = self._agents.pop(agent_id, None)
            if entry is None:
                return False
            self._release(entry)
            self._evict_hot_module(entry)

        # meta=None is the removal signal: subscribers that only care about
        # re-reading live config (the Router) can ignore the argument entirely.
        # Without this, a removed Agent stays in Router.dynamic_agents and keeps
        # being routed to (see Router._agent_specs, which merges that dict back).
        self._notify(agent_id, None)
        return True

    @staticmethod
    def _release(entry: dict) -> None:
        """Best-effort teardown of an Agent instance before it is dropped."""
        instance = entry.get("instance")
        if instance is None:
            return
        for attr in ("close", "shutdown", "aclose"):
            closer = getattr(instance, attr, None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    _log.debug("Agent teardown via %s() failed", attr, exc_info=True)
                break
        entry["instance"] = None
        entry["status"] = "removed"

    @staticmethod
    def _drop_module(name: Optional[str]) -> None:
        """Remove one module from the import cache, if present."""
        if name and sys.modules.pop(name, None) is not None:
            _log.debug("Evicted hot module %s", name)

    @classmethod
    def _evict_hot_module(cls, entry: dict) -> None:
        """Drop a hot-swapped module from sys.modules.

        Safe by construction: deleting the cache entry does not invalidate the
        module object a running Agent instance already holds, so requests that
        started on the previous version keep executing it to completion. The
        object is reclaimed by refcounting once the last such request lets go,
        which is what keeps repeated reloads from accumulating old code.
        """
        cls._drop_module(entry.get("hot_module"))

    def get(self, agent_id: str) -> Optional[dict]:
        with self._lock:
            return self._agents.get(agent_id)

    def entries(self) -> dict:
        with self._lock:
            return dict(self._agents)

    def get_instance(self, agent_id: str):
        entry = self.get(agent_id)
        return entry.get("instance") if entry else None

    def get_or_create(self, agent_id: str):
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                raise AgentLoadError(f"agent '{agent_id}' is not registered")

            health = entry.setdefault("health", self._new_health())

            # Tripped open: stay out of service until the cooldown elapses, then
            # let exactly this call through as a half-open probe.
            if health.get("state") == "open":
                if not self._cooldown_elapsed(entry):
                    raise AgentLoadError(
                        f"agent '{agent_id}' is tripped open: "
                        f"{health.get('last_error', '')}"
                    )
                health["state"] = "half_open"
                _log.info("Agent '%s' 冷却结束，进入半开探测", agent_id)

            if entry.get("instance") is not None:
                return entry["instance"]

            if entry.get("status") == "unavailable" and not self._cooldown_elapsed(
                entry
            ):
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
                entry["opened_at"] = time.monotonic()
                raise AgentLoadError(
                    f"agent '{agent_id}' failed to load: {exc}"
                ) from exc

            entry["instance"] = instance
            entry["status"] = "ready"
            entry["error"] = ""
            return instance

    def record_failure(
        self,
        agent_id: str,
        error,
        epoch: int = None,
        severity: str = HARD_FAILURE,
    ) -> dict:
        """Report a failed execution and trip the breaker if it keeps happening.

        ``epoch`` is the generation the caller was serving. A report whose epoch
        no longer matches is dropped: an instance that was replaced mid-flight
        must not put a failure on the account of the version that replaced it.

        Only ``severity="hard"`` counts toward tripping -- the output was
        unusable. Semantic ("soft") failures judged by a model are recorded but
        never trip on their own, because a wrong judgement there would swap a
        correct answer for nothing.
        """
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return {}
            health = entry.setdefault("health", self._new_health())
            if epoch is not None and epoch != health.get("epoch"):
                return dict(health)
            if severity != HARD_FAILURE:
                health["last_error"] = str(error)
                return dict(health)

            health["consecutive_failures"] += 1
            health["last_error"] = str(error)
            # A probe that fails goes straight back out of service.
            if (
                health["state"] == "half_open"
                or health["consecutive_failures"] >= CIRCUIT_FAILURE_THRESHOLD
            ):
                health["state"] = "open"
                entry["opened_at"] = time.monotonic()
                _log.warning(
                    "Agent '%s' 熔断（连续失败 %d 次）: %s",
                    agent_id,
                    health["consecutive_failures"],
                    error,
                )
            return dict(health)

    def record_success(self, agent_id: str, epoch: int = None) -> dict:
        """Report a healthy execution: close the breaker and clear the count."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return {}
            health = entry.setdefault("health", self._new_health())
            if epoch is not None and epoch != health.get("epoch"):
                return dict(health)
            if health.get("state") != "closed" or health["consecutive_failures"]:
                _log.info("Agent '%s' 恢复正常", agent_id)
            health["state"] = "closed"
            health["consecutive_failures"] = 0
            health["last_error"] = ""
            entry["opened_at"] = 0.0
            return dict(health)

    def health_of(self, agent_id: str) -> dict:
        entry = self.get(agent_id)
        return dict(entry.get("health") or {}) if entry else {}

    def epoch_of(self, agent_id: str) -> int:
        """Generation counter, used by callers when reporting outcomes."""
        entry = self.get(agent_id)
        return int(((entry or {}).get("health") or {}).get("epoch", 0))

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

    def _load_factory_from_path(self, path: str, factory: str):
        """Load an Agent factory from its file under a fresh, versioned name.

        ``importlib.import_module`` returns whatever is already cached in
        ``sys.modules``, so re-reading an Agent whose source was edited on disk
        silently yields the OLD code: discovery reports success while the
        implementation never changes. Loading from the file path under a unique
        module name is what makes a replacement actually take effect.
        """
        if not isinstance(factory, str) or ":" not in factory:
            raise AgentLoadError(f"invalid Agent factory '{factory}'")
        module_name, attribute = factory.split(":", 1)

        try:
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
        except OSError as exc:
            raise AgentLoadError(f"cannot read Agent source {path}: {exc}") from exc

        versioned = f"{module_name}__hot{next(_HOT_SEQ)}"
        spec = importlib.util.spec_from_file_location(versioned, path)
        if spec is None:
            raise AgentLoadError(f"cannot build a module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[versioned] = module
        try:
            # Compile the source ourselves rather than delegating to the import
            # system's loader. Its .pyc cache is keyed on the source PATH (not the
            # module name) and validated by (mtime, size), so an edit that keeps
            # the same byte length inside the same clock second silently executes
            # stale bytecode -- precisely the failure a reload exists to remove.
            exec(compile(source, path, "exec"), module.__dict__)
            resolved = getattr(module, attribute, None)
            if not callable(resolved):
                raise AgentLoadError(
                    f"factory '{attribute}' in {path} is not callable"
                )
        except Exception:
            # Never leave a half-executed module behind in the import cache.
            sys.modules.pop(versioned, None)
            raise
        return resolved, versioned

    def reload_agent(self, agent_id: str) -> dict:
        """Hot-swap one Agent's implementation without restarting the process.

        Build-then-swap: the replacement is fully constructed BEFORE the registry
        is mutated. A broken edit -- syntax error, missing dependency, factory
        that raises -- therefore cannot take down a healthy Agent; the existing
        entry is left exactly as it was and AgentLoadError is raised.
        """
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                raise AgentLoadError(f"agent '{agent_id}' is not registered")
            path = entry.get("path") or ""

        if not path or not os.path.isfile(path):
            raise AgentLoadError(f"agent '{agent_id}' has no reloadable source file")

        try:
            meta, _ = self._read_manifest(path)
        except Exception as exc:
            # A source that cannot even be parsed must not escape as a raw
            # SyntaxError; the swap contract is that the old Agent survives.
            raise AgentLoadError(
                f"hot reload of '{agent_id}' failed, keeping the running "
                f"version: {exc}"
            ) from exc
        if meta is None:
            raise AgentLoadError(f"agent '{agent_id}' source declares no AGENT_SPEC")
        if meta["id"] != agent_id:
            raise AgentLoadError(
                f"reload id mismatch: {path} declares '{meta['id']}', "
                f"expected '{agent_id}'"
            )

        hot_module = None
        try:
            factory, hot_module = self._load_factory_from_path(
                path, meta.get("factory", "")
            )
            instance = factory()
            if instance is None:
                raise RuntimeError("factory returned None")
        except Exception as exc:
            # The candidate module compiled but was never adopted. Drop it, or
            # every failed reload leaves a copy of the Agent in memory.
            self._drop_module(hot_module)
            raise AgentLoadError(
                f"hot reload of '{agent_id}' failed, keeping the running "
                f"version: {exc}"
            ) from exc

        normalized = self._validate_meta(dict(meta), default_id=agent_id)
        with self._lock:
            previous = self._agents.get(agent_id)
            # A brand-new entry means a brand-new incarnation: fresh breaker
            # state (a fixed Agent should not inherit the old failure count)
            # and a new epoch (in-flight reports from the old one are dropped).
            self._agents[agent_id] = self._entry(
                meta=normalized,
                factory=factory,
                source="builtin",
                path=path,
                instance=instance,
                sig=self._source_sig(path),
                hot_module=hot_module,
            )
            new_epoch = self._agents[agent_id]["health"]["epoch"]
        if previous is not None:
            self._release(previous)
            self._evict_hot_module(previous)
        self._notify(agent_id, normalized)
        return {
            "agent_id": agent_id,
            "status": "reloaded",
            "module": hot_module,
            "epoch": new_epoch,
        }

    def rescan(self) -> dict:
        """Apply only what changed under agents/, and report what it did.

        ``discover()`` rebuilds every builtin entry from scratch; this is the
        runtime-safe counterpart -- new manifests are registered, edited ones
        hot-reloaded, deleted ones unregistered, and untouched ones left alone so
        their warm instances survive.
        """
        report = {
            "added": [],
            "reloaded": [],
            "removed": [],
            "unchanged": [],
            "errors": {},
        }
        if not os.path.isdir(self._agents_dir):
            return report

        on_disk: dict[str, str] = {}
        unparseable: set[str] = set()
        for filename in sorted(os.listdir(self._agents_dir)):
            if not filename.endswith("_agent.py") or filename.startswith("_"):
                continue
            path = os.path.join(self._agents_dir, filename)
            try:
                meta, _ = self._read_manifest(path)
            except Exception as exc:
                report["errors"][filename] = str(exc)
                # The file is still there, it just will not parse right now.
                unparseable.add(path)
                continue
            if meta is not None:
                on_disk[meta["id"]] = path

        with self._lock:
            known = {
                agent_id: entry
                for agent_id, entry in self._agents.items()
                if entry.get("source") == "builtin"
            }

        for agent_id, path in on_disk.items():
            entry = known.get(agent_id)
            try:
                if entry is None:
                    meta, factory_spec = self._read_manifest(path)
                    factory, hot_module = self._load_factory_from_path(
                        path, factory_spec
                    )
                    try:
                        self.register(
                            agent_id,
                            meta,
                            factory=factory,
                            source="builtin",
                            path=path,
                            hot_module=hot_module,
                        )
                    except Exception:
                        self._drop_module(hot_module)
                        raise
                    report["added"].append(agent_id)
                elif entry.get("sig") != self._source_sig(path):
                    self.reload_agent(agent_id)
                    report["reloaded"].append(agent_id)
                else:
                    report["unchanged"].append(agent_id)
            except Exception as exc:
                report["errors"][agent_id] = str(exc)

        for agent_id, entry in known.items():
            if agent_id in on_disk:
                continue
            # A source that is present but currently unparseable is a failed
            # reload, not a deletion: treating it as removed would let one bad
            # edit unregister a perfectly healthy, still-serving Agent.
            if entry.get("path") in unparseable:
                report["errors"].setdefault(
                    agent_id,
                    "source present but unparseable; keeping current version",
                )
                continue
            self.unregister(agent_id)
            report["removed"].append(agent_id)

        return report

    def get_capabilities(self, agent_id: str) -> dict:
        entry = self.get(agent_id)
        if not entry:
            return {}
        instance = entry.get("instance")
        if instance is not None and hasattr(instance, "capabilities"):
            return instance.capabilities
        return entry["meta"].get("capabilities", {})

    def list_all(self, include_unavailable: bool = False) -> dict:
        with self._lock:
            return {
                agent_id: entry["meta"]
                for agent_id, entry in self._agents.items()
                if include_unavailable or self._is_serving(entry)
            }

    def get_router_config(self) -> dict:
        # Snapshot under the lock: reload_agent() rebinds an entry mid-flight,
        # and readers must never observe a half-swapped registry.
        with self._lock:
            snapshot = dict(self._agents)
        config = {}
        for agent_id, entry in snapshot.items():
            # A tripped or unavailable Agent simply stops being a routing
            # candidate -- it is hidden, never deleted, so it can come back.
            if not self._is_serving(entry):
                continue
            meta = entry["meta"]
            config[agent_id] = {
                "id": meta.get("id", agent_id),
                "name": meta.get("name", agent_id),
                "description": meta.get("description", ""),
                "keywords": meta.get("keywords", []),
                "intents": meta.get("intents", []),
                "examples": meta.get("examples", []),
                "few_shots": meta.get("few_shots", []),
                "requires": meta.get("requires", []),
                "exports": meta.get("exports", []),
                "icon": meta.get("icon", "🔧"),
                "source": entry.get("source", "builtin"),
                "status": entry.get("status", "discovered"),
                "capabilities": meta.get("capabilities", {}),
                "input_schema": meta.get("input_schema", {}),
                "output_schema": meta.get("output_schema", {}),
            }
        return config

    def loader_registry(self) -> dict:
        with self._lock:
            ready = [
                agent_id
                for agent_id, entry in self._agents.items()
                if self._is_serving(entry)
            ]
        return {
            agent_id: (lambda current=agent_id: self.get_or_create(current))
            for agent_id in ready
        }

    def on_register(self, callback: Callable, owner: str = None) -> None:
        if owner is not None:
            self._callbacks = [
                (registered_owner, registered_callback)
                for registered_owner, registered_callback in self._callbacks
                if registered_owner != owner
            ]
        self._callbacks.append((owner, callback))

    def _notify(self, agent_id: str, meta: Optional[dict]) -> None:
        """Fan out a registration/removal event to subscribers.

        ``meta is None`` means the Agent was unregistered. Subscribers that only
        rebuild from live registry state (the Router) ignore the argument, which
        is what lets one channel serve both events.
        """
        for _, callback in list(self._callbacks):
            try:
                callback(agent_id, meta)
            except Exception:
                _log.debug("Agent registry callback failed", exc_info=True)

    def register_skill_loader(self, skill_loader) -> int:
        """Register every loaded Skill as an Agent; returns how many were added.

        Already-registered ids are skipped so this is safe to call again after a
        runtime re-discovery or a crystallize, when most Skills are unchanged.
        """
        added = 0
        for skill_id, skill in skill_loader.skills.items():
            if skill_id in self._agents:
                continue
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
            added += 1
        return added

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
