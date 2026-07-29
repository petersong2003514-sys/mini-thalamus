"""Shared Agent runtime used by CLI, web endpoints and tests."""

import os
import sys
import threading

from agent_registry import AgentLoadError, AgentRegistry


# The delivery bundle contains two sibling top-level packages. The host runtime
# owns this one-time package-root bootstrap; child Agents must not mutate it.
_MINICODE_ROOT = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_ROOT = os.path.dirname(_MINICODE_ROOT)
if _BUNDLE_ROOT not in sys.path:
    sys.path.insert(0, _BUNDLE_ROOT)


class AgentRuntime:
    def __init__(self, registry: AgentRegistry = None):
        self.registry = registry or AgentRegistry()
        self._discovered = False

    def discover(self) -> int:
        count = self.registry.discover()
        self._discovered = True
        return count

    def ensure_discovered(self) -> None:
        if not self._discovered:
            self.discover()

    def list_agent_ids(self, include_unavailable: bool = False) -> list[str]:
        self.ensure_discovered()
        result = []
        for agent_id, entry in self.registry.entries().items():
            if include_unavailable or entry["status"] != "unavailable":
                result.append(agent_id)
        return sorted(result)

    def get_agent(self, agent_id: str):
        self.ensure_discovered()
        return self.registry.get_or_create(agent_id)

    def load_all(self) -> dict:
        self.ensure_discovered()
        agents = {}
        for agent_id in self.list_agent_ids():
            try:
                agents[agent_id] = self.registry.get_or_create(agent_id)
            except AgentLoadError:
                continue
        return agents

    def loader_registry(self) -> dict:
        self.ensure_discovered()
        return {
            agent_id: (lambda current=agent_id: self.get_agent(current))
            for agent_id in self.list_agent_ids()
        }


_runtime = None
_runtime_lock = threading.Lock()


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = AgentRuntime()
                _runtime.discover()
    return _runtime


def reset_runtime_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None


__all__ = ["AgentRuntime", "get_runtime", "reset_runtime_for_tests"]
