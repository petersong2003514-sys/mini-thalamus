"""Uniform input/output adapter shared by every Agent instance."""

from base_agent import AgentContext, AgentResult


STANDARD_STATUSES = {"success", "partial", "fallback", "error", "blocked", "skipped"}
USABLE_PARTIAL_STATUSES = {"partial", "fallback"}


def normalize_agent_result(raw, *, agent_id: str = "") -> dict:
    """Return a canonical AgentResult dict with normalized status semantics."""
    if isinstance(raw, AgentResult):
        result = raw.to_dict()
    elif isinstance(raw, dict):
        result = AgentResult.from_dict(raw).to_dict()
    else:
        raise TypeError(f"Agent '{agent_id}' must return AgentResult or dict")

    metadata = dict(result.get("metadata") or {})
    status = result.get("status", "success") or "success"
    answer = result.get("answer") or ""
    data = result.get("data") or {}

    # A partial/fallback result with usable content should not break DAG
    # dependencies; keep the original status visible in metadata.
    if status in USABLE_PARTIAL_STATUSES and (answer or data):
        metadata["original_status"] = status
        metadata["usable_partial"] = True
        status = "success"

    if status not in STANDARD_STATUSES:
        status = "error" if status in {"failed", "failure"} else "success"

    result["status"] = status
    if agent_id and not metadata.get("agent_id"):
        metadata["agent_id"] = agent_id
    result["metadata"] = metadata
    return result


class StandardAgentAdapter:
    """Wrap any agent behind a single run(query, context) -> AgentResult contract."""

    def __init__(self, agent, agent_id: str = ""):
        self._agent = agent
        self.agent_id = (
            agent_id
            or getattr(agent, "agent_id", "")
            or type(agent).__name__
        )

    def run(self, query: str, context: dict = None) -> dict:
        if isinstance(context, AgentContext):
            context = context.to_dict()
        raw = self._agent.run(query, context=context)
        return normalize_agent_result(raw, agent_id=self.agent_id)

    def __getattr__(self, name):
        return getattr(self._agent, name)


def adapt_agent(agent, agent_id: str = ""):
    """Idempotently wrap an Agent instance with the standard adapter."""
    if isinstance(agent, StandardAgentAdapter):
        return agent
    return StandardAgentAdapter(agent, agent_id=agent_id)


__all__ = ["StandardAgentAdapter", "adapt_agent", "normalize_agent_result"]
