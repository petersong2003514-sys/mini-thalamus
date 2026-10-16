"""Uniform input/output adapter shared by every Agent instance."""

import logging

from mini_thalamus.base_agent import AgentContext, AgentResult


_log = logging.getLogger(__name__)

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


def _has_payload(result: dict) -> bool:
    """A result with neither an answer nor data is unusable to the caller."""
    return bool(result.get("answer") or result.get("data"))


class StandardAgentAdapter:
    """Wrap any agent behind a single run(query, context) -> AgentResult contract.

    Every Agent invocation in the platform funnels through here, which makes it
    the one place that can report an execution outcome to the registry -- and
    that is what feeds the circuit breaker. A failure anywhere downstream becomes
    a recorded failure; a healthy run closes the breaker again.
    """

    def __init__(
        self,
        agent,
        agent_id: str = "",
        registry=None,
        epoch: int = None,
    ):
        self._agent = agent
        self._registry = registry
        self.agent_id = (
            agent_id
            or getattr(agent, "agent_id", "")
            or type(agent).__name__
        )
        # The generation this adapter was handed. Outcomes reported later are
        # tagged with it so that an instance replaced mid-flight cannot put its
        # failure on the account of the version that replaced it.
        self._epoch = epoch
        if self._registry is not None and self._epoch is None:
            self._epoch = self._registry.epoch_of(self.agent_id)

    def run(self, query: str, context: dict = None) -> dict:
        if isinstance(context, AgentContext):
            context = context.to_dict()
        try:
            raw = self._agent.run(query, context=context)
        except Exception as exc:
            self._report_failure(exc)
            raise

        result = normalize_agent_result(raw, agent_id=self.agent_id)
        if result.get("status") == "error" or not _has_payload(result):
            self._report_failure(result.get("answer") or "empty result")
        else:
            self._report_success()
        return result

    def _report_failure(self, error) -> None:
        if self._registry is None:
            return
        try:
            self._registry.record_failure(
                self.agent_id, error, epoch=self._epoch
            )
        except Exception:
            # Reporting must never be the thing that breaks a request.
            _log.debug("record_failure failed", exc_info=True)

    def _report_success(self) -> None:
        if self._registry is None:
            return
        try:
            self._registry.record_success(self.agent_id, epoch=self._epoch)
        except Exception:
            _log.debug("record_success failed", exc_info=True)

    def __getattr__(self, name):
        return getattr(self._agent, name)


def adapt_agent(agent, agent_id: str = "", registry=None, epoch: int = None):
    """Idempotently wrap an Agent instance with the standard adapter."""
    if isinstance(agent, StandardAgentAdapter):
        return agent
    return StandardAgentAdapter(
        agent, agent_id=agent_id, registry=registry, epoch=epoch
    )


__all__ = ["StandardAgentAdapter", "adapt_agent", "normalize_agent_result"]
