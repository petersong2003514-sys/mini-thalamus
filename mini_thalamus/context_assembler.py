"""Deterministic, capability-aware Agent memory assembly."""

import json
import re

from base_agent import AgentContext


class ContextAssembler:
    def __init__(self, memory_store, session_manager):
        self.store = memory_store
        self.sessions = session_manager

    @staticmethod
    def _terms(text: str) -> set:
        return set(
            re.findall(
                r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fff]{2,4}",
                (text or "").lower(),
            )
        )

    @staticmethod
    def _turn_dict(turn) -> dict:
        return {
            "id": turn.turn_id,
            "q": turn.query,
            "a": turn.answer,
            "agent": turn.agent_id or "",
            "tags": list(turn.tags),
            "status": turn.status,
        }

    @staticmethod
    def _memory_size(history_text, relevant, long_term) -> int:
        return (
            len(history_text)
            + len(json.dumps(relevant, ensure_ascii=False))
            + len(json.dumps(long_term, ensure_ascii=False))
        )

    def build(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        capabilities: dict,
        current_query: str,
        previous_result: dict = None,
        workflow_inputs: dict = None,
    ) -> AgentContext:
        capabilities = capabilities or {}
        budget = max(0, int(capabilities.get("max_context_chars", 4000)))
        session = self.sessions.resolve(user_id, session_id)
        relevant = []
        long_term = {}
        history_text = ""
        tags = set()

        turns = []
        if capabilities.get("needs_history", True):
            turns = self.store.list_turns(
                session_id,
                statuses=("success", "error"),
            )
            for turn in turns[-20:]:
                tags.update(turn.tags)

            if session.summary:
                candidate = f"[会话摘要] {session.summary}"
                available = max(0, budget - 4)
                history_text = candidate[:available]

            query_terms = self._terms(current_query)
            relevant_turns = []
            for turn in reversed(turns[-10:]):
                turn_terms = self._terms(turn.query)
                turn_terms.update(turn.tags)
                if query_terms & turn_terms:
                    relevant_turns.append(turn)

            used_ids = set()
            for turn in reversed(relevant_turns[:3]):
                item = self._turn_dict(turn)
                candidate = [*relevant, item]
                if self._memory_size(
                    history_text,
                    candidate,
                    long_term,
                ) <= budget:
                    relevant = candidate
                    used_ids.add(turn.turn_id)

            for turn in reversed(turns[-5:]):
                if turn.turn_id in used_ids:
                    continue
                block = (
                    f"\n[{turn.turn_id}][{turn.status}] "
                    f"用户: {turn.query}\n助手: {turn.answer[:500]}"
                )
                candidate = history_text + block
                if self._memory_size(
                    candidate,
                    relevant,
                    long_term,
                ) <= budget:
                    history_text = candidate

        if capabilities.get("needs_long_term", False):
            for memory in reversed(self.store.lookup_long_term(user_id)):
                candidate = {**long_term, memory.memory_key: memory.gist}
                if self._memory_size(
                    history_text,
                    relevant,
                    candidate,
                ) <= budget:
                    long_term = candidate

        files = (
            self.store.get_files(session_id)
            if capabilities.get("needs_files", False)
            else []
        )
        return AgentContext(
            current_query=current_query,
            relevant_history=relevant,
            tags=sorted(tags),
            long_term=long_term,
            files=files,
            session_id=session_id,
            previous_result=previous_result,
            conversation_history=history_text,
            workflow_inputs=dict(workflow_inputs or {}),
        )


__all__ = ["ContextAssembler"]
