"""Standard Agent capability contract used by Router and Workflow Planner."""


class AgentContractError(ValueError):
    """Raised when an Agent profile/contract is invalid."""


def _string_list(value, *, field: str, agent_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AgentContractError(f"agent '{agent_id}' {field} must be a list of strings")
    return list(value)


def _normalize_few_shots(meta: dict, intents: list[str], agent_id: str) -> list[dict]:
    raw = meta.get("few_shots")
    if raw is None:
        default_intent = intents[0] if intents else "general_task"
        raw = [
            {"query": example, "expected_intent": default_intent}
            for example in meta.get("examples", []) or []
        ]

    if not isinstance(raw, list):
        raise AgentContractError(f"agent '{agent_id}' few_shots must be a list")

    few_shots = []
    allowed_intents = set(intents)
    for item in raw:
        if isinstance(item, str):
            expected_intent = intents[0] if intents else "general_task"
            item = {"query": item, "expected_intent": expected_intent}
        if not isinstance(item, dict):
            raise AgentContractError(
                f"agent '{agent_id}' few_shots entries must be objects"
            )
        query = item.get("query")
        expected_intent = item.get("expected_intent")
        if not isinstance(query, str) or not query:
            raise AgentContractError(
                f"agent '{agent_id}' few_shots.query must be a non-empty string"
            )
        if not isinstance(expected_intent, str) or not expected_intent:
            raise AgentContractError(
                f"agent '{agent_id}' few_shots.expected_intent must be a non-empty string"
            )
        if allowed_intents and expected_intent not in allowed_intents:
            raise AgentContractError(
                f"agent '{agent_id}' few_shot intent '{expected_intent}' is not registered"
            )
        few_shots.append({
            "query": query,
            "expected_intent": expected_intent,
        })
    return few_shots


def normalize_agent_contract(meta: dict, default_id: str = "") -> dict:
    """Return a normalized Agent contract.

    The contract keeps legacy fields (`examples`, `input_schema`, `output_schema`)
    while adding the standard fields consumed by the next DAG planning stage:
    `few_shots`, `requires`, and `exports`.
    """
    if not isinstance(meta, dict):
        raise AgentContractError("Agent contract must be an object")

    agent_id = meta.get("id") or default_id
    if not isinstance(agent_id, str) or not agent_id:
        raise AgentContractError("Agent id must be a non-empty string")

    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise AgentContractError(f"agent '{agent_id}' requires a name")

    description = meta.get("description", "")
    if not isinstance(description, str):
        raise AgentContractError(f"agent '{agent_id}' description must be a string")

    normalized = dict(meta)
    normalized["id"] = agent_id
    normalized["description"] = description
    normalized["keywords"] = _string_list(meta.get("keywords", []), field="keywords", agent_id=agent_id)
    normalized["intents"] = _string_list(meta.get("intents", []), field="intents", agent_id=agent_id)
    normalized["examples"] = _string_list(meta.get("examples", []), field="examples", agent_id=agent_id)
    normalized["few_shots"] = _normalize_few_shots(
        meta,
        normalized["intents"],
        agent_id,
    )
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
    normalized["requires"] = _string_list(
        meta.get("requires", normalized["input_schema"].get("required", [])),
        field="requires",
        agent_id=agent_id,
    )
    normalized["exports"] = _string_list(
        meta.get("exports", normalized["output_schema"].get("exports", [])),
        field="exports",
        agent_id=agent_id,
    )
    normalized.setdefault("capabilities", {})
    return normalized


__all__ = ["AgentContractError", "normalize_agent_contract"]
