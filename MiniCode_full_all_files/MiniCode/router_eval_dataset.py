"""Strict JSONL contract for layered Router evaluation cases."""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


CASE_FIELDS = {
    "id",
    "query",
    "session_context",
    "file_context",
    "expected",
    "expected_stage",
    "difficulty",
    "tags",
}
EXPECTED_FIELDS = {"mode", "calls", "acceptable_agents"}
CALL_FIELDS = {"agent_id", "depends_on", "input_refs"}
VALID_MODES = {"single", "workflow"}
VALID_STAGES = {"rule", "semantic", "llm", "fallback"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class RouterEvalDatasetError(ValueError):
    """Raised when a benchmark case violates the evaluation contract."""


@dataclass(frozen=True)
class ExpectedCall:
    agent_id: str
    depends_on: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouterEvalCase:
    case_id: str
    query: str
    session_context: str
    file_context: dict
    expected_mode: str
    expected_calls: tuple[ExpectedCall, ...]
    acceptable_agents: tuple[str, ...]
    expected_stage: str
    difficulty: str
    tags: tuple[str, ...]


def _require_fields(value: dict, expected: set[str], path: str):
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RouterEvalDatasetError(
            f"{path} fields mismatch; missing={missing}, extra={extra}"
        )


def _string_list(value, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RouterEvalDatasetError(f"{path} must be a list of strings")
    if len(set(value)) != len(value):
        raise RouterEvalDatasetError(f"{path} contains duplicates")
    return tuple(value)


def _validate_dag(calls: tuple[ExpectedCall, ...], case_id: str):
    step_ids = tuple(f"step_{index}" for index in range(1, len(calls) + 1))
    dependencies = {
        step_id: set(call.depends_on)
        for step_id, call in zip(step_ids, calls)
    }
    valid = set(step_ids)
    for step_id, refs in dependencies.items():
        unknown = refs - valid
        if unknown:
            raise RouterEvalDatasetError(
                f"case {case_id} has unknown dependencies: {sorted(unknown)}"
            )
        if step_id in refs:
            raise RouterEvalDatasetError(
                f"case {case_id} dependency cycle at {step_id}"
            )

    remaining = {key: set(value) for key, value in dependencies.items()}
    completed = set()
    while remaining:
        ready = [
            step_id
            for step_id, refs in remaining.items()
            if refs.issubset(completed)
        ]
        if not ready:
            raise RouterEvalDatasetError(
                f"case {case_id} dependency cycle"
            )
        for step_id in ready:
            completed.add(step_id)
            del remaining[step_id]

    for index, call in enumerate(calls, start=1):
        dependencies_for_call = set(call.depends_on)
        for ref in call.input_refs:
            source, separator, path = ref.partition(".")
            if not separator or not path:
                raise RouterEvalDatasetError(
                    f"case {case_id} has invalid input_ref: {ref}"
                )
            if source not in dependencies_for_call:
                raise RouterEvalDatasetError(
                    f"case {case_id} input_ref {ref} is not a dependency of step_{index}"
                )


def _parse_case(value, line_number: int, registered_agents: set[str]):
    path = f"line {line_number}"
    if not isinstance(value, dict):
        raise RouterEvalDatasetError(f"{path} must be a JSON object")
    _require_fields(value, CASE_FIELDS, path)

    case_id = value["id"]
    query = value["query"]
    if not isinstance(case_id, str) or not case_id:
        raise RouterEvalDatasetError(f"{path}.id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise RouterEvalDatasetError(f"case {case_id} query is empty")
    if not isinstance(value["session_context"], str):
        raise RouterEvalDatasetError(
            f"case {case_id} session_context must be a string"
        )
    if not isinstance(value["file_context"], dict):
        raise RouterEvalDatasetError(
            f"case {case_id} file_context must be an object"
        )

    expected = value["expected"]
    if not isinstance(expected, dict):
        raise RouterEvalDatasetError(f"case {case_id} expected must be an object")
    _require_fields(expected, EXPECTED_FIELDS, f"case {case_id}.expected")
    mode = expected["mode"]
    if mode not in VALID_MODES:
        raise RouterEvalDatasetError(f"case {case_id} has invalid mode {mode!r}")
    raw_calls = expected["calls"]
    if not isinstance(raw_calls, list) or not raw_calls:
        raise RouterEvalDatasetError(f"case {case_id} calls must be non-empty")
    if mode == "single" and len(raw_calls) != 1:
        raise RouterEvalDatasetError(
            f"case {case_id} single mode requires exactly one call"
        )
    if mode == "workflow" and len(raw_calls) < 2:
        raise RouterEvalDatasetError(
            f"case {case_id} workflow mode requires at least two calls"
        )

    calls = []
    for index, raw_call in enumerate(raw_calls, start=1):
        if not isinstance(raw_call, dict):
            raise RouterEvalDatasetError(
                f"case {case_id}.step_{index} must be an object"
            )
        _require_fields(
            raw_call,
            CALL_FIELDS,
            f"case {case_id}.step_{index}",
        )
        agent_id = raw_call["agent_id"]
        if agent_id not in registered_agents:
            raise RouterEvalDatasetError(
                f"case {case_id} references unknown agent {agent_id!r}"
            )
        calls.append(ExpectedCall(
            agent_id=agent_id,
            depends_on=_string_list(
                raw_call["depends_on"],
                f"case {case_id}.step_{index}.depends_on",
            ),
            input_refs=_string_list(
                raw_call["input_refs"],
                f"case {case_id}.step_{index}.input_refs",
            ),
        ))
    calls_tuple = tuple(calls)
    _validate_dag(calls_tuple, case_id)

    acceptable_agents = _string_list(
        expected["acceptable_agents"],
        f"case {case_id}.acceptable_agents",
    )
    unknown_acceptable = set(acceptable_agents) - registered_agents
    if unknown_acceptable:
        raise RouterEvalDatasetError(
            f"case {case_id} has unknown acceptable agents: "
            f"{sorted(unknown_acceptable)}"
        )
    stage = value["expected_stage"]
    difficulty = value["difficulty"]
    tags = _string_list(value["tags"], f"case {case_id}.tags")
    if stage not in VALID_STAGES:
        raise RouterEvalDatasetError(
            f"case {case_id} has invalid expected_stage {stage!r}"
        )
    if difficulty not in VALID_DIFFICULTIES:
        raise RouterEvalDatasetError(
            f"case {case_id} has invalid difficulty {difficulty!r}"
        )
    if not tags:
        raise RouterEvalDatasetError(f"case {case_id} requires at least one tag")

    return RouterEvalCase(
        case_id=case_id,
        query=query,
        session_context=value["session_context"],
        file_context=dict(value["file_context"]),
        expected_mode=mode,
        expected_calls=calls_tuple,
        acceptable_agents=acceptable_agents,
        expected_stage=stage,
        difficulty=difficulty,
        tags=tags,
    )


def load_cases(
    path: str,
    registered_agents: Iterable[str],
    expected_count: int = 200,
) -> list[RouterEvalCase]:
    agents = set(registered_agents)
    cases = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RouterEvalDatasetError(
                        f"line {line_number} is invalid JSON: {exc.msg}"
                    ) from exc
                case = _parse_case(value, line_number, agents)
                if case.case_id in seen:
                    raise RouterEvalDatasetError(
                        f"duplicate case id: {case.case_id}"
                    )
                seen.add(case.case_id)
                cases.append(case)
    except OSError as exc:
        raise RouterEvalDatasetError(f"cannot read dataset {path}: {exc}") from exc

    if len(cases) != int(expected_count):
        raise RouterEvalDatasetError(
            f"expected {expected_count} cases, found {len(cases)}"
        )
    return cases


def dataset_fingerprint(cases: Iterable[RouterEvalCase]) -> str:
    normalized = [asdict(case) for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ExpectedCall",
    "RouterEvalCase",
    "RouterEvalDatasetError",
    "dataset_fingerprint",
    "load_cases",
]
