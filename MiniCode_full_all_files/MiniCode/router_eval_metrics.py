"""Pure scoring functions for Router evaluation predictions."""

from collections import Counter, defaultdict
import math

from router_eval_dataset import RouterEvalCase


QUALITY_GATES = {
    "rule_accuracy": (">=", 1.0),
    "false_rule_capture_rate": ("<=", 0.01),
    "semantic_top3_recall": (">=", 0.90),
    "single_intent_accuracy": (">=", 0.85),
    "multi_step_recall": (">=", 0.80),
    "schema_valid_rate": (">=", 1.0),
    "fallback_accuracy": (">=", 0.90),
}


def _safe_div(numerator: int | float, denominator: int | float):
    return numerator / denominator if denominator else None


def _f1(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _counter_counts(expected: list[str], predicted: list[str]):
    expected_counter = Counter(expected)
    predicted_counter = Counter(predicted)
    true_positive = sum((expected_counter & predicted_counter).values())
    return true_positive, sum(predicted_counter.values()), sum(expected_counter.values())


def _prf(expected: list[str], predicted: list[str]):
    true_positive, predicted_count, expected_count = _counter_counts(
        expected,
        predicted,
    )
    precision = _safe_div(true_positive, predicted_count)
    recall = _safe_div(true_positive, expected_count)
    return precision, recall, _f1(precision, recall), true_positive


def _candidate_ids(value) -> list[str]:
    result = []
    for item in value or []:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and isinstance(item.get("agent_id"), str):
            result.append(item["agent_id"])
    return result


def score_case(case: RouterEvalCase, prediction: dict) -> dict:
    raw_calls = prediction.get("calls", [])
    predicted_calls = [call for call in raw_calls if isinstance(call, dict)]
    expected_agents = [call.agent_id for call in case.expected_calls]
    predicted_agents = [
        call.get("agent_id", "")
        for call in predicted_calls
        if call.get("agent_id")
    ]
    step_precision, step_recall, step_f1, step_tp = _prf(
        expected_agents,
        predicted_agents,
    )
    sequence_exact = predicted_agents == expected_agents

    expected_edges = {
        (dependency, f"step_{index}")
        for index, call in enumerate(case.expected_calls, start=1)
        for dependency in call.depends_on
    }
    predicted_edges = {
        (dependency, call.get("call_id", f"step_{index}"))
        for index, call in enumerate(predicted_calls, start=1)
        for dependency in call.get("depends_on", [])
        if isinstance(dependency, str)
    }
    edge_tp = len(expected_edges & predicted_edges)
    edge_precision = _safe_div(edge_tp, len(predicted_edges))
    edge_recall = _safe_div(edge_tp, len(expected_edges))
    if not expected_edges and not predicted_edges:
        edge_precision = edge_recall = 1.0
    edge_f1 = _f1(edge_precision, edge_recall)

    expected_refs = {
        (f"step_{index}", ref)
        for index, call in enumerate(case.expected_calls, start=1)
        for ref in call.input_refs
    }
    predicted_refs = {
        (call.get("call_id", f"step_{index}"), ref)
        for index, call in enumerate(predicted_calls, start=1)
        for ref in call.get("input_refs", [])
        if isinstance(ref, str)
    }
    input_refs_exact = expected_refs == predicted_refs

    expected_primary = expected_agents[0]
    predicted_primary = predicted_agents[0] if predicted_agents else None
    allowed = {expected_primary, *case.acceptable_agents}
    single_correct = (
        predicted_primary in allowed
        if case.expected_mode == "single"
        else None
    )
    mode_correct = prediction.get("mode") == case.expected_mode
    workflow_exact = (
        mode_correct
        and sequence_exact
        and expected_edges == predicted_edges
        and input_refs_exact
        if case.expected_mode == "workflow"
        else None
    )
    candidates = _candidate_ids(prediction.get("candidates"))
    top3_hit = bool(allowed & set(candidates[:3]))

    return {
        "case_id": case.case_id,
        "expected_mode": case.expected_mode,
        "expected_primary_agent": expected_primary,
        "acceptable_agents": list(case.acceptable_agents),
        "predicted_primary_agent": predicted_primary,
        "expected_stage": case.expected_stage,
        "difficulty": case.difficulty,
        "tags": list(case.tags),
        "method": prediction.get("method", "error"),
        "mode_correct": mode_correct,
        "single_correct": single_correct,
        "step_precision": step_precision,
        "step_recall": step_recall,
        "step_f1": step_f1,
        "sequence_exact": sequence_exact,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "input_refs_exact": input_refs_exact,
        "workflow_exact": workflow_exact,
        "top3_hit": top3_hit,
        "schema_valid": bool(prediction.get("schema_valid", False)),
        "latency_ms": float(prediction.get("latency_ms", 0.0) or 0.0),
        "error": prediction.get("error"),
        "_step_tp": step_tp,
        "_step_predicted": len(predicted_agents),
        "_step_expected": len(expected_agents),
        "_edge_tp": edge_tp,
        "_edge_predicted": len(predicted_edges),
        "_edge_expected": len(expected_edges),
    }


def _average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _macro_f1(scores: list[dict]):
    singles = [score for score in scores if score["single_correct"] is not None]
    labels = sorted({score["expected_primary_agent"] for score in singles})
    if not labels:
        return None
    values = []
    for label in labels:
        true_positive = sum(
            score["expected_primary_agent"] == label
            and score["predicted_primary_agent"] == label
            for score in singles
        )
        false_positive = sum(
            score["expected_primary_agent"] != label
            and score["predicted_primary_agent"] == label
            for score in singles
        )
        false_negative = sum(
            score["expected_primary_agent"] == label
            and score["predicted_primary_agent"] != label
            for score in singles
        )
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        values.append(_f1(precision or 0.0, recall or 0.0))
    return sum(values) / len(values)


def _slice_summary(scores: list[dict]):
    singles = [score for score in scores if score["single_correct"] is not None]
    workflows = [score for score in scores if score["workflow_exact"] is not None]
    return {
        "case_count": len(scores),
        "single_intent_accuracy": _average(
            [float(score["single_correct"]) for score in singles]
        ),
        "workflow_exact_rate": _average(
            [float(score["workflow_exact"]) for score in workflows]
        ),
        "method_distribution": dict(Counter(score["method"] for score in scores)),
    }


def aggregate_scores(scored_cases: list[dict]) -> dict:
    scores = list(scored_cases)
    singles = [score for score in scores if score["single_correct"] is not None]
    workflows = [score for score in scores if score["workflow_exact"] is not None]
    rule_cases = [score for score in scores if score["expected_stage"] == "rule"]
    non_rule_cases = [score for score in scores if score["expected_stage"] != "rule"]
    semantic_cases = [
        score for score in scores if score["expected_stage"] == "semantic"
    ]
    fallback_cases = [
        score for score in scores if score["expected_stage"] == "fallback"
    ]
    step_tp = sum(score["_step_tp"] for score in workflows)
    step_predicted = sum(score["_step_predicted"] for score in workflows)
    step_expected = sum(score["_step_expected"] for score in workflows)
    multi_precision = _safe_div(step_tp, step_predicted)
    multi_recall = _safe_div(step_tp, step_expected)

    dimensions = {
        "difficulty": defaultdict(list),
        "expected_stage": defaultdict(list),
        "tag": defaultdict(list),
        "agent": defaultdict(list),
    }
    for score in scores:
        dimensions["difficulty"][score["difficulty"]].append(score)
        dimensions["expected_stage"][score["expected_stage"]].append(score)
        dimensions["agent"][score["expected_primary_agent"]].append(score)
        for tag in score["tags"]:
            dimensions["tag"][tag].append(score)

    return {
        "case_count": len(scores),
        "single_intent_accuracy": _average(
            [float(score["single_correct"]) for score in singles]
        ),
        "macro_f1": _macro_f1(scores),
        "workflow_exact_rate": _average(
            [float(score["workflow_exact"]) for score in workflows]
        ),
        "multi_step_precision": multi_precision,
        "multi_step_recall": multi_recall,
        "multi_step_f1": _f1(multi_precision, multi_recall),
        "rule_accuracy": _average([
            float(
                score["method"] == "rule"
                and bool(score["single_correct"] or score["workflow_exact"])
            )
            for score in rule_cases
        ]),
        "false_rule_capture_rate": _average([
            float(score["method"] == "rule") for score in non_rule_cases
        ]),
        "semantic_top3_recall": _average([
            float(score["top3_hit"]) for score in semantic_cases
        ]),
        "schema_valid_rate": _average([
            float(score["schema_valid"]) for score in scores
        ]),
        "fallback_accuracy": _average([
            float(bool(score["single_correct"])) for score in fallback_cases
        ]),
        "method_distribution": dict(Counter(score["method"] for score in scores)),
        "latency_ms": {
            "p50": _percentile([score["latency_ms"] for score in scores], 0.50),
            "p95": _percentile([score["latency_ms"] for score in scores], 0.95),
        },
        "error_count": sum(bool(score["error"]) for score in scores),
        "slices": {
            dimension: {
                key: _slice_summary(items)
                for key, items in sorted(groups.items())
            }
            for dimension, groups in dimensions.items()
        },
    }


def evaluate_quality_gates(metrics: dict) -> dict:
    results = {}
    available_passed = []
    for name, (operator, target) in QUALITY_GATES.items():
        value = metrics.get(name)
        if value is None:
            status = "skipped"
        else:
            passed = value >= target if operator == ">=" else value <= target
            status = "passed" if passed else "failed"
            available_passed.append(passed)
        results[name] = {
            "status": status,
            "value": value,
            "operator": operator,
            "target": target,
        }
    results["all_available_passed"] = all(available_passed)
    return results


__all__ = [
    "QUALITY_GATES",
    "aggregate_scores",
    "evaluate_quality_gates",
    "score_case",
]
