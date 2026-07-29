"""Layered, non-executing Router evaluation CLI."""

import argparse
from datetime import datetime, timezone
import json
import os
import time

from agent_runtime import AgentRuntime
from router import _COMPOUND_PATTERN, FALLBACK_AGENT, Router
from router_eval_dataset import dataset_fingerprint, load_cases
from router_eval_metrics import aggregate_scores, evaluate_quality_gates, score_case
from semantic_router import SemanticMatcher


DEFAULT_CASES = os.path.join(os.path.dirname(__file__), "evals", "router_cases.jsonl")
DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "evals", "results")


class UnavailableSemanticMatcher:
    """Explicit no-op matcher for environments with an unsafe native runtime."""

    def __init__(self, reason: str):
        self.last_error = reason

    def rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
        return []


def normalize_decision(decision, latency_ms: float) -> dict:
    raw = decision.to_dict() if hasattr(decision, "to_dict") else decision
    if not isinstance(raw, dict):
        raise TypeError("route decision must be an object")
    calls = []
    for index, call in enumerate(raw.get("calls", []), start=1):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        calls.append({
            "call_id": call.get("call_id", f"step_{index}"),
            "agent_id": function.get("name", ""),
            "depends_on": list(call.get("depends_on", [])),
            "input_refs": list(arguments.get("input_refs", [])),
        })
    return {
        "mode": raw.get("mode"),
        "calls": calls,
        "candidates": list(raw.get("candidates", [])),
        "method": raw.get("method", "unknown"),
        "schema_valid": True,
        "latency_ms": float(latency_ms),
        "error": None,
    }


def _safe_prediction_error(elapsed_ms: float, exc: Exception) -> dict:
    return {
        "mode": None,
        "calls": [],
        "candidates": [],
        "method": "error",
        "schema_valid": False,
        "latency_ms": elapsed_ms,
        "error": type(exc).__name__,
    }


def evaluate_cases(cases, *, layer: str, predictor, dataset_hash: str) -> dict:
    predictions = []
    scores = []
    for case in cases:
        started = time.perf_counter()
        try:
            prediction = predictor(case)
            if not isinstance(prediction, dict):
                prediction = normalize_decision(
                    prediction,
                    (time.perf_counter() - started) * 1000,
                )
        except Exception as exc:
            prediction = _safe_prediction_error(
                (time.perf_counter() - started) * 1000,
                exc,
            )
        prediction = dict(prediction)
        prediction["case_id"] = case.case_id
        predictions.append(prediction)
        scores.append(score_case(case, prediction))
    metrics = aggregate_scores(scores)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "dataset_fingerprint": dataset_hash,
        "metrics": metrics,
        "quality_gates": evaluate_quality_gates(metrics),
        "predictions": predictions,
        "scores": [
            {key: value for key, value in score.items() if not key.startswith("_")}
            for score in scores
        ],
    }


def _markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Router Evaluation",
        "",
        f"- Layer: `{report['layer']}`",
        f"- Cases: {metrics['case_count']}",
        f"- Dataset: `{report['dataset_fingerprint']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in (
        "single_intent_accuracy", "macro_f1", "workflow_exact_rate",
        "multi_step_recall", "rule_accuracy", "false_rule_capture_rate",
        "semantic_top3_recall", "schema_valid_rate", "fallback_accuracy",
        "error_count",
    ):
        value = metrics.get(name)
        rendered = "n/a" if value is None else (
            f"{value:.4f}" if isinstance(value, float) else str(value)
        )
        lines.append(f"| {name} | {rendered} |")
    lines.extend(["", "## Quality gates", "", "| Gate | Status |", "|---|---|"])
    for name, result in report["quality_gates"].items():
        if name != "all_available_passed":
            lines.append(f"| {name} | {result['status']} |")
    return "\n".join(lines) + "\n"


def write_report(report: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"router-{report['layer']}-{stamp}"
    json_path = os.path.join(output_dir, stem + ".json")
    markdown_path = os.path.join(output_dir, stem + ".md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown(report))
    return json_path, markdown_path


def _rule_predictor(router):
    specs = router._agent_specs()

    def predict(case):
        started = time.perf_counter()
        agent_id, _ = router._rule_match(case.query, case.file_context, specs)
        if _COMPOUND_PATTERN.search(case.query):
            agent_id = None
        latency = (time.perf_counter() - started) * 1000
        return {
            "mode": "single" if agent_id else None,
            "calls": ([{"call_id": "step_1", "agent_id": agent_id,
                        "depends_on": [], "input_refs": []}] if agent_id else []),
            "candidates": ([{"agent_id": agent_id, "score": 1.0}] if agent_id else []),
            "method": "rule" if agent_id else "no_rule",
            "schema_valid": True,
            "latency_ms": latency,
            "error": None,
        }
    return predict


def _semantic_predictor(router, matcher):
    specs = {
        key: value for key, value in router._agent_specs().items()
        if key != FALLBACK_AGENT
    }

    def predict(case):
        started = time.perf_counter()
        candidates = matcher.rank(case.query, specs, top_k=3)
        latency = (time.perf_counter() - started) * 1000
        best = candidates[0]["agent_id"] if candidates else None
        error = "SemanticUnavailable" if not candidates and matcher.last_error else None
        return {
            "mode": "single" if best else None,
            "calls": ([{"call_id": "step_1", "agent_id": best,
                        "depends_on": [], "input_refs": []}] if best else []),
            "candidates": candidates,
            "method": "semantic" if best else "unavailable",
            "schema_valid": True,
            "latency_ms": latency,
            "error": error,
        }
    return predict


def _cascade_predictor(router):
    def predict(case):
        started = time.perf_counter()
        decision = router.route_decision(
            case.query,
            session_context=case.session_context,
            file_context=case.file_context,
        )
        return normalize_decision(decision, (time.perf_counter() - started) * 1000)
    return predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("rules", "semantic", "cascade"), default="rules")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument(
        "--disable-semantic",
        action="store_true",
        help="record semantic routing as unavailable without importing its native runtime",
    )
    args = parser.parse_args(argv)

    runtime = AgentRuntime()
    runtime.discover()
    registered = set(runtime.list_agent_ids(include_unavailable=True)) | {FALLBACK_AGENT}
    cases = load_cases(args.cases, registered)
    matcher = (
        UnavailableSemanticMatcher("native_runtime_disabled")
        if args.disable_semantic
        else SemanticMatcher()
    )
    if args.layer == "rules":
        router = Router(registry=runtime.registry, semantic_matcher=matcher, llm_router=lambda **_: {})
        predictor = _rule_predictor(router)
    elif args.layer == "semantic":
        router = Router(registry=runtime.registry, semantic_matcher=matcher, llm_router=lambda **_: {})
        predictor = _semantic_predictor(router, matcher)
    else:
        llm_override = None if args.live_llm else (lambda **_: {})
        router = Router(registry=runtime.registry, semantic_matcher=matcher, llm_router=llm_override)
        predictor = _cascade_predictor(router)
    report = evaluate_cases(
        cases,
        layer=args.layer + ("-live" if args.live_llm else "-offline"),
        predictor=predictor,
        dataset_hash=dataset_fingerprint(cases),
    )
    paths = write_report(report, args.output_dir)
    print(json.dumps({"json": paths[0], "markdown": paths[1], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
