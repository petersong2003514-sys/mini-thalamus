"""Layered, non-executing Router evaluation CLI."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
import uuid

from mini_thalamus.agent_runtime import AgentRuntime
from mini_thalamus.config import get_llm_config
from mini_thalamus.router import _is_compound_intent, FALLBACK_AGENT, Router
from mini_thalamus.router_eval_dataset import dataset_fingerprint, load_cases
from mini_thalamus.router_eval_metrics import aggregate_scores, evaluate_quality_gates, score_case
from mini_thalamus.route_schema import RouteDecision
from mini_thalamus.semantic_router import SemanticMatcher


DEFAULT_CASES = os.path.join(os.path.dirname(__file__), "evals", "router_cases.jsonl")
DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "evals", "results")


def _environment_metadata() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class UnavailableSemanticMatcher:
    """Explicit no-op matcher for environments with an unsafe native runtime."""

    def __init__(self, reason: str):
        self.last_error = reason

    def rank(self, query: str, agent_specs: dict, top_k: int = 3) -> list[dict]:
        return []


def select_llm_cases(cases) -> list:
    hard = [case for case in cases if case.difficulty == "hard"]
    preferred = [case for case in hard if case.expected_stage == "llm"]
    remaining = [case for case in hard if case.expected_stage != "llm"]
    return (preferred + remaining)[:50]


def expand_layers(layer: str) -> list[str]:
    return ["rules", "semantic", "llm", "cascade"] if layer == "all" else [layer]


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
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "dataset_fingerprint": dataset_hash,
        "environment": _environment_metadata(),
        "metrics": metrics,
        "quality_gates": evaluate_quality_gates(metrics),
        "predictions": predictions,
        "scores": [
            {key: value for key, value in score.items() if not key.startswith("_")}
            for score in scores
        ],
    }


def skipped_report(*, layer: str, dataset_hash: str, case_count: int, reason: str) -> dict:
    metrics = aggregate_scores([])
    metrics["case_count"] = case_count
    return {
        "schema_version": 1,
        "status": "skipped",
        "skip_reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        "dataset_fingerprint": dataset_hash,
        "environment": _environment_metadata(),
        "metrics": metrics,
        "quality_gates": evaluate_quality_gates(metrics),
        "predictions": [],
        "scores": [],
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
        if _is_compound_intent(case.query):
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


def _cascade_predictor(router, strict_cascade: bool = False):
    def predict(case):
        started = time.perf_counter()
        decision = router.route_decision(
            case.query,
            session_context=case.session_context,
            file_context=case.file_context,
            strict_cascade=strict_cascade,
        )
        return normalize_decision(decision, (time.perf_counter() - started) * 1000)
    return predict


def _llm_predictor(router):
    specs = router._agent_specs()

    def predict(case):
        started = time.perf_counter()
        candidates = router._lexical_candidates(case.query, specs)
        payload = router._call_llm_router(
            query=case.query,
            context=case.session_context,
            file_context=case.file_context,
            candidates=candidates,
            specs=specs,
        )
        payload = dict(payload)
        payload["method"] = "llm"
        payload["trace_id"] = uuid.uuid4().hex
        if not payload.get("candidates"):
            payload["candidates"] = candidates
        decision = RouteDecision.from_dict(payload, agent_specs=specs)
        return normalize_decision(decision, (time.perf_counter() - started) * 1000)
    return predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("rules", "semantic", "llm", "cascade", "all"), default="rules")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument(
        "--disable-semantic",
        action="store_true",
        help="record semantic routing as unavailable without importing its native runtime",
    )
    parser.add_argument(
        "--strict-cascade",
        action="store_true",
        help="force cascade evaluation through schema-bound LLM retry flow",
    )
    args = parser.parse_args(argv)

    if args.layer == "all":
        common = ["--cases", args.cases, "--output-dir", args.output_dir]
        if args.live_llm:
            common.append("--live-llm")
        if args.disable_semantic:
            common.append("--disable-semantic")
        if args.strict_cascade:
            common.append("--strict-cascade")
        return max(main(["--layer", layer, *common]) for layer in expand_layers("all"))

    runtime = AgentRuntime()
    runtime.discover()
    registered = set(runtime.list_agent_ids(include_unavailable=True)) | {FALLBACK_AGENT}
    cases = load_cases(args.cases, registered)
    llm_key_available = bool(get_llm_config().get("api_key"))
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
    elif args.layer == "cascade":
        llm_override = None if args.live_llm else (lambda **_: {})
        router = Router(registry=runtime.registry, semantic_matcher=matcher, llm_router=llm_override)
        predictor = _cascade_predictor(router, strict_cascade=args.strict_cascade)
    else:
        llm_override = None if args.live_llm and llm_key_available else (lambda **_: {})
        router = Router(
            registry=runtime.registry,
            semantic_matcher=matcher,
            llm_router=llm_override,
        )
        predictor = _llm_predictor(router)
    layer_name = args.layer + ("-live" if args.live_llm else "-offline")
    fingerprint = dataset_fingerprint(cases)
    live_unavailable = args.live_llm and not llm_key_available
    llm_disabled = args.layer == "llm" and not args.live_llm
    if args.layer == "semantic" and args.disable_semantic:
        eligible = [case for case in cases if case.expected_stage == "semantic"]
        report = skipped_report(
            layer=layer_name,
            dataset_hash=fingerprint,
            case_count=len(eligible),
            reason="native_runtime_disabled",
        )
    elif live_unavailable or llm_disabled:
        eligible = select_llm_cases(cases) if args.layer == "llm" else cases
        report = skipped_report(
            layer=layer_name,
            dataset_hash=fingerprint,
            case_count=len(eligible),
            reason=("missing_llm_credential" if live_unavailable else "live_llm_not_enabled"),
        )
    else:
        eligible = (
            [case for case in cases if case.expected_stage == "semantic"]
            if args.layer == "semantic"
            else (select_llm_cases(cases) if args.layer == "llm" else cases)
        )
        report = evaluate_cases(
            eligible,
            layer=layer_name,
            predictor=predictor,
            dataset_hash=fingerprint,
        )
    report["reproducibility"] = {
        "git_commit": _git_commit(),
        "registered_agents": sorted(registered),
        "semantic_model": getattr(matcher, "model_name", None),
        "router_thresholds": {
            "semantic": router.semantic_threshold,
            "semantic_margin": router.semantic_margin,
            "semantic_fallback": router.semantic_fallback_threshold,
            "top_k": router.top_k,
        },
        "live_llm_requested": bool(args.live_llm),
        "llm_credential_available": llm_key_available,
        "semantic_disabled": bool(args.disable_semantic),
        "strict_cascade": bool(args.strict_cascade),
        "eligible_case_count": len(eligible),
        "token_usage": None,
        "estimated_cost": None,
    }
    report["component_status"] = {
        "rules": "available",
        "semantic": "unavailable" if args.disable_semantic else "available",
        "llm": (
            "unavailable" if live_unavailable
            else ("available" if args.live_llm else "not_run")
        ),
    }
    paths = write_report(report, args.output_dir)
    print(json.dumps({"json": paths[0], "markdown": paths[1], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
