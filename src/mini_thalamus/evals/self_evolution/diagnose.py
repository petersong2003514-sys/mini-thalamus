"""Diagnose whether keyword patches can move the router metrics at all.

The self-evolution loop wants to search over `AGENT_SPEC.keywords`. Before
building that search we need to know where the routing failures actually come
from: if they are agent-selection errors the semantic profile can influence,
keyword search is worth building; if they are mode/DAG-shape errors, it is not.

Runs the offline cascade (stubbed LLM, so no network) over the router eval set
and buckets every failing case by cause.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mini_thalamus.agent_runtime import AgentRuntime
from mini_thalamus.router import FALLBACK_AGENT, Router
from mini_thalamus.router_eval import _cascade_predictor
from mini_thalamus.router_eval_dataset import dataset_fingerprint, load_cases
from mini_thalamus.router_eval_metrics import aggregate_scores, evaluate_quality_gates, score_case
from mini_thalamus.semantic_router import SemanticMatcher


DEFAULT_CASES = os.path.join(PROJECT_ROOT, "evals", "router_cases.jsonl")
DEFAULT_OUTPUT = os.path.join(HERE, "reports")

# Causes a keyword patch could plausibly fix: the correct agent was reachable
# (it was ranked somewhere in the candidate list) but did not win.
FIXABLE_BY_KEYWORDS = {"wrong_rank", "missing_from_topk"}
# Causes keywords cannot fix: the agent was right and the shape was wrong, or
# the router never produced a candidate at all.
NOT_FIXABLE_BY_KEYWORDS = {"fallback", "mode_mismatch", "workflow_shape"}


def _passed(score: dict, expected_mode: str):
    if expected_mode == "single":
        return bool(score.get("single_correct"))
    if expected_mode == "workflow":
        return bool(score.get("workflow_exact"))
    if expected_mode == "fallback":
        return bool(score.get("fallback_correct"))
    return False


def classify(score: dict, expected_mode: str) -> str:
    """Bucket one scored case. Assumes the case did NOT pass."""
    method = score.get("method") or ""
    primary = score.get("predicted_primary_agent")

    if method == "fallback" or not primary or primary == FALLBACK_AGENT:
        return "fallback"
    if not score.get("mode_correct"):
        return "mode_mismatch"
    if expected_mode == "workflow":
        # Right agents, wrong wiring: dependencies / input_refs / ordering.
        return "workflow_shape"
    if score.get("top3_hit"):
        # The correct agent WAS in the candidate list, it just did not win.
        return "wrong_rank"
    return "missing_from_topk"


def diagnose(cases_path: str = DEFAULT_CASES, limit: int = 0) -> dict:
    runtime = AgentRuntime()
    runtime.discover()
    registered = set(runtime.list_agent_ids(include_unavailable=True)) | {FALLBACK_AGENT}

    cases = load_cases(cases_path, registered)
    if limit:
        cases = cases[:limit]

    matcher = SemanticMatcher()
    router = Router(
        registry=runtime.registry,
        semantic_matcher=matcher,
        llm_router=lambda **_: {},
    )
    predictor = _cascade_predictor(router)
    specs = {
        key: value
        for key, value in router._agent_specs().items()
        if key != FALLBACK_AGENT
    }

    buckets: dict[str, list[str]] = {}
    scores = []
    rows = []
    semantic_top1 = 0
    semantic_top3 = 0
    lexical_hit = 0

    for case in cases:
        started = time.perf_counter()
        prediction = predictor(case)
        prediction = dict(prediction)
        prediction["case_id"] = case.case_id
        score = score_case(case, prediction)
        scores.append(score)

        allowed = {score["expected_primary_agent"]} | set(score["acceptable_agents"])
        allowed.discard(None)

        ranked = matcher.rank(case.query, specs, top_k=3)
        ranked_ids = [item["agent_id"] for item in ranked]
        lexical = router._lexical_candidates(case.query, specs)
        lexical_ids = [item["agent_id"] for item in lexical]

        # Leverage probes: could a better keyword set reach the right agent?
        if ranked_ids and ranked_ids[0] in allowed:
            semantic_top1 += 1
        if allowed & set(ranked_ids[:3]):
            semantic_top3 += 1
        if allowed & set(lexical_ids):
            lexical_hit += 1

        passed = _passed(score, case.expected_mode)
        cause = "pass" if passed else classify(score, case.expected_mode)
        buckets.setdefault(cause, []).append(case.case_id)

        rows.append({
            "case_id": case.case_id,
            "expected_mode": case.expected_mode,
            "expected": score["expected_primary_agent"],
            "acceptable": score["acceptable_agents"],
            "predicted": score["predicted_primary_agent"],
            "method": score["method"],
            "cause": cause,
            "semantic_top3": ranked_ids[:3],
            "lexical_top3": lexical_ids[:3],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })

    total = len(cases)
    failures = total - len(buckets.get("pass", []))
    fixable = sum(len(v) for k, v in buckets.items() if k in FIXABLE_BY_KEYWORDS)
    not_fixable = sum(len(v) for k, v in buckets.items() if k in NOT_FIXABLE_BY_KEYWORDS)

    metrics = aggregate_scores(scores)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases_path": cases_path,
        "dataset_fingerprint": dataset_fingerprint(cases),
        "case_count": total,
        "metrics": metrics,
        "quality_gates": evaluate_quality_gates(metrics),
        "buckets": {key: {"count": len(value), "case_ids": value[:30]}
                    for key, value in sorted(buckets.items())},
        "summary": {
            "failures": failures,
            "fixable_by_keywords": fixable,
            "not_fixable_by_keywords": not_fixable,
            "fixable_share": round(fixable / failures, 4) if failures else None,
            "semantic_top1_hit": semantic_top1,
            "semantic_top3_hit": semantic_top3,
            "lexical_hit": lexical_hit,
            "semantic_top1_rate": round(semantic_top1 / total, 4) if total else None,
            "semantic_top3_rate": round(semantic_top3 / total, 4) if total else None,
            "lexical_rate": round(lexical_hit / total, 4) if total else None,
        },
        "rows": rows,
    }


def _print_report(report: dict) -> None:
    summary = report["summary"]
    metrics = report["metrics"]
    print("=" * 68)
    print(f"cases: {report['case_count']}   failures: {summary['failures']}")
    print("-" * 68)
    print("cause buckets")
    for cause, payload in report["buckets"].items():
        bar = "#" * min(40, payload["count"])
        print(f"  {cause:<20} {payload['count']:>4}  {bar}")
    print("-" * 68)
    print("leverage probes (is the right agent reachable at all?)")
    print(f"  semantic top-1 hit   {summary['semantic_top1_rate']}")
    print(f"  semantic top-3 hit   {summary['semantic_top3_rate']}")
    print(f"  lexical  any  hit    {summary['lexical_rate']}")
    print("-" * 68)
    print("cascade metrics")
    for name in (
        "single_intent_accuracy", "macro_f1", "workflow_exact_rate",
        "semantic_top3_recall", "rule_accuracy", "schema_valid_rate",
    ):
        print(f"  {name:<24} {metrics.get(name)}")
    print("-" * 68)
    share = summary["fixable_share"]
    verdict = "GO" if (share or 0) >= 0.30 else "NO-GO"
    print(f"keyword-fixable share: {share}   -> {verdict} (threshold 0.30)")
    print("=" * 68)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="diagnose only the first N cases")
    args = parser.parse_args(argv)

    report = diagnose(args.cases, limit=args.limit)
    _print_report(report)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.output_dir, f"diagnose-{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
