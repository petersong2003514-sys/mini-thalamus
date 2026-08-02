"""Lightweight capability evaluation for every Agent.

Each case is a labeled task with a deterministic scoring rule. Metrics include
task pass rate, exact/substring match, structure checks, source hit rate,
latency, and error rate.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from agent_runtime import AgentRuntime


DEFAULT_CASES = os.path.join("evals", "capability_cases.jsonl")
DEFAULT_OUTPUT = os.path.join("evals", "results")


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _resolve_file(path):
    prefix = "MiniCode_full_all_files/MiniCode/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _score(case, result):
    answer = result.get("answer", "") or ""
    data = result.get("data", {}) or {}
    status = result.get("status", "")
    metric = case.get("metric", "status_success")
    expected = case.get("expected", []) or []
    min_chars = int(case.get("min_chars", 0))

    if len(answer) < min_chars:
        return False, f"answer too short ({len(answer)} < {min_chars})"
    if status == "error":
        return False, answer[:200]

    searchable = answer + "\n" + json.dumps(data, ensure_ascii=False)
    if metric == "contains_any":
        return any(exp in searchable for exp in expected), f"expected any of {expected}"
    if metric == "all_contains":
        return all(exp in searchable for exp in expected), f"expected all of {expected}"
    if metric == "structure":
        headings = [
            line for line in answer.splitlines()
            if line.strip().startswith("#")
        ]
        return (
            len(headings) >= expected[0] and "##" in answer,
            f"headings={len(headings)}",
        )
    if metric == "sources_min":
        sources = data.get("sources") or []
        return len(sources) >= expected[0], f"sources={len(sources)}"
    if metric == "http_ok":
        return (
            data.get("status_code") == 200 or "HTTP 200" in answer
        ), "http status"
    if metric == "status_success":
        return status == "success", f"status={status}"
    return status == "success", f"status={status}"


def run_case(runtime, case):
    agent = runtime.get_agent(case["agent"])
    context = {"session_id": "eval-" + case["id"], "files": []}
    file_rel = case.get("file")
    if file_rel:
        path = _resolve_file(file_rel)
        with open(path, "rb") as handle:
            content = handle.read()
        context["files"] = [{
            "name": os.path.basename(path),
            "content": content,
            "size": len(content),
        }]

    try:
        from memory_service import get_memory_service
        get_memory_service().resolve_session("eval", context["session_id"])
    except Exception:
        pass

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.run, case["query"], context)
        try:
            result = future.result(timeout=float(case.get("timeout", 120)))
        except FutureTimeout:
            return {
                "status": "timeout",
                "answer": "",
                "data": {},
            }, (time.perf_counter() - started) * 1000, "timeout"
        except Exception as exc:
            return {
                "status": "error",
                "answer": str(exc),
                "data": {},
            }, (time.perf_counter() - started) * 1000, type(exc).__name__
    latency_ms = (time.perf_counter() - started) * 1000
    return result, latency_ms, ""


def aggregate(records):
    by_agent = {}
    overall_pass = 0
    total = len(records)
    total_latency = 0.0
    errors = 0
    for record in records:
        agent = record["agent"]
        bucket = by_agent.setdefault(agent, {
            "case_count": 0,
            "pass_count": 0,
            "latency_sum": 0.0,
            "errors": 0,
        })
        bucket["case_count"] += 1
        bucket["latency_sum"] += record["latency_ms"]
        if record["passed"]:
            overall_pass += 1
            bucket["pass_count"] += 1
        if not record["passed"]:
            errors += 1
            bucket["errors"] += 1
        total_latency += record["latency_ms"]

    agents = {}
    for agent, bucket in by_agent.items():
        agents[agent] = {
            "case_count": bucket["case_count"],
            "pass_rate": round(
                bucket["pass_count"] / bucket["case_count"], 4
            ),
            "avg_latency_ms": round(
                bucket["latency_sum"] / bucket["case_count"], 1
            ),
            "errors": bucket["errors"],
        }
    return {
        "case_count": total,
        "pass_rate": round(overall_pass / total, 4) if total else 0,
        "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        "error_rate": round(errors / total, 4) if total else 0,
        "agents": agents,
    }


def write_report(records, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(output_dir, f"capability-{stamp}")
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump({
            "metrics": metrics,
            "records": records,
        }, handle, ensure_ascii=False, indent=2)
    lines = [
        "# Capability Evaluation",
        "",
        f"- Cases: {metrics['case_count']}",
        f"- Pass rate: {metrics['pass_rate']}",
        f"- Avg latency: {metrics['avg_latency_ms']} ms",
        f"- Error rate: {metrics['error_rate']}",
        "",
        "## By agent",
        "",
        "| Agent | Cases | Pass rate | Avg latency (ms) | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent, meta in sorted(metrics["agents"].items()):
        lines.append(
            f"| {agent} | {meta['case_count']} | {meta['pass_rate']} | "
            f"{meta['avg_latency_ms']} | {meta['errors']} |"
        )
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return stem + ".json", stem + ".md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if args.skip_network:
        cases = [case for case in cases if not case.get("network")]

    runtime = AgentRuntime()
    runtime.discover()
    records = []
    for case in cases:
        result, latency_ms, runner_error = run_case(runtime, case)
        passed, detail = _score(case, result)
        if runner_error:
            passed = False
        records.append({
            "id": case["id"],
            "agent": case["agent"],
            "query": case["query"],
            "status": result.get("status"),
            "passed": passed,
            "latency_ms": round(latency_ms, 1),
            "detail": detail,
            "answer": (result.get("answer") or "")[:300],
        })
        print(f"{case['id']} [{case['agent']}] pass={passed} "
              f"latency={round(latency_ms, 1)}ms detail={detail}")

    metrics = aggregate(records)
    json_path, md_path = write_report(records, metrics, args.output_dir)
    print(json.dumps({
        "json": json_path,
        "markdown": md_path,
        "metrics": metrics,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
