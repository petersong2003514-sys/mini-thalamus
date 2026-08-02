"""One-command regression suite.

Runs unit tests, router evaluation (semantic and optional live LLM), and the
capability evaluation, then writes a dated report under evals/reports so every
major update can be compared against previous runs.
"""

import argparse
import json
import os
import subprocess
import sys
import time


REPORTS_DIR = os.path.join("evals", "reports")


def run(cmd, timeout=900):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def last_json(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def latest_previous():
    if not os.path.isdir(REPORTS_DIR):
        return None
    files = sorted(
        name for name in os.listdir(REPORTS_DIR)
        if name.startswith("regression-") and name.endswith(".json")
    )
    if not files:
        return None
    with open(os.path.join(REPORTS_DIR, files[-1]), encoding="utf-8") as handle:
        return json.load(handle)


def diff_value(now, before):
    if now is None or before is None:
        return None
    return round(float(now) - float(before), 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run live LLM router eval")
    args = parser.parse_args(argv)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unit_tests": None,
        "router_semantic": None,
        "router_live": None,
        "capability": None,
    }

    print("== unit tests ==")
    proc = run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"])
    report["unit_tests"] = {
        "ok": proc.returncode == 0,
        "summary": (proc.stdout or proc.stderr).strip().splitlines()[-3:],
    }
    print("unit ok:", proc.returncode == 0)

    print("== router semantic ==")
    proc = run([sys.executable, "-u", "router_eval.py", "--layer", "semantic",
                "--output-dir", os.path.join("evals", "results")])
    payload = last_json(proc.stdout)
    report["router_semantic"] = payload.get("metrics") if payload else None
    print("router semantic done")

    if args.live:
        print("== router live LLM ==")
        proc = run([sys.executable, "-u", "router_eval.py", "--layer", "llm",
                    "--live-llm", "--output-dir", os.path.join("evals", "results")])
        payload = last_json(proc.stdout)
        report["router_live"] = payload.get("metrics") if payload else None
        print("router live done")

    print("== capability eval ==")
    proc = run([sys.executable, "-u", "capability_eval.py",
                "--output-dir", os.path.join("evals", "results")])
    payload = last_json(proc.stdout)
    report["capability"] = payload.get("metrics") if payload else None
    print("capability done")

    previous = latest_previous()
    report["diff"] = {}
    if previous:
        report["diff"]["capability_pass_rate"] = diff_value(
            report["capability"]["pass_rate"] if report["capability"] else None,
            previous.get("capability", {}).get("pass_rate") if previous.get("capability") else None,
        )
        report["diff"]["semantic_top3_recall"] = diff_value(
            report["router_semantic"].get("semantic_top3_recall") if report["router_semantic"] else None,
            previous.get("router_semantic", {}).get("semantic_top3_recall") if previous.get("router_semantic") else None,
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(REPORTS_DIR, f"regression-{stamp}")
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Regression Report",
        "",
        f"- Time: {report['timestamp']}",
        "",
        "## Unit tests",
        "",
        f"- OK: {report['unit_tests']['ok']}",
        "",
        "## Router semantic",
        "",
    ]
    if report["router_semantic"]:
        m = report["router_semantic"]
        lines += [
            f"- single_intent_accuracy: {m.get('single_intent_accuracy')}",
            f"- semantic_top3_recall: {m.get('semantic_top3_recall')}",
            f"- macro_f1: {m.get('macro_f1')}",
            "",
        ]
    if report["router_live"]:
        m = report["router_live"]
        lines += [
            "## Router live LLM",
            "",
            f"- single_intent_accuracy: {m.get('single_intent_accuracy')}",
            f"- multi_step_recall: {m.get('multi_step_recall')}",
            f"- workflow_exact_rate: {m.get('workflow_exact_rate')}",
            f"- schema_valid_rate: {m.get('schema_valid_rate')}",
            "",
        ]
    if report["capability"]:
        m = report["capability"]
        lines += [
            "## Capability",
            "",
            f"- case_count: {m.get('case_count')}",
            f"- pass_rate: {m.get('pass_rate')}",
            f"- avg_latency_ms: {m.get('avg_latency_ms')}",
            f"- error_rate: {m.get('error_rate')}",
            "",
            "| Agent | Cases | Pass rate | Latency |",
            "|---|---:|---:|---:|",
        ]
        for agent, meta in sorted(m.get("agents", {}).items()):
            lines.append(
                f"| {agent} | {meta['case_count']} | {meta['pass_rate']} | "
                f"{meta['avg_latency_ms']} ms |"
            )
        lines.append("")
    if report.get("diff"):
        lines += ["## Diff vs previous", "", f"- capability_pass_rate: {report['diff'].get('capability_pass_rate')}",
                  f"- semantic_top3_recall: {report['diff'].get('semantic_top3_recall')}", ""]
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    ok = (
        report["unit_tests"]["ok"]
        and bool(report["capability"])
        and report["capability"]["pass_rate"] >= 0.8
    )
    print(json.dumps({
        "report": stem + ".json",
        "ok": ok,
        "capability_pass_rate": report["capability"]["pass_rate"] if report["capability"] else None,
    }, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
