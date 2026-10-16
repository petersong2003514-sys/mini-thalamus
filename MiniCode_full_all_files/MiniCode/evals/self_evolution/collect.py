"""Collect the newest external-benchmark metrics and failure cases."""

import argparse
import json
import os
import re
import time


EVALS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(EVALS_ROOT, "results")
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "reports",
)


def _timestamp(name, path):
    match = re.search(r"(\d{8}-\d{6})", name)
    if match:
        return match.group(1)
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(path)))


def find_reports(results_dir):
    reports = []
    if not os.path.isdir(results_dir):
        return reports
    for name in os.listdir(results_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        reports.append(
            {
                "path": path,
                "name": name,
                "timestamp": _timestamp(name, path),
                "data": data,
            }
        )
    return reports


def newest_by_key(reports, key):
    selected = {}
    for report in reports:
        data = report["data"]
        if key == "bfcl":
            if data.get("benchmark") != "BFCL_v4":
                continue
            group = (data.get("category"), data.get("mode"))
        elif key == "beir":
            if data.get("benchmark") != "BEIR":
                continue
            group = (data.get("dataset"),)
        elif key == "spider":
            if data.get("benchmark") != "Spider":
                continue
            group = (data.get("split"),)
        else:
            continue
        if group not in selected or report["timestamp"] > selected[group]["timestamp"]:
            selected[group] = report
    return selected


def extract_failures(data, benchmark):
    failures = []
    records = data.get("records", []) or []
    if benchmark == "BFCL_v4":
        for record in records:
            if record.get("passed"):
                continue
            failures.append(
                {
                    "id": record.get("id", "?"),
                    "category": data.get("category", "?"),
                    "mode": data.get("mode", "?"),
                    "error_type": record.get("error_type", ""),
                    "answer": record.get("answer", "")[:500],
                    "question": record.get("question", "")[:500],
                }
            )
    elif benchmark == "BEIR":
        for record in records:
            if record.get("ndcg_at_10", 1.0) >= 0.999:
                continue
            failures.append(
                {
                    "id": record.get("qid", "?"),
                    "dataset": data.get("dataset", "?"),
                    "error_type": "retrieval_miss",
                    "answer": record.get("query", "")[:500],
                    "predicted": record.get("predicted", [])[:10],
                    "relevant": record.get("relevant", [])[:10],
                }
            )
    elif benchmark == "Spider":
        for record in records:
            if record.get("passed"):
                continue
            failures.append(
                {
                    "id": f"{record.get('db_id', '?')}:{record.get('question', '?')[:40]}",
                    "dataset": record.get("db_id", "?"),
                    "error_type": record.get("error_type", "execution_error"),
                    "answer": record.get("predicted_sql", "")[:500],
                    "query": record.get("query", "")[:500],
                }
            )
    return failures


def collect(results_dir=DEFAULT_RESULTS, output_dir=DEFAULT_OUTPUT, benchmark=None):
    reports = find_reports(results_dir)
    keys = (
        [benchmark.lower()] if benchmark else ["bfcl", "beir", "spider"]
    )
    groups = {}
    for key in keys:
        groups.update(newest_by_key(reports, key))

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
        "failures": [],
    }
    for group, report in sorted(groups.items(), key=lambda item: item[1]["timestamp"]):
        data = report["data"]
        benchmark_name = data.get("benchmark", "?")
        metrics = dict(data.get("metrics", {}))
        failures = extract_failures(data, benchmark_name)
        key_name = "_".join(str(item) for item in group)
        summary["benchmarks"][key_name] = {
            "report": report["path"],
            "timestamp": report["timestamp"],
            "benchmark": benchmark_name,
            "metrics": metrics,
            "failure_count": len(failures),
        }
        summary["failures"].extend(failures)

    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"collect-{stamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", choices=["bfcl", "beir", "spider"])
    args = parser.parse_args(argv)
    collect(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
