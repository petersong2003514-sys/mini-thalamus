"""Regression gate for self-evolution patches.

Compares the metric before and after applying a patch.  The gate only accepts
the new run when ``after - before >= threshold`` (default 0.0).
"""

import argparse
import json
import subprocess
import sys


def metric_for(data):
    benchmark = data.get("benchmark", "")
    metrics = data.get("metrics", {})
    if benchmark == "BFCL_v4":
        return metrics.get("pass_rate", 0)
    if benchmark == "BEIR":
        return metrics.get("ndcg_at_10", 0)
    if benchmark == "Spider":
        return metrics.get("execution_accuracy", 0)
    return metrics.get("pass_rate", metrics.get("ndcg_at_10", 0))


def load_metrics(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data, metric_for(data)


def guard(before_path, after_path, threshold=0.0):
    before_data, before_metric = load_metrics(before_path)
    after_data, after_metric = load_metrics(after_path)
    delta = after_metric - before_metric
    passed = delta >= threshold
    return {
        "benchmark": after_data.get("benchmark", before_data.get("benchmark", "?")),
        "before_metric": before_metric,
        "after_metric": after_metric,
        "delta": delta,
        "threshold": threshold,
        "passed": bool(passed),
        "before_report": before_path,
        "after_report": after_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--eval-cmd", help="optional command used to regenerate --after")
    args = parser.parse_args(argv)

    if args.eval_cmd:
        result = subprocess.run(
            args.eval_cmd,
            shell=True,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "error": "eval command failed",
                        "stdout": result.stdout[-2000:],
                        "stderr": result.stderr[-2000:],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

    outcome = guard(args.before, args.after, threshold=args.threshold)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    if not outcome["passed"]:
        print(
            json.dumps(
                {
                    "suggested_rollback": [
                        "git diff --stat",
                        "git checkout -- agents/",
                    ]
                },
                ensure_ascii=False,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
