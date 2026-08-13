"""Direct DeepSeek function-calling probe for BFCL multi-turn."""
import argparse
import json
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EVALS_ROOT = os.path.join(PROJECT_ROOT, "evals")
DATA_DIR = os.path.join(EVALS_ROOT, "external", "bfcl_data")
REPORT_DIR = os.path.join(EVALS_ROOT, "results")
sys.path.insert(0, EVALS_ROOT)

from multi_turn_runner import DeepSeekFCModelStep, run_multi_turn_entry  # noqa: E402


def load_entries(category, limit=0):
    path = os.path.join(DATA_DIR, f"BFCL_v4_{category}.json")
    with open(path, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if limit:
        entries = entries[:limit]
    return entries


def load_ground_truth(category, entry_id):
    path = os.path.join(
        DATA_DIR,
        "possible_answer",
        f"BFCL_v4_{category}.json",
    )
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            answer = json.loads(line)
            if answer.get("id") == entry_id:
                return answer.get("ground_truth", [])
    return []


def deepseek_client():
    from openai import OpenAI

    sys.path.insert(0, PROJECT_ROOT)
    from config import get_llm_config

    cfg = get_llm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


def run_entry(entry, client, model):
    ground_truth = load_ground_truth(entry["id"].rsplit("_", 1)[0], entry["id"])
    model_step = DeepSeekFCModelStep(client, model)
    checker, model_results, latency = run_multi_turn_entry(
        entry,
        model_step,
        ground_truth,
    )
    return checker, model_results, latency


def write_report(category, records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    metrics = {
        "case_count": total,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_latency_ms": round(
            sum(record["latency_s"] for record in records) / total * 1000,
            1,
        )
        if total
        else 0,
    }
    path = os.path.join(
        output_dir,
        f"bfcl-direct-fc-{category}-{stamp}.json",
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "BFCL_v4",
                "category": category,
                "mode": "deepseek-fc-direct",
                "metrics": metrics,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return path, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    args = parser.parse_args(argv)

    client, model = deepseek_client()
    entries = load_entries(args.category, limit=args.limit)
    records = []
    for entry in entries:
        checker, model_results, latency = run_entry(entry, client, model)
        record = {
            "id": entry["id"],
            "passed": bool(checker["valid"]),
            "latency_s": round(latency, 2),
            "error_type": checker.get("error_type", ""),
            "error_message": checker.get("error_message", ""),
            "model_results": model_results,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "id": entry["id"],
                    "passed": record["passed"],
                    "error_type": record["error_type"],
                },
                ensure_ascii=False,
            )
        )
    path, metrics = write_report(args.category, records, args.output_dir)
    print(json.dumps({"report": path, "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
