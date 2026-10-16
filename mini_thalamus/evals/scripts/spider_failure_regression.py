"""Build a deduplicated failure regression set from Spider eval reports."""
import glob
import json
import os


EVALS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EVALS_ROOT, "results")
SPIDER_ROOT = os.path.join(
    EVALS_ROOT,
    "external",
    "spider_data",
    "spider",
)


def collect_failures(results_dir=RESULTS_DIR):
    seen = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "spider-dev-*.json"))):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        for record in report.get("records", []):
            if record.get("passed") is not False:
                continue
            key = (record.get("db_id"), record.get("query"))
            if key not in seen:
                seen[key] = {
                    "db_id": record.get("db_id"),
                    "query": record.get("query"),
                    "question": record.get("question", ""),
                    "predicted_sql": record.get("predicted_sql", ""),
                }
    return list(seen.values())


def main():
    failures = collect_failures()
    os.makedirs(SPIDER_ROOT, exist_ok=True)
    out = os.path.join(SPIDER_ROOT, "dev_failure_regression.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=1)
    print(out, len(failures))


if __name__ == "__main__":
    main()
