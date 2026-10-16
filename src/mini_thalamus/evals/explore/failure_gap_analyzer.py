"""Group benchmark failures by category and error_type."""
import glob
import json
import os


def summarize_failures(results_dir):
    summary = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        records = report.get("records")
        if not isinstance(records, list):
            continue
        category = report.get("category", report.get("benchmark", "unknown"))
        for record in records:
            if record.get("passed") is not False:
                continue
            key = (category, record.get("error_type", "unknown"))
            summary[key] = summary.get(key, 0) + 1
    return [
        {"category": category, "error_type": error_type, "count": count}
        for (category, error_type), count in sorted(
            summary.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]
