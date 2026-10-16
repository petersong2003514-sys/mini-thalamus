"""Spider dev-set NL2SQL evaluation for the Mini Thalamus stack.

The runner downloads the official Spider zip, builds the existing NL2SQL peer
agent against each SQLite database, and reports Execution Accuracy by comparing
the predicted result set with the gold query result set.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL2SQL_ROOT = os.path.dirname(PROJECT_ROOT)
EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVALS_ROOT, "external", "spider_data")
REPORT_DIR = os.path.join(EVALS_ROOT, "results")
SPIDER_URL = (
    "https://drive.google.com/uc?export=download&id="
    "1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"
)


def download_spider(data_dir=DATA_DIR, zip_url=None):
    os.makedirs(data_dir, exist_ok=True)
    if zip_url and zip_url.startswith(("http://", "https://")):
        zip_path = os.path.join(data_dir, "spider.zip")
        print(json.dumps({"downloading": zip_url}, ensure_ascii=False))
        request = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "mini-thalamus-eval/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
            with open(zip_path, "wb") as handle:
                handle.write(payload)
        except Exception as exc:
            raise RuntimeError(
                f"Spider download failed from {zip_url}: {exc}"
            ) from exc
    elif zip_url:
        zip_path = zip_url
        if not os.path.exists(zip_path):
            raise RuntimeError(f"local Spider zip not found: {zip_path}")
    else:
        zip_path = os.path.join(data_dir, "spider.zip")
        if not os.path.exists(zip_path):
            print(json.dumps({"downloading": SPIDER_URL}, ensure_ascii=False))
            request = urllib.request.Request(
                SPIDER_URL,
                headers={"User-Agent": "mini-thalamus-eval/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = response.read()
                with open(zip_path, "wb") as handle:
                    handle.write(payload)
            except Exception as exc:
                raise RuntimeError(
                    "official Spider Google Drive link is unavailable; "
                    f"provide a local zip under {data_dir} or an alternative "
                    f"mirror URL (last error: {exc})"
                ) from exc
    target = os.path.abspath(data_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = os.path.abspath(os.path.join(data_dir, member.filename))
            if not resolved.startswith(target + os.sep):
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        archive.extractall(data_dir)
    return data_dir


def find_spider_root(data_dir=DATA_DIR):
    for candidate in (
        os.path.join(data_dir, "spider"),
        data_dir,
    ):
        if os.path.exists(os.path.join(candidate, "dev.json")):
            return candidate
    raise RuntimeError(
        "Spider data not found; run `python evals/spider_eval.py --download` "
        "or provide a local zip via `python evals/spider_eval.py "
        "--zip-url <local.zip> --check-data`"
    )


def load_dev(data_dir=DATA_DIR, limit=0, offset=0, db_filter=None):
    root = find_spider_root(data_dir)
    with open(os.path.join(root, "dev.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    if db_filter:
        rows = [row for row in rows if row.get("db_id") == db_filter]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return root, rows


def fetch_rows(path, sql):
    uri = f"file:{urllib.parse.quote(path)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=15)
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [item[0] for item in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, [_normalize_row(row) for row in rows]
    finally:
        connection.close()


def _normalize_row(row):
    return tuple(value for value in row)


def execution_match(expected_rows, actual_rows):
    return set(expected_rows) == set(actual_rows)


def extract_sql_text(answer):
    if not answer:
        return ""
    match = re.search(r"SQL:\s*`([^`]+)`", answer)
    if match:
        return match.group(1).strip()
    match = re.search(r"```sql\s*(.*?)```", answer, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer.strip()


def build_agent(db_path, schema_strategy):
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, NL2SQL_ROOT)
    from nl2sql_agent.agent_factory import build_agent, load_dotenv
    from query_disambiguator import SmartDisambiguator

    load_dotenv()
    agent, db, schema_provider = build_agent(
        schema_strategy=schema_strategy,
        db_path=db_path,
    )
    disambiguator = SmartDisambiguator(db)
    return agent, disambiguator


def run_case(row, root, schema_strategy, disambiguate=True):
    db_id = row["db_id"]
    db_path = os.path.join(root, "database", db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return {
            "question": row["question"],
            "query": row["query"],
            "predicted_sql": "",
            "passed": False,
            "error_type": "missing_database",
            "error": f"missing database: {db_path}",
        }

    agent, disambiguator = build_agent(db_path, schema_strategy)
    question = row["question"]
    if disambiguate:
        disambiguation = disambiguator.disambiguate(question)
        if disambiguation.get("action") == "enhance":
            question = disambiguation.get("enhanced_query", question)

    started = time.perf_counter()
    try:
        result = agent.run(question, mode="auto")
        success = bool(getattr(result, "success", False))
        predicted_sql = getattr(result, "final_sql", "") or ""
    except Exception as exc:
        result = None
        success = False
        predicted_sql = ""
        error = str(exc)

    latency = time.perf_counter() - started
    error_type = ""
    error = ""
    if success and predicted_sql:
        try:
            gold_columns, gold_rows = fetch_rows(db_path, row["query"])
            pred_columns, pred_rows = fetch_rows(db_path, predicted_sql)
            passed = execution_match(gold_rows, pred_rows)
        except Exception as exc:
            passed = False
            error_type = "execution_error"
            error = str(exc)[:300]
    else:
        passed = False
        error_type = "sql_generation_failed"
        error = (
            str(error or getattr(result, "error", ""))[:300]
            or "agent did not produce a final SQL"
        )

    return {
        "db_id": db_id,
        "question": row["question"],
        "query": row["query"],
        "predicted_sql": predicted_sql[:1000],
        "passed": bool(passed),
        "latency_s": round(latency, 2),
        "error_type": error_type,
        "error": error,
    }


def aggregate(records):
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    latency = sum(record.get("latency_s", 0) for record in records)
    return {
        "case_count": total,
        "execution_accuracy": round(passed / total, 4) if total else 0,
        "passed": passed,
        "avg_latency_ms": round(latency / total * 1000, 1) if total else 0,
    }


def write_report(metrics, records, output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(output_dir, f"spider-dev-{stamp}")
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "Spider",
                "split": "dev",
                "metrics": metrics,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    lines = [
        "# Spider Dev NL2SQL Evaluation",
        "",
        f"- Cases: {metrics['case_count']}",
        f"- Execution accuracy: {metrics['execution_accuracy']}",
        f"- Passed: {metrics['passed']}",
        f"- Avg latency: {metrics['avg_latency_ms']} ms",
        "",
        "| db_id | passed | latency_s | error_type |",
        "|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['db_id']} | {record['passed']} | "
            f"{record['latency_s']} | {record.get('error_type', '')} |"
        )
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--zip-url",
        help="local path or URL to a Spider zip (defaults to SPIDER_URL)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--db-id")
    parser.add_argument("--rows-file")
    parser.add_argument("--schema-strategy", default="full")
    parser.add_argument("--skip-disambiguate", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args(argv)

    if args.download:
        download_spider(args.data_dir, zip_url=args.zip_url)
    root, rows = load_dev(
        data_dir=args.data_dir,
        limit=args.limit,
        offset=args.offset,
        db_filter=args.db_id,
    )
    if args.rows_file:
        with open(args.rows_file, encoding="utf-8") as handle:
            rows = json.load(handle)
        if args.offset:
            rows = rows[args.offset :]
        if args.limit:
            rows = rows[: args.limit]
    if args.check_data:
        databases = [
            name for name in os.listdir(os.path.join(root, "database"))
            if os.path.isdir(os.path.join(root, "database", name))
        ]
        print(
            json.dumps(
                {
                    "root": root,
                    "cases": len(rows),
                    "databases": len(databases),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    records = []
    for index, row in enumerate(rows, start=1):
        record = run_case(
            row,
            root,
            args.schema_strategy,
            disambiguate=not args.skip_disambiguate,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "index": index,
                    "db_id": row.get("db_id"),
                    "passed": record["passed"],
                    "error_type": record.get("error_type", ""),
                },
                ensure_ascii=False,
            )
        )
    metrics = aggregate(records)
    stem = write_report(metrics, records, args.output_dir)
    print(
        json.dumps(
            {"report": stem + ".json", "metrics": metrics},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
