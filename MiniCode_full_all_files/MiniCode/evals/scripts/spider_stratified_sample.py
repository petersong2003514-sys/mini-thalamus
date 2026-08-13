"""Pick one dev case per Spider database for a 20-case stratified smoke."""
import json
import os


ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external",
    "spider_data",
    "spider",
)


def main():
    with open(os.path.join(ROOT, "dev.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    by_db = {}
    for row in rows:
        by_db.setdefault(row["db_id"], row)
    sample = [by_db[db_id] for db_id in sorted(by_db)]
    out = os.path.join(ROOT, "dev_stratified_20.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=1)
    print(out, len(sample))


if __name__ == "__main__":
    main()
