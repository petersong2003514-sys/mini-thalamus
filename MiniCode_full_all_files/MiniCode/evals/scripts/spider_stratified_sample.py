"""Pick a stratified dev sample across Spider databases."""
import argparse
import json
import os


ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external",
    "spider_data",
    "spider",
)


def pick_sample(rows, count):
    by_db = {}
    for row in rows:
        by_db.setdefault(row["db_id"], []).append(row)
    sample = []
    indexes = {db_id: 0 for db_id in sorted(by_db)}
    while len(sample) < count:
        for db_id in sorted(by_db):
            if len(sample) >= count:
                break
            bucket = by_db[db_id]
            if indexes[db_id] < len(bucket):
                sample.append(bucket[indexes[db_id]])
                indexes[db_id] += 1
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    with open(os.path.join(ROOT, "dev.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    sample = pick_sample(rows, args.count)
    out = os.path.join(ROOT, f"dev_stratified_{args.count}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=1)
    print(out, len(sample))


if __name__ == "__main__":
    main()
