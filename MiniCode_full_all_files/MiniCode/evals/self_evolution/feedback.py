"""Turn collected external-benchmark failures into auditable evolution patches."""

import argparse
import json
import os
import re
import time


EVALS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "patches",
)

STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "with", "and",
    "what", "who", "where", "when", "which", "is", "are", "was", "were",
    "please", "can", "could", "how", "do", "does", "did", "this", "that",
    "查", "一下", "请", "帮我", "什么", "哪些", "哪个", "多少", "如何",
    "收到", "关于", "可以", "尝试", "更", "具体", "描述", "需求", "例如",
    "查询", "数据库", "分析", "文件", "生成", "报告", "用户",
}


def extract_keywords(text, limit=4):
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*|[\u4e00-\u9fff]{2,8}", text)
    seen = set()
    keywords = []
    for token in tokens:
        lowered = token.lower()
        if lowered in STOPWORDS or len(token) < 2:
            continue
        if lowered not in seen:
            seen.add(lowered)
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def target_for_failure(failure):
    benchmark = failure.get("category") or failure.get("dataset") or ""
    if failure.get("error_type") == "retrieval_miss":
        return "agents/rag_agent.py", "keywords"
    if "web_search" in benchmark or "memory" in benchmark:
        return "agents/websearch_agent.py", "keywords"
    if failure.get("dataset") and failure.get("query"):
        return "agents/nl2sql_agent.py", "keywords"
    return "agents/api_caller_agent.py", "keywords"


def generate_patches(collected, output_dir=DEFAULT_OUTPUT):
    failures = collected.get("failures", [])
    grouped = {}
    for failure in failures:
        file_path, field = target_for_failure(failure)
        values = extract_keywords(
            failure.get("question")
            or failure.get("answer")
            or failure.get("query")
            or failure.get("id", "")
        )
        key = (file_path, field)
        entry = grouped.setdefault(
            key,
            {
                "file": file_path,
                "field": field,
                "values": [],
                "sources": [],
                "error_types": {},
            },
        )
        error_type = failure.get("error_type", "unknown")
        entry["error_types"][error_type] = entry["error_types"].get(error_type, 0) + 1
        entry["sources"].append(failure.get("id", "?"))
        for value in values:
            if value not in entry["values"]:
                entry["values"].append(value)

    patches = []
    for index, ((file_path, field), entry) in enumerate(sorted(grouped.items()), start=1):
        patches.append(
            {
                "id": f"patch-{time.strftime('%Y%m%d-%H%M%S')}-{index}",
                "file": file_path,
                "field": field,
                "values": entry["values"][:8],
                "reason": (
                    f"{len(entry['sources'])} failures; "
                    + ", ".join(
                        f"{name}={count}"
                        for name, count in sorted(entry["error_types"].items())
                    )
                ),
                "sources": entry["sources"][:20],
            }
        )

    document = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patch_count": len(patches),
        "patches": patches,
    }
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"feedback-{stamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with open(args.collected, encoding="utf-8") as handle:
        collected = json.load(handle)
    generate_patches(collected, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
