"""Download official BFCL v4 subsets into ``evals/external/bfcl_data``.

The BFCL v4 information dataset mirrors the official Berkeley Function-Calling
Leaderboard files.  This helper prefers the Hugging Face hub and keeps the
``possible_answer`` mirror so ``bfcl_eval.py`` can score with the vendored
official checkers.
"""

import argparse
import json
import os
import sys
import urllib.request


EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVALS_ROOT, "external", "bfcl_data")
ANSWER_DIR = os.path.join(DATA_DIR, "possible_answer")
HF_REPO = "tuandunghcmut/BFCL_v4_information"
GITHUB_BASE = (
    "https://raw.githubusercontent.com/HuanzhiMao/gorilla/master/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

OFFLINE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "memory",
]
NO_ANSWER_CATEGORIES = {"irrelevance"}


def _hf_file_maps():
    from huggingface_hub import list_repo_files

    files = list_repo_files(HF_REPO, repo_type="dataset")
    primary = {}
    answers = {}
    for path in files:
        base = os.path.basename(path)
        if not base.startswith("BFCL_v4_") or not base.endswith(".json"):
            continue
        target = answers if "/possible_answer/" in path else primary
        target.setdefault(base, path)
    return primary, answers


def _github_candidates(base):
    return [
        (
            f"{GITHUB_BASE}/{base}",
            os.path.join(DATA_DIR, base),
        ),
        (
            f"{GITHUB_BASE}/possible_answer/{base}",
            os.path.join(ANSWER_DIR, base),
        ),
    ]


def _download(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-thalamus-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with open(destination, "wb") as handle:
        handle.write(payload)
    return len(payload)


def download_categories(categories):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ANSWER_DIR, exist_ok=True)

    downloaded = []
    missing = []
    for category in categories:
        base = f"BFCL_v4_{category}.json"
        destinations = []
        candidates = _github_candidates(base)
        if category in NO_ANSWER_CATEGORIES:
            candidates = candidates[:1]
        for url, destination in candidates:
            try:
                size = _download(url, destination)
                destinations.append(
                    {
                        "category": category,
                        "file": os.path.relpath(destination, EVALS_ROOT),
                        "bytes": size,
                    }
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "category": category,
                            "url": url,
                            "error": str(exc)[:200],
                        },
                        ensure_ascii=False,
                    )
                )
        if len(destinations) < (1 if category in NO_ANSWER_CATEGORIES else 2):
            missing.append(category)
            continue
        downloaded.extend(destinations)
        print(
            json.dumps(
                {"category": category, "files": destinations, "ok": True},
                ensure_ascii=False,
            )
        )

    summary = {"downloaded": downloaded, "missing": missing}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default=",".join(OFFLINE_CATEGORIES),
        help="comma-separated categories to download (default: offline set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="download every BFCL_v4 JSON file found in the HF repo",
    )
    args = parser.parse_args(argv)

    if args.all:
        categories = OFFLINE_CATEGORIES + [
            "web_search",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ]
    else:
        categories = [
            item.strip() for item in args.category.split(",") if item.strip()
        ]
    return download_categories(categories)


if __name__ == "__main__":
    raise SystemExit(main())
