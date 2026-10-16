"""Run sampled or full external-benchmark evidence with a single log."""
import argparse
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PYTHON = sys.executable
MULTI_TURN_CATEGORIES = (
    "multi_turn_base,multi_turn_long_context,multi_turn_miss_func,"
    "multi_turn_miss_param"
)


def load_env():
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_file):
        return
    with open(env_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "DEEPSEEK_API_KEY":
                os.environ.setdefault("NL2SQL_API_KEY", value)
    os.environ.setdefault("NL2SQL_MODEL", "deepseek-chat")
    os.environ.setdefault("NL2SQL_BASE_URL", "https://api.deepseek.com/v1")


def run(cmd, log_path):
    print(json.dumps({"run": cmd}, ensure_ascii=False))
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} "
            + " ".join(cmd)
            + "\n"
        )
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        handle.write(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}")


def build_full_commands(available_categories):
    cmds = []
    for category in available_categories:
        if category.startswith("multi_turn_"):
            cmds.append(
                [
                    PYTHON,
                    "evals/bfcl_eval.py",
                    "--category",
                    category,
                    "--mode",
                    "bfcl-fc",
                ]
            )
        elif category in ("memory", "web_search"):
            cmds.append(
                [
                    PYTHON,
                    "evals/bfcl_eval.py",
                    "--category",
                    category,
                    "--mode",
                    "mini-thalamus",
                ]
            )
        else:
            cmds.append(
                [
                    PYTHON,
                    "evals/bfcl_eval.py",
                    "--category",
                    category,
                    "--mode",
                    "deepseek-fc",
                ]
            )
    cmds.append(
        [PYTHON, "evals/beir_eval.py", "--dataset", "nfcorpus,scifact", "--bm25"]
    )
    cmds.append([PYTHON, "evals/spider_eval.py", "--schema-strategy", "full"])
    return cmds


def list_available_categories():
    proc = subprocess.run(
        [PYTHON, "evals/bfcl_eval.py", "--list-categories"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(proc.stdout)
    return payload["available"]


def commands(stage, bfcl_limit, spider_limit):
    if stage == "full":
        return build_full_commands(list_available_categories())

    cmds = []
    multi_turn_args = [
        PYTHON,
        "evals/bfcl_eval.py",
        "--category",
        MULTI_TURN_CATEGORIES,
        "--mode",
        "mini-thalamus",
    ]
    if stage == "sample":
        multi_turn_args += ["--limit", str(bfcl_limit)]
    cmds.append(multi_turn_args)

    if stage == "sample":
        cmds.append(
            [
                PYTHON,
                "evals/scripts/spider_stratified_sample.py",
                "--count",
                str(spider_limit),
            ]
        )
        cmds.append(
            [
                PYTHON,
                "evals/spider_eval.py",
                "--rows-file",
                os.path.join(
                    "evals",
                    "external",
                    "spider_data",
                    "spider",
                    f"dev_stratified_{spider_limit}.json",
                ),
                "--schema-strategy",
                "full",
            ]
        )
    else:
        cmds.append([PYTHON, "evals/spider_eval.py", "--schema-strategy", "full"])
        cmds.append([PYTHON, "evals/beir_eval.py", "--dataset", "nfcorpus", "--bm25"])
        cmds.append([PYTHON, "evals/beir_eval.py", "--dataset", "scifact", "--bm25"])
    return cmds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sample", "full"], default="sample")
    parser.add_argument("--bfcl-limit", type=int, default=3)
    parser.add_argument("--spider-limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log")
    args = parser.parse_args(argv)

    load_env()
    log_path = args.log or os.path.join(
        PROJECT_ROOT,
        "logs",
        f"full_evidence_{time.strftime('%Y%m%d-%H%M%S')}.log",
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmds = commands(args.stage, args.bfcl_limit, args.spider_limit)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "commands": cmds}, ensure_ascii=False, indent=2))
        return 0
    for cmd in cmds:
        run(cmd, log_path)
    print(json.dumps({"log": log_path, "status": "done"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
