"""Vendor the official BFCL multi-turn evaluator subset from the bfcl-eval wheel."""
import argparse
import glob
import os
import zipfile


TARGET_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external",
    "bfcl_official",
)
PACKAGE_ROOT = "bfcl_official_impl"

MEMBERS = [
    "bfcl_eval/__init__.py",
    "bfcl_eval/constants/__init__.py",
    "bfcl_eval/constants/executable_backend_config.py",
    "bfcl_eval/eval_checker/__init__.py",
    "bfcl_eval/eval_checker/multi_turn_eval/__init__.py",
    "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py",
    "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py",
    "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/__init__.py",
]
for name in [
    "gorilla_file_system",
    "long_context",
    "math_api",
    "message_api",
    "posting_api",
    "ticket_api",
    "trading_bot",
    "travel_booking",
    "vehicle_control",
]:
    MEMBERS.append(
        f"bfcl_eval/eval_checker/multi_turn_eval/func_source_code/{name}.py"
    )


def find_wheel(wheel):
    if wheel and os.path.exists(wheel):
        return wheel
    for pattern in [
        os.path.join(os.environ.get("TEMP", ""), "bfcl_pkg", "*.whl"),
        os.path.expanduser("~/bfcl_pkg/*.whl"),
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise SystemExit(
        "bfcl-eval wheel not found; run: "
        "python -m pip download bfcl-eval --no-deps -d <dir>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel")
    args = parser.parse_args(argv)
    wheel = find_wheel(args.wheel)
    with zipfile.ZipFile(wheel) as archive:
        for member in MEMBERS:
            try:
                archive.extract(member, TARGET_ROOT)
            except KeyError as exc:
                raise SystemExit(f"missing member in wheel: {member}") from exc
    source = os.path.join(TARGET_ROOT, "bfcl_eval")
    target = os.path.join(TARGET_ROOT, PACKAGE_ROOT)
    if os.path.isdir(source) and not os.path.exists(target):
        os.rename(source, target)
    for dirpath, _, filenames in os.walk(target):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
            updated = content.replace("bfcl_eval.", "bfcl_official_impl.")
            if updated != content:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(updated)
    print(f"vendored {len(MEMBERS)} files to {target}")


if __name__ == "__main__":
    main()
