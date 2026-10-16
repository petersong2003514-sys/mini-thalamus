"""Apply auditable evolution patches to agent AGENT_SPEC keyword lists."""

import argparse
import ast
import difflib
import json
import os
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "applied",
)


def _byte_position(lines, lineno, col):
    return sum(len(line.encode("utf-8")) for line in lines[: lineno - 1]) + col


def _char_offset(source, byte_offset):
    return len(source.encode("utf-8")[:byte_offset].decode("utf-8"))


def find_list_node(source, field, target_name=None):
    tree = ast.parse(source)
    names = {target_name} if target_name else {"AGENT_SPEC", "AGENT_META"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == field
                and isinstance(value, ast.List)
            ):
                return value
    return None


def read_values(node):
    values = []
    for element in node.elts:
        if isinstance(element, ast.Constant):
            values.append(element.value)
    return values


def replace_list(source, node, values):
    lines = source.splitlines(keepends=True)
    start_byte = _byte_position(lines, node.lineno, node.col_offset)
    end_byte = _byte_position(lines, node.end_lineno, node.end_col_offset)
    start = _char_offset(source, start_byte)
    end = _char_offset(source, end_byte)
    rendered = "[" + ", ".join(json.dumps(value, ensure_ascii=False) for value in values) + "]"
    return source[:start] + rendered + source[end:]


def apply_patch_file(patch, target_root, apply_write=False):
    file_path = os.path.abspath(os.path.join(target_root, patch["file"]))
    root = os.path.abspath(target_root)
    if not file_path.startswith(root + os.sep):
        return {
            "id": patch.get("id", "?"),
            "ok": False,
            "error": "patch path escapes target root",
        }
    if not os.path.exists(file_path):
        return {
            "id": patch.get("id", "?"),
            "ok": False,
            "error": f"missing file: {patch['file']}",
        }
    if patch.get("field") not in ("keywords", "few_shots"):
        return {
            "id": patch.get("id", "?"),
            "ok": False,
            "error": f"unsupported field {patch.get('field')}",
        }

    with open(file_path, encoding="utf-8") as handle:
        source = handle.read()
    node = find_list_node(source, patch["field"])
    if node is None:
        return {
            "id": patch.get("id", "?"),
            "ok": False,
            "error": f"AGENT_SPEC.{patch['field']} list not found",
        }

    existing = read_values(node)
    existing_lookup = {str(value).lower() for value in existing}
    added = [
        value for value in patch.get("values", [])
        if str(value).lower() not in existing_lookup
    ]
    if not added:
        return {
            "id": patch.get("id", "?"),
            "ok": True,
            "changed": False,
            "message": "no new values to add",
        }

    updated = existing + added
    after = replace_list(source, node, updated)
    compile(after, file_path, "exec")
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(True),
            after.splitlines(True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    if apply_write:
        with open(file_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(after)
    return {
        "id": patch.get("id", "?"),
        "ok": True,
        "changed": True,
        "file": patch["file"],
        "field": patch["field"],
        "added": added,
        "diff": diff,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--target-root", default=PROJECT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write changes; default is dry-run",
    )
    args = parser.parse_args(argv)

    with open(args.patches, encoding="utf-8") as handle:
        document = json.load(handle)
    results = []
    for patch in document.get("patches", []):
        result = apply_patch_file(patch, args.target_root, apply_write=args.apply)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    summary = {
        "dry_run": not args.apply,
        "total": len(results),
        "changed": sum(1 for result in results if result.get("changed")),
        "failed": sum(1 for result in results if not result.get("ok")),
        "results": results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    import time

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"apply-{stamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"report": output_path, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
