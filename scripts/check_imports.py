#!/usr/bin/env python
"""Import every module in the package and report the ones that fail.

The unit suite exercises a fraction of the tree. A stale import left behind by
a refactor — a module that moved, a directory that stopped being a package —
stays invisible until someone happens to run that code path. This walks the
whole tree and imports everything, including the eval harnesses the tests never
touch.

    python scripts/check_imports.py

Exits non-zero if any module fails to import, so it works as a CI gate. A
module that only needs an uninstalled optional dependency (`pip install -e
".[docparse]"` and friends) is reported separately and is not a failure.
"""

import importlib
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

# Directories that are data, generated output, or not Python at all.
SKIP_DIRS = {
    "__pycache__", "archive", "external", "node_modules", "web",
    "logs", "reports", "results", "data", "static", "skills",
}
PACKAGES = ("mini_thalamus", "nl2sql_agent")


def module_names():
    names = set()
    for package in PACKAGES:
        for parent, dirs, files in os.walk(os.path.join(SRC_ROOT, package)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if not name.endswith(".py"):
                    continue
                relative = os.path.relpath(os.path.join(parent, name), SRC_ROOT)
                parts = relative[:-3].split(os.sep)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                names.add(".".join(parts))
    return sorted(names)


def main():
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)

    names = module_names()
    failed = []
    needs_extra = []

    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing = getattr(exc, "name", "") or ""
            # A missing third-party package is the documented behaviour of an
            # uninstalled extra, not a broken import.
            if missing and not missing.startswith(PACKAGES):
                needs_extra.append((name, missing))
            else:
                failed.append((name, traceback.format_exc()))
        except Exception:
            failed.append((name, traceback.format_exc()))

    print("modules checked      : %d" % len(names))
    print("missing optional deps: %d" % len(needs_extra))
    for name, missing in needs_extra:
        print("    %-58s needs %s" % (name, missing))

    if failed:
        print("FAILED               : %d" % len(failed))
        for name, tb in failed:
            print("\n=== %s\n%s" % (name, tb))
        return 1

    print("FAILED               : 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
