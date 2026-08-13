"""BFCL v4 multi-turn path scoring.

BFCL multi-turn entries carry an expected ``path`` such as
``["GorillaFileSystem.find", "GorillaFileSystem.mv"]``.  This scorer compares
the tool methods emitted by the agent against that ordered path.
"""

import re
import os
import sys


EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_tool_path(answer: str) -> list[str]:
    """Extract ordered ClassName.method paths from a model response."""
    return re.findall(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", answer or "")


def score_multi_turn(entry: dict, answer: str) -> dict:
    """Return a checker-style result for one multi-turn entry."""
    expected = entry.get("path", [])
    actual = parse_tool_path(answer)
    valid = actual == expected
    return {
        "valid": valid,
        "error_type": "" if valid else "multi_turn:path_mismatch",
        "expected": expected,
        "actual": actual,
    }


def score_official_multi_turn(
    entry,
    ground_truth_list,
    model_result_list_decoded,
    category="multi_turn_base",
    model_name="shushu",
):
    """Run the official BFCL executable multi-turn checker."""
    official_root = os.path.join(EVALS_ROOT, "external", "bfcl_official")
    if official_root not in sys.path:
        sys.path.insert(0, official_root)
    from bfcl_official_impl.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    result = multi_turn_checker(
        multi_turn_model_result_list_decoded=model_result_list_decoded,
        multi_turn_ground_truth_list=ground_truth_list,
        test_entry=entry,
        test_category=category,
        model_name=model_name,
    )
    return {
        "valid": bool(result.get("valid", False)),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "details": result.get("details", {}),
    }


__all__ = [
    "parse_tool_path",
    "score_multi_turn",
    "score_official_multi_turn",
]
