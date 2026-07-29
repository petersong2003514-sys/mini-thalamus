"""Validation and containment for caller-supplied session identifiers."""

import os
import re


_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class InvalidSessionId(ValueError):
    """Raised when a session ID is unsafe or outside the public contract."""


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
        session_id
    ):
        raise InvalidSessionId(
            "session_id must be 1-128 safe ASCII characters"
        )
    return session_id


def contained_session_dir(root: str, session_id: str) -> str:
    sid = validate_session_id(session_id)
    sessions_root = os.path.abspath(os.path.join(root, "sessions"))
    target = os.path.abspath(os.path.join(sessions_root, sid))
    if os.path.commonpath((sessions_root, target)) != sessions_root:
        raise InvalidSessionId("session path escapes the session root")
    return target


__all__ = ["InvalidSessionId", "contained_session_dir", "validate_session_id"]
