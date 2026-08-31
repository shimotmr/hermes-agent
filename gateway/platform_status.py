"""Pure live views over persisted Gateway platform-writer history."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def project_live_platforms(
    platforms: object,
    *,
    current_pid: object,
    current_start_time: object,
) -> dict[str, Any]:
    """Return a non-mutating live writer inventory for one serving identity.

    Persisted platform state intentionally includes forensic history. A record is
    safe to remove from the live view only when both writer identity fields are
    exact integers and both unambiguously belong to a prior process. Ambiguous,
    partial, and malformed records remain visible so strict validators fail
    closed rather than hiding authority conflicts.
    """
    if not isinstance(platforms, Mapping):
        raise TypeError("platforms must be a mapping")
    if type(current_pid) is not int or type(current_start_time) is not int:
        raise TypeError("current identity must use exact integers")

    return {
        name: record
        for name, record in platforms.items()
        if not (
            isinstance(record, dict)
            and type(record.get("writer_pid")) is int
            and record.get("writer_pid") != current_pid
            and type(record.get("writer_start_time")) is int
            and record.get("writer_start_time") != current_start_time
        )
    }
