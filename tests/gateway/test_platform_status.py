from __future__ import annotations

from copy import deepcopy

import pytest

from gateway.platform_status import project_live_platforms


@pytest.mark.parametrize(
    ("writer", "visible"),
    [
        ({"writer_pid": 10, "writer_start_time": 20}, False),
        ({"writer_pid": 100, "writer_start_time": 20}, True),
        ({"writer_pid": 10, "writer_start_time": 200}, True),
        ({"writer_pid": 100, "writer_start_time": 200}, True),
        ({"writer_pid": True, "writer_start_time": 20}, True),
        ({"writer_pid": 10, "writer_start_time": False}, True),
        ({"writer_pid": "10", "writer_start_time": 20}, True),
        ({"writer_pid": 10, "writer_start_time": 20.0}, True),
        ({"writer_pid": 10}, True),
        ({"writer_start_time": 20}, True),
        ({}, True),
        (None, True),
        ("malformed", True),
    ],
)
def test_project_live_platforms_filters_only_unambiguous_prior_writer(
    writer: object,
    visible: bool,
) -> None:
    platforms = {"telegram": writer}

    projected = project_live_platforms(
        platforms,
        current_pid=100,
        current_start_time=200,
    )

    assert ("telegram" in projected) is visible


def test_project_live_platforms_does_not_mutate_input() -> None:
    platforms = {
        "telegram": {"writer_pid": 10, "writer_start_time": 20},
        "api_server": {"writer_pid": 100, "writer_start_time": 200},
    }
    original = deepcopy(platforms)

    projected = project_live_platforms(
        platforms,
        current_pid=100,
        current_start_time=200,
    )

    assert platforms == original
    assert projected is not platforms
    assert projected["api_server"] is platforms["api_server"]


@pytest.mark.parametrize("platforms", [None, [], "bad", 1, True])
def test_project_live_platforms_rejects_non_mapping_inventory(platforms: object) -> None:
    with pytest.raises(TypeError, match="platforms must be a mapping"):
        project_live_platforms(
            platforms,
            current_pid=100,
            current_start_time=200,
        )


@pytest.mark.parametrize(
    ("current_pid", "current_start_time"),
    [(True, 200), (100, False), ("100", 200), (100, 200.0)],
)
def test_project_live_platforms_rejects_non_exact_current_identity(
    current_pid: object,
    current_start_time: object,
) -> None:
    with pytest.raises(TypeError, match="current identity must use exact integers"):
        project_live_platforms(
            {},
            current_pid=current_pid,
            current_start_time=current_start_time,
        )
