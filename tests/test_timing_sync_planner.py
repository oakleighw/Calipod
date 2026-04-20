import pytest

from circuit_management.timing_sync_planner import build_timing_sync_plan, format_ms_as_clock, render_timing_sync_table


def test_timing_sync_plan_for_100fps_and_120hz_matches_expected_interval():
    plan = build_timing_sync_plan(100, 120, checkpoint_count=4)

    assert plan.frame_interval == 5
    assert plan.interval_ms == pytest.approx(50.0)
    assert [cp.frame_number for cp in plan.checkpoints] == [0, 5, 10, 15]
    assert [cp.expected_phone_ms_rounded for cp in plan.checkpoints] == [0, 50, 100, 150]


def test_timing_sync_plan_respects_start_frame():
    plan = build_timing_sync_plan(90, 120, checkpoint_count=3, start_frame=2)

    assert plan.frame_interval == 3
    assert [cp.frame_number for cp in plan.checkpoints] == [2, 5, 8]
    assert [cp.expected_phone_ms_rounded for cp in plan.checkpoints] == [22, 56, 89]


def test_format_ms_as_clock_formats_expected_pattern():
    assert format_ms_as_clock(0) == "00:00:000"
    assert format_ms_as_clock(75_437) == "01:15:437"


def test_render_timing_sync_table_in_ms_mode_includes_offset_values():
    plan = build_timing_sync_plan(100, 120, checkpoint_count=3)

    table_text = render_timing_sync_table(plan, first_frame_time_ms=123, use_clock_format=False)

    assert "Expected Phone Timer (ms)" in table_text
    assert "123.000" in table_text
    assert "173.000" in table_text


def test_render_timing_sync_table_in_clock_mode_uses_clock_header_and_values():
    plan = build_timing_sync_plan(100, 120, checkpoint_count=2)

    table_text = render_timing_sync_table(plan, first_frame_time_ms=123, use_clock_format=True)

    assert "Expected Phone Timer (mins:seconds:ms)" in table_text
    assert "00:00:123" in table_text
    assert "00:00:173" in table_text


@pytest.mark.parametrize(
    ("fps", "hz", "count", "start_frame"),
    [
        (0, 120, 10, 0),
        (100, 0, 10, 0),
        (100, 120, 0, 0),
        (100, 120, 5, -1),
    ],
)
def test_timing_sync_plan_rejects_invalid_inputs(fps, hz, count, start_frame):
    with pytest.raises(ValueError):
        build_timing_sync_plan(fps, hz, checkpoint_count=count, start_frame=start_frame)
