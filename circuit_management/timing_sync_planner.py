"""Utilities for planning timestamp checks in camera/phone sync validation."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class TimingCheckpoint:
    """A frame to inspect and the expected timer reading on the phone display."""

    frame_number: int
    expected_phone_ms: float
    expected_phone_ms_rounded: int


@dataclass(frozen=True)
class TimingSyncPlan:
    """Plan describing clear-read sampling points for sync verification."""

    video_fps: float
    phone_refresh_hz: float
    frame_interval: int
    interval_ms: float
    checkpoints: list[TimingCheckpoint]


def format_ms_as_clock(total_ms: int) -> str:
    """Format milliseconds as mm:ss:ms for easier visual comparison."""
    if total_ms < 0:
        raise ValueError("Milliseconds must be zero or greater")

    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"


def render_timing_sync_table(
    plan: TimingSyncPlan,
    first_frame_time_ms: int = 0,
    use_clock_format: bool = False,
) -> str:
    """Render a fixed-width timing table for GUI display."""
    if first_frame_time_ms < 0:
        raise ValueError("First-frame time must be zero or greater")

    expected_header = (
        "Expected Phone Timer (mins:seconds:ms)"
        if use_clock_format
        else "Expected Phone Timer (ms)"
    )
    rounded_header = "Rounded to nearest ms"

    table_rows: list[tuple[str, str, str]] = []
    for checkpoint in plan.checkpoints:
        expected_ms = checkpoint.expected_phone_ms + first_frame_time_ms
        expected_ms_rounded = round(expected_ms)
        expected_text = (
            format_ms_as_clock(expected_ms_rounded)
            if use_clock_format
            else f"{expected_ms:.3f}"
        )
        rounded_text = (
            format_ms_as_clock(expected_ms_rounded)
            if use_clock_format
            else str(expected_ms_rounded)
        )
        table_rows.append((str(checkpoint.frame_number), expected_text, rounded_text))

    frame_col_width = max(len("Frame"), *(len(row[0]) for row in table_rows)) + 2
    expected_col_width = max(len(expected_header), *(len(row[1]) for row in table_rows)) + 2
    rounded_col_width = max(len(rounded_header), *(len(row[2]) for row in table_rows))

    lines = [
        f"{'Frame':<{frame_col_width}}"
        f"{expected_header:<{expected_col_width}}"
        f"{rounded_header:<{rounded_col_width}}",
        f"{'-' * (frame_col_width - 1):<{frame_col_width}}"
        f"{'-' * (expected_col_width - 1):<{expected_col_width}}"
        f"{'-' * rounded_col_width}",
    ]

    for frame_text, expected_text, rounded_text in table_rows:
        lines.append(
            f"{frame_text:<{frame_col_width}}"
            f"{expected_text:<{expected_col_width}}"
            f"{rounded_text:<{rounded_col_width}}"
        )

    return "\n".join(lines)


def build_timing_sync_plan(
    video_fps: float,
    phone_refresh_hz: float,
    checkpoint_count: int = 20,
    start_frame: int = 0,
) -> TimingSyncPlan:
    """Build a sampling plan where camera and phone-refresh phases realign.

    Args:
        video_fps: Camera recording frame rate.
        phone_refresh_hz: Phone display refresh rate.
        checkpoint_count: Number of frame checkpoints to generate.
        start_frame: First frame index used for checkpoint generation.

    Returns:
        TimingSyncPlan with interval and generated checkpoints.
    """
    if video_fps <= 0:
        raise ValueError("Video FPS must be greater than zero")
    if phone_refresh_hz <= 0:
        raise ValueError("Phone refresh Hz must be greater than zero")
    if checkpoint_count <= 0:
        raise ValueError("Checkpoint count must be greater than zero")
    if start_frame < 0:
        raise ValueError("Start frame must be zero or greater")

    fps_fraction = Fraction(str(video_fps)).limit_denominator(1_000_000)
    hz_fraction = Fraction(str(phone_refresh_hz)).limit_denominator(1_000_000)

    # Minimal frame step n where n * (hz / fps) is an integer.
    frame_interval = (hz_fraction / fps_fraction).denominator
    interval_ms = float(Fraction(frame_interval, 1) / fps_fraction * 1000)

    checkpoints: list[TimingCheckpoint] = []
    for sample_idx in range(checkpoint_count):
        frame_number = start_frame + sample_idx * frame_interval
        expected_ms = float(Fraction(frame_number, 1) / fps_fraction * 1000)
        checkpoints.append(
            TimingCheckpoint(
                frame_number=frame_number,
                expected_phone_ms=expected_ms,
                expected_phone_ms_rounded=round(expected_ms),
            )
        )

    return TimingSyncPlan(
        video_fps=video_fps,
        phone_refresh_hz=phone_refresh_hz,
        frame_interval=frame_interval,
        interval_ms=interval_ms,
        checkpoints=checkpoints,
    )
