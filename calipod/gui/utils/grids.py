"""Shared helpers for explicit 3D grid rendering and scale labels."""

from __future__ import annotations

import numpy as np


def nice_step(value: float) -> float:
    """Round a spacing value to a human-friendly 1/2/5*10^n step."""
    value = max(float(value), 1e-6)
    exponent = int(np.floor(np.log10(value)))
    base = 10.0 ** exponent
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        candidate = multiplier * base
        if candidate >= value:
            return candidate
    return 10.0 ** (exponent + 1)


def adaptive_grid_spacing(visible_extent: float, target_major_intervals: float = 20.0) -> tuple[float, float]:
    """Return minor/major spacing for an explicit grid based on visible extent.

    The result is stable and readable across different scene scales.
    """
    visible_extent = max(float(visible_extent), 1e-6)
    raw_major = visible_extent / max(float(target_major_intervals), 1.0)
    major = nice_step(raw_major)
    minor = nice_step(major / 5.0)
    if minor >= major:
        minor = max(1e-6, major / 5.0)
    return minor, major


def build_plane_grid_lines(plane: str, spacing: float, half_extent: float):
    """Build endpoints for a plane-aligned explicit grid."""
    lines = []
    spacing = max(float(spacing), 1e-6)
    half_extent = max(float(half_extent), 0.0)
    steps = int(np.floor((2.0 * half_extent) / spacing))
    for i in range(-steps // 2, steps // 2 + 1):
        value = round(i * spacing, 6)
        if abs(value) > half_extent + 1e-9:
            continue
        if plane == "xy":
            lines.append([[-half_extent, value, 0.0], [half_extent, value, 0.0]])
            lines.append([[value, -half_extent, 0.0], [value, half_extent, 0.0]])
        elif plane == "xz":
            lines.append([[-half_extent, 0.0, value], [half_extent, 0.0, value]])
            lines.append([[value, 0.0, -half_extent], [value, 0.0, half_extent]])
        else:
            raise ValueError(f"Unsupported plane '{plane}'. Expected 'xy' or 'xz'.")
    return lines


def format_axis_label(scene_value: float, scene_scale: float) -> str:
    """Format a scene-space value using mm/cm/m depending on magnitude."""
    mm_value = float(scene_value) / max(float(scene_scale), 1e-12)
    abs_mm = abs(mm_value)
    if abs_mm >= 1000.0:
        return f"{(mm_value / 1000.0):.2f}".rstrip("0").rstrip(".") + " m"
    if abs_mm >= 10.0:
        return f"{(mm_value / 10.0):.1f}".rstrip("0").rstrip(".") + " cm"
    return f"{mm_value:.0f} mm"


def scale_label_values(visible_extent: float, scene_scale: float, target_major_intervals: float = 20.0) -> tuple[float, float, float]:
    """Return minor spacing, major spacing, and label interval in scene units."""
    minor, major = adaptive_grid_spacing(visible_extent, target_major_intervals=target_major_intervals)
    label_interval = major
    return minor, major, label_interval


def build_edge_tick_specs(
    half_extent: float,
    label_interval: float,
    label_formatter,
    offset_ratio: float = 0.03,
    interval_offset_ratio: float = 0.25,
    diagonal_ratio: float = 0.35,
):
    """Return edge-only tick label specs for XY/XZ grids.

    Returns a list of tuples: ((x, y, z), text)
    """
    specs = []
    half_extent = float(half_extent)
    label_interval = max(float(label_interval), 1e-12)
    offset = max(half_extent * offset_ratio, label_interval * interval_offset_ratio)
    diagonal = offset * diagonal_ratio

    steps = int(np.floor((2.0 * half_extent) / label_interval))
    for i in range(-steps // 2, steps // 2 + 1):
        value = i * label_interval
        if abs(value) > half_extent + 1e-9:
            continue

        sign = 0.0 if abs(value) < 1e-9 else np.sign(value)
        text = label_formatter(value)

        # X ticks on lower edge of XY plane.
        specs.append(((value + sign * diagonal, -half_extent - offset, 0.0), text))
        # Y ticks on left edge of XY plane.
        specs.append(((-half_extent - offset, value + sign * diagonal, 0.0), text))
        # Z ticks on left edge of XZ plane.
        specs.append(((-half_extent - offset, 0.0, value + sign * diagonal), text))

    return specs


def build_axis_label_specs(
    plane: str,
    half_extent: float,
    offset_ratio: float = 0.05,
    interval_offset_ratio: float = 0.25,
) -> list[tuple[tuple[float, float, float], str]]:
    """Return axis end label specs for plane-aligned grids.

    Places "X", "Y", or "Z" labels at the end of each axis, parallel to their plane.
    For xy plane, labels X and Y (z=0). For xz plane, labels Z (y=0).
    Returns a list of tuples: ((x, y, z), text)
    """
    specs = []
    half_extent = float(half_extent)
    offset = max(half_extent * offset_ratio, half_extent * interval_offset_ratio)

    if plane == "xy":
        # X label at the end of x-axis, in xy plane (z=0)
        specs.append(((half_extent + offset, 0.0, 0.0), "X"))
        # Y label at the end of y-axis, in xy plane (z=0)
        specs.append(((0.0, half_extent + offset, 0.0), "Y"))
    elif plane == "xz":
        # Z label at the end of z-axis, in xz plane (y=0)
        specs.append(((0.0, 0.0, half_extent + offset), "Z"))
    else:
        raise ValueError(f"Unsupported plane '{plane}'. Expected 'xy' or 'xz'.")

    return specs


def build_complete_grid_label_specs(
    half_extent: float,
    label_interval: float,
    label_formatter,
    offset_ratio: float = 0.03,
    interval_offset_ratio: float = 0.25,
    diagonal_ratio: float = 0.35,
) -> list[tuple[tuple[float, float, float], str]]:
    """Return complete label specs combining tick specs and axis labels for both xy/xz planes.

    Returns a list of tuples: ((x, y, z), text) with both tick labels and axis end labels.
    """
    specs = []
    
    # Add edge tick specs
    specs.extend(build_edge_tick_specs(
        half_extent=half_extent,
        label_interval=label_interval,
        label_formatter=label_formatter,
        offset_ratio=offset_ratio,
        interval_offset_ratio=interval_offset_ratio,
        diagonal_ratio=diagonal_ratio,
    ))
    
    # Add axis labels for both planes
    for plane in ("xy", "xz"):
        specs.extend(build_axis_label_specs(
            plane=plane,
            half_extent=half_extent,
            offset_ratio=offset_ratio,
            interval_offset_ratio=interval_offset_ratio,
        ))
    
    return specs
