"""Shared camera colour utilities used across calibration and GUI views."""

from __future__ import annotations

import colorsys


def camera_rgba(index: int, total: int, *, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Return a deterministic camera colour in RGBA (0-1 range).

    Colors are evenly distributed in hue using HLS parameters consistent with
    existing arena and camera-array rendering defaults.
    """
    hue = 0.0 if total <= 0 else index / total
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.9)
    return (r, g, b, alpha)


def rgba_to_hex(color: tuple[float, float, float, float]) -> str:
    """Convert an RGBA 0-1 tuple into a hex string like #RRGGBB."""
    r, g, b, _ = color
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def rgba_to_css(color: tuple[float, float, float, float]) -> str:
    """Convert an RGBA 0-1 tuple into CSS rgb(r, g, b)."""
    r, g, b, _ = color
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def rgba_with_alpha(color: tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]:
    """Return the same RGB values with a different alpha."""
    r, g, b, _ = color
    return (r, g, b, max(0.0, min(1.0, alpha)))


def darken_rgba(
    color: tuple[float, float, float, float],
    factor: float = 0.45,
    alpha: float | None = None,
) -> tuple[float, float, float, float]:
    """Return a darker variant of an RGBA color."""
    r, g, b, a = color
    darkened = (
        max(0.0, min(1.0, r * factor)),
        max(0.0, min(1.0, g * factor)),
        max(0.0, min(1.0, b * factor)),
        a if alpha is None else max(0.0, min(1.0, alpha)),
    )
    return darkened


def camera_color_maps_for_ports(
    ports: list[int],
) -> tuple[dict[int, tuple[float, float, float, float]], dict[int, str]]:
    """Build RGBA and hex colour maps for the supplied camera port ordering."""
    cam_colors: dict[int, tuple[float, float, float, float]] = {}
    cam_hexes: dict[int, str] = {}
    num_colors = len(ports)

    for i, port in enumerate(ports):
        color = camera_rgba(i, num_colors)
        cam_colors[port] = color
        cam_hexes[port] = rgba_to_hex(color)

    return cam_colors, cam_hexes
