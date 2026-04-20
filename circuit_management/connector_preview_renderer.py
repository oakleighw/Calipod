"""Render numbered connector previews for the circuit management tab."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

# Pin numbering is in connector face-view order used by the UI table.
CONNECTOR_PIN_LAYOUTS: dict[str, tuple[tuple[int, int], ...]] = {
    "Hirose 4 Pin": (
        (415, 422),
        (415, 610),
        (610, 610),
        (609, 421),
    ),
    "Hirose 6 Pin": (
        (313, 268),
        (256, 365),
        (311, 463),
        (422, 463),
        (477, 365),
        (427, 267),
    ),
}

NUMBER_CIRCLE_RADIUS = 24
NUMBER_FONT = cv2.FONT_HERSHEY_SIMPLEX
NUMBER_FONT_SCALE = 0.7
NUMBER_THICKNESS = 2


def _hex_to_bgra(colour_hex: str | None, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not colour_hex:
        return fallback

    colour_hex = colour_hex.lstrip("#")
    if len(colour_hex) != 6:
        return fallback

    try:
        red = int(colour_hex[0:2], 16)
        green = int(colour_hex[2:4], 16)
        blue = int(colour_hex[4:6], 16)
    except ValueError:
        return fallback

    return blue, green, red, 255


def _text_colour_for_bgra(colour_bgra: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    blue, green, red, _alpha = colour_bgra
    luminance = (0.114 * blue) + (0.587 * green) + (0.299 * red)
    return (255, 255, 255, 255) if luminance < 140 else (0, 0, 0, 255)


def _load_connector_frame(image_path: Path) -> np.ndarray:
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Could not read connector image: {image_path}")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    return frame


def render_connector_preview_frame(
    image_path: Path,
    connector_name: str,
    wire_colours: Mapping[int, str | None],
) -> np.ndarray:
    """Return a BGRA frame with numbered pins rendered on top of the connector art."""
    frame = _load_connector_frame(image_path)
    pin_centres = CONNECTOR_PIN_LAYOUTS.get(connector_name)
    if not pin_centres:
        return frame

    fill_fallback = (184, 184, 184, 255)

    for pin_number, (centre_x, centre_y) in enumerate(pin_centres, start=1):
        fill_colour = _hex_to_bgra(wire_colours.get(pin_number), fill_fallback)
        text_colour = _text_colour_for_bgra(fill_colour)

        cv2.circle(frame, (centre_x, centre_y), NUMBER_CIRCLE_RADIUS, fill_colour, thickness=-1, lineType=cv2.LINE_AA)

        text = str(pin_number)
        text_size, _baseline = cv2.getTextSize(text, NUMBER_FONT, NUMBER_FONT_SCALE, NUMBER_THICKNESS)
        text_x = int(centre_x - (text_size[0] / 2))
        text_y = int(centre_y + (text_size[1] / 2))
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            NUMBER_FONT,
            NUMBER_FONT_SCALE,
            text_colour,
            thickness=NUMBER_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return frame
