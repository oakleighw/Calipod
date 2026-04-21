from pathlib import Path

from calipod import __root__
from circuit_management.connector_preview_renderer import CONNECTOR_PIN_LAYOUTS, render_connector_preview_frame


def _bgra_from_hex(colour_hex: str) -> tuple[int, int, int, int]:
    colour_hex = colour_hex.lstrip("#")
    red = int(colour_hex[0:2], 16)
    green = int(colour_hex[2:4], 16)
    blue = int(colour_hex[4:6], 16)
    return blue, green, red, 255


def test_render_connector_preview_frame_places_colours_on_pin_centres():
    image_path = Path(__root__, "calipod", "gui", "icons", "connectors", "4_pin_hirose_female.png")
    colours = {
        1: "#D32F2F",
        2: "#2E7D32",
        3: "#1976D2",
        4: "#C9A200",
    }

    frame = render_connector_preview_frame(image_path, "Hirose 4 Pin", colours)

    assert frame.shape[:2] == (1027, 1024)

    for pin_number, (centre_x, centre_y) in enumerate(CONNECTOR_PIN_LAYOUTS["Hirose 4 Pin"], start=1):
        assert tuple(frame[centre_y, centre_x + 8]) == _bgra_from_hex(colours[pin_number])


def test_render_connector_preview_frame_handles_six_pin_layout():
    image_path = Path(__root__, "calipod", "gui", "icons", "connectors", "6_pin_hirose_female.png")
    colours = {pin_number: "#616161" for pin_number in range(1, 7)}

    frame = render_connector_preview_frame(image_path, "Hirose 6 Pin", colours)

    assert frame.shape[:2] == (736, 736)

    for centre_x, centre_y in CONNECTOR_PIN_LAYOUTS["Hirose 6 Pin"]:
        assert tuple(frame[centre_y, centre_x + 8]) == _bgra_from_hex("#616161")
