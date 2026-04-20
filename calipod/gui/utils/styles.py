"""Helper functions for consistent styling across the GUI."""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from calipod.core.camera_colours import camera_rgba, rgba_to_css


# Helper function to create section titles
def create_section_font(point_size: int = 11) -> QFont:
    """Create a styled font for section headers."""
    font = QFont()
    font.setBold(True)
    font.setPointSize(point_size)
    return font


def create_section_title(text: str) -> QLabel:
    """Create a styled section title label."""
    title = QLabel(text)
    title.setFont(create_section_font(11))
    return title


# Helper function to create subsection titles
def create_subsection_title(text: str, color: str | None = None) -> QLabel:
    """Create a styled subsection title label."""
    title = QLabel(text)
    title.setFont(create_section_font(10))
    if color is not None:
        title.setStyleSheet(f"color: {color};")
    return title


def create_subsubsection_title(text: str, color: str | None = None) -> QLabel:
    """Create a styled subsubsection title label."""
    title = QLabel(text)
    title.setFont(create_section_font(9))
    if color is not None:
        title.setStyleSheet(f"color: {color};")
    return title


def resolve_camera_title_color(
    camera_index: int,
    camera_count: int,
    camera_data=None,
    *,
    zero_based_index: bool = False,
) -> str:
    """Resolve camera title color from camera metadata or deterministic fallback."""
    if camera_data is not None:
        if getattr(camera_data, "color_hex", None):
            return camera_data.color_hex
        if getattr(camera_data, "color", None):
            return rgba_to_css(camera_data.color)

    palette_index = camera_index if zero_based_index else camera_index - 1
    return rgba_to_css(camera_rgba(palette_index, max(1, int(camera_count or 0))))


def create_styled_groupbox(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    """Create a QGroupBox with styled section title (returns group and layout)."""
    group = QGroupBox()
    layout = QVBoxLayout()

    # Add styled title as a label, not as QGroupBox title
    title_label = create_section_title(title)
    layout.addWidget(title_label)

    group.setLayout(layout)
    return group, layout
