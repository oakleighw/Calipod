"""Helpers for creating collapsible container sections."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget


def create_collapsible_container(
    parent: QWidget,
    parent_layout,
    title: str,
    expanded: bool = False,
    font_weight: int = 700,
    content_left_indent: int = 8,
) -> QVBoxLayout:
    """Create a collapsible section and return the layout for section content."""
    section_container = QWidget(parent)
    section_layout = QVBoxLayout(section_container)
    section_layout.setContentsMargins(0, 0, 0, 0)
    section_layout.setSpacing(4)

    header_button = QPushButton(parent)
    header_button.setFlat(True)
    header_button.setCheckable(True)
    header_button.setChecked(expanded)
    header_button.setCursor(Qt.CursorShape.PointingHandCursor)
    header_button.setStyleSheet(
        "QPushButton {"
        " text-align: left;"
        " border: none;"
        " background: transparent;"
        " padding: 2px 0;"
        f" font-weight: {font_weight};"
        "}"
    )

    content_widget = QWidget(parent)
    content_layout = QVBoxLayout(content_widget)
    content_layout.setContentsMargins(content_left_indent, 0, 0, 0)
    content_layout.setSpacing(4)

    def _apply_state(is_expanded: bool):
        arrow = "▼" if is_expanded else "▶"
        header_button.setText(f"{arrow} {title}")
        content_widget.setVisible(is_expanded)

    _apply_state(expanded)
    header_button.toggled.connect(_apply_state)

    section_layout.addWidget(header_button)
    section_layout.addWidget(content_widget)
    parent_layout.addWidget(section_container)

    return content_layout
