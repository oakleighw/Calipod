"""Helpers for a reusable path URL entry row with browse button."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


@dataclass(slots=True)
class PathUrlEntryWidgets:
    """Widgets returned from create_path_url_entry."""

    container: QWidget
    line_edit: QLineEdit
    browse_button: QPushButton


def create_path_url_entry(
    initial_path: str | None = None,
    parent: QWidget | None = None,
    dialog_caption: str = "Select Path",
    placeholder_text: str = "Enter or browse for a path",
    browse_button_text: str = "Browse...",
    select_directory: bool = True,
    file_filter: str = "All Files (*)",
    on_path_selected: Callable[[str], str | None] | None = None,
) -> PathUrlEntryWidgets:
    """Create a styled path entry row with a browse button.

    The text box is pre-populated if initial_path is provided.
    """
    line_edit = QLineEdit(parent)
    line_edit.setClearButtonEnabled(True)
    line_edit.setPlaceholderText(placeholder_text)
    line_edit.setStyleSheet("QLineEdit { padding: 5px 8px; border: 1px solid #b8b8b8; border-radius: 4px;}")

    if initial_path:
        line_edit.setText(initial_path)

    browse_button = QPushButton(browse_button_text, parent)
    browse_button.setAutoDefault(False)

    def _browse_start_path() -> str:
        text = line_edit.text().strip()
        if " (" in text:
            text = text.split(" (", 1)[0].strip()
        return text

    def on_browse_clicked() -> None:
        start_path = _browse_start_path()

        if start_path:
            start_path = str(Path(start_path))

        if select_directory:
            selected_path = QFileDialog.getExistingDirectory(
                parent,
                dialog_caption,
                start_path,
                options=QFileDialog.Option.ShowDirsOnly,
            )
        else:
            selected_path, _ = QFileDialog.getOpenFileName(
                parent,
                dialog_caption,
                start_path,
                file_filter,
            )

        if selected_path and on_path_selected is not None:
            selected_path = on_path_selected(selected_path)

        if selected_path:
            line_edit.setText(selected_path)

    browse_button.clicked.connect(on_browse_clicked)

    container = QWidget(parent)
    row_layout = QHBoxLayout(container)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(line_edit, stretch=1)
    row_layout.addWidget(browse_button)

    return PathUrlEntryWidgets(
        container=container,
        line_edit=line_edit,
        browse_button=browse_button,
    )
