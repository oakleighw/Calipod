"""Helpers for building reusable filesystem tree widgets."""

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileSystemModel, QMenu, QTreeView, QWidget


def create_project_file_tree_view(root_path: Path, parent: QWidget | None = None) -> tuple[QTreeView, QFileSystemModel]:
    """Create a file tree view rooted at the provided workspace path."""
    model = QFileSystemModel(parent)
    root_str = str(root_path)
    model.setRootPath(root_str)

    tree = QTreeView(parent)
    tree.setModel(model)
    tree.setRootIndex(model.index(root_str))

    # Keep the initial structure-focused UI minimal.
    tree.setHeaderHidden(True)
    tree.setColumnHidden(1, True)
    tree.setColumnHidden(2, True)
    tree.setColumnHidden(3, True)
    tree.setAnimated(True)
    tree.setIndentation(18)

    def _path_from_index(index) -> Path | None:
        if not index.isValid():
            return None
        return Path(model.filePath(index))

    def _open_path(path: Path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_file_on_double_click(index):
        path = _path_from_index(index)
        if path is None or not path.is_file():
            return
        _open_path(path)

    def _reveal_in_system_file_browser(path: Path):
        if sys.platform.startswith("win"):
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                subprocess.Popen(["explorer", str(path)])
            return

        if sys.platform == "darwin":
            if path.is_file():
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["open", str(path)])
            return

        target = path.parent if path.is_file() else path
        subprocess.Popen(["xdg-open", str(target)])

    def _copy_full_path(path: Path):
        clipboard = QApplication.clipboard()
        clipboard.setText(str(path))

    def _show_context_menu(point: QPoint):
        index = tree.indexAt(point)
        if not index.isValid():
            return

        path = _path_from_index(index)
        if path is None:
            return

        menu = QMenu(tree)
        open_action = menu.addAction("Open")
        reveal_action = menu.addAction("Reveal in Explorer")
        copy_action = menu.addAction("Copy Full Path")

        selected_action = menu.exec(tree.viewport().mapToGlobal(point))
        if selected_action == open_action:
            _open_path(path)
        elif selected_action == reveal_action:
            _reveal_in_system_file_browser(path)
        elif selected_action == copy_action:
            _copy_full_path(path)

    tree.doubleClicked.connect(_open_file_on_double_click)
    tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    tree.customContextMenuRequested.connect(_show_context_menu)

    return tree, model
