"""Helpers for building reusable filesystem tree widgets."""

from pathlib import Path

from PySide6.QtWidgets import QFileSystemModel, QTreeView, QWidget


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

    return tree, model