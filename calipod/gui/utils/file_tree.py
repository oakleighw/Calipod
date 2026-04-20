"""Helpers for building reusable filesystem tree widgets."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileSystemModel, QMenu, QStyle, QTreeView, QWidget

ORGANISATION_DIR_NAME = "organisation"
EXTERNAL_LINKS_MANIFEST_NAME = "external_file_links.json"


class ProjectFileTreeModel(QFileSystemModel):
    """Filesystem model that surfaces link status in icon and tooltip data."""

    def __init__(self, workspace_root: Path, parent: QWidget | None = None):
        """Initialise the tree model and manifest cache state."""
        super().__init__(parent)
        self._workspace_root = workspace_root.resolve()
        self._style = QApplication.style()
        self._manifest_cache: dict = {}
        self._manifest_mtime: float | None = None

    def _path_from_index(self, index) -> Path | None:
        """Return the filesystem path for a first-column model index."""
        if not index.isValid() or index.column() != 0:
            return None
        return Path(self.filePath(index))

    def _hard_link_count(self, path: Path) -> int:
        """Return the hard link count for a file path when available."""
        try:
            return os.stat(path).st_nlink
        except OSError:
            return 0

    def _manifest_path(self) -> Path:
        """Return the path to the workspace external-link manifest."""
        return self._workspace_root / ORGANISATION_DIR_NAME / EXTERNAL_LINKS_MANIFEST_NAME

    def _manifest_entries(self) -> dict:
        """Load and cache manifest entries, reloading only when file mtime changes."""
        manifest_file = self._manifest_path()
        if not manifest_file.exists():
            self._manifest_cache = {}
            self._manifest_mtime = None
            return self._manifest_cache

        try:
            mtime = manifest_file.stat().st_mtime
        except OSError:
            self._manifest_cache = {}
            self._manifest_mtime = None
            return self._manifest_cache

        if self._manifest_mtime == mtime:
            return self._manifest_cache

        try:
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        self._manifest_cache = entries if isinstance(entries, dict) else {}
        self._manifest_mtime = mtime
        return self._manifest_cache

    def _manifest_key_for_path(self, path: Path) -> str:
        """Convert an absolute path into the manifest key format."""
        try:
            return path.relative_to(self._workspace_root).as_posix()
        except ValueError:
            return str(path)

    def _manifest_entry_for_path(self, path: Path) -> dict | None:
        """Find manifest metadata for a tree path by key or destination path."""
        entries = self._manifest_entries()
        key = self._manifest_key_for_path(path)
        candidate = entries.get(key)
        if isinstance(candidate, dict):
            return candidate

        path_str = str(path)
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("destination_path") == path_str:
                return entry
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Provide custom icon and tooltip metadata for managed link items."""
        path = self._path_from_index(index)
        manifest_entry = self._manifest_entry_for_path(path) if path is not None else None

        if path is not None and role == Qt.ItemDataRole.DecorationRole:
            if isinstance(manifest_entry, dict) and manifest_entry.get("link_type") in {
                "symlink",
                "junction",
                "hardlink",
                "external-reference",
            }:
                if path.is_dir():
                    pixmap_name = "SP_DirLinkIcon"
                else:
                    pixmap_name = "SP_FileLinkIcon"
                standard_pixmap = getattr(QStyle.StandardPixmap, pixmap_name, None)
                if standard_pixmap is not None:
                    return self._style.standardIcon(standard_pixmap)

            if path.is_symlink():
                if path.is_dir():
                    pixmap_name = "SP_DirLinkIcon"
                else:
                    pixmap_name = "SP_FileLinkIcon"
                standard_pixmap = getattr(QStyle.StandardPixmap, pixmap_name, None)
                if standard_pixmap is not None:
                    return self._style.standardIcon(standard_pixmap)

        if path is not None and role == Qt.ItemDataRole.ToolTipRole:
            tooltip_parts = []
            if path.is_symlink():
                try:
                    target = path.resolve(strict=True)
                    tooltip_parts.append(f"Symbolic link\nTarget: {target}")
                except OSError:
                    tooltip_parts.append("Broken symbolic link")

            if path.is_file():
                hard_link_count = self._hard_link_count(path)
                if hard_link_count > 1:
                    tooltip_parts.append(
                        "Hard link file\n"
                        f"Shared path count: {hard_link_count}\n"
                        "This is not a copy; multiple paths reference the same file data."
                    )

            if isinstance(manifest_entry, dict):
                source_path = manifest_entry.get("source_path")
                link_type = manifest_entry.get("link_type")
                updated_utc = manifest_entry.get("updated_utc")
                tooltip_parts.append("Managed external link")
                if link_type:
                    tooltip_parts.append(f"Type: {link_type}")
                if source_path:
                    tooltip_parts.append(f"Source: {source_path}")
                if updated_utc:
                    tooltip_parts.append(f"Updated: {updated_utc}")

            if tooltip_parts:
                return "\n\n".join(tooltip_parts)

        return super().data(index, role)


def create_project_file_tree_view(root_path: Path, parent: QWidget | None = None) -> tuple[QTreeView, QFileSystemModel]:
    """Create a file tree view rooted at the provided workspace path."""
    model = ProjectFileTreeModel(workspace_root=root_path, parent=parent)
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
        """Resolve a tree index into a filesystem path."""
        if not index.isValid():
            return None
        return Path(model.filePath(index))

    def _open_path(path: Path):
        """Open a file or directory using the platform default handler."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_file_on_double_click(index):
        """Open files on double click while leaving directory expansion unchanged."""
        path = _path_from_index(index)
        if path is None or not path.is_file():
            return
        _open_path(path)

    def _reveal_in_system_file_browser(path: Path):
        """Reveal the target path in the system file browser for each OS."""
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
        """Copy the absolute path to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(str(path))

    def _show_context_menu(point: QPoint):
        """Display contextual file actions for the selected tree item."""
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
