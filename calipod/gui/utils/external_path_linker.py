"""Helpers for linking externally selected paths into a workspace layout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)

ORGANISATION_DIR_NAME = "organisation"
EXTERNAL_LINKS_MANIFEST_NAME = "external_file_links.json"


def _utc_timestamp() -> str:
    """Return current UTC timestamp string for manifest events."""
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(workspace_root: Path) -> Path:
    """Return the manifest file path inside the workspace organisation folder."""
    return workspace_root / ORGANISATION_DIR_NAME / EXTERNAL_LINKS_MANIFEST_NAME


def _default_manifest() -> dict:
    """Create a new empty manifest payload."""
    return {
        "version": 1,
        "updated_utc": _utc_timestamp(),
        "entries": {},
    }


def _load_manifest(path: Path) -> dict:
    """Load manifest JSON with safe fallbacks for missing/corrupt files."""
    if not path.exists():
        return _default_manifest()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read link manifest at %s (%s); recreating", path, exc)
        return _default_manifest()

    if not isinstance(payload, dict):
        return _default_manifest()

    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    payload.setdefault("version", 1)
    payload.setdefault("updated_utc", _utc_timestamp())
    return payload


def _save_manifest(path: Path, payload: dict) -> None:
    """Persist the manifest payload to disk with a refreshed update timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_utc"] = _utc_timestamp()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _manifest_key(destination: Path, workspace_root: Path) -> str:
    """Convert destination path into a stable manifest key."""
    destination_abs = destination if destination.is_absolute() else workspace_root / destination
    try:
        return destination_abs.relative_to(workspace_root).as_posix()
    except ValueError:
        return str(destination_abs)


def _record_manifest_entry(
    workspace_root: Path,
    destination: Path,
    target_label: str,
    source_path: Path,
    link_type: str,
    result: str,
    details: str | None = None,
) -> None:
    """Create or update the manifest entry for a managed destination path."""
    manifest_file = _manifest_path(workspace_root)
    payload = _load_manifest(manifest_file)
    entry_key = _manifest_key(destination, workspace_root)
    entry_payload = {
        "target_label": target_label,
        "destination_path": str(destination),
        "source_path": str(source_path),
        "link_type": link_type,
        "last_result": result,
        "updated_utc": _utc_timestamp(),
    }
    if details:
        entry_payload["last_details"] = details

    payload["entries"][entry_key] = entry_payload
    _save_manifest(manifest_file, payload)


def _remove_manifest_entry(
    workspace_root: Path,
    destination: Path,
    target_label: str,
) -> None:
    """Remove a manifest entry for a destination path."""
    manifest_file = _manifest_path(workspace_root)
    payload = _load_manifest(manifest_file)
    entry_key = _manifest_key(destination, workspace_root)
    payload["entries"].pop(entry_key, None)
    _save_manifest(manifest_file, payload)


def _os_error_details(exc: OSError) -> str:
    """Format useful OS error details for troubleshooting filesystem link issues."""
    return (
        f"{exc} "
        f"(errno={getattr(exc, 'errno', None)}, "
        f"winerror={getattr(exc, 'winerror', None)})"
    )


def _create_windows_directory_junction(destination: Path, source: Path) -> None:
    """Create a Windows directory junction at destination pointing to source."""
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "Unknown mklink failure"
        raise OSError(f"mklink /J failed: {detail}")


def link_external_path_selection(
    parent: QWidget,
    selected_path: str,
    link_destination: Path | None,
    workspace_root: Path,
    expect_directory: bool,
    target_label: str | None = None,
    on_link_created: Callable[[], None] | None = None,
) -> str:
    """Link an external file/directory into a workspace destination if needed."""
    source_path = Path(selected_path).expanduser()
    label = target_label or "Selected path"
    workspace_root = workspace_root.resolve()

    logger.info(
        "Link selection requested for '%s': source=%s destination=%s expect_directory=%s",
        label,
        source_path,
        link_destination,
        expect_directory,
    )

    if link_destination is None:
        return str(source_path)

    try:
        source_resolved = source_path.resolve(strict=True)
    except FileNotFoundError:
        logger.warning("Link selection failed for '%s': source not found: %s", label, source_path)
        QMessageBox.warning(parent, "Path Not Found", f"Selected path does not exist:\n{source_path}")
        return ""

    is_dir = source_resolved.is_dir()
    if expect_directory and not is_dir:
        logger.warning("Link selection failed for '%s': expected directory but got file: %s", label, source_resolved)
        QMessageBox.warning(parent, "Invalid Selection", "Please choose a directory for this field.")
        return ""
    if not expect_directory and not source_resolved.is_file():
        logger.warning("Link selection failed for '%s': expected file but got directory: %s", label, source_resolved)
        QMessageBox.warning(parent, "Invalid Selection", "Please choose a file for this field.")
        return ""

    # Keep workspace-local paths untouched; no link required.
    try:
        source_resolved.relative_to(workspace_root.resolve())
        logger.info("Link not needed for '%s': source already inside workspace (%s)", label, source_resolved)
        _remove_manifest_entry(
            workspace_root,
            destination=link_destination,
            target_label=label,
        )
        return str(source_resolved)
    except ValueError:
        pass

    destination = link_destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            try:
                current_target = destination.resolve(strict=True)
            except OSError:
                current_target = None

            if current_target == source_resolved:
                logger.info("Existing link already correct for '%s': %s -> %s", label, destination, source_resolved)
                _record_manifest_entry(
                    workspace_root,
                    destination=destination,
                    target_label=label,
                    source_path=source_resolved,
                    link_type="symlink",
                    result="link_unchanged",
                )
                return str(destination)

            replace_choice = QMessageBox.question(
                parent,
                "Replace Existing Link?",
                f"{label} is currently linked to:\n{current_target or destination}\n\n"
                f"Replace it with:\n{source_resolved}\n\n"
                "Select No to keep the current link.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if replace_choice != QMessageBox.StandardButton.Yes:
                logger.info("User kept existing link for '%s' at %s", label, destination)
                if current_target is not None:
                    _record_manifest_entry(
                        workspace_root,
                        destination=destination,
                        target_label=label,
                        source_path=current_target,
                        link_type="symlink",
                        result="replace_cancelled",
                    )
                else:
                    _remove_manifest_entry(
                        workspace_root,
                        destination=destination,
                        target_label=label,
                    )
                return str(destination)

            destination.unlink()
            logger.info("Removed existing link for '%s' at %s before replacement", label, destination)
        else:
            try:
                if destination.resolve(strict=True) == source_resolved:
                    logger.info(
                        "Destination already contains expected path for '%s': %s",
                        label,
                        destination,
                    )
                    QMessageBox.information(
                        parent,
                        "Already In Project Folder",
                        f"{label} is already in the expected project location:\n{destination}",
                    )
                    _remove_manifest_entry(
                        workspace_root,
                        destination=destination,
                        target_label=label,
                    )
                    return str(destination)
            except OSError:
                pass

            QMessageBox.warning(
                parent,
                "Destination Already Exists",
                f"{label} already exists in the project folder:\n{destination}\n\n"
                "No link was created, and the existing project file/folder was kept.",
            )
            logger.warning(
                "Link creation skipped for '%s': destination exists and is not a link (%s)",
                label,
                destination,
            )
            _remove_manifest_entry(
                workspace_root,
                destination=destination,
                target_label=label,
            )
            return str(destination)

    try:
        destination.symlink_to(source_resolved, target_is_directory=is_dir)
        logger.info("Created symlink for '%s': %s -> %s", label, destination, source_resolved)
        _record_manifest_entry(
            workspace_root,
            destination=destination,
            target_label=label,
            source_path=source_resolved,
            link_type="symlink",
            result="link_created",
        )
        if on_link_created is not None:
            on_link_created()
        return str(destination)
    except OSError as exc:
        logger.warning(
            "Symlink creation failed for '%s': %s -> %s; reason=%s",
            label,
            destination,
            source_resolved,
            _os_error_details(exc),
        )
        if is_dir and sys.platform.startswith("win"):
            try:
                _create_windows_directory_junction(destination=destination, source=source_resolved)
                logger.warning(
                    "Created directory junction fallback for '%s': %s -> %s",
                    label,
                    destination,
                    source_resolved,
                )
                _record_manifest_entry(
                    workspace_root,
                    destination=destination,
                    target_label=label,
                    source_path=source_resolved,
                    link_type="junction",
                    result="junction_fallback_created",
                    details=_os_error_details(exc),
                )
                if on_link_created is not None:
                    on_link_created()
                return str(destination)
            except OSError as junction_exc:
                logger.warning(
                    "Directory junction fallback failed for '%s': %s -> %s; reason=%s",
                    label,
                    destination,
                    source_resolved,
                    junction_exc,
                )

        if not is_dir:
            try:
                os.link(source_resolved, destination)
                logger.warning(
                    "Created hard link fallback for '%s': %s -> %s",
                    label,
                    destination,
                    source_resolved,
                )
                _record_manifest_entry(
                    workspace_root,
                    destination=destination,
                    target_label=label,
                    source_path=source_resolved,
                    link_type="hardlink",
                    result="hardlink_fallback_created",
                    details=_os_error_details(exc),
                )
                if on_link_created is not None:
                    on_link_created()
                return str(destination)
            except OSError as hard_link_exc:
                logger.warning(
                    "Hard link fallback failed for '%s': %s -> %s; reason=%s",
                    label,
                    destination,
                    source_resolved,
                    _os_error_details(hard_link_exc),
                )

        logger.warning("Failed to create link from %s to %s: %s", source_resolved, destination, exc)
        if is_dir:
            if sys.platform.startswith(("linux", "darwin")):
                message = (
                    "Could not create a project-side directory link on this platform. "
                    "Annotations and other directory-backed data need to live at the project path "
                    "so the rest of the app can find them. Please copy/import the directory into the project instead."
                )
            else:
                message = (
                    "Could not create a project-side directory link. On this system, the app cannot safely "
                    "fall back to a direct external directory reference for downstream readers. Please copy/import "
                    "the directory into the project instead, or resolve the filesystem permissions/link support issue."
                )

            QMessageBox.critical(parent, "Directory Link Failed", message)
            _record_manifest_entry(
                workspace_root,
                destination=destination,
                target_label=label,
                source_path=source_resolved,
                link_type="external-reference",
                result="directory_link_failed",
                details=_os_error_details(exc),
            )
            return ""

        QMessageBox.warning(
            parent,
            "Link Creation Failed",
            "Could not create a filesystem link at the project location. "
            "The selected external path will be used directly instead.",
        )
        _record_manifest_entry(
            workspace_root,
            destination=destination,
            target_label=label,
            source_path=source_resolved,
            link_type="external-reference",
            result="external_reference_used",
            details="Filesystem link creation failed; using direct external path.",
        )
        return str(source_resolved)
