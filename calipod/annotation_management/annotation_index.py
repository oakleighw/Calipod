"""Shared annotation indexing and cache utilities.

This module scans annotation directories once per workspace and caches the
result so multiple widgets can reuse the same annotation metadata without
re-walking the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional

from calipod.annotation_management.annotations_config_manager import (
    AnnotationsConfigManager,
)
from calipod.core import logger as calipod_logger
from calipod.core.annotation_checker import AnnotationFormatChecker

logger = calipod_logger.get(__name__)


class AnnotationIndex:
    """Cache annotation directory metadata for a workspace."""

    def __init__(self, workspace_dir: Path | str):
        """Initialize the cache for a single workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        self.annotations_dir = self.workspace_dir / "annotations"
        self.ground_truth_dir = self.annotations_dir / "ground_truth"
        self.predictions_dir = self.annotations_dir / "predictions"

        self._annotation_info_cache: Optional[Dict] = None
        self._annotation_scan_in_progress = False
        self._annotation_lock = Lock()

    def _empty_annotation_info(self) -> Dict:
        """Return the default empty annotation metadata structure."""
        return {
            "has_ground_truth": False,
            "has_predictions": False,
            "gt_subdirs": [],
            "gt_format": None,
            "gt_has_bboxes": False,
            "gt_classes": set(),
            "pred_subdirs": [],
            "pred_format": None,
            "pred_has_bboxes": False,
            "pred_classes": set(),
        }

    def _scan_annotation_dirs(self) -> Dict:
        """Scan annotation directories and build cached metadata."""
        anno_info = self._empty_annotation_info()

        if not self.annotations_dir.exists():
            logger.info(f"Annotations directory does not exist: {self.annotations_dir}")
            return anno_info

        logger.info(f"Found annotations directory: {self.annotations_dir}")

        try:
            if self.ground_truth_dir.exists():
                self._scan_annotation_root(self.ground_truth_dir, anno_info, "gt")

            if self.predictions_dir.exists():
                self._scan_annotation_root(self.predictions_dir, anno_info, "pred")

            if anno_info["has_ground_truth"] or anno_info["has_predictions"]:
                config_manager = AnnotationsConfigManager(self.workspace_dir)
                config_manager.save_annotations_config(
                    ground_truth_labels=set(anno_info["gt_classes"]),
                    ground_truth_format=anno_info["gt_format"],
                    predictions_labels=set(anno_info["pred_classes"]),
                    predictions_format=anno_info["pred_format"],
                )

            anno_info["gt_subdirs"] = sorted(anno_info["gt_subdirs"])
            anno_info["pred_subdirs"] = sorted(anno_info["pred_subdirs"])
            anno_info["gt_classes"] = sorted(list(anno_info["gt_classes"]))
            anno_info["pred_classes"] = sorted(list(anno_info["pred_classes"]))
        except Exception as e:
            logger.info(f"Error reading annotation directories: {e}")

        return anno_info

    def _scan_annotation_root(self, root_dir: Path, anno_info: Dict, key_prefix: str) -> None:
        """Scan one annotation root and merge the discovered metadata into the cache."""
        subdir_key = f"{key_prefix}_subdirs"
        format_key = f"{key_prefix}_format"
        has_bboxes_key = f"{key_prefix}_has_bboxes"
        classes_key = f"{key_prefix}_classes"

        for port_dir in root_dir.iterdir():
            if not port_dir.is_dir() or not port_dir.name.startswith("port_"):
                continue

            format_info = None
            all_files = []

            for pattern in ["**/*.txt", "**/*.json", "**/*.xml", "**/*.csv"]:
                for file in port_dir.rglob(pattern):
                    if not file.is_file():
                        continue

                    all_files.append(file)
                    if format_info is None:
                        detected_info = AnnotationFormatChecker.detect_format(file)
                        if detected_info["format"] != "Unknown":
                            format_info = detected_info

            if format_info is None:
                continue

            anno_info[f"has_{'ground_truth' if key_prefix == 'gt' else 'predictions'}"] = True
            anno_info[format_key] = format_info["format"]
            anno_info[has_bboxes_key] = format_info["has_bboxes"]
            anno_info[classes_key].update(format_info["classes"])

            for file in all_files[1:]:
                classes = AnnotationFormatChecker._extract_classes_only(file, format_info["format"])
                anno_info[classes_key].update(classes)

            if port_dir.name not in anno_info[subdir_key]:
                anno_info[subdir_key].append(port_dir.name)

    def _scan_worker(self) -> None:
        """Background worker that populates the cached annotation metadata."""
        try:
            annotation_info = self._scan_annotation_dirs()
        except Exception as e:
            logger.debug(f"Error while scanning annotations in background: {e}")
            annotation_info = self._empty_annotation_info()

        with self._annotation_lock:
            self._annotation_info_cache = annotation_info
            self._annotation_scan_in_progress = False

    def start_scan_if_needed(self) -> None:
        """Start the background scan unless one is already running or cached."""
        with self._annotation_lock:
            if self._annotation_scan_in_progress or self._annotation_info_cache is not None:
                return
            self._annotation_scan_in_progress = True

        worker = Thread(target=self._scan_worker, daemon=True)
        worker.start()

    def annotation_scan_in_progress(self) -> bool:
        """Return whether the background annotation scan is still running."""
        with self._annotation_lock:
            return self._annotation_scan_in_progress

    def get_cached_annotation_info(self) -> Optional[Dict]:
        """Return cached annotation metadata, starting the scan if needed."""
        self.start_scan_if_needed()
        with self._annotation_lock:
            return self._annotation_info_cache

    def get_cached_annotation_dir_text(self) -> str:
        """Return cached annotation summary text or a loading placeholder."""
        annotation_info = self.get_cached_annotation_info()
        if annotation_info is None:
            return "Checking for annotations..."

        if not annotation_info["has_ground_truth"] and not annotation_info["has_predictions"]:
            return "NONE"

        text_parts = ["annotation directories:"]

        if annotation_info["has_ground_truth"]:
            text_parts.append("  ground truth:")
            text_parts.append(f"    subdirectories: {', '.join(annotation_info['gt_subdirs'])}")
            text_parts.append(f"    format: {annotation_info['gt_format']}")
            text_parts.append(f"    classes: {annotation_info['gt_classes']}")

        if annotation_info["has_predictions"]:
            text_parts.append("  predictions:")
            text_parts.append(f"    subdirectories: {', '.join(annotation_info['pred_subdirs'])}")
            text_parts.append(f"    format: {annotation_info['pred_format']}")
            text_parts.append(f"    classes: {annotation_info['pred_classes']}")

        return "\n".join(text_parts)

    def get_cached_annotation_classes(self) -> Optional[List[str]]:
        """Return cached annotation class labels in displayable form."""
        annotation_info = self.get_cached_annotation_info()
        if annotation_info is None:
            return None

        classes: List[str] = []
        classes.extend(f"groundtruth/{class_id}" for class_id in annotation_info["gt_classes"])
        classes.extend(f"predictions/{class_id}" for class_id in annotation_info["pred_classes"])
        return sorted(classes)

    def get_cached_label_maps(self) -> tuple[dict, dict] | None:
        """Return label name maps for ground truth and predictions once available."""
        annotation_info = self.get_cached_annotation_info()
        if annotation_info is None:
            return None

        config_manager = AnnotationsConfigManager(self.workspace_dir)
        gt_labels = config_manager.get_ground_truth_labels() or {}
        pred_labels = config_manager.get_predictions_labels() or {}
        return gt_labels, pred_labels
