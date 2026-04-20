"""
Utility module for detecting and validating annotation formats.
Supports: YOLO, COCO, Pascal VOC, CSV, and custom formats.
"""

import json
from pathlib import Path
from typing import Dict

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)


class AnnotationFormatChecker:
    """Detect annotation format and extract metadata."""

    @staticmethod
    def detect_format(file_path: Path) -> Dict:
        """
        Detect the annotation format of a file.

        Returns dict with:
        - format: Detected format name (YOLO, COCO, Pascal_VOC, CSV, Unknown)
        - is_valid: Whether file appears to be valid for detected format
        - has_bboxes: Whether annotations contain bounding boxes
        - classes: List of detected classes
        - details: Additional format-specific details
        """
        if not file_path.exists():
            return {
                "format": "Unknown",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": "File does not exist",
            }

        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return AnnotationFormatChecker._check_json(file_path)
        elif suffix == ".xml":
            return AnnotationFormatChecker._check_pascal_voc(file_path)
        elif suffix == ".txt":
            return AnnotationFormatChecker._check_yolo(file_path)
        elif suffix == ".csv":
            return AnnotationFormatChecker._check_csv(file_path)
        else:
            return {
                "format": "Unknown",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Unsupported file type: {suffix}",
            }

    @staticmethod
    def _check_json(file_path: Path) -> Dict:
        """Check if JSON file is COCO format."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Check COCO format structure
            if isinstance(data, dict):
                has_images = "images" in data
                has_annotations = "annotations" in data
                has_categories = "categories" in data

                if has_images and has_annotations and has_categories:
                    classes = []
                    if isinstance(data["categories"], list):
                        classes = [cat.get("name", str(cat.get("id"))) for cat in data["categories"]]

                    return {
                        "format": "COCO",
                        "is_valid": True,
                        "has_bboxes": True,
                        "classes": sorted(set(classes)),
                        "details": (
                            f"{len(data.get('images', []))} images, "
                            f"{len(data.get('annotations', []))} annotations"
                        ),
                    }

            # Check if it's a list of objects (custom JSON lines or array format)
            elif isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "bbox" in data[0]:
                    classes = list(set(ann.get("class") for ann in data if "class" in ann))
                    return {
                        "format": "JSON",
                        "is_valid": True,
                        "has_bboxes": True,
                        "classes": sorted(classes),
                        "details": f"{len(data)} annotations",
                    }

            return {
                "format": "JSON",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": "JSON structure not recognized",
            }
        except Exception as e:
            logger.debug(f"Error checking JSON format: {e}")
            return {
                "format": "JSON",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Parse error: {str(e)}",
            }

    @staticmethod
    def _check_yolo(file_path: Path) -> Dict:
        """Check if text file is YOLO format (class_id x_center y_center width height)."""
        try:
            classes = set()
            line_count = 0
            valid_lines = 0

            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    line_count += 1
                    parts = line.split()

                    # YOLO format: class_id x_center y_center width height (5+ values)
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x, y, w, h = [float(p) for p in parts[1:5]]

                            # Check if normalized (0-1 range)
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                classes.add(class_id)
                                valid_lines += 1
                        except ValueError:
                            pass

            if valid_lines > 0 and valid_lines / max(line_count, 1) > 0.8:
                return {
                    "format": "YOLO",
                    "is_valid": True,
                    "has_bboxes": True,
                    "classes": sorted(list(classes)),
                    "details": f"{line_count} lines, {valid_lines} valid YOLO format annotations",
                }

            return {
                "format": "Text",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Text file, but not valid YOLO format ({valid_lines}/{line_count} valid lines)",
            }
        except Exception as e:
            logger.debug(f"Error checking YOLO format: {e}")
            return {
                "format": "Text",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Parse error: {str(e)}",
            }

    @staticmethod
    def _check_pascal_voc(file_path: Path) -> Dict:
        """Check if XML file is Pascal VOC format."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Pascal VOC structure: root tag should be "annotation"
            if root.tag == "annotation":
                objects = root.findall("object")
                classes = set()

                for obj in objects:
                    class_elem = obj.find("name")
                    if class_elem is not None and class_elem.text:
                        classes.add(class_elem.text)

                return {
                    "format": "Pascal_VOC",
                    "is_valid": True,
                    "has_bboxes": True,
                    "classes": sorted(list(classes)),
                    "details": f"{len(objects)} objects",
                }

            return {
                "format": "XML",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": "XML file, but not Pascal VOC format",
            }
        except Exception as e:
            logger.debug(f"Error checking Pascal VOC format: {e}")
            return {
                "format": "XML",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Parse error: {str(e)}",
            }

    @staticmethod
    def _check_csv(file_path: Path) -> Dict:
        """Check CSV file for annotation structure."""
        try:
            import pandas as pd

            df = pd.read_csv(file_path)
            columns = df.columns.tolist()

            # Check for bounding box columns
            bbox_patterns = [
                ["x", "y", "width", "height"],
                ["x1", "y1", "x2", "y2"],
                ["xmin", "ymin", "xmax", "ymax"],
                ["left", "top", "right", "bottom"],
            ]

            has_bboxes = False
            for pattern in bbox_patterns:
                if all(col in columns for col in pattern):
                    has_bboxes = True
                    break

            # Extract classes
            classes = []
            for col in ["class", "label", "class_id", "category", "species"]:
                if col in columns:
                    classes = sorted(df[col].unique().tolist())
                    break

            format_type = "CSV_BBox" if has_bboxes else "CSV"

            return {
                "format": format_type,
                "is_valid": True,
                "has_bboxes": has_bboxes,
                "classes": [str(c) for c in classes],
                "details": f"{len(df)} rows, columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}",
            }
        except Exception as e:
            logger.debug(f"Error checking CSV format: {e}")
            return {
                "format": "CSV",
                "is_valid": False,
                "has_bboxes": False,
                "classes": [],
                "details": f"Parse error: {str(e)}",
            }

    @staticmethod
    def _extract_classes_only(file_path: Path, file_format: str) -> list:
        """
        Fast extraction of classes from a file, assuming format is already known.
        Skips format validation for speed.

        Returns list of class IDs/names found in the file.
        """
        classes = set()
        try:
            if file_format == "YOLO":
                # YOLO: just read first column as class_id
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    classes.add(class_id)
                                except ValueError:
                                    pass
            elif file_format == "COCO":
                # COCO: parse JSON and extract category IDs
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "annotations" in data:
                        for ann in data.get("annotations", []):
                            if "category_id" in ann:
                                classes.add(ann["category_id"])
            elif file_format == "Pascal_VOC":
                # Pascal VOC: extract class names from XML
                import xml.etree.ElementTree as ET

                tree = ET.parse(file_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    class_elem = obj.find("name")
                    if class_elem is not None and class_elem.text:
                        classes.add(class_elem.text)
            elif file_format in ["CSV", "CSV_BBox"]:
                # CSV: extract from class column
                import pandas as pd

                df = pd.read_csv(file_path)
                for col in ["class", "label", "class_id", "category", "species"]:
                    if col in df.columns:
                        classes.update(df[col].unique().tolist())
                        break
        except Exception as e:
            logger.debug(f"Error extracting classes from {file_path}: {e}")

        return list(classes)
