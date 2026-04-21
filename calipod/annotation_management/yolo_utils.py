"""
YOLO annotation parsing utilities.

Provides shared functions for parsing YOLO format annotations across the codebase.
Used by FlyTracker, AnnotationChecker, BGSProcessor, and other modules.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float, Optional[float]]]:
    """
    Parse a single YOLO format line.

    YOLO format: class_id x_center y_center width height [confidence]

    Args:
        line: A line from a YOLO label file

    Returns:
        Tuple of (class_id, x_center, y_center, width, height, confidence) in normalized coords,
        or None if parsing failed. Confidence defaults to 1.0 if not provided.
    """
    try:
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0

        return (class_id, x_center, y_center, width, height, confidence)
    except (ValueError, IndexError):
        return None


def denormalize_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    frame_shape: Tuple[int, int, int],
) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO coordinates to pixel coordinates.

    Args:
        x_center, y_center, width, height: Normalized coordinates (0-1 range)
        frame_shape: Tuple of (height, width, channels)

    Returns:
        Tuple of (x_center_px, y_center_px, width_px, height_px) in pixel coordinates
    """
    frame_height, frame_width = frame_shape[0], frame_shape[1]
    x_center_px = x_center * frame_width
    y_center_px = y_center * frame_height
    width_px = width * frame_width
    height_px = height * frame_height
    return (x_center_px, y_center_px, width_px, height_px)


def load_yolo_file(
    text_file: Path,
    frame_shape: Optional[Tuple[int, int, int]] = None,
    highest_confidence_only: bool = False,
    add_corner_points: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and parse a YOLO label file to extract point IDs, locations, and bounding boxes.

    If highest_confidence_only set to true, only the highest confidence
    detection per class is kept per frame (for single-object scenarios).

    For labels 9 (fruit) and 10 (leaves), also generates corner points for proper 3D triangulation.
    Corner points use IDs: base_id * 1000 + corner_index (0=TL, 1=TR, 2=BR, 3=BL)

    Args:
        text_file: Path to YOLO label file
        frame_shape: Optional tuple of (height, width, channels). If provided, normalizes to pixel coords.
        highest_confidence_only: If True, only keep highest confidence detection per class.
        add_corner_points: If True, for classes 9 (fruit) and 10 (leaves), generate 4 corner points
                          for proper 3D triangulation. Corner IDs: class_id*1000 + corner_index (0=TL, 1=TR, 2=BR, 3=BL)

    Returns:
        Tuple of:
        - ids: numpy array of class IDs (including corner point IDs if add_corner_points=True)
        - img_loc: numpy array of (x, y) positions in pixel coords (or normalized if frame_shape=None)
        - bboxes: numpy array of (width, height) for each detection in pixel coords
    """
    ids = np.array([], dtype=int)
    img_loc = np.array([], dtype=float)
    bboxes = np.array([], dtype=float)

    if not text_file.exists():
        logger.warning(f"YOLO label file not found: {text_file}")
        return ids, img_loc, bboxes

    try:
        with open(text_file, "r") as txt:
            lines = txt.readlines()

        if not lines:
            return ids, img_loc, bboxes

        # If highest_confidence_only, filter to keep only best detection per class
        if highest_confidence_only:
            class_detections = {}
            for line in lines:
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue

                class_id, x, y, w, h, confidence = parsed
                if class_id not in class_detections or confidence > class_detections[class_id][0]:
                    class_detections[class_id] = (confidence, line)

            lines = [entry[1] for entry in class_detections.values()]

        # Parse lines
        ids_list = []
        img_loc_list = []
        bboxes_list = []

        for line in lines:
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue

            class_id, x_center, y_center, width, height, _ = parsed

            # Convert to pixel coordinates if frame_shape provided
            if frame_shape is not None:
                x_center, y_center, width, height = denormalize_bbox(
                    x_center, y_center, width, height, frame_shape
                )

            # Add center point
            ids_list.append(class_id)
            img_loc_list.append((x_center, y_center))
            bboxes_list.append((width, height))

            logger.debug(
                f"Parsed YOLO point: class={class_id}, pos=({x_center}, {y_center}), "
                f"bbox=({width}, {height}) from {text_file.name}"
            )

            # For fruit (9) and leaves (10), add 4 corner points for proper 3D triangulation
            if add_corner_points and class_id in [9, 10]:
                half_w = width / 2.0
                half_h = height / 2.0

                # Corner positions: TL, TR, BR, BL
                corners = [
                    (x_center - half_w, y_center - half_h),  # Top-left (0)
                    (x_center + half_w, y_center - half_h),  # Top-right (1)
                    (x_center + half_w, y_center + half_h),  # Bottom-right (2)
                    (x_center - half_w, y_center + half_h),  # Bottom-left (3)
                ]

                for corner_idx, (cx, cy) in enumerate(corners):
                    corner_id = class_id * 1000 + corner_idx  # e.g., 9000, 9001, 9002, 9003
                    ids_list.append(corner_id)
                    img_loc_list.append((cx, cy))
                    bboxes_list.append((width, height))

                logger.debug(
                    f"Added 4 corner points for class_id={class_id} with IDs "
                    f"{class_id * 1000} to {class_id * 1000 + 3}"
                )

        return (
            np.array(ids_list, dtype=int),
            np.array(img_loc_list, dtype=np.float32),
            np.array(bboxes_list, dtype=np.float32),
        )

    except FileNotFoundError:
        logger.warning(f"YOLO label file not found: {text_file}")
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing YOLO label file {text_file}: {e}")

    return ids, img_loc, bboxes
