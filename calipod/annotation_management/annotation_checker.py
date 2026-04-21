"""
Annotation checker utilities for loading and visualizing annotations with video frames.

Provides functionality to load annotations, extract frames, and visualize
bounding boxes overlaid on video frames.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calipod.annotation_management.file_utils import extract_frame_index_from_filename
from calipod.annotation_management.yolo_utils import parse_yolo_line
from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)


class AnnotationChecker:
    """Utilities for examining annotations visually with video frames."""

    def __init__(self, controller):
        """
        Initialize the annotation checker.

        Args:
            controller: Controller instance providing workspace_guide with directory paths
        """
        self.controller = controller
        self.workspace_guide = controller.workspace_guide

    def get_available_classes(
        self, include_ground_truth: bool = True, include_predictions: bool = True
    ) -> List[str]:
        """
        Get all available annotation classes in format 'groundtruth/ID' or 'predictions/ID'.

        Args:
            include_ground_truth: Include ground truth annotations
            include_predictions: Include prediction annotations

        Returns:
            Sorted list of class identifiers
        """
        classes = []

        if include_ground_truth:
            classes.extend(self._get_classes_from_dir(self.workspace_guide.ground_truth_dir, "groundtruth"))

        if include_predictions:
            classes.extend(self._get_classes_from_dir(self.workspace_guide.predictions_dir, "predictions"))

        return sorted(classes)

    def _get_classes_from_dir(self, anno_dir: Path, prefix: str) -> List[str]:
        """Extract class IDs from annotation directory."""
        classes = set()

        if not anno_dir.exists():
            return []

        # Look for port_X directories
        for port_dir in anno_dir.iterdir():
            if not port_dir.is_dir() or not port_dir.name.startswith("port_"):
                continue

            # Look for annotation files in labels subdirectories
            for anno_file in port_dir.rglob("*.txt"):
                try:
                    with open(anno_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                classes.add(class_id)
                except (ValueError, IOError):
                    pass

        return [f"{prefix}/{cls_id}" for cls_id in sorted(classes)]

    def find_recording_for_port(self, port: int) -> Optional[Path]:
        """
        Find recording directory that contains video for the given port.

        Args:
            port: Port number to search for

        Returns:
            Path to recording directory containing port_X.mp4, or None
        """
        recording_root = self.workspace_guide.recording_dir
        if not recording_root.exists():
            return None

        # Look through recording directories for port_X.mp4
        for recording_dir in recording_root.iterdir():
            if not recording_dir.is_dir():
                continue

            video_path = recording_dir / f"port_{port}.mp4"
            if video_path.exists():
                return recording_dir

        return None

    def extract_frame_from_video(self, video_path: Path, frame_index: int) -> Optional[np.ndarray]:
        """
        Extract a single frame from a video file.

        Args:
            video_path: Path to video file
            frame_index: Frame number to extract (0-indexed)

        Returns:
            Frame as numpy array (BGR), or None if extraction failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.warning(f"Could not read frame {frame_index} from {video_path}")
                return None

            return frame
        except Exception as e:
            logger.error(f"Error extracting frame from {video_path}: {e}")
            return None

    def load_annotations_for_class(
        self, class_label: str, port: int, recording_dir: Optional[Path] = None
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Load all annotations for a specific class from a port.

        Args:
            class_label: Class identifier in format 'groundtruth/ID' or 'predictions/ID'
            port: Port number
            recording_dir: Recording directory (used to validate video exists)

        Returns:
            Dict mapping frame_index -> list of (x_center, y_center, width, height)
            Coordinates are in normalized format (0-1 range)
        """
        annotations = {}

        # Parse class label format
        parts = class_label.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid class label format: {class_label}")
            return annotations

        anno_type, class_id_str = parts
        try:
            class_id = int(class_id_str)
        except ValueError:
            logger.error(f"Invalid class ID: {class_id_str}")
            return annotations

        # Determine annotation directory
        if anno_type == "groundtruth":
            anno_dir = self.workspace_guide.ground_truth_dir / f"port_{port}"
        elif anno_type == "predictions":
            anno_dir = self.workspace_guide.predictions_dir / f"port_{port}"
        else:
            logger.error(f"Unknown annotation type: {anno_type}")
            return annotations

        if not anno_dir.exists():
            logger.warning(f"Annotation directory not found: {anno_dir}")
            return annotations

        # Load all annotation files
        for anno_file in anno_dir.rglob("*.txt"):
            try:
                # Parse frame index from filename using shared utility
                frame_idx = extract_frame_index_from_filename(anno_file.name)
                if frame_idx is None:
                    continue

                with open(anno_file, "r") as f:
                    for line in f:
                        parsed = parse_yolo_line(line)
                        if parsed is None:
                            continue

                        file_class_id, x_center, y_center, width, height, _ = parsed
                        if file_class_id == class_id:
                            if frame_idx not in annotations:
                                annotations[frame_idx] = []
                            annotations[frame_idx].append((x_center, y_center, width, height))
            except Exception as e:
                logger.debug(f"Error loading annotations from {anno_file}: {e}")
                continue

        return annotations

    def get_frames_for_class(
        self, class_label: str, port: int, recording_dir: Optional[Path] = None
    ) -> List[int]:
        """
        Get all frame indices that contain annotations for a specific class.

        Args:
            class_label: Class identifier
            port: Port number
            recording_dir: Recording directory

        Returns:
            Sorted list of frame indices
        """
        annotations = self.load_annotations_for_class(class_label, port, recording_dir)
        return sorted(annotations.keys())

    def get_frames_with_annotations(
        self,
        class_label: str,
        port: int,
        recording_dir: Optional[Path] = None,
        num_frames: int = 3,
        area_ratio: Optional[float] = 3.0,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Get frames with annotation bounding boxes drawn evenly distributed throughout the video.

        If fewer frames than requested are available, returns only the available ones.

        Args:
            class_label: Class identifier
            port: Port number
            recording_dir: Recording directory (auto-detected if not provided)
            num_frames: Number of frames to display (default 3)
            area_ratio: Ratio of display area to bbox area for cropping (e.g., 3.0 = 3x area).
                       If None, returns full frame without cropping (default 3.0 for zoomed view)

        Returns:
            List of tuples (frame_array, frame_index) where frame_array is BGR numpy array,
            or empty list if visualization failed
        """
        # Find recording directory if not provided
        if recording_dir is None:
            recording_dir = self.find_recording_for_port(port)

        if recording_dir is None:
            logger.error(f"Could not find recording directory for port {port}")
            return []

        # Get video path
        video_path = recording_dir / f"port_{port}.mp4"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []

        # Load annotations
        annotations = self.load_annotations_for_class(class_label, port, recording_dir)
        if not annotations:
            logger.warning(f"No annotations found for class {class_label} on port {port}")
            return []

        # Get frames to display
        frame_indices = sorted(annotations.keys())
        frames_to_display = self._select_evenly_distributed_frames(frame_indices, num_frames)

        if not frames_to_display:
            logger.error("No frames available to display")
            return []

        # Extract frames with annotations drawn
        result_frames = []
        for frame_idx in frames_to_display:
            frame = self.extract_frame_from_video(video_path, frame_idx)
            if frame is None:
                logger.warning(f"Could not extract frame {frame_idx}")
                continue

            # Draw bounding boxes
            frame_h, frame_w = frame.shape[:2]
            frame_with_boxes = frame.copy()

            for x_center, y_center, width, height in annotations[frame_idx]:
                # Convert normalized coordinates to pixel coordinates
                x_pixel = int(x_center * frame_w)
                y_pixel = int(y_center * frame_h)
                w_pixel = int(width * frame_w)
                h_pixel = int(height * frame_h)

                # Calculate top-left corner of bbox
                x_top_left = x_pixel - w_pixel // 2
                y_top_left = y_pixel - h_pixel // 2

                # Draw bounding box (BGR format, green)
                cv2.rectangle(
                    frame_with_boxes,
                    (x_top_left, y_top_left),
                    (x_top_left + w_pixel, y_top_left + h_pixel),
                    (0, 255, 0),
                    2,
                )

            # Crop frame around first bounding box (if available)
            # Display area size controlled by area_ratio parameter
            if annotations[frame_idx] and area_ratio is not None:
                x_center, y_center, width, height = annotations[frame_idx][0]
                frame_with_boxes = self._crop_frame_around_bbox(
                    frame_with_boxes,
                    (x_center, y_center, width, height),
                    frame_shape=(frame_h, frame_w),
                    area_ratio=area_ratio,
                )

            result_frames.append((frame_with_boxes, frame_idx))

        return result_frames

    def _crop_frame_around_bbox(
        self,
        frame: np.ndarray,
        bbox_normalized: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        area_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Crop a frame around a bounding box with optional zoom.

        Args:
            frame: Frame to crop (BGR numpy array)
            bbox_normalized: Tuple of (x_center, y_center, width, height) in normalized 0-1 coordinates
            frame_shape: Tuple of (height, width) in pixels
            area_ratio: Ratio of display area to bbox area (e.g., 3.0 = 3x area).
                       If None, returns full frame. If 1.0, returns frame with just bbox.

        Returns:
            Cropped frame (numpy array)
        """
        if area_ratio is None or area_ratio <= 1.0:
            # Return full frame without cropping
            return frame

        frame_h, frame_w = frame_shape
        x_center, y_center, width, height = bbox_normalized

        # Convert normalized to pixel coordinates
        x_pixel = int(x_center * frame_w)
        y_pixel = int(y_center * frame_h)
        w_pixel = int(width * frame_w)
        h_pixel = int(height * frame_h)

        # Calculate crop size: area_ratio = (crop_w / w_pixel)^2
        # So scale_factor = sqrt(area_ratio)
        scale_factor = area_ratio ** 0.5
        crop_w = int(w_pixel * scale_factor)
        crop_h = int(h_pixel * scale_factor)

        # Center crop on bbox center
        crop_left = int(x_pixel - crop_w // 2)
        crop_top = int(y_pixel - crop_h // 2)
        crop_right = crop_left + crop_w
        crop_bottom = crop_top + crop_h

        # Clamp to frame boundaries
        crop_left = max(0, min(crop_left, frame_w - 1))
        crop_top = max(0, min(crop_top, frame_h - 1))
        crop_right = max(crop_left + 1, min(crop_right, frame_w))
        crop_bottom = max(crop_top + 1, min(crop_bottom, frame_h))

        # Return cropped frame
        return frame[crop_top:crop_bottom, crop_left:crop_right]

    def _select_evenly_distributed_frames(
        self, frame_indices: List[int], num_frames: int
    ) -> List[int]:
        """
        Select frames evenly distributed throughout the list.

        Args:
            frame_indices: Available frame indices
            num_frames: Number of frames to select

        Returns:
            List of selected frame indices
        """
        if not frame_indices:
            return []

        if len(frame_indices) <= num_frames:
            return frame_indices

        # Select evenly distributed indices
        selected = []
        for i in range(num_frames):
            idx = int(i * (len(frame_indices) - 1) / (num_frames - 1))
            selected.append(frame_indices[idx])

        return selected
