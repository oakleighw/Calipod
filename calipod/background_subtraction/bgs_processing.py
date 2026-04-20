"""
Background Subtraction processing module for motion detection in video frames.

Implements adaptive background modeling with Gaussian statistics, incorporating
glare suppression and morphological operations for robust foreground detection.
"""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from tqdm import tqdm

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)


class BGSProcessor(QThread):
    """
    Background Subtraction processor that runs in a separate thread.

    Emits signals for progress updates and frame outputs during processing.
    """

    progress_updated = Signal(int, float)  # frame_number, timestamp
    frame_processed = Signal(np.ndarray)  # Processed frame for display
    processing_complete = Signal(str)  # Output file path
    processing_error = Signal(str)  # Error message

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        alpha: float,
        n_sigma: float,
        bright_cutoff: int,
        replacement: int,
        start_sec: float,
        end_sec: float,
        opening_size: int = 0,
        warmup_secs: float = 5,
        bounding_box: tuple = None,
        show_ground_truth: bool = False,
        annotations_dir: Path = None,
        save_detections: bool = False,
    ):
        """
        Initialize the BGS processor.

        Parameters:
            video_path: Path to input video file
            output_dir: Directory to save processed video
            alpha: Learning rate for background adaptation (0.0-1.0)
            n_sigma: Sensitivity threshold multiplier (typically 3.0-6.0)
            bright_cutoff: Brightness threshold for glare suppression (0-255)
            replacement: Value to replace suppressed pixels (typically 0)
            start_sec: Start time of analysis window (seconds)
            end_sec: End time of analysis window (seconds)
            opening_size: Morphological kernel size for noise reduction
            warmup_secs: Seconds before analysis window for model initialization
            bounding_box: Tuple of (x1, y1, x2, y2) in pixel coordinates, or None for full frame
            show_ground_truth: Whether to overlay ground truth annotations
            annotations_dir: Directory containing YOLO annotation files
            save_detections: Whether to save detected centroids in YOLO format
        """
        super().__init__()
        self.video_path = video_path
        self.show_ground_truth = show_ground_truth
        self.annotations_dir = annotations_dir
        self.save_detections = save_detections
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.n_sigma = n_sigma
        self.bright_cutoff = bright_cutoff
        self.replacement = replacement
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.opening_size = opening_size
        self.warmup_secs = warmup_secs
        self.bounding_box = bounding_box

        self.kernel = np.ones((opening_size, opening_size), np.uint8) if (opening_size and opening_size > 0) else None
        self._is_running = True

    def stop(self):
        """Signal the processor to stop"""
        self._is_running = False

    def _get_ground_truth_centroid(
        self, frame_idx: int, x1: int, y1: int, x2: int, y2: int, target_class: int = 0
    ) -> tuple:
        """
        Extract ground truth centroid from YOLO label file for a specific frame.

        Parameters:
            frame_idx: Frame index in the video
            x1, y1, x2, y2: Bounding box in pixel coordinates
            target_class: YOLO class ID to extract (default 0 for fruit)

        Returns:
            Centroid as (x, y) in ROI coordinates, or None if not found
        """
        if not self.annotations_dir:
            return None

        # Extract port number from video filename
        import re

        video_name = Path(self.video_path).stem
        match = re.search(r"port_(\d+)", video_name)
        if not match:
            return None

        port = match.group(1)

        # Build label file path
        label_file = self.annotations_dir / f"port_{port}" / "labels" / "train" / f"frame_{frame_idx:06d}.txt"

        if not label_file.exists():
            return None

        try:
            # Get frame dimensions
            cap = cv2.VideoCapture(self.video_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                data = line.split()
                if not data:
                    continue

                class_id = int(data[0])
                if class_id == target_class:
                    # YOLO format: class_id center_x center_y width height (normalized 0-1)
                    gx_px = float(data[1]) * frame_w
                    gy_px = float(data[2]) * frame_h

                    # Check if centroid is within bounding box
                    if x1 <= gx_px <= x2 and y1 <= gy_px <= y2:
                        # Return coordinates relative to ROI
                        return (int(gx_px - x1), int(gy_px - y1))
        except Exception as e:
            logger.warning(f"Error extracting ground truth from {label_file}: {e}")

        return None

    def _save_detections_yolo(self, detections: list, frame_w: int, frame_h: int):
        """
        Save detected centroids in YOLO format with true frame numbers.
        Organizes labels in port-specific directories to avoid conflicts.

        Parameters:
            detections: List of (frame_idx, x_pixel, y_pixel) tuples in full frame coordinates
            frame_w, frame_h: Original video dimensions for normalization
        """
        try:
            # Extract port number from video filename (e.g., "port_1.mp4" -> "1")
            import re

            video_name = Path(self.video_path).stem
            match = re.search(r"port_(\d+)", video_name)
            port_num = match.group(1) if match else "0"

            # Create port-specific output directory structure
            # FLY/bgs/port_X/labels/train/
            labels_dir = self.output_dir / f"port_{port_num}" / "labels" / "train"
            labels_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx, det_x, det_y in detections:
                # Normalize coordinates to 0-1 range
                norm_x = det_x / frame_w
                norm_y = det_y / frame_h

                # Clamp to valid range (in case of edge cases)
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                # YOLO format: class_id center_x center_y width height
                # For point detections, use fixed small width/height
                label_file = labels_dir / f"frame_{frame_idx:06d}.txt"

                with open(label_file, "w") as f:
                    # class_id=0 for BGS detections, fixed width/height of 0.1
                    f.write(f"0 {norm_x:.6f} {norm_y:.6f} 0.1 0.1\n")

            logger.info(f"Saved {len(detections)} detection labels to {labels_dir}")

        except Exception as e:
            logger.warning(f"Error saving detections in YOLO format: {e}")

    def run(self):
        """Main processing loop"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit(f"Could not open video: {self.video_path}")
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Determine output frame dimensions
            if self.bounding_box:
                x1, y1, x2, y2 = self.bounding_box
                out_w = x2 - x1
                out_h = y2 - y1
            else:
                out_w = frame_w
                out_h = frame_h
                self.bounding_box = (0, 0, frame_w, frame_h)

            logger.info(f"Processing video: {self.video_path}")
            logger.info(f"Input resolution: {frame_w}x{frame_h}, Output resolution: {out_w}x{out_h}, FPS: {fps}")
            logger.info(f"Bounding box: {self.bounding_box}")

            # Calculate frame indices
            warmup_start = max(0, int((self.start_sec - self.warmup_secs) * fps))
            analysis_start = int(self.start_sec * fps)
            end_frame = int(self.end_sec * fps)

            logger.info(
                "Frame calculations: "
                f"warmup_start={warmup_start}, analysis_start={analysis_start}, end_frame={end_frame}"
            )
            logger.info(f"Expected total frames: {end_frame - warmup_start} (including warmup)")

            # Initialize background model
            cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
            ret, frame = cap.read()
            if not ret:
                self.processing_error.emit("Could not read first frame")
                return

            # Crop to bounding box
            x1, y1, x2, y2 = self.bounding_box
            cropped_frame = frame[y1:y2, x1:x2]

            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean_bg = gray_frame.copy()
            mean_sq_im = np.square(gray_frame)

            # Set up output video writer
            output_path = self.output_dir / f"{Path(self.video_path).stem}_bgs.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Note: output video will only contain analysis frames, not warmup frames
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

            logger.info(f"Output will be saved to: {output_path}")

            # Process frames
            pbar = tqdm(total=(end_frame - warmup_start), desc="BGS Processing", unit="fr")
            frame_idx = warmup_start

            # Storage for detections (if saving enabled)
            detections = []  # List of (frame_idx, x_pixel, y_pixel) tuples

            while cap.isOpened() and frame_idx <= end_frame and self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                pbar.update(1)

                # Crop frame to bounding box
                x1, y1, x2, y2 = self.bounding_box
                cropped_frame = frame[y1:y2, x1:x2]

                curr_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

                # Update running background statistics
                cv2.accumulateWeighted(curr_gray, mean_bg, self.alpha)
                cv2.accumulateWeighted(np.square(curr_gray), mean_sq_im, self.alpha)

                # Always process and write frames (warmup and analysis)
                # Compute standard deviation and create foreground mask
                std_dev = np.sqrt(np.abs(mean_sq_im - np.square(mean_bg)))
                diff = np.abs(curr_gray - mean_bg)
                mask = (diff > (self.n_sigma * std_dev)).astype(np.uint8) * 255

                # Suppress detections in bright areas to avoid glare artifacts
                mask[mean_bg > self.bright_cutoff] = self.replacement

                # Apply morphological opening to remove small noise
                if self.kernel is not None:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

                # Find the most statistically significant contour (only for analysis phase)
                det_point = None
                if frame_idx >= analysis_start:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        fly_candidates = [c for c in contours if 2 < cv2.contourArea(c) < 100]

                        if fly_candidates:
                            # Calculate statistical significance for each candidate
                            def contour_significance(contour):
                                contour_mask = np.zeros_like(mask, dtype=np.uint8)
                                cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                                contour_pixels = curr_gray[contour_mask == 255]
                                bg_mean_pixels = mean_bg[contour_mask == 255]
                                bg_var_pixels = mean_sq_im[contour_mask == 255] - mean_bg[contour_mask == 255] ** 2

                                if len(contour_pixels) == 0:
                                    return 0.0

                                pixel_deviations = np.abs(contour_pixels - bg_mean_pixels)
                                bg_std = np.sqrt(np.maximum(bg_var_pixels, 1e-6))
                                z_scores = pixel_deviations / bg_std
                                return np.median(z_scores)

                            target_contour = max(fly_candidates, key=contour_significance)
                            M = cv2.moments(target_contour)
                            if M["m00"] > 5:
                                det_x_roi = int(M["m10"] / M["m00"])
                                det_y_roi = int(M["m01"] / M["m00"])
                                det_point = (det_x_roi, det_y_roi)

                                # Store detection if saving is enabled (convert ROI to full frame coords)
                                if self.save_detections:
                                    x1, y1, x2, y2 = self.bounding_box
                                    det_x_full = det_x_roi + x1
                                    det_y_full = det_y_roi + y1
                                    detections.append((frame_idx, det_x_full, det_y_full))

                # Create visualization
                display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                timestamp = frame_idx / fps

                if det_point:
                    cv2.circle(display, det_point, 6, (0, 255, 0), -1)

                # Draw ground truth if enabled
                if self.show_ground_truth and self.annotations_dir:
                    gt_point = self._get_ground_truth_centroid(frame_idx, x1, y1, x2, y2)
                    if gt_point:
                        cv2.drawMarker(display, gt_point, (255, 255, 0), cv2.MARKER_CROSS, 15, 2)

                # Add frame label indicating warmup or analysis phase
                if frame_idx < analysis_start:
                    phase_label = "WARMUP"
                    label_color = (0, 165, 255)  # Orange for warmup
                else:
                    phase_label = "SIGNAL"
                    label_color = (0, 255, 0)  # Green for analysis

                text_label = f"Frame: {frame_idx} ({timestamp:.1f}s) [{phase_label}]"
                cv2.putText(display, text_label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Write all frames (warmup + analysis) to output video
                out.write(display)

            cap.release()
            out.release()
            pbar.close()

            logger.info(f"Processing complete. Output saved to: {output_path}")
            logger.info(
                "Output video contains approximately "
                f"{frame_idx - warmup_start} frames "
                f"(from frame {warmup_start} to {frame_idx})"
            )

            # Save detections in YOLO format if enabled
            if self.save_detections and detections:
                self._save_detections_yolo(detections, frame_w, frame_h)
                logger.info(f"Saved {len(detections)} detections in YOLO format")
            elif self.save_detections:
                logger.info("Save detections enabled but no detections found during processing")

            # Save metadata file
            metadata_path = output_path.with_name(f"{output_path.stem}_metadata.txt")
            try:
                with open(metadata_path, "w") as f:
                    f.write("BGS Processing Parameters\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Video: {Path(self.video_path).name}\n")
                    f.write(f"Output: {output_path.name}\n\n")
                    f.write("Processing Parameters:\n")
                    f.write(f"  Alpha (learning rate): {self.alpha}\n")
                    f.write(f"  N-Sigma (sensitivity): {self.n_sigma}\n")
                    f.write(f"  Bright Cutoff: {self.bright_cutoff}\n")
                    f.write(f"  Replacement: {self.replacement}\n")
                    f.write(
                        "  Opening Size: "
                        f"{self.opening_size if self.opening_size and self.opening_size > 0 else 'None (disabled)'}\n"
                    )
                    f.write(f"  Warmup Duration: {self.warmup_secs} seconds\n\n")
                    f.write("Timing:\n")
                    f.write(f"  Start (analysis): {self.start_sec} seconds (frame {analysis_start})\n")
                    f.write(f"  End (analysis): {self.end_sec} seconds (frame {end_frame})\n")
                    f.write(f"  Warmup Start: {self.start_sec - self.warmup_secs} seconds (frame {warmup_start})\n\n")
                    f.write("Output:\n")
                    f.write(f"  Video FPS: {fps}\n")
                    f.write(f"  Total Frames Written: {frame_idx - warmup_start}\n")
                    f.write(f"  Output Resolution: {out_w}x{out_h}\n")
                    if self.bounding_box != (0, 0, frame_w, frame_h):
                        f.write(
                            "  Bounding Box (Region): "
                            f"({self.bounding_box[0]}, {self.bounding_box[1]}, "
                            f"{self.bounding_box[2]}, {self.bounding_box[3]})\n"
                        )
                    f.write("\n")
                    from datetime import datetime

                    f.write(f"Processing Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                logger.info(f"Metadata saved to: {metadata_path}")
            except Exception as e:
                logger.warning(f"Could not save metadata file: {e}")

            self.processing_complete.emit(str(output_path))

        except Exception as e:
            logger.error(f"Error during BGS processing: {str(e)}")
            self.processing_error.emit(str(e))


def process_bgs_silent(
    video_path: str,
    output_dir: str,
    alpha: float,
    n_sigma: float,
    bright_cutoff: int,
    replacement: int,
    start_sec: float,
    end_sec: float,
    opening_size: int = 0,
    warmup_secs: float = 5,
) -> Path:
    """
    Synchronous BGS processing without threading (for non-GUI use).

    Returns the path to the output video.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = np.ones((opening_size, opening_size), np.uint8) if (opening_size and opening_size > 0) else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Processing: {video_path} ({frame_w}x{frame_h}, {fps}fps)")

    warmup_start = max(0, int((start_sec - warmup_secs) * fps))
    analysis_start = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_bg = gray_frame.copy()
    mean_sq_im = np.square(gray_frame)

    output_path = output_dir / f"{Path(video_path).stem}_bgs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    pbar = tqdm(total=(end_frame - warmup_start), desc="BGS Processing", unit="fr")
    frame_idx = warmup_start

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cv2.accumulateWeighted(curr_gray, mean_bg, alpha)
        cv2.accumulateWeighted(np.square(curr_gray), mean_sq_im, alpha)

        # Always process and write frames (warmup and analysis)
        # Compute standard deviation and create foreground mask
        std_dev = np.sqrt(np.abs(mean_sq_im - np.square(mean_bg)))
        diff = np.abs(curr_gray - mean_bg)
        mask = (diff > (n_sigma * std_dev)).astype(np.uint8) * 255

        # Suppress detections in bright areas to avoid glare artifacts
        mask[mean_bg > bright_cutoff] = replacement

        # Apply morphological opening to remove small noise
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find the most statistically significant contour (only for analysis phase)
        det_point = None
        if frame_idx >= analysis_start:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                fly_candidates = [c for c in contours if 2 < cv2.contourArea(c) < 100]

                if fly_candidates:

                    def contour_significance(contour):
                        contour_mask = np.zeros_like(mask, dtype=np.uint8)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                        contour_pixels = curr_gray[contour_mask == 255]
                        bg_mean_pixels = mean_bg[contour_mask == 255]
                        bg_var_pixels = mean_sq_im[contour_mask == 255] - mean_bg[contour_mask == 255] ** 2

                        if len(contour_pixels) == 0:
                            return 0.0

                        pixel_deviations = np.abs(contour_pixels - bg_mean_pixels)
                        bg_std = np.sqrt(np.maximum(bg_var_pixels, 1e-6))
                        z_scores = pixel_deviations / bg_std
                        return np.median(z_scores)

                    target_contour = max(fly_candidates, key=contour_significance)
                    M = cv2.moments(target_contour)
                    if M["m00"] > 5:
                        det_point = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Create visualization
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        timestamp = frame_idx / fps

        if det_point:
            cv2.circle(display, det_point, 6, (0, 255, 0), -1)

        # Add frame label indicating warmup or analysis phase
        if frame_idx < analysis_start:
            phase_label = "WARMUP"
            label_color = (0, 165, 255)  # Orange for warmup
        else:
            phase_label = "SIGNAL"
            label_color = (0, 255, 0)  # Green for analysis

        text_label = f"Frame: {frame_idx} ({timestamp:.1f}s) [{phase_label}]"
        cv2.putText(display, text_label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # Write all frames (warmup + analysis) to output video
        out.write(display)

    cap.release()
    out.release()
    pbar.close()

    logger.info(f"Processing complete: {output_path}")
    return output_path
