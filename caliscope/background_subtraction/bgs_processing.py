"""
Background Subtraction processing module for motion detection in video frames.

Implements adaptive background modeling with Gaussian statistics, incorporating
glare suppression and morphological operations for robust foreground detection.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PySide6.QtCore import QThread, Signal

import caliscope.logger

logger = caliscope.logger.get(__name__)


class BGSProcessor(QThread):
    """
    Background Subtraction processor that runs in a separate thread.
    
    Emits signals for progress updates and frame outputs during processing.
    """
    progress_updated = Signal(int, float)  # frame_number, timestamp
    frame_processed = Signal(np.ndarray)  # Processed frame for display
    processing_complete = Signal(str)  # Output file path
    processing_error = Signal(str)  # Error message
    
    def __init__(self, video_path: str, output_dir: str, alpha: float, n_sigma: float,
                 bright_cutoff: int, replacement: int, start_sec: float, end_sec: float,
                 opening_size: int = 0, warmup_secs: float = 5, bounding_box: tuple = None):
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
        """
        super().__init__()
        self.video_path = video_path
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
    
    def run(self):
        """Main processing loop"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit(f"Could not open video: {self.video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            progress_counter = 0
            
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
                                bg_var_pixels = mean_sq_im[contour_mask == 255] - mean_bg[contour_mask == 255]**2
                                
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
                
                # Write to output video
                out.write(display)
            
            cap.release()
            out.release()
            pbar.close()
            
            logger.info(f"Processing complete. Output saved to: {output_path}")
            self.processing_complete.emit(str(output_path))
            
        except Exception as e:
            logger.error(f"Error during BGS processing: {str(e)}")
            self.processing_error.emit(str(e))


def process_bgs_silent(video_path: str, output_dir: str, alpha: float, n_sigma: float,
                       bright_cutoff: int, replacement: int, start_sec: float, end_sec: float,
                       opening_size: int = 0, warmup_secs: float = 5) -> Path:
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                        bg_var_pixels = mean_sq_im[contour_mask == 255] - mean_bg[contour_mask == 255]**2
                        
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
        
        # Write to output video
        out.write(display)
    
    cap.release()
    out.release()
    pbar.close()
    
    logger.info(f"Processing complete: {output_path}")
    return output_path
