"""Reusable focus-analysis helpers for the Focus widget."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


FOCUS_RESULT_COLUMNS = [
    "video_name",
    "video_path",
    "frame_index",
    "roi",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "mean_brightness",
    "variance",
    "entropy",
    "canny_edges_count",
    "average_sobel_magnitude",
    "laplacian_zero_crossings",
    "timestamp",
]


@dataclass(slots=True)
class FocusAnalysisResult:
    video_name: str
    video_path: str
    frame_index: int
    roi: str
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    mean_brightness: float
    variance: float
    entropy: float
    canny_edges_count: int
    average_sobel_magnitude: float
    laplacian_zero_crossings: int
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "roi": self.roi,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "mean_brightness": self.mean_brightness,
            "variance": self.variance,
            "entropy": self.entropy,
            "canny_edges_count": self.canny_edges_count,
            "average_sobel_magnitude": self.average_sobel_magnitude,
            "laplacian_zero_crossings": self.laplacian_zero_crossings,
            "timestamp": self.timestamp,
        }


class LensFocusAnalyzer:
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "focus_results.csv"

    def load_results(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []

        data_frame = pd.read_csv(self.csv_path)
        if data_frame.empty:
            return []
        return data_frame.to_dict(orient="records")

    def write_results(self, results: list[dict[str, Any]]) -> None:
        data_frame = pd.DataFrame(results, columns=FOCUS_RESULT_COLUMNS)
        data_frame.to_csv(self.csv_path, index=False)

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        source_path: Path | str,
        frame_index: int,
    ) -> FocusAnalysisResult:
        x1, y1, x2, y2 = roi
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(x1 + 1, int(x2))
        y2 = max(y1 + 1, int(y2))

        height_limit, width_limit = frame_bgr.shape[:2]
        x1 = min(x1, width_limit - 1)
        y1 = min(y1, height_limit - 1)
        x2 = min(x2, width_limit)
        y2 = min(y2, height_limit)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Selected region is empty. Draw a larger rectangle before analyzing.")

        cropped = frame_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            raise ValueError("Selected region is empty. Draw a larger rectangle before analyzing.")

        grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(cv2.mean(grey)[0])

        _, stddev = cv2.meanStdDev(grey)
        variance = float(stddev[0, 0] ** 2)

        hist = cv2.calcHist([grey], [0], None, [256], [0, 256]).astype(np.float64)
        total_pixels = float(np.sum(hist))
        probabilities = hist / total_pixels if total_pixels > 0 else hist
        probabilities = np.where(probabilities > 0, probabilities, 1e-10)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        edges = cv2.Canny(grey, 1, 25)
        canny_edges_count = int(cv2.countNonZero(edges))

        sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
        average_sobel_magnitude = float(np.mean(cv2.magnitude(sobel_x, sobel_y)))

        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        laplacian_zero_crossings = int(np.count_nonzero(np.abs(np.diff(laplacian)) > 0.01))

        source_path = Path(source_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return FocusAnalysisResult(
            video_name=source_path.name,
            video_path=str(source_path),
            frame_index=int(frame_index),
            roi=f"({x1}, {y1})-({x2}, {y2})",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=int(x2 - x1),
            height=int(y2 - y1),
            mean_brightness=mean_brightness,
            variance=variance,
            entropy=entropy,
            canny_edges_count=canny_edges_count,
            average_sobel_magnitude=average_sobel_magnitude,
            laplacian_zero_crossings=laplacian_zero_crossings,
            timestamp=timestamp,
        )