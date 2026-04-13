"""Metrics computation and display for motion tracking analysis."""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from calipod.core import logger as calipod_logger

logger = calipod_logger.get(__name__)


class MetricsComputer:
    """Handles performance metrics computation and display."""

    def __init__(self, parent_widget):
        """
        Initialize metrics computer.

        Args:
            parent_widget: PlaybackTriangulationWidget instance for UI operations and state access.
        """
        self.parent = parent_widget

    def compute_and_display(self, motion_trial, use_filtered: bool):
        """
        Compute metrics and display in a dialog (synchronous, no threading).

        Args:
            motion_trial: MotionTrial instance with predictions and ground truth.
            use_filtered: Whether to use filtered predictions.
        """
        # Validation
        if motion_trial is None or motion_trial.is_empty:
            QMessageBox.warning(self.parent, "No Data", "No motion trial loaded; cannot compute metrics.")
            return

        if not hasattr(motion_trial, "predictions_df") or motion_trial.predictions_df.empty:
            QMessageBox.warning(
                self.parent, "No Predictions", "Predictions are empty; load predictions before computing metrics."
            )
            return

        if not hasattr(motion_trial, "xyz_df") or motion_trial.xyz_df.empty:
            QMessageBox.warning(
                self.parent, "No Ground Truth", "Ground-truth xyz data are empty; cannot compute metrics."
            )
            return

        try:
            # Disable button during computation
            self.parent.compute_metrics_button.setEnabled(False)

            # Log which predictions are active
            pred_source = "filtered" if use_filtered else "raw"
            logger.info(f"Computing metrics with {pred_source} predictions...")

            # Compute metrics using internal calculation logic
            metrics = self._compute_metrics_from_dataframes(motion_trial.predictions_df, motion_trial.xyz_df)

            # Compute per-source metrics if available
            metrics_by_source = {}
            if (
                use_filtered
                and hasattr(motion_trial, "predictions_df")
                and "measurement_source" in motion_trial.predictions_df.columns
            ):
                for source in ["YOLO", "BGS", "filter_only"]:
                    source_df = motion_trial.predictions_df[motion_trial.predictions_df["measurement_source"] == source]
                    if not source_df.empty:
                        # Save originals
                        orig_pred_df = motion_trial.predictions_df.copy()
                        orig_gt_df = motion_trial.xyz_df.copy()

                        try:
                            # Filter to point_id == 0 and compute metrics
                            motion_trial.predictions_df = (
                                source_df[source_df["point_id"] == 0] if "point_id" in source_df.columns else source_df
                            )
                            motion_trial.xyz_df = (
                                orig_gt_df[orig_gt_df["point_id"] == 0] if not orig_gt_df.empty else orig_gt_df
                            )

                            source_metrics = self._compute_metrics_from_dataframes(
                                motion_trial.predictions_df, motion_trial.xyz_df
                            )
                            source_metrics["point_count"] = len(
                                source_df[source_df["point_id"] == 0] if "point_id" in source_df.columns else source_df
                            )
                            source_metrics["frame_count"] = source_df["sync_index"].nunique()
                            metrics_by_source[source] = source_metrics
                        except Exception as e:
                            logger.warning(f"Could not compute metrics for source {source}: {e}")
                        finally:
                            # Restore originals
                            motion_trial.predictions_df = orig_pred_df
                            motion_trial.xyz_df = orig_gt_df

            # Display the results
            self._display_metrics_dialog(metrics, metrics_by_source, motion_trial)

        except Exception as exc:
            logger.error(f"Error computing performance metrics: {exc}", exc_info=True)
            QMessageBox.critical(self.parent, "Metrics Error", f"Failed to compute metrics:\n{str(exc)}")
        finally:
            self.parent.compute_metrics_button.setEnabled(True)

    def _compute_metrics_from_dataframes(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
        """
        Computes performance metrics from predictions and ground truth dataframes.

        Metrics:
        - RMSE: Root Mean Square Error - squares errors so penalizes large errors more (mm)
        - MOTP: Multiple Object Tracking Precision - mean Euclidean distance error (mm)
        - MOTA: Multiple Object Tracking Accuracy - detection accuracy per frame (0-1 scale)
        - Median_Error: Median Euclidean distance error - robust to outliers (mm)

        Args:
            pred_df: Predictions dataframe with columns: sync_index, point_id, x_coord, y_coord, z_coord
            gt_df: Ground truth dataframe with columns: sync_index, point_id, x_coord, y_coord, z_coord

        Returns:
            Dictionary of computed metrics
        """
        if pred_df.empty or gt_df.empty:
            logger.debug("Empty predictions or ground truth; returning empty performance metrics.")
            return {}

        # Debug: Check data structure
        logger.info(f"xyz_df shape: {gt_df.shape}, columns: {list(gt_df.columns)[:5]}")
        logger.info(f"predictions_df shape: {pred_df.shape}, columns: {list(pred_df.columns)[:5]}")
        if "point_id" in gt_df.columns:
            logger.info(f"GT unique point_ids: {sorted(gt_df['point_id'].unique())}")
        if "point_id" in pred_df.columns:
            logger.info(f"Pred unique point_ids: {sorted(pred_df['point_id'].unique())}")

        # Merge ground truth and predictions on sync_index and point_id
        # Filters to only those points present in both ground truth and predictions (does not include false positives/negatives)
        merged_df = pd.merge(gt_df, pred_df, on=["sync_index", "point_id"], suffixes=("_gt", "_pred"))

        if merged_df.empty:
            logger.debug(
                "No matching points between ground truth and predictions; returning empty performance metrics."
            )
            return {}

        # Calculate RMSE (element-wise across all coordinates)
        rmse = np.sqrt(
            np.mean(
                (
                    merged_df[["x_coord_gt", "y_coord_gt", "z_coord_gt"]].values
                    - merged_df[["x_coord_pred", "y_coord_pred", "z_coord_pred"]].values
                )
                ** 2
            )
        )

        # Calculate Euclidean distances (errors) for each matched point
        distances = np.linalg.norm(
            merged_df[["x_coord_gt", "y_coord_gt", "z_coord_gt"]].values
            - merged_df[["x_coord_pred", "y_coord_pred", "z_coord_pred"]].values,
            axis=1,
        )

        # Calculate multiple object tracking precision (MOTP) - mean Euclidean error
        motp = np.mean(distances)

        # Median error - robust to outliers
        median_error = np.median(distances)

        # Calculate multiple object tracking accuracy (MOTA) - PER FRAME
        # MOTA measures detection accuracy, not distance accuracy
        # Filter GT to only point_ids that exist in predictions (to compare apples-to-apples)
        # Example: if predictions only track point_id 0, don't count missing point_ids 1-10 as false negatives
        relevant_point_ids = pred_df["point_id"].unique()
        gt_filtered = gt_df[gt_df["point_id"].isin(relevant_point_ids)]

        logger.info(
            f"Filtered GT from {len(gt_df)} to {len(gt_filtered)} rows (only point_ids: {sorted(relevant_point_ids)})"
        )

        # Count detections per frame (only for relevant point_ids)
        gt_counts_per_frame = gt_filtered.groupby("sync_index").size()
        pred_counts_per_frame = pred_df.groupby("sync_index").size()
        matched_counts_per_frame = merged_df.groupby("sync_index").size()

        # Get all unique frame indices
        all_frames = sorted(set(gt_counts_per_frame.index) | set(pred_counts_per_frame.index))

        # Calculate FN and FP per frame, then sum
        total_false_negatives = 0
        total_false_positives = 0
        total_gt_detections = 0

        for frame_idx in all_frames:
            gt_count = gt_counts_per_frame.get(frame_idx, 0)
            pred_count = pred_counts_per_frame.get(frame_idx, 0)
            matched_count = matched_counts_per_frame.get(frame_idx, 0)

            fn = gt_count - matched_count  # Missed detections in this frame
            fp = pred_count - matched_count  # Extra detections in this frame

            total_false_negatives += fn
            total_false_positives += fp
            total_gt_detections += gt_count

        # MOTA formula: 1 - (FN + FP + ID switches) / total_GT
        # ID switches not implemented yet
        mota = (
            1 - (total_false_negatives + total_false_positives) / total_gt_detections
            if total_gt_detections > 0
            else 0.0
        )

        # Convert distances from meters to millimeters
        rmse_mm = rmse * 1000
        motp_mm = motp * 1000
        median_error_mm = median_error * 1000

        metrics = {"RMSE_mm": rmse_mm, "MOTP_mm": motp_mm, "MOTA": mota, "Median_Error_mm": median_error_mm}

        logger.debug(f"Computed performance metrics: RMSE={rmse_mm:.2f}mm, MOTP={motp_mm:.2f}mm, MOTA={mota:.4f}")
        return metrics

    def _display_metrics_dialog(self, metrics: dict, metrics_by_source: dict, motion_trial=None):
        """Display computed metrics in a dialog (runs in main thread)."""
        if not metrics:
            QMessageBox.warning(self.parent, "No Metrics", "No metrics were computed.")
            return

        # Build metrics text
        lines = []

        pred_source = "filtered" if self.parent.toggle_filtered_button.isChecked() else "raw"
        lines.append(f"[Using {pred_source.upper()} predictions]")

        # Debug: Show extend and frame range info
        extend_enabled = getattr(self.parent, "extend_filtered_track", False)
        lines.append(f"[Extend filter: {'ON' if extend_enabled else 'OFF'}]")

        # Use motion_trial if available, otherwise fall back to parent widget state
        if motion_trial is not None:
            pred_min_frame = (
                motion_trial.predictions_df["sync_index"].min() if not motion_trial.predictions_df.empty else 0
            )
            pred_max_frame = (
                motion_trial.predictions_df["sync_index"].max() if not motion_trial.predictions_df.empty else 0
            )
            gt_min_frame = motion_trial.xyz_df["sync_index"].min() if not motion_trial.xyz_df.empty else 0
            gt_max_frame = motion_trial.xyz_df["sync_index"].max() if not motion_trial.xyz_df.empty else 0

            lines.append(
                f"[Pred frame range: {int(pred_min_frame)}-{int(pred_max_frame)} | GT frame range: {int(gt_min_frame)}-{int(gt_max_frame)}]"
            )
            lines.append("")
            lines.append("=== OVERALL METRICS ===")

            # Get start/end frame filtering from spinboxes
            start_frame = self.parent.filter_start_spin.value()
            end_frame = self.parent.filter_end_spin.value()

            # Show frame and point counts for both GT and predictions
            total_gt_frames = motion_trial.xyz_df["sync_index"].nunique() if not motion_trial.xyz_df.empty else 0
            # Only count GT points for point_id == 0 (the tracked fly)
            gt_fly_df = (
                motion_trial.xyz_df[motion_trial.xyz_df["point_id"] == 0]
                if not motion_trial.xyz_df.empty
                else pd.DataFrame()
            )
            total_gt_fly_points = len(gt_fly_df) if not gt_fly_df.empty else 0
            # Only count pred points for point_id == 0 (the tracked fly)
            pred_fly_df = (
                motion_trial.predictions_df[motion_trial.predictions_df["point_id"] == 0]
                if not motion_trial.predictions_df.empty
                else pd.DataFrame()
            )
            total_pred_frames = pred_fly_df["sync_index"].nunique() if not pred_fly_df.empty else 0
            total_pred_fly_points = len(pred_fly_df) if not pred_fly_df.empty else 0

            # Apply frame range filtering for percentage calculations
            gt_df_for_range = motion_trial.xyz_df
            if start_frame > 0 or end_frame > 0:
                if start_frame > 0:
                    gt_df_for_range = gt_df_for_range[gt_df_for_range["sync_index"] >= start_frame]
                if end_frame > 0:
                    gt_df_for_range = gt_df_for_range[gt_df_for_range["sync_index"] <= end_frame]

            # Count GT frames that actually have the tracked fly (point_id == 0)
            gt_fly_df_in_range = (
                gt_df_for_range[gt_df_for_range["point_id"] == 0] if not gt_df_for_range.empty else pd.DataFrame()
            )
            total_gt_frames_in_range = gt_fly_df_in_range["sync_index"].nunique() if not gt_fly_df_in_range.empty else 0

            # Display frame ranges
            if start_frame > 0 or end_frame > 0:
                range_str = f" (frames {start_frame}" if start_frame > 0 else " (all start"
                if end_frame > 0:
                    range_str += f" to {end_frame})"
                else:
                    range_str += " onward)"
                lines.append(f"Total GT Frames with fly point{range_str}: {int(total_gt_frames_in_range)}")
            else:
                lines.append(f"Total GT Frames with fly point: {int(total_gt_frames_in_range)}")

            lines.append(f"Total GT Fly Points (point_id=0): {int(total_gt_fly_points)}")
            lines.append(f"Total Pred Frames: {int(total_pred_frames)}")
            lines.append(f"Total Pred Fly Points (point_id=0): {int(total_pred_fly_points)}")
        else:
            lines.append("")
            lines.append("=== OVERALL METRICS ===")

        # Show performance metrics
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        # Add per-source metrics if available
        if metrics_by_source and motion_trial is not None:
            lines.append("")
            lines.append("=== METRICS BY MEASUREMENT SOURCE ===")
            # Add clarification about which GT frames are being used for percentages
            start_frame = self.parent.filter_start_spin.value()
            end_frame = self.parent.filter_end_spin.value()
            total_gt_frames_in_range = total_gt_frames_in_range if "total_gt_frames_in_range" in locals() else 0

            if start_frame > 0 or end_frame > 0:
                lines.append(
                    f"(% based on GT frames with fly point in range {start_frame}-{end_frame}: {int(total_gt_frames_in_range)} frames)"
                )
            else:
                lines.append(
                    f"(% based on GT frames with fly point in entire video: {int(total_gt_frames_in_range)} frames)"
                )

            total_pred_fly_points = total_pred_fly_points if "total_pred_fly_points" in locals() else 1
            total_pred_points = total_pred_fly_points if total_pred_fly_points > 0 else 1  # Avoid division by zero

            for source in ["YOLO", "BGS", "filter_only"]:
                if source in metrics_by_source:
                    source_metrics = metrics_by_source[source]
                    point_count = source_metrics.get("point_count", 0)
                    frame_count = source_metrics.get("frame_count", 0)
                    pct_of_pred = (point_count / total_pred_points * 100) if total_pred_points > 0 else 0
                    pct_of_frames = (
                        (frame_count / total_gt_frames_in_range * 100) if total_gt_frames_in_range > 0 else 0
                    )
                    lines.append(
                        f"\n--- {source} ({frame_count} frames, {point_count} points = {pct_of_pred:.1f}% of predictions, {pct_of_frames:.1f}% of GT frames) ---"
                    )
                    # Display accuracy metrics, skip context metrics (point_count, frame_count)
                    skip_keys = {"point_count", "frame_count"}
                    for key, value in source_metrics.items():
                        if key not in skip_keys:
                            if isinstance(value, float):
                                lines.append(f"  {key}: {value:.4f}")
                            else:
                                lines.append(f"  {key}: {value}")

        # Create scrollable dialog
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Performance Metrics")
        dialog.setGeometry(100, 100, 600, 500)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText("\n".join(lines))
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec()
