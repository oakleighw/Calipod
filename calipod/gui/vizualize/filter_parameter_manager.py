"""Filter parameter management for tracking and hybrid filtering."""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd

from calipod.core import logger as calipod_logger
from calipod.gui.vizualize.metrics_computer import MetricsComputer

logger = calipod_logger.get(__name__)


class FilterParameterManager:
    """Manages tracking filter parameters (YOLO, BGS, hybrid) and their persistence."""
    
    def __init__(
        self,
        # YOLO filter parameters
        kalman_process_noise_scale: float = 1.0,
        kalman_measurement_noise_std: float = 0.01,
        gate_distance_sigma: float = 3.0,
        max_distance_threshold: float = 0.1,
        # BGS filter parameters
        bgs_kalman_process_noise_scale: float = 1.0,
        bgs_kalman_measurement_noise_std: float = 0.01,
        bgs_gate_distance_sigma: float = 3.0,
        bgs_max_distance_threshold: float = 0.1,
        # Flags
        gap_fill_only: bool = False,
        extend_filtered_track: bool = False,
    ):
        """Initialize filter parameters with default or provided values."""
        # YOLO parameters
        self.kalman_process_noise_scale = kalman_process_noise_scale
        self.kalman_measurement_noise_std = kalman_measurement_noise_std
        self.gate_distance_sigma = gate_distance_sigma
        self.max_distance_threshold = max_distance_threshold
        
        # BGS parameters
        self.bgs_kalman_process_noise_scale = bgs_kalman_process_noise_scale
        self.bgs_kalman_measurement_noise_std = bgs_kalman_measurement_noise_std
        self.bgs_gate_distance_sigma = bgs_gate_distance_sigma
        self.bgs_max_distance_threshold = bgs_max_distance_threshold
        
        # Flags
        self.gap_fill_only = gap_fill_only
        self.extend_filtered_track = extend_filtered_track
        
        # UI widget references (set via setters if needed)
        self.ui_widgets = {}
    
    def set_ui_widgets(self, **widgets):
        """Store references to UI widgets for updating."""
        self.ui_widgets.update(widgets)
    
    def update_gate_label(self, value: int):
        """Update gate distance label when slider changes."""
        sigma = value / 10.0
        if "gate_distance_label" in self.ui_widgets:
            self.ui_widgets["gate_distance_label"].setText(f"{sigma:.1f}σ")
    
    def update_bgs_gate_label(self, value: int):
        """Update BGS gate distance label when slider changes."""
        sigma = value / 10.0
        if "bgs_gate_distance_label" in self.ui_widgets:
            self.ui_widgets["bgs_gate_distance_label"].setText(f"{sigma:.1f}σ")
    
    def toggle_bgs_filter_row(self):
        """Enable/disable BGS filter parameter row based on hybrid checkbox state."""
        is_hybrid_enabled = False
        if "use_hybrid_bgs" in self.ui_widgets and self.ui_widgets["use_hybrid_bgs"] is not None:
            is_hybrid_enabled = self.ui_widgets["use_hybrid_bgs"].isChecked()
        
        # Enable/disable all BGS-related widgets and labels
        for widget_name in [
            "bgs_process_noise_spin",
            "bgs_meas_noise_spin",
            "bgs_gate_distance_slider",
            "bgs_max_distance_spin",
            "bgs_gate_distance_label",
        ]:
            if widget_name in self.ui_widgets:
                self.ui_widgets[widget_name].setEnabled(is_hybrid_enabled)
    
    def save_metadata(self, filtered_csv_path: Path, motion_trial=None):
        """Save filter parameters as JSON metadata alongside the filtered predictions CSV."""
        # Compute performance metrics for the filtered predictions
        overall_metrics = {}
        metrics_by_source = {}
        
        try:
            if motion_trial is not None and not motion_trial.is_empty:
                computer = MetricsComputer(None)
                metrics = computer._compute_metrics_from_dataframes(motion_trial.predictions_df, motion_trial.xyz_df)
                # Convert numpy types to native Python types for JSON serialization
                overall_metrics = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()
                }
                
                # Compute separate metrics for each measurement source
                if filtered_csv_path.exists():
                    filtered_df = pd.read_csv(filtered_csv_path, engine="pyarrow")
                    if "measurement_source" in filtered_df.columns:
                        for source in ["YOLO", "BGS", "filter_only"]:
                            source_df = filtered_df[filtered_df["measurement_source"] == source]
                            if not source_df.empty:
                                # Temporarily swap in source data to compute metrics
                                orig_pred_df = motion_trial.predictions_df
                                orig_gt_df = motion_trial.xyz_df
                                # Filter both to only point_id == 0 (the tracked fly)
                                motion_trial.predictions_df = (
                                    source_df[source_df["point_id"] == 0]
                                    if "point_id" in source_df.columns
                                    else source_df
                                )
                                motion_trial.xyz_df = (
                                    orig_gt_df[orig_gt_df["point_id"] == 0]
                                    if not orig_gt_df.empty
                                    else orig_gt_df
                                )
                                try:
                                    source_metrics = computer._compute_metrics_from_dataframes(motion_trial.predictions_df, motion_trial.xyz_df)
                                    # Convert numpy types to native Python types
                                    source_data = {
                                        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                        for k, v in source_metrics.items()
                                    }
                                    # Add point and frame counts for context
                                    source_data["point_count"] = int(
                                        len(
                                            source_df[source_df["point_id"] == 0]
                                            if "point_id" in source_df.columns
                                            else source_df
                                        )
                                    )
                                    source_data["frame_count"] = int(source_df["sync_index"].nunique())
                                    metrics_by_source[source] = source_data
                                except:
                                    pass
                                finally:
                                    motion_trial.predictions_df = orig_pred_df
                                    motion_trial.xyz_df = orig_gt_df
        except Exception as e:
            logger.debug(f"Could not compute performance metrics for metadata: {e}")
        
        # Get frame and point counts
        total_gt_frames = 0
        total_gt_fly_points = 0
        total_pred_frames = 0
        total_pred_fly_points = 0
        
        if motion_trial is not None and not motion_trial.is_empty:
            total_gt_frames = (
                motion_trial.xyz_df["sync_index"].nunique() if not motion_trial.xyz_df.empty else 0
            )
            gt_fly_df = (
                motion_trial.xyz_df[motion_trial.xyz_df["point_id"] == 0]
                if not motion_trial.xyz_df.empty
                else pd.DataFrame()
            )
            total_gt_fly_points = len(gt_fly_df) if not gt_fly_df.empty else 0
            pred_fly_df = (
                motion_trial.predictions_df[motion_trial.predictions_df["point_id"] == 0]
                if not motion_trial.predictions_df.empty
                else pd.DataFrame()
            )
            total_pred_frames = pred_fly_df["sync_index"].nunique() if not pred_fly_df.empty else 0
            total_pred_fly_points = len(pred_fly_df) if not pred_fly_df.empty else 0
        
        # Get hybrid BGS checkbox state
        use_hybrid_bgs = False
        if "use_hybrid_bgs" in self.ui_widgets and self.ui_widgets["use_hybrid_bgs"] is not None:
            use_hybrid_bgs = self.ui_widgets["use_hybrid_bgs"].isChecked()
        
        # Read filter frame range from spinboxes
        start_frame_to_save = 0
        end_frame_to_save = 0
        if "filter_start_spin" in self.ui_widgets:
            start_frame_to_save = self.ui_widgets["filter_start_spin"].value()
        if "filter_end_spin" in self.ui_widgets:
            end_frame_to_save = self.ui_widgets["filter_end_spin"].value()
        logger.debug(f"Saving filter frame range to metadata: start={start_frame_to_save}, end={end_frame_to_save}")
        
        metadata = {
            "kalman_process_noise_scale": float(self.kalman_process_noise_scale),
            "kalman_measurement_noise_std_mm": float(self.kalman_measurement_noise_std * 1000.0),
            "gate_distance_sigma": float(self.gate_distance_sigma),
            "max_distance_threshold_mm": float(self.max_distance_threshold),
            "bgs_kalman_process_noise_scale": float(self.bgs_kalman_process_noise_scale),
            "bgs_kalman_measurement_noise_std_mm": float(self.bgs_kalman_measurement_noise_std * 1000.0),
            "bgs_gate_distance_sigma": float(self.bgs_gate_distance_sigma),
            "bgs_max_distance_threshold_mm": float(self.bgs_max_distance_threshold),
            "gap_fill_only": bool(self.gap_fill_only),
            "extend_filtered_track": bool(self.extend_filtered_track),
            "use_hybrid_bgs": bool(use_hybrid_bgs),
            "filter_start_frame": int(start_frame_to_save),
            "filter_end_frame": int(end_frame_to_save),
            "total_ground_truth_frames": int(total_gt_frames),
            "total_ground_truth_fly_points": int(total_gt_fly_points),
            "total_prediction_frames": int(total_pred_frames),
            "total_prediction_fly_points": int(total_pred_fly_points),
            "performance_metrics": overall_metrics,
            "performance_metrics_by_source": metrics_by_source,
        }
        metadata_path = filtered_csv_path.with_stem(filtered_csv_path.stem + "_metadata")
        metadata_path = metadata_path.with_suffix(".json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved filter metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save filter metadata: {e}")
    
    def load_metadata(self, filtered_pred_path: Optional[Path]):
        """Load filter metadata from JSON file if available and update UI."""
        if filtered_pred_path is None:
            return
        
        metadata_path = filtered_pred_path.with_stem(filtered_pred_path.stem + "_metadata")
        metadata_path = metadata_path.with_suffix(".json")
        
        if not metadata_path.exists():
            logger.debug(f"No filter metadata found at {metadata_path}; using software defaults")
            return
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Load YOLO filter parameters
            self.kalman_process_noise_scale = metadata.get(
                "kalman_process_noise_scale", self.kalman_process_noise_scale
            )
            self.kalman_measurement_noise_std = (
                metadata.get("kalman_measurement_noise_std_mm", self.kalman_measurement_noise_std * 1000.0)
                / 1000.0
            )
            self.gate_distance_sigma = metadata.get("gate_distance_sigma", self.gate_distance_sigma)
            self.max_distance_threshold = metadata.get(
                "max_distance_threshold_mm", self.max_distance_threshold
            )
            
            # Load BGS filter parameters
            self.bgs_kalman_process_noise_scale = metadata.get(
                "bgs_kalman_process_noise_scale", self.bgs_kalman_process_noise_scale
            )
            self.bgs_kalman_measurement_noise_std = (
                metadata.get("bgs_kalman_measurement_noise_std_mm", self.bgs_kalman_measurement_noise_std * 1000.0)
                / 1000.0
            )
            self.bgs_gate_distance_sigma = metadata.get(
                "bgs_gate_distance_sigma", self.bgs_gate_distance_sigma
            )
            self.bgs_max_distance_threshold = metadata.get(
                "bgs_max_distance_threshold_mm", self.bgs_max_distance_threshold
            )
            
            # Load boolean flags
            self.gap_fill_only = metadata.get("gap_fill_only", self.gap_fill_only)
            self.extend_filtered_track = metadata.get("extend_filtered_track", self.extend_filtered_track)
            
            # Load frame range parameters
            start_frame = metadata.get("filter_start_frame", 0)
            end_frame = metadata.get("filter_end_frame", 0)
            logger.debug(f"Loaded filter frame range from metadata: start={start_frame}, end={end_frame}")
            
            # Update UI controls to reflect loaded values
            if "filter_start_spin" in self.ui_widgets:
                logger.debug(f"Setting filter_start_spin to {start_frame}")
                self.ui_widgets["filter_start_spin"].setValue(start_frame)
            else:
                logger.debug("WARNING: filter_start_spin not in ui_widgets during metadata load")
            if "filter_end_spin" in self.ui_widgets:
                logger.debug(f"Setting filter_end_spin to {end_frame}")
                self.ui_widgets["filter_end_spin"].setValue(end_frame)
            else:
                logger.debug("WARNING: filter_end_spin not in ui_widgets during metadata load")
            if "meas_noise_spin" in self.ui_widgets:
                self.ui_widgets["meas_noise_spin"].setValue(self.kalman_measurement_noise_std * 1000.0)
            if "gate_distance_slider" in self.ui_widgets:
                self.ui_widgets["gate_distance_slider"].setValue(int(self.gate_distance_sigma * 10))
            if "max_distance_spin" in self.ui_widgets:
                self.ui_widgets["max_distance_spin"].setValue(self.max_distance_threshold)
            
            if "bgs_process_noise_spin" in self.ui_widgets:
                self.ui_widgets["bgs_process_noise_spin"].setValue(self.bgs_kalman_process_noise_scale)
            if "bgs_meas_noise_spin" in self.ui_widgets:
                self.ui_widgets["bgs_meas_noise_spin"].setValue(self.bgs_kalman_measurement_noise_std * 1000.0)
            if "bgs_gate_distance_slider" in self.ui_widgets:
                self.ui_widgets["bgs_gate_distance_slider"].setValue(int(self.bgs_gate_distance_sigma * 10))
            if "bgs_max_distance_spin" in self.ui_widgets:
                self.ui_widgets["bgs_max_distance_spin"].setValue(self.bgs_max_distance_threshold)
            
            if "filter_start_spin" in self.ui_widgets:
                self.ui_widgets["filter_start_spin"].setValue(start_frame)
            if "filter_end_spin" in self.ui_widgets:
                self.ui_widgets["filter_end_spin"].setValue(end_frame)
            
            if "gap_fill_only_checkbox" in self.ui_widgets:
                self.ui_widgets["gap_fill_only_checkbox"].setChecked(self.gap_fill_only)
            if "extend_filtered_track_checkbox" in self.ui_widgets:
                self.ui_widgets["extend_filtered_track_checkbox"].setChecked(self.extend_filtered_track)
            
            # Load use_hybrid_bgs state and apply it
            use_hybrid_bgs = metadata.get("use_hybrid_bgs", False)
            if "use_hybrid_bgs" in self.ui_widgets and self.ui_widgets["use_hybrid_bgs"] is not None:
                self.ui_widgets["use_hybrid_bgs"].setChecked(use_hybrid_bgs)
            
            # Apply the hybrid state to enable/disable BGS controls
            self.toggle_bgs_filter_row()
            
            logger.info(f"Loaded filter metadata from {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to load filter metadata from {metadata_path}: {e}; using software defaults")


