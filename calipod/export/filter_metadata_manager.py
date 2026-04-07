"""Filter metadata persistence for Kalman filtering parameters and performance metrics."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import calipod.logger
from calipod.gui.vizualize.metrics_computer import MetricsComputer

logger = calipod.logger.get(__name__)


class FilterMetadataManager:
    """Manages persistence of filter parameters and performance metrics as JSON."""
    
    @staticmethod
    def save_filter_metadata(
        filtered_csv_path: Path,
        motion_trial,
        kalman_process_noise_scale: float,
        kalman_measurement_noise_std: float,
        gate_distance_sigma: float,
        max_distance_threshold: float,
        bgs_kalman_process_noise_scale: float,
        bgs_kalman_measurement_noise_std: float,
        bgs_gate_distance_sigma: float,
        bgs_max_distance_threshold: float,
        gap_fill_only: bool,
        extend_filtered_track: bool,
        filter_start_frame: int,
        filter_end_frame: int,
    ):
        """
        Save filter parameters and performance metrics as JSON metadata.
        
        Args:
            filtered_csv_path: Path to filtered predictions CSV
            motion_trial: MotionTrial object with xyz_df and predictions_df
            kalman_process_noise_scale: Process noise parameter
            kalman_measurement_noise_std: Measurement noise in meters
            gate_distance_sigma: Mahalanobis gating threshold
            max_distance_threshold: Maximum distance threshold in mm
            bgs_kalman_process_noise_scale: BGS process noise
            bgs_kalman_measurement_noise_std: BGS measurement noise in meters
            bgs_gate_distance_sigma: BGS gating threshold
            bgs_max_distance_threshold: BGS max distance in mm
            gap_fill_only: Whether gap-fill-only mode was used
            extend_filtered_track: Whether track extension was enabled
            filter_start_frame: Start frame for filtering
            filter_end_frame: End frame for filtering
        """
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
                    if 'measurement_source' in filtered_df.columns:
                        for source in ['YOLO', 'BGS', 'filter_only']:
                            source_df = filtered_df[filtered_df['measurement_source'] == source]
                            if not source_df.empty:
                                # Temporarily swap in source data
                                orig_pred_df = motion_trial.predictions_df
                                orig_gt_df = motion_trial.xyz_df
                                
                                motion_trial.predictions_df = (
                                    source_df[source_df['point_id'] == 0] 
                                    if 'point_id' in source_df.columns else source_df
                                )
                                motion_trial.xyz_df = (
                                    orig_gt_df[orig_gt_df['point_id'] == 0] 
                                    if not orig_gt_df.empty else orig_gt_df
                                )
                                try:
                                    source_metrics = computer._compute_metrics_from_dataframes(motion_trial.predictions_df, motion_trial.xyz_df)
                                    source_data = {
                                        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                                        for k, v in source_metrics.items()
                                    }
                                    source_data['point_count'] = int(
                                        len(source_df[source_df['point_id'] == 0] 
                                            if 'point_id' in source_df.columns else source_df)
                                    )
                                    source_data['frame_count'] = int(source_df['sync_index'].nunique())
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
                motion_trial.xyz_df['sync_index'].nunique() 
                if not motion_trial.xyz_df.empty else 0
            )
            gt_fly_df = (
                motion_trial.xyz_df[motion_trial.xyz_df['point_id'] == 0] 
                if not motion_trial.xyz_df.empty else pd.DataFrame()
            )
            total_gt_fly_points = len(gt_fly_df) if not gt_fly_df.empty else 0
            pred_fly_df = (
                motion_trial.predictions_df[motion_trial.predictions_df['point_id'] == 0] 
                if not motion_trial.predictions_df.empty else pd.DataFrame()
            )
            total_pred_frames = pred_fly_df['sync_index'].nunique() if not pred_fly_df.empty else 0
            total_pred_fly_points = len(pred_fly_df) if not pred_fly_df.empty else 0
        
        metadata = {
            "kalman_process_noise_scale": float(kalman_process_noise_scale),
            "kalman_measurement_noise_std_mm": float(kalman_measurement_noise_std * 1000.0),
            "gate_distance_sigma": float(gate_distance_sigma),
            "max_distance_threshold_mm": float(max_distance_threshold),
            "bgs_kalman_process_noise_scale": float(bgs_kalman_process_noise_scale),
            "bgs_kalman_measurement_noise_std_mm": float(bgs_kalman_measurement_noise_std * 1000.0),
            "bgs_gate_distance_sigma": float(bgs_gate_distance_sigma),
            "bgs_max_distance_threshold_mm": float(bgs_max_distance_threshold),
            "gap_fill_only": bool(gap_fill_only),
            "extend_filtered_track": bool(extend_filtered_track),
            "filter_start_frame": int(filter_start_frame),
            "filter_end_frame": int(filter_end_frame),
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
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved filter metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save filter metadata: {e}")
    
    @staticmethod
    def load_filter_metadata(
        filtered_pred_path: Optional[Path] = None,
        motion_trial = None,
    ) -> dict:
        """
        Load filter metadata from JSON file if available.
        
        Args:
            filtered_pred_path: Path to filtered predictions CSV
            motion_trial: MotionTrial object (for auto-detection)
            
        Returns:
            Dictionary of metadata values, or empty dict if not found
        """
        # Auto-detect path if not provided
        if filtered_pred_path is None and motion_trial is not None:
            if hasattr(motion_trial, 'predictions_csv') and motion_trial.predictions_csv is not None:
                pred_path = Path(motion_trial.predictions_csv)
                if pred_path.exists():
                    parent = pred_path.parent
                    stem = pred_path.stem
                    if "_predictions" in stem and "_predictions_filtered" not in stem:
                        filtered_stem = stem.replace("_predictions", "_predictions_filtered")
                        potential_path = parent / f"{filtered_stem}.csv"
                        if potential_path.exists():
                            filtered_pred_path = potential_path
                            logger.debug(f"Auto-detected filtered predictions path: {filtered_pred_path}")
        
        if filtered_pred_path is None:
            return {}
        
        metadata_path = filtered_pred_path.with_stem(filtered_pred_path.stem + "_metadata")
        metadata_path = metadata_path.with_suffix(".json")
        
        if not metadata_path.exists():
            logger.debug(f"No filter metadata found at {metadata_path}")
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded filter metadata from {metadata_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load filter metadata from {metadata_path}: {e}")
            return {}
