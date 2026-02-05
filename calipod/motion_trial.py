from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional # <--- ADD THIS IMPORT

import numpy as np
import pandas as pd

from calipod.packets import XYZPacket
from calipod.trackers.tracker_enum import TrackerEnum
from calipod import logger


@dataclass
class MotionTrial:
    """
    Motion trial loaded in from output csv
    """

    # MODIFIED: Make xyz_csv optional, default to None
    xyz_csv: Optional[Path] = None
    predictions_csv: Optional[Path] = None  # Optional predictions for comparison (auto-detected if None)
    xyz_packets: dict = field(default_factory=dict[int:XYZPacket])

    def __post_init__(self):
        # Always initialize xyz_packets to an empty dictionary
        self.xyz_packets = {}

        # MODIFIED: Handle the case where no CSV path is provided
        if self.xyz_csv is None:
            self.is_empty = True
            self.xyz_df = pd.DataFrame() # Initialize with an empty DataFrame
            self.predictions_df = pd.DataFrame()  # Initialize predictions as empty
            self.start_index = 0
            self.end_index = 0
            self.tracker = None # No tracker associated if no CSV
            self.wireframe = None
            logger.get(__name__).debug("MotionTrial initialized as empty (no CSV path provided).")
            return # Exit __post_init__ early if no CSV

        # Assertions for when xyz_csv IS provided
        assert isinstance(self.xyz_csv, Path), f"xyz_csv must be a Path object, got {type(self.xyz_csv)}"
        assert self.xyz_csv.exists(), f"XYZ CSV file not found at: {self.xyz_csv}"

        tracker_name = self.xyz_csv.stem[4:]  # peel off "xyz_"
        self.tracker = TrackerEnum[tracker_name].value()

        if hasattr(self.tracker, "wireframe"):
            self.wireframe = self.tracker.wireframe
        else:
            self.wireframe = None

        self.xyz_df = pd.read_csv(self.xyz_csv, engine="pyarrow")
        
        # Auto-detect and load predictions if not explicitly provided
        if self.predictions_csv is None:
            # Try to find predictions CSV in the same directory as ground truth
            tracker_name = self.xyz_csv.stem[4:]  # peel off "xyz_"
            auto_pred_path = self.xyz_csv.parent / f"xyz_{tracker_name}_predictions.csv"
            if auto_pred_path.exists():
                self.predictions_csv = auto_pred_path
                logger.get(__name__).debug(f"Auto-detected predictions CSV at {auto_pred_path}")
        
        # Load predictions if provided or auto-detected
        if self.predictions_csv is not None and self.predictions_csv.exists():
            try:
                self.predictions_df = pd.read_csv(self.predictions_csv, engine="pyarrow")
                logger.get(__name__).info(f"Loaded predictions from {self.predictions_csv}")
                try:
                    pred_count = len(self.predictions_df)
                    if pred_count > 0 and {"sync_index","point_id","x_coord","y_coord","z_coord"}.issubset(self.predictions_df.columns):
                        syncs = self.predictions_df["sync_index"].unique()
                        logger.get(__name__).info(f"Predictions summary: rows={pred_count}, sync_index range={syncs.min()}..{syncs.max()}, point_id counts={self.predictions_df['point_id'].value_counts().to_dict()}")
                    else:
                        logger.get(__name__).info(f"Predictions loaded but missing expected columns or empty; columns={list(self.predictions_df.columns)}")
                except Exception as e2:
                    logger.get(__name__).warning(f"Error summarizing predictions_df: {e2}")
            except Exception as e:
                logger.get(__name__).warning(f"Failed to load predictions from {self.predictions_csv}: {e}")
                self.predictions_df = pd.DataFrame()
        else:
            self.predictions_df = pd.DataFrame()
        
        sync_indices = self.xyz_df["sync_index"].unique()

        # MODIFIED: Ensure start/end indices are 0 if no sync_indices are found even if CSV exists
        self.start_index = sync_indices.min() if len(sync_indices) > 0 else 0
        self.end_index = sync_indices.max() if len(sync_indices) > 0 else 0
        
        self.is_empty = len(sync_indices) == 0
        if self.is_empty:
             logger.get(__name__).debug(f"MotionTrial initialized from {self.xyz_csv} but found no sync_indices, marking as empty.")
        else:
             logger.get(__name__).debug(f"MotionTrial initialized from {self.xyz_csv} with data.")

    def get_xyz(self, sync_index: int) -> XYZPacket:
        """
        Cache packets as they are initially read off.
        Returns an empty XYZPacket if the MotionTrial is empty or if the sync_index
        has no triangulated points in the loaded data.
        """
        if self.is_empty:
            logger.get(__name__).debug(f"MotionTrial is empty. Returning empty XYZPacket for sync index {sync_index}.")
            # Return an XYZPacket with empty arrays for points
            return XYZPacket(sync_index=sync_index, point_ids=np.array([]), point_xyz=np.array([]).reshape(0,3))

        if sync_index not in self.xyz_packets:
            current_sync_index = self.xyz_df["sync_index"] == sync_index
            
            # MODIFIED: Handle case where sync_index might not be in DataFrame for a loaded trial
            if not current_sync_index.any():
                logger.get(__name__).debug(f"Sync index {sync_index} not found in MotionTrial DataFrame. Returning empty XYZPacket.")
                self.xyz_packets[sync_index] = XYZPacket(sync_index=sync_index, point_ids=np.array([]), point_xyz=np.array([]).reshape(0,3))
            else:
                point_ids = self.xyz_df["point_id"][current_sync_index]
                x = self.xyz_df["x_coord"][current_sync_index]
                y = self.xyz_df["y_coord"][current_sync_index]
                z = self.xyz_df["z_coord"][current_sync_index]

                xyz = np.column_stack([x, y, z])
                self.xyz_packets[sync_index] = XYZPacket(sync_index=sync_index, point_ids=point_ids, point_xyz=xyz)

        return self.xyz_packets[sync_index]

    def update_wireframe(self, sync_index: int):
        xyz_packet = self.get_xyz(sync_index)

        if self.wireframe is not None:
            self.wireframe.set_points(xyz_packet)
        else:
            logger.get(__name__).debug(f"Skipping wireframe update for sync index {sync_index}: self.wireframe is None (no tracker or no wireframe for tracker).")

    def performance_metrics(self) -> dict:
        """
        Computes performance metrics if predictions are available.
        Returns an empty dictionary if no predictions are loaded.
        Only single object class & instance capability implemented for now.
        
        Metrics:
        - RMSE: Root Mean Square Error - squares errors so penalizes large errors more (mm)
        - MOTP: Multiple Object Tracking Precision - mean Euclidean distance error (mm)
        - MOTA: Multiple Object Tracking Accuracy - detection accuracy per frame (0-1 scale)
        - Median_Error: Median Euclidean distance error - robust to outliers (mm)
        """
        if self.predictions_df.empty:
            logger.get(__name__).debug("No predictions loaded; returning empty performance metrics.")
            return {}

        # Debug: Check data structure
        logger.get(__name__).info(f"xyz_df shape: {self.xyz_df.shape}, columns: {list(self.xyz_df.columns)[:5]}")
        logger.get(__name__).info(f"predictions_df shape: {self.predictions_df.shape}, columns: {list(self.predictions_df.columns)[:5]}")
        if 'point_id' in self.xyz_df.columns:
            logger.get(__name__).info(f"GT unique point_ids: {sorted(self.xyz_df['point_id'].unique())}")
        if 'point_id' in self.predictions_df.columns:
            logger.get(__name__).info(f"Pred unique point_ids: {sorted(self.predictions_df['point_id'].unique())}")

        # Merge ground truth and predictions on sync_index and point_id
        # Filters to only those points present in both ground truth and predictions (does not include false positives/negatives)
        merged_df = pd.merge(
            self.xyz_df,
            self.predictions_df,
            on=["sync_index", "point_id"],
            suffixes=('_gt', '_pred')
        )

        if merged_df.empty:
            logger.get(__name__).debug("No matching points between ground truth and predictions; returning empty performance metrics.")
            return {}

        # Calculate RMSE (element-wise across all coordinates)
        rmse = np.sqrt(np.mean(
            (merged_df[['x_coord_gt', 'y_coord_gt', 'z_coord_gt']].values -
             merged_df[['x_coord_pred', 'y_coord_pred', 'z_coord_pred']].values) ** 2
        ))

        # Calculate Euclidean distances (errors) for each matched point
        distances = np.linalg.norm(
            merged_df[['x_coord_gt', 'y_coord_gt', 'z_coord_gt']].values -
            merged_df[['x_coord_pred', 'y_coord_pred', 'z_coord_pred']].values, axis=1
        )
        
        # Calculate multiple object tracking precision (MOTP) - mean Euclidean error
        motp = np.mean(distances)
        
        # Median error - robust to outliers
        median_error = np.median(distances)

        # Calculate multiple object tracking accuracy (MOTA) - PER FRAME
        # MOTA measures detection accuracy, not distance accuracy
        # Filter GT to only point_ids that exist in predictions (to compare apples-to-apples)
        # Example: if predictions only track point_id 0, don't count missing point_ids 1-10 as false negatives
        relevant_point_ids = self.predictions_df['point_id'].unique()
        gt_filtered = self.xyz_df[self.xyz_df['point_id'].isin(relevant_point_ids)]
        
        logger.get(__name__).info(f"Filtered GT from {len(self.xyz_df)} to {len(gt_filtered)} rows (only point_ids: {sorted(relevant_point_ids)})")
        
        # Count detections per frame (only for relevant point_ids)
        gt_counts_per_frame = gt_filtered.groupby('sync_index').size()
        pred_counts_per_frame = self.predictions_df.groupby('sync_index').size()
        matched_counts_per_frame = merged_df.groupby('sync_index').size()
        
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
        mota = 1 - (total_false_negatives + total_false_positives) / total_gt_detections if total_gt_detections > 0 else 0.0

        # Convert distances from meters to millimeters
        rmse_mm = rmse * 1000
        motp_mm = motp * 1000
        median_error_mm = median_error * 1000
        
        metrics = {
            "RMSE_mm": rmse_mm,
            "MOTP_mm": motp_mm,
            "MOTA": mota,
            "Median_Error_mm": median_error_mm
        }

        logger.get(__name__).debug(f"Computed performance metrics: RMSE={rmse_mm:.2f}mm, MOTP={motp_mm:.2f}mm, MOTA={mota:.4f}")
        return metrics
