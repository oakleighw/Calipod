from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional # <--- ADD THIS IMPORT

import numpy as np
import pandas as pd

from caliscope.packets import XYZPacket
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope import logger


@dataclass
class MotionTrial:
    """
    Motion trial loaded in from output csv
    """

    # MODIFIED: Make xyz_csv optional, default to None
    xyz_csv: Optional[Path] = None
    xyz_packets: dict = field(default_factory=dict[int:XYZPacket])

    def __post_init__(self):
        # Always initialize xyz_packets to an empty dictionary
        self.xyz_packets = {}

        # MODIFIED: Handle the case where no CSV path is provided
        if self.xyz_csv is None:
            self.is_empty = True
            self.xyz_df = pd.DataFrame() # Initialize with an empty DataFrame
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