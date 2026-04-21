import shutil
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import pandas as pd

from calipod.annotation_management.file_utils import extract_frame_index_from_filename
from calipod.annotation_management.yolo_utils import load_yolo_file
from calipod.cameras.camera_array import CameraArray
from calipod.core import logger as calipod_logger
from calipod.export import xyz_to_trc, xyz_to_wide_labelled
from calipod.post_processing.gap_filling import gap_fill_xy, gap_fill_xyz
from calipod.post_processing.smoothing import smooth_xyz
from calipod.synchronized_stream_manager import SynchronizedStreamManager
from calipod.trackers.tracker_enum import TrackerEnum
from calipod.triangulate.triangulation import triangulate_xy

logger = calipod_logger.get(__name__)

# gap filling and filtering is outside the current scope of the project so I'm toggling this off for now
APPLY_EXPERIMENTAL_POST_PROCESSING = False


class PostProcessor:
    """
    The post processer operates independently of the session. It does not need to worry about camera management.
    Provide it with a path to the directory that contains the following:
    - config.toml
    - frame_time.csv
    - .mp4 files

    The post processor will archive the active config.toml file into the subdirectory
    """

    def __init__(
        self,
        camera_array: CameraArray,
        recording_path: Path,
        ground_truth_path: Path = None,
        predictions_path: Path = None,
        tracker_enum: TrackerEnum = TrackerEnum.HAND,
        annotations_path: Path = None,
    ):
        # Backward compatibility: older callers passed `annotations_path`.
        if ground_truth_path is None and annotations_path is not None:
            ground_truth_path = annotations_path

        self.camera_array = camera_array
        self.recording_path = recording_path
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path
        self.tracker_enum = tracker_enum
        self.tracker_name = tracker_enum.name
        self.tracker = tracker_enum.value()

        logger.info(f"!!!!!!!!!SET ANNOTATIONS name here {self.tracker_name} !!!!!!!!!")
        logger.info("!!!!!!!!!SET ANNOTATIONS DIR!!!!!!!!!")
        logger.info("!!!!!!!!!SET ANNOTATIONS DIR!!!!!!!!!")
        logger.info("!!!!!!!!!SET ANNOTATIONS DIR!!!!!!!!!")
        logger.info("!!!!!!!!!SET ANNOTATIONS DIR!!!!!!!!!")
        logger.info("!!!!!!!!!SET ANNOTATIONS DIR!!!!!!!!!")
        if self.tracker_name == "FLY" and self.ground_truth_path is not None:
            # Instantiate FlyTracker WITHOUT arguments
            # NOW SET THE PROPERTIES

            self.tracker.annotations_dir = self.ground_truth_path

        # save out current camera array to output folder
        tracker_subdirectory = Path(self.recording_path, self.tracker_name)
        tracker_subdirectory.mkdir(exist_ok=True, parents=True)
        shutil.copy(Path(self.recording_path.parent.parent, "config.toml"), Path(tracker_subdirectory, "config.toml"))

        logger.info(f"Creating sync stream manager for videos stored in {self.recording_path}")
        self.sync_stream_manager = SynchronizedStreamManager(
            self.recording_path, self.camera_array.cameras, self.tracker
        )

    def create_xy(self, fps_target=100, include_video=True):
        """
        Reads through all .mp4  files in the recording path and applies the tracker to them
        The xy_TrackerName.csv file is saved out to the same directory by the VideoRecorder

        Note that high fps target and including video will increase processing overhead

        For FlyTracker with annotations, automatically uses annotations-only mode for faster processing.
        """
        # Auto-detect if we should use annotations-only mode
        if self.tracker_name == "FLY" and hasattr(self.tracker, "check_annotations_available"):
            if self.tracker.check_annotations_available(self.recording_path, self.camera_array.cameras):
                logger.info("✓ Using annotations-only mode for faster post-processing")
                include_video = False

        self.sync_stream_manager.process_streams(include_video=include_video, fps_target=fps_target)

        while self.sync_stream_manager.recorder.recording:
            sleep(1)
            percent_complete = int(
                (self.sync_stream_manager.recorder.sync_index / self.sync_stream_manager.mean_frame_count) * 100
            )
            logger.info(f"(Stage 1 of 2): {percent_complete}% of frames processed for (x,y) landmark detection")

    def create_xyz(self, xy_gap_fill=3, xyz_gap_fill=3, cutoff_freq=6, include_trc=True) -> None:
        """
        creates xyz_{tracker name}.csv file within the recording_path directory

        Uses the two functions above, first creating the xy points based on the tracker if they
        don't already exist, the triangulating them. Makes use of an internal method self.triangulate_xy_data

        """
        logger.info("=" * 80)
        logger.info("CREATE_XYZ STARTING - Predictions pipeline enabled and active")
        logger.info("=" * 80)

        tracker_output_path = Path(self.recording_path, self.tracker_name)
        xy_csv_path = Path(tracker_output_path, f"xy_{self.tracker_name}.csv")

        # create if it doesn't already exist
        if not xy_csv_path.exists():
            self.create_xy()

        # load in 2d data and triangulate it
        logger.info("Reading in (x,y) data..")
        xy = pd.read_csv(xy_csv_path)
        if xy.shape[0] > 0:
            logger.info("Filling small gaps in (x,y) data")
            xy = gap_fill_xy(xy, max_gap_size=xy_gap_fill)
            logger.info("Beginning data triangulation")
            xyz = triangulate_xy(xy, self.camera_array)
        else:
            logger.warning("No points tracked. Terminating post-processing early.")
            return

        if xyz.shape[0] > 0:
            if APPLY_EXPERIMENTAL_POST_PROCESSING:
                logger.info("Filling small gaps in (x,y,z) data")
                xyz = gap_fill_xyz(xyz, max_gap_size=xyz_gap_fill)
                logger.info("Smoothing (x,y,z) using butterworth filter with cutoff frequency of 6hz")
                xyz = smooth_xyz(xyz, order=2, fps=self.sync_stream_manager.mean_fps, cutoff=cutoff_freq)

            logger.info("Saving (x,y,z) to csv file")
            xyz_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.csv")
            xyz.to_csv(xyz_csv_path)
            xyz_wide_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}_labelled.csv")
            xyz_labelled = xyz_to_wide_labelled(xyz, self.tracker_enum.value())
            xyz_labelled.to_csv(xyz_wide_csv_path)

        else:
            logger.warning("No points triangulated. Terminating post-processing early.")
            return

        # only include trc if wanted and only if there is actually good data to export
        if include_trc and xyz.shape[0] > 0:
            try:
                trc_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.trc")
                time_history_path = Path(tracker_output_path, "frame_time_history.csv")
                xyz_to_trc(
                    xyz,
                    tracker=self.tracker_enum.value(),
                    time_history_path=time_history_path,
                    target_path=trc_path,
                )
                logger.info(f"TRC file exported to {trc_path}")
            except Exception as e:
                logger.warning(f"Failed to export TRC file: {type(e).__name__}: {e}")
                logger.warning("Continuing with predictions pipeline despite TRC export failure...")

        # Attempt to build and triangulate predictions if available
        logger.info("(Predictions) Checking for prediction label files to process...")
        logger.info(f"(Predictions) predictions_path={self.predictions_path}")
        try:
            self.create_xy_predictions()
            logger.info("(Predictions) create_xy_predictions() completed successfully")
        except Exception as e:
            logger.error(f"(Predictions) EXCEPTION in create_xy_predictions(): {type(e).__name__}: {e}", exc_info=True)

        xy_pred_path = Path(tracker_output_path, f"xy_predictions_{self.tracker_name}.csv")
        if xy_pred_path.exists():
            logger.info(f"(Predictions) XY predictions file confirmed to exist at {xy_pred_path}")
        else:
            logger.info(f"(Predictions) XY predictions file NOT found at {xy_pred_path}")

        logger.info("(Predictions) Attempting triangulation of predictions (if XY exists)...")
        try:
            self.triangulate_predictions()
            logger.info("(Predictions) triangulate_predictions() completed successfully")
        except Exception as e:
            logger.error(
                f"(Predictions) EXCEPTION in triangulate_predictions(): {type(e).__name__}: {e}", exc_info=True
            )

        xyz_pred_path = Path(tracker_output_path, f"xyz_{self.tracker_name}_predictions.csv")
        if xyz_pred_path.exists():
            logger.info(f"(Predictions) XYZ predictions file confirmed to exist at {xyz_pred_path}")
        else:
            logger.info(f"(Predictions) XYZ predictions file NOT found at {xyz_pred_path}")

    def create_xy_predictions(self) -> None:
        """Create xy_predictions_{tracker_name}.csv from YOLO prediction label files.

        Expects prediction labels under: {predictions_path}/port_{port}/labels/frame_######.txt
        Writes consolidated XY CSV to: {recording_path}/{tracker_name}/xy_predictions_{tracker_name}.csv
        """
        logger.info(f"(Predictions) create_xy_predictions() called; tracker_name={self.tracker_name}")
        if self.tracker_name != "FLY":
            logger.info("(Predictions) XY creation currently implemented for FLY tracker only; skipping.")
            return

        tracker_output_path = Path(self.recording_path, self.tracker_name)
        xy_pred_path = Path(tracker_output_path, f"xy_predictions_{self.tracker_name}.csv")

        # If already exists and non-empty, skip to avoid rework
        if xy_pred_path.exists():
            try:
                if pd.read_csv(xy_pred_path, nrows=1).shape[0] > 0:
                    logger.info(f"(Predictions) XY predictions already exist at {xy_pred_path}; skipping rebuild.")
                    return
            except Exception:
                pass

        if not hasattr(self, "predictions_path") or self.predictions_path is None:
            logger.info("(Predictions) No predictions_path set; skipping predictions XY creation.")
            return

        logger.info(f"(Predictions) predictions_path is set: {self.predictions_path}")

        # Determine frame dimensions per port from recorded videos
        port_frame_size: dict[int, tuple[int, int]] = {}
        for port, cam in self.camera_array.cameras.items():
            mp4_path = Path(self.recording_path, f"port_{cam.port}.mp4")
            try:
                cap = cv2.VideoCapture(str(mp4_path))
                if not cap.isOpened():
                    logger.warning(
                        f"(Predictions) Could not open video {mp4_path} to read frame size; "
                        "predictions will be skipped for this port."
                    )
                    continue
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                port_frame_size[cam.port] = (width, height)
            except Exception as e:
                logger.warning(f"(Predictions) Failed reading video properties for port {cam.port}: {e}")

        logger.info(f"(Predictions) Frame sizes determined: {port_frame_size}")
        logger.info(f"(Predictions) Recording path: {self.recording_path}")
        logger.info(f"(Predictions) Predictions path: {self.predictions_path}")

        rows = {
            "sync_index": [],
            "port": [],
            "frame_index": [],
            "frame_time": [],
            "point_id": [],
            "img_loc_x": [],
            "img_loc_y": [],
            "obj_loc_x": [],
            "obj_loc_y": [],
        }

        # Walk prediction label files per port
        any_found = False
        for port, cam in self.camera_array.cameras.items():
            labels_dir = Path(self.predictions_path, f"port_{port}", "labels")
            logger.info(f"(Predictions) Checking for labels at port {port}:")
            logger.info(f"  Full path: {labels_dir.resolve()}")
            logger.info(f"  Path exists: {labels_dir.exists()}")

            if not labels_dir.exists():
                logger.info(f"(Predictions) No labels directory for port {port}")
                continue

            if port not in port_frame_size:
                logger.info(f"(Predictions) Missing frame size for port {port}; skipping its predictions.")
                continue

            width, height = port_frame_size[port]
            frame_shape = (height, width, 3)

            # Support multiple naming conventions
            txt_files = sorted(labels_dir.glob("*.txt"))
            logger.info(f"(Predictions) Port {port}: found {len(txt_files)} prediction label files")

            for txt_path in txt_files:
                # Parse frame index from filename using shared utility
                idx = extract_frame_index_from_filename(txt_path.name)
                if idx is None:
                    logger.warning(
                        f"(Predictions) Cannot parse frame index from filename: {txt_path.name}; skipping"
                    )
                    continue

                # Reuse shared YOLO parser for consistency (adds corners for classes 9/10)
                try:
                    ids, img_loc, _ = load_yolo_file(txt_path, frame_shape=frame_shape)
                except Exception as e:
                    logger.warning(f"(Predictions) Failed parsing {txt_path}: {e}")
                    continue

                if ids.size == 0:
                    continue

                any_found = True
                for k in range(len(ids)):
                    rows["sync_index"].append(idx)
                    rows["port"].append(port)
                    rows["frame_index"].append(idx)
                    rows["frame_time"].append(np.nan)
                    rows["point_id"].append(int(ids[k]))
                    rows["img_loc_x"].append(float(img_loc[k][0]))
                    rows["img_loc_y"].append(float(img_loc[k][1]))
                    rows["obj_loc_x"].append(np.nan)
                    rows["obj_loc_y"].append(np.nan)

        if not any_found:
            logger.info("(Predictions) No prediction labels found across ports; skipping XY predictions creation.")
            return

        df_xy_pred = pd.DataFrame(rows)
        tracker_output_path.mkdir(exist_ok=True, parents=True)
        df_xy_pred.to_csv(xy_pred_path, index=False)
        try:
            sync_min = (
                int(df_xy_pred["sync_index"].min())
                if "sync_index" in df_xy_pred.columns and len(df_xy_pred) > 0
                else None
            )
            sync_max = (
                int(df_xy_pred["sync_index"].max())
                if "sync_index" in df_xy_pred.columns and len(df_xy_pred) > 0
                else None
            )
            logger.info(
                f"(Predictions) XY predictions written to {xy_pred_path} "
                f"(rows={len(df_xy_pred)}; sync_index range={sync_min}..{sync_max})"
            )
        except Exception:
            logger.info(f"(Predictions) XY predictions written to {xy_pred_path}")

    def triangulate_predictions(self) -> None:
        """
        Triangulate predictions from 2D to 3D coordinates using the same process as ground truth.

        Looks for xy_predictions_{tracker_name}.csv in the tracker output directory.
        Saves the triangulated predictions to xyz_{tracker_name}_predictions.csv.
        """
        tracker_output_path = Path(self.recording_path, self.tracker_name)
        xy_pred_path = Path(tracker_output_path, f"xy_predictions_{self.tracker_name}.csv")

        if not xy_pred_path.exists():
            logger.warning(f"Predictions XY file not found at {xy_pred_path}. Skipping prediction triangulation.")
            return

        logger.info(f"Loading predictions from {xy_pred_path}")
        try:
            xy_pred = pd.read_csv(xy_pred_path)
        except Exception as e:
            logger.error(f"Failed to read predictions CSV: {e}")
            return

        if xy_pred.shape[0] == 0:
            logger.warning("Predictions CSV is empty. Skipping prediction triangulation.")
            return

        logger.info(f"Triangulating {xy_pred.shape[0]} prediction points using camera array")
        try:
            xyz_pred = triangulate_xy(xy_pred, self.camera_array)
        except Exception as e:
            logger.error(f"Failed to triangulate predictions: {e}")
            return

        if xyz_pred.shape[0] > 0:
            xyz_pred_path = Path(tracker_output_path, f"xyz_{self.tracker_name}_predictions.csv")
            xyz_pred.to_csv(xyz_pred_path, index=False)
            try:
                sync_min = int(xyz_pred["sync_index"].min())
                sync_max = int(xyz_pred["sync_index"].max())
                logger.info(
                    f"Predictions triangulated and saved to {xyz_pred_path} "
                    f"(rows={len(xyz_pred)}; sync_index range={sync_min}..{sync_max})"
                )
            except Exception:
                logger.info(f"Predictions triangulated and saved to {xyz_pred_path}")
        else:
            logger.warning("No prediction points were successfully triangulated.")


if __name__ == "__main__":
    from calipod.core.controller import Controller

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\caliscope\demo")
    controller = Controller(workspace_dir)
    controller.load_camera_array()

    camera_aray = controller.camera_array
    recording_dir = Path(workspace_dir, "recordings", "STS")

    post_processor = PostProcessor(
        camera_array=camera_aray,
        recording_path=recording_dir,
        tracker_enum=TrackerEnum.HAND,
    )

    post_processor.create_xyz()
