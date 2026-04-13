import json
from pathlib import Path

from calipod.core import logger as calipod_logger
from calipod.post_processing.distance_metrics import calculate_distance
from calipod.trackers.tracker_enum import TrackerEnum

logger = calipod_logger.get(__name__)


def generate_metarig_config(tracker_enum: TrackerEnum, xyz_csv_path: Path):
    """
    Stores metarig config json file within the tracker sub-directory within a recording folder
    """
    tracker = tracker_enum.value()

    import pandas as pd

    xyz_trajectories = pd.read_csv(xyz_csv_path)
    json_path = Path(xyz_csv_path.parent, f"metarig_config_{tracker.name}.json")

    # for testing purposes, need to make sure that this file is not there before proceeding
    json_path.unlink(missing_ok=True)
    assert not json_path.exists()

    autorig_config = {}  # Dictionary that will become json

    # average distances across the bilateral measures
    for measure, points in tracker.metarig_bilateral_measures.items():
        logger.info(f"Calculating mean distance (IQR) for {measure}")

        mean_distance = 0
        for side in ["left", "right"]:
            point1 = f"{side}_{points[0]}"
            point2 = f"{side}_{points[1]}"

            distance = calculate_distance(xyz_trajectories, point1, point2)
            logger.info(f"Between {point1} and {point2} the mean distance is {distance}")
            mean_distance += distance / 2
        autorig_config[measure] = round(mean_distance, 4)

    # calculate distance across symmetrical measures
    for measure, points in tracker.metarig_symmetrical_measures.items():
        distance = calculate_distance(xyz_trajectories, points[0], points[1])
        autorig_config[measure] = round(distance, 4)

    # save output to file that will be in same folder as the associated xy_csv data
    with open(json_path, "w") as f:
        json.dump(autorig_config, f, indent=4)
