import numpy as np
import pandas as pd


def calculate_distance(xyz_trajectory_data: pd.DataFrame, point1: str, point2: str):
    """
    Given a set of xyz trajectories from tracked and triangulated landmarks, calculate the
    mean distance between two named points, excluding outlier points for data cleanliness

    """
    # calculate the distance
    distances = np.sqrt(
        (xyz_trajectory_data[point1 + "_x"] - xyz_trajectory_data[point2 + "_x"]) ** 2
        + (xyz_trajectory_data[point1 + "_y"] - xyz_trajectory_data[point2 + "_y"]) ** 2
        + (xyz_trajectory_data[point1 + "_z"] - xyz_trajectory_data[point2 + "_z"]) ** 2
    )

    # calculate Q1, Q3 and IQR for outlier detection
    Q1 = distances.quantile(0.25)
    Q3 = distances.quantile(0.75)
    IQR = Q3 - Q1

    # filter out the outliers
    filtered_distances = distances[(distances >= Q1 - 1.5 * IQR) & (distances <= Q3 + 1.5 * IQR)]

    # calculate the average length
    average_length = filtered_distances.mean()
    return average_length
