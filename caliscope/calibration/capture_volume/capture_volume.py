# %%

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

import caliscope.logger
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.set_origin_functions import (
    get_board_origin_transform,
)
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray

logger = caliscope.logger.get(__name__)

CAMERA_PARAM_COUNT = 6


@dataclass
class CaptureVolume:
    camera_array: CameraArray
    point_estimates: PointEstimates
    stage: int = 0
    origin_sync_index: int = None

    def __post__init__():
        logger.info("Creating capture volume from estimated camera array and stereotriangulated points...")

    def _save(self, directory: Path, descriptor: str = None):
        if descriptor is None:
            pkl_name = "capture_volume_stage_" + str(self.stage) + ".pkl"
        else:
            pkl_name = "capture_volume_stage_" + str(self.stage) + "_" + descriptor + ".pkl"
        logger.info(f"Saving stage {str(self.stage)} capture volume to {directory}")
        with open(Path(directory, pkl_name), "wb") as file:
            pickle.dump(self, file)

    def get_vectorized_params(self):
        """
        Convert the parameters of the camera array and the point estimates into one long array.
        This is the required data format of the least squares optimization
        """
        camera_params = self.camera_array.get_extrinsic_params()
        combined = np.hstack((camera_params.ravel(), self.point_estimates.obj.ravel()))

        return combined
    
    def get_cam_origins(self):
        """
        Gets the camera origins (optical centres) in the world plane
        """
        cam_world_origins = self.camera_array.get_world_origins()

        for i, origin in enumerate(cam_world_origins):
            logger.info(f"Camera {i} Origin: X={origin[0]:.4f}, Y={origin[1]:.4f}, Z={origin[2]:.4f}")

        return cam_world_origins

    @property
    def rmse(self):
        if hasattr(self, "least_sq_result"):
            rmse = rms_reproj_error(self.least_sq_result.fun, self.point_estimates.camera_indices)
        else:
            param_estimates = self.get_vectorized_params()
            xy_reproj_error = xy_reprojection_error(param_estimates, self)
            rmse = rms_reproj_error(xy_reproj_error, self.point_estimates.camera_indices)

        return rmse
    
    def cam_distances(self,type = None):
        """
        Returns a dictionary that shows the distance between each optical centre
        """
        #get camera origins
        interdist = {}
        cam_origins = self.get_cam_origins()
        
        for i, port in enumerate(np.unique(self.point_estimates.camera_indices)):
            
            if i ==0: #log first port number
                first_port = port

            logger.info(f"port, i, len unique -1 {port} {i} {len(np.unique(self.point_estimates.camera_indices))}")
            if i == len(np.unique(self.point_estimates.camera_indices))-1: #if end of list (last port), get distance between last port (current) and first port
                interdist[f"{str(port)}-{str(first_port)}"] = self.cam_dist(cam_origins[i],cam_origins[0],type)
            else:#get distance between current and next port cam origin
                interdist[f"{str(port)}-{str(port+1)}"] = self.cam_dist(cam_origins[i],cam_origins[i+1],type)
                
        return interdist

    def get_rmse_summary(self):
        rmse_string = f"RMSE of Reprojection Overall: {round(self.rmse['overall'],2)}\n"
        rmse_string += "    by camera:\n"
        for key, value in self.rmse.items():
            if key == "overall":
                pass
            else:
                rmse_string += f"    {key: >9}: {round(float(value),2)}\n"

        return rmse_string
    
    def cam_dist(self,origin1,origin2, type = None): #distance between two 3d points
        if type == 'vertical': #z
            p1 = np.array(origin1[2])
            p2 = np.array(origin2[2])
        elif type == 'horizontal': #xy
            p1 = np.array(origin1[0:2])
            p2 = np.array(origin2[0:2])
        else: #full 3D distance
            p1 = origin1
            p2 = origin2
        cam_dist = np.linalg.norm(p1-p2)
        return cam_dist
    
    def get_cam_distance_summary(self):
        cam_dist_string = f"Distance between lens optical centres (Total): \n"
        cam_dist_string += "    by camera:\n"
        for key, value in self.cam_distances().items():
            cam_dist_string += f"    {key: >9}: {round(float(value)*100,3)} cm\n"# x100 for cm

        cam_dist_string += f"(xy): \n"
        cam_dist_string += "    by camera:\n"
        for key, value in self.cam_distances('horizontal').items():
            cam_dist_string += f"    {key: >9}: {round(float(value)*100,3)} cm\n"# x100 for cm

        cam_dist_string += f"(z): \n"
        cam_dist_string += "    by camera:\n"
        for key, value in self.cam_distances('vertical').items():
            cam_dist_string += f"    {key: >9}: {round(float(value)*100,3)} cm\n"# x100 for cm

        return cam_dist_string

    def get_xy_reprojection_error(self):
        vectorized_params = self.get_vectorized_params()
        error = xy_reprojection_error(vectorized_params, self)

        return error

    def optimize(self):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        initial_param_estimate = self.get_vectorized_params()

        # get a snapshot of where things are at the start
        # initial_xy_error = xy_reprojection_error(initial_param_estimate, self)

        # logger.info(
        #     f"Prior to bundle adjustment (stage {str(self.stage)}), RMSE is: {self.rmse}"
        # )
        logger.info(f"Beginning bundle adjustment to calculated stage {self.stage+1}")
        self.least_sq_result = least_squares(
            xy_reprojection_error,
            initial_param_estimate,
            jac_sparsity=self.point_estimates.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=1e-8,
            method="trf",
            # xy_reprojection error takes the vectorized param estimates as first arg and capture volume as second
            args=(self,),
        )

        self.camera_array.update_extrinsic_params(self.least_sq_result.x)
        self.point_estimates.update_obj_xyz(self.least_sq_result.x)
        self.stage += 1

        logger.info(f"Following bundle adjustment (stage {str(self.stage)}), RMSE is: {self.rmse['overall']}")

    def get_xyz_points(self):
        """Get 3d positions arrived at by bundle adjustment"""
        n_cameras = len(self.camera_array.cameras)
        xyz = self.get_vectorized_params()[n_cameras * CAMERA_PARAM_COUNT :]
        xyz = xyz.reshape(-1, 3)

        return xyz

    def shift_origin(self, origin_shift_transform: np.ndarray):
        # update 3d point estimates
        xyz = self.point_estimates.obj
        scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
        xyzh = np.hstack([xyz, scale])

        new_origin_xyzh = np.matmul(np.linalg.inv(origin_shift_transform), xyzh.T).T
        self.point_estimates.obj = new_origin_xyzh[:, 0:3]

        # update camera array
        for port, camera_data in self.camera_array.cameras.items():
            camera_data.transformation = np.matmul(camera_data.transformation, origin_shift_transform)

    def set_origin_to_board(self, sync_index, charuco: Charuco):
        """
        Find the pose of the charuco (rvec and tvec) from a given frame
        Transform stereopairs and 3d point estimates for this new origin
        """
        self.origin_sync_index = sync_index

        logger.info(f"Capture volume origin set to board position at sync index {sync_index}")

        origin_transform = get_board_origin_transform(self.camera_array, self.point_estimates, sync_index, charuco)
        self.shift_origin(origin_transform)


def xy_reprojection_error(current_param_estimates, capture_volume: CaptureVolume):
    """
    current_param_estimates:
        the current iteration of the vector that was originally initialized for the x0 input of least squares

    This function exists outside of the CaptureVolume class because the first argument must be the vector of parameters
    that is being adjusted by the least_squares optimization.

    """
    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera parameters (could be extr. or intr.)
    camera_params = current_param_estimates[: capture_volume.point_estimates.n_cameras * CAMERA_PARAM_COUNT].reshape(
        (capture_volume.point_estimates.n_cameras, CAMERA_PARAM_COUNT)
    )

    ## similarly unpack the 3d point location estimates
    points_3d = current_param_estimates[capture_volume.point_estimates.n_cameras * CAMERA_PARAM_COUNT :].reshape(
        (capture_volume.point_estimates.n_obj_points, 3)
    )

    ## create zero columns as placeholders for the reprojected 2d points
    rows = capture_volume.point_estimates.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [
            np.array([capture_volume.point_estimates.camera_indices]).T,
            points_3d[capture_volume.point_estimates.obj_indices],
            capture_volume.point_estimates.img,
            blanks,
        ]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...

    for port, cam in capture_volume.camera_array.cameras.items():
        cam_points = np.where(capture_volume.point_estimates.camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]

        # if a camera is being ignored, it will not show up on the parameter list
        # so you must use the port index to make sure you read the correct params
        port_index = capture_volume.camera_array.port_index[port]
        cam_matrix = cam.matrix
        rvec = camera_params[port_index][0:3]
        tvec = camera_params[port_index][3:6]
        distortions = cam.distortions

        # get the projection of the 2d points on the image plane; ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(object_points.astype(np.float64), rvec, tvec, cam_matrix, distortions)

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]

    xy_reprojection_error = (points_proj - capture_volume.point_estimates.img).ravel()
    # capture_volume.rmse = rms_reproj_error(xy_reprojection_error)
    # if round(perf_counter(), 2) * 10 % 1 == 0: # log less frequently
    #     logger.info(
    #         f"Optimizing... RMSE of reprojection = {capture_volume.rmse}"
    #     )

    # reshape the x,y reprojection error to a single vector
    return xy_reprojection_error


def rms_reproj_error(xy_reproj_error, camera_indices):
    """
    Returns a dictionary that shows the overal rmse & for each camera
    """
    rmse = {}
    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))
    rmse["overall"] = np.sqrt(np.mean(euclidean_distance_error**2))

    for port in np.unique(camera_indices):
        camera_errors = euclidean_distance_error[camera_indices == port]
        rmse[str(port)] = np.sqrt(np.mean(camera_errors**2))
    # for port in camera_indices
    # logger.info(f"Optimization run with {xy_reproj_error.shape[0]} image points")
    # logger.info(f"RMSE of reprojection is {rmse}")
    return rmse





# def load_capture_volume(session_path:Path):
#     config = get_config(session_directory)

#     camera_array = get_camera_array(config)
#     point_estimates = load_point_estimates(config)

#     capture_volume = CaptureVolume(camera_array, point_estimates)
#     capture_volume.stage = config["capture_volume"]["stage"]
#     # capture_volume.origin_sync_index = config["capture_volume"]["origin_sync_index"]

#     return capture_volume
