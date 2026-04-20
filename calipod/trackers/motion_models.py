"""Reusable motion models for tracker filtering.

Currently includes a constant-velocity 3D model that produces the
state transition matrix (A) and process noise covariance (Q) for
position/velocity states [x y z vx vy vz]^T given a timestep ``dt``.
"""

from __future__ import annotations

import numpy as np


class ConstantVelocity3DModel:
    """Simple constant-velocity motion model in 3D.

    State vector: [x, y, z, vx, vy, vz]^T

    Parameters
    ----------
    motion_noise_scale: float
        Multiplier applied to the process noise covariance ``Q``.
    """

    def __init__(self, motion_noise_scale: float = 1.0) -> None:
        self.motion_noise_scale = motion_noise_scale

    def calc_for_dt(self, dt: float) -> dict[str, np.ndarray]:
        """Return transition and noise matrices for a given timestep.

        Returns
        -------
        dict
            ``transition_model`` (A), ``transition_model_transpose`` (A.T),
            and ``transition_noise_covariance`` (Q).
        """
        # Transition matrix maps velocity into position with dt
        A = np.eye(6)
        A[0, 3] = A[1, 4] = A[2, 5] = dt

        # Process noise terms for CV model
        t33 = (dt**3) / 3.0
        t22 = (dt**2) / 2.0

        q_pos = np.eye(3) * t33
        q_vel = np.eye(3) * dt
        q_corr = np.eye(3) * t22

        Q = (
            np.block(
                [
                    [q_pos, q_corr],
                    [q_corr, q_vel],
                ]
            )
            * self.motion_noise_scale
        )

        return {
            "transition_model": A,
            "transition_model_transpose": A.T,
            "transition_noise_covariance": Q,
        }
