"""Interactive 3D trajectory visualization using matplotlib."""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calipod.cameras.camera_array import CameraArray
from calipod.core import logger as calipod_logger
from calipod.motion_trial import MotionTrial

logger = calipod_logger.get(__name__)


class Interactive3DGraphWindow(QWidget):
    """An interactive matplotlib 3D graph visualization in a separate window."""

    def __init__(
        self,
        motion_trial: MotionTrial,
        camera_array: CameraArray,
        xyz_history_path: Path = None,
        is_filtered: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.motion_trial = motion_trial
        self.camera_array = camera_array
        self.xyz_history_path = xyz_history_path
        self.is_filtered = is_filtered
        self.setWindowTitle("3D Trajectory Graph (Interactive)" + (" - Filtered" if is_filtered else ""))
        self.setGeometry(100, 100, 1200, 900)

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            logger.error("Matplotlib not available; cannot create interactive graph.")
            return

        self.plt = plt
        self.FigureCanvas = FigureCanvas
        self.Poly3DCollection = Poly3DCollection

        layout = QVBoxLayout()

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 9), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Button row
        button_row = QHBoxLayout()
        export_button = QPushButton("Export Current View as PNG")
        export_button.clicked.connect(self.export_view_as_png)
        button_row.addWidget(export_button)
        layout.addLayout(button_row)

        self.setLayout(layout)
        self.plot_3d_graph()

    def plot_3d_graph(self):
        """Plot the 3D graph with all objects."""
        MM_PER_M = 1000

        # Clear previous plot
        self.ax.clear()

        # Collect all fly xyz data (point_id 0)
        all_xyz = []
        if hasattr(self.motion_trial, "xyz_df") and not self.motion_trial.xyz_df.empty:
            fly_mask = self.motion_trial.xyz_df["point_id"] == 0
            fly_data = self.motion_trial.xyz_df[fly_mask].sort_values("sync_index")

            if not fly_data.empty:
                x = fly_data["x_coord"].values * MM_PER_M
                y = fly_data["y_coord"].values * MM_PER_M
                z = fly_data["z_coord"].values * MM_PER_M
                all_xyz = list(zip(x, y, z))

        # Plot arena (corner points: 1-8)
        arena_points = {i: None for i in range(1, 9)}
        if hasattr(self.motion_trial, "xyz_df") and not self.motion_trial.xyz_df.empty:
            for pid in range(1, 9):
                mask = self.motion_trial.xyz_df["point_id"] == pid
                if mask.any():
                    data = self.motion_trial.xyz_df[mask]
                    x_mean = data["x_coord"].mean() * MM_PER_M
                    y_mean = data["y_coord"].mean() * MM_PER_M
                    z_mean = data["z_coord"].mean() * MM_PER_M
                    arena_points[pid] = np.array([x_mean, y_mean, z_mean])

        # Draw arena wireframe edges
        if all(arena_points[i] is not None for i in range(1, 9)):
            edge_pairs = [
                (1, 3),
                (2, 4),
                (5, 8),
                (6, 7),  # Vertical
                (1, 2),
                (1, 5),
                (2, 6),
                (5, 6),  # Top
                (3, 4),
                (3, 8),
                (4, 7),
                (8, 7),  # Bottom
            ]
            for i, j in edge_pairs:
                if arena_points[i] is not None and arena_points[j] is not None:
                    x = [arena_points[i][0], arena_points[j][0]]
                    y = [arena_points[i][1], arena_points[j][1]]
                    z = [arena_points[i][2], arena_points[j][2]]
                    self.ax.plot(x, y, z, "k-", alpha=0.3, linewidth=1)

            # Draw arena floor as filled polygon (bottom 4 corners)
            if all(arena_points[i] is not None for i in [3, 4, 7, 8]):
                floor_verts = [
                    [arena_points[3], arena_points[4], arena_points[7]],
                    [arena_points[3], arena_points[7], arena_points[8]],
                ]
                floor_collection = self.Poly3DCollection(floor_verts, alpha=0.2, facecolor="gray", edgecolor="black")
                self.ax.add_collection3d(floor_collection)

        # Plot leaves as filled square (point_id 10 center + corners 10000-10003)
        leaves_center = None
        leaves_corners = [None] * 4
        if hasattr(self.motion_trial, "xyz_df") and not self.motion_trial.xyz_df.empty:
            center_mask = self.motion_trial.xyz_df["point_id"] == 10
            if center_mask.any():
                center_data = self.motion_trial.xyz_df[center_mask]
                leaves_center = np.array(
                    [
                        center_data["x_coord"].mean() * MM_PER_M,
                        center_data["y_coord"].mean() * MM_PER_M,
                        center_data["z_coord"].mean() * MM_PER_M,
                    ]
                )

            for i, corner_id in enumerate([10000, 10001, 10002, 10003]):
                corner_mask = self.motion_trial.xyz_df["point_id"] == corner_id
                if corner_mask.any():
                    corner_data = self.motion_trial.xyz_df[corner_mask]
                    leaves_corners[i] = np.array(
                        [
                            corner_data["x_coord"].mean() * MM_PER_M,
                            corner_data["y_coord"].mean() * MM_PER_M,
                            corner_data["z_coord"].mean() * MM_PER_M,
                        ]
                    )

        if leaves_center is not None and all(c is not None for c in leaves_corners):
            leaves_verts = [
                [leaves_corners[0], leaves_corners[1], leaves_corners[2]],
                [leaves_corners[0], leaves_corners[2], leaves_corners[3]],
            ]
            leaves_collection = self.Poly3DCollection(
                leaves_verts,
                alpha=0.35,
                facecolor="green",
                edgecolor="darkgreen",
                linewidth=1.5,
                zorder=1,
            )
            self.ax.add_collection3d(leaves_collection)

        # Plot strawberry as hemisphere (point_id 9 center + corners)
        fruit_center = None
        fruit_corners = [None] * 4
        if hasattr(self.motion_trial, "xyz_df") and not self.motion_trial.xyz_df.empty:
            center_mask = self.motion_trial.xyz_df["point_id"] == 9
            if center_mask.any():
                center_data = self.motion_trial.xyz_df[center_mask]
                fruit_center = np.array(
                    [
                        center_data["x_coord"].mean() * MM_PER_M,
                        center_data["y_coord"].mean() * MM_PER_M,
                        center_data["z_coord"].mean() * MM_PER_M,
                    ]
                )

            for i, corner_id in enumerate([9000, 9001, 9002, 9003]):
                corner_mask = self.motion_trial.xyz_df["point_id"] == corner_id
                if corner_mask.any():
                    corner_data = self.motion_trial.xyz_df[corner_mask]
                    fruit_corners[i] = np.array(
                        [
                            corner_data["x_coord"].mean() * MM_PER_M,
                            corner_data["y_coord"].mean() * MM_PER_M,
                            corner_data["z_coord"].mean() * MM_PER_M,
                        ]
                    )

        if fruit_center is not None and all(c is not None for c in fruit_corners):
            # Create oriented hemisphere from corners using SVD plane
            corners_array = np.array(fruit_corners)

            # Order corners by angle on their plane
            centroid = corners_array.mean(axis=0)
            centered = corners_array - centroid
            _, _, vh = np.linalg.svd(centered)
            plane_normal = vh[2]
            axis_x = vh[0]
            axis_y = np.cross(plane_normal, axis_x)

            # Calculate radius from bounding box
            width_3d = np.linalg.norm(corners_array[1] - corners_array[0])
            height_3d = np.linalg.norm(corners_array[3] - corners_array[0])
            radius = min(width_3d, height_3d) * 0.5 * 1.05

            # Generate hemisphere vertices oriented along plane normal
            u = np.linspace(0, 2 * np.pi, 16)
            v = np.linspace(0, np.pi / 2, 8)

            fruit_verts = []
            for i in range(len(u) - 1):
                for j in range(len(v) - 1):
                    # Generate vertices for this quad
                    for ui, vi in [(u[i], v[j]), (u[i + 1], v[j]), (u[i], v[j + 1])]:
                        height = radius * np.cos(vi)
                        ring_r = radius * np.sin(vi)
                        offset = axis_x * (ring_r * np.cos(ui)) + axis_y * (ring_r * np.sin(ui))
                        vertex = fruit_center + plane_normal * height + offset
                        fruit_verts.append(vertex)

            # Reshape verts for Poly3DCollection (triangles)
            fruit_tris = [fruit_verts[i * 3 : (i + 1) * 3] for i in range(len(fruit_verts) // 3)]
            fruit_collection = self.Poly3DCollection(
                fruit_tris,
                alpha=0.9,
                facecolor="red",
                edgecolor="darkred",
                linewidth=0.5,
                zorder=3,
            )
            self.ax.add_collection3d(fruit_collection)

        # Plot camera origin points
        if self.camera_array and hasattr(self.camera_array, "cameras"):
            for port, cam in self.camera_array.cameras.items():
                if hasattr(cam, "extrinsic_matrix") and cam.extrinsic_matrix is not None:
                    ext = cam.extrinsic_matrix
                    cam_pos = -ext[:3, :3].T @ ext[:3, 3]
                    self.ax.scatter(
                        [cam_pos[0] * MM_PER_M],
                        [cam_pos[1] * MM_PER_M],
                        [cam_pos[2] * MM_PER_M],
                        s=100,
                        marker="*",
                        label=f"Camera {port}",
                    )

        # Plot fly track last so it renders on top of arena and objects
        if all_xyz:
            x_fly, y_fly, z_fly = zip(*all_xyz)
            self.ax.plot(x_fly, y_fly, z_fly, "b-", linewidth=2, alpha=0.6, label="Fly Track (Ground Truth)")
            self.ax.scatter([x_fly[0]], [y_fly[0]], [z_fly[0]], color="green", s=10, marker="o", label="Start")
            self.ax.scatter([x_fly[-1]], [y_fly[-1]], [z_fly[-1]], color="black", s=10, marker="s", label="End")

        # Plot predictions if available
        if hasattr(self.motion_trial, "predictions_df") and not self.motion_trial.predictions_df.empty:
            pred_mask = self.motion_trial.predictions_df["point_id"] == 0
            pred_data = self.motion_trial.predictions_df[pred_mask].sort_values("sync_index")

            if not pred_data.empty:
                x_pred = pred_data["x_coord"].values * MM_PER_M
                y_pred = pred_data["y_coord"].values * MM_PER_M
                z_pred = pred_data["z_coord"].values * MM_PER_M
                self.ax.plot(
                    x_pred,
                    y_pred,
                    z_pred,
                    color="orange",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.9,
                    label="Fly Track (Predictions)",
                )
                self.ax.scatter(
                    [x_pred[0]], [y_pred[0]], [z_pred[0]], color="lightsalmon", s=10, marker="o", label="Pred Start"
                )
                self.ax.scatter(
                    [x_pred[-1]], [y_pred[-1]], [z_pred[-1]], color="darkorange", s=10, marker="s", label="Pred End"
                )

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        title = "3D Trajectory with Arena, Leaves, and Strawberry"
        if self.is_filtered:
            title += " (Filtered Track)"
        self.ax.set_title(title)
        self.ax.legend()

        # Set camera view: elevation 20°, azimuth -60° (front-facing, looking slightly down)
        self.ax.view_init(elev=20, azim=-60)

        # Set equal aspect ratio for all axes to prevent distortion
        self.ax.set_box_aspect([1, 1, 1])

        if all_xyz:
            all_points = np.array(all_xyz + [p for p in arena_points.values() if p is not None])
        else:
            all_points = np.array([p for p in arena_points.values() if p is not None])

        if len(all_points) > 0:
            max_range = (
                np.array(
                    [
                        all_points[:, 0].max() - all_points[:, 0].min(),
                        all_points[:, 1].max() - all_points[:, 1].min(),
                        all_points[:, 2].max() - all_points[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.canvas.draw()

    def export_view_as_png(self):
        """Export the current view as PNG."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Graph As", "", "PNG Images (*.png)")

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=150, bbox_inches="tight")
                logger.info(f"Graph exported to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to export graph: {e}")
