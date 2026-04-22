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

from calipod.annotation_management.annotations_config_manager import (
    STRUCTURE_GEOMETRY_FLAT,
    STRUCTURE_GEOMETRY_SEMI_SPHERE,
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
        frame_roi_structures: list[dict] | None = None,
        arena_vertices: list[dict] | None = None,
        show_environment_structures: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.motion_trial = motion_trial
        self.camera_array = camera_array
        self.xyz_history_path = xyz_history_path
        self.is_filtered = is_filtered
        self.frame_roi_structures = frame_roi_structures or []
        self.arena_vertices = arena_vertices or []
        self.show_environment_structures = bool(show_environment_structures)
        self.setWindowTitle("3D Trajectory Graph (Interactive)" + (" - Filtered" if is_filtered else ""))
        self.setGeometry(100, 100, 1200, 900)

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
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

        structure_geometry_by_id = {
            int(item["id"]): item.get("geometry", STRUCTURE_GEOMETRY_FLAT)
            for item in self.frame_roi_structures
            if item.get("enabled", True)
        }

        # Draw arena perimeter and floor from user-selected arena-vertex metadata.
        if self.show_environment_structures and hasattr(self.motion_trial, "xyz_df") \
        and not self.motion_trial.xyz_df.empty:
            enabled_ids = [int(item["id"]) for item in self.arena_vertices if item.get("enabled", True)]
            floor_ids = [
                int(item["id"])
                for item in self.arena_vertices
                if item.get("enabled", True) and item.get("add_to_floor", False)
            ]

            enabled_points = []
            for pid in enabled_ids:
                mask = self.motion_trial.xyz_df["point_id"] == pid
                if not mask.any():
                    continue
                data = self.motion_trial.xyz_df[mask]
                point = np.array(
                    [
                        data["x_coord"].mean() * MM_PER_M,
                        data["y_coord"].mean() * MM_PER_M,
                        data["z_coord"].mean() * MM_PER_M,
                    ]
                )
                enabled_points.append((pid, point))
                self.ax.scatter([point[0]], [point[1]], [point[2]], color="black", s=14)

            if len(enabled_points) >= 3:
                id_to_coord = {pid: coord for pid, coord in enabled_points}
                edge_pairs = self._build_arena_edge_pairs(enabled_points)
                for point_id_a, point_id_b in edge_pairs:
                    a = id_to_coord[point_id_a]
                    b = id_to_coord[point_id_b]
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "k-", alpha=0.3, linewidth=1)

            floor_points = []
            for pid in floor_ids:
                mask = self.motion_trial.xyz_df["point_id"] == pid
                if not mask.any():
                    continue
                data = self.motion_trial.xyz_df[mask]
                floor_points.append(
                    np.array(
                        [
                            data["x_coord"].mean() * MM_PER_M,
                            data["y_coord"].mean() * MM_PER_M,
                            data["z_coord"].mean() * MM_PER_M,
                        ]
                    )
                )

            if len(floor_points) >= 4:
                ordered_floor = self._order_corners_on_plane(np.array(floor_points, dtype=np.float32))
                floor_tris = []
                for tri_idx in range(1, len(ordered_floor) - 1):
                    floor_tris.append([ordered_floor[0], ordered_floor[tri_idx], ordered_floor[tri_idx + 1]])
                floor_collection = self.Poly3DCollection(floor_tris, alpha=0.2, facecolor="gray", edgecolor="black")
                self.ax.add_collection3d(floor_collection)

        # Plot configured frame-ROI structures from annotation labels.
        if self.show_environment_structures and hasattr(self.motion_trial, "xyz_df") \
        and not self.motion_trial.xyz_df.empty:
            for point_id, geometry in structure_geometry_by_id.items():
                center_mask = self.motion_trial.xyz_df["point_id"] == point_id
                if not center_mask.any():
                    continue

                center_data = self.motion_trial.xyz_df[center_mask]
                center = np.array(
                    [
                        center_data["x_coord"].mean() * MM_PER_M,
                        center_data["y_coord"].mean() * MM_PER_M,
                        center_data["z_coord"].mean() * MM_PER_M,
                    ]
                )

                corner_ids = [point_id * 1000 + i for i in range(4)]
                corners = []
                for corner_id in corner_ids:
                    corner_mask = self.motion_trial.xyz_df["point_id"] == corner_id
                    if not corner_mask.any():
                        continue
                    corner_data = self.motion_trial.xyz_df[corner_mask]
                    corners.append(
                        np.array(
                            [
                                corner_data["x_coord"].mean() * MM_PER_M,
                                corner_data["y_coord"].mean() * MM_PER_M,
                                corner_data["z_coord"].mean() * MM_PER_M,
                            ]
                        )
                    )

                if len(corners) != 4:
                    continue

                ordered = self._order_corners_on_plane(np.array(corners, dtype=np.float32))

                if geometry == STRUCTURE_GEOMETRY_FLAT:
                    flat_verts = [
                        [ordered[0], ordered[1], ordered[2]],
                        [ordered[0], ordered[2], ordered[3]],
                    ]
                    flat_collection = self.Poly3DCollection(
                        flat_verts,
                        alpha=0.35,
                        facecolor="green",
                        edgecolor="darkgreen",
                        linewidth=1.5,
                        zorder=1,
                    )
                    self.ax.add_collection3d(flat_collection)
                    continue

                if geometry == STRUCTURE_GEOMETRY_SEMI_SPHERE:
                    corners_array = ordered
                    centroid = corners_array.mean(axis=0)
                    centered = corners_array - centroid
                    _, _, vh = np.linalg.svd(centered)
                    plane_normal = vh[2]
                    axis_x = vh[0]
                    axis_y = np.cross(plane_normal, axis_x)

                    width_3d = np.linalg.norm(corners_array[1] - corners_array[0])
                    height_3d = np.linalg.norm(corners_array[3] - corners_array[0])
                    radius = min(width_3d, height_3d) * 0.5 * 1.05

                    u = np.linspace(0, 2 * np.pi, 16)
                    v = np.linspace(0, np.pi / 2, 8)

                    dome_verts = []
                    for i in range(len(u) - 1):
                        for j in range(len(v) - 1):
                            for ui, vi in [(u[i], v[j]), (u[i + 1], v[j]), (u[i], v[j + 1])]:
                                height = radius * np.cos(vi)
                                ring_r = radius * np.sin(vi)
                                offset = axis_x * (ring_r * np.cos(ui)) + axis_y * (ring_r * np.sin(ui))
                                vertex = center + plane_normal * height + offset
                                dome_verts.append(vertex)

                    dome_tris = [dome_verts[i * 3 : (i + 1) * 3] for i in range(len(dome_verts) // 3)]
                    dome_collection = self.Poly3DCollection(
                        dome_tris,
                        alpha=0.9,
                        facecolor="red",
                        edgecolor="darkred",
                        linewidth=0.5,
                        zorder=3,
                    )
                    self.ax.add_collection3d(dome_collection)

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

        arena_points_for_bounds = []
        if hasattr(self.motion_trial, "xyz_df") and not self.motion_trial.xyz_df.empty:
            for item in self.arena_vertices:
                pid = int(item["id"])
                mask = self.motion_trial.xyz_df["point_id"] == pid
                if mask.any():
                    data = self.motion_trial.xyz_df[mask]
                    arena_points_for_bounds.append(
                        np.array(
                            [
                                data["x_coord"].mean() * MM_PER_M,
                                data["y_coord"].mean() * MM_PER_M,
                                data["z_coord"].mean() * MM_PER_M,
                            ]
                        )
                    )

        if all_xyz:
            all_points = np.array(all_xyz + arena_points_for_bounds)
        else:
            all_points = np.array(arena_points_for_bounds)

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

    def _order_corners_on_plane(self, corners: np.ndarray) -> np.ndarray:
        """Return corners ordered consistently around their best-fit plane."""
        centroid = corners.mean(axis=0)
        centered = corners - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        axis_x = vh[0]
        axis_y = np.cross(normal, axis_x)
        proj_x = centered @ axis_x
        proj_y = centered @ axis_y
        angles = np.arctan2(proj_y, proj_x)
        return corners[np.argsort(angles)]

    def _order_ids_on_plane(self, id_coord_pairs: list[tuple[int, np.ndarray]]) -> list[int]:
        """Order point IDs around the best-fit plane of their coordinates."""
        ids = [pair[0] for pair in id_coord_pairs]
        coords = np.array([pair[1] for pair in id_coord_pairs], dtype=np.float32)
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        axis_x = vh[0]
        axis_y = np.cross(normal, axis_x)
        proj_x = centered @ axis_x
        proj_y = centered @ axis_y
        angles = np.arctan2(proj_y, proj_x)
        order = np.argsort(angles)
        return [ids[idx] for idx in order]

    def _build_arena_edge_pairs(self, id_coord_pairs: list[tuple[int, np.ndarray]]) -> list[tuple[int, int]]:
        """Build robust arena edges from selected vertices.

        - Coplanar selections: connect as a closed ordered loop.
        - Non-coplanar selections: connect each point to its nearest neighbors.
        """
        if len(id_coord_pairs) < 2:
            return []

        ids = [pair[0] for pair in id_coord_pairs]
        coords = np.array([pair[1] for pair in id_coord_pairs], dtype=np.float32)

        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        plane_distances = np.abs(centered @ normal)
        max_plane_dist = float(np.max(plane_distances)) if len(plane_distances) else 0.0

        if max_plane_dist < 0.01 or len(ids) <= 4:
            ordered_ids = self._order_ids_on_plane(id_coord_pairs)
            return [
                (ordered_ids[i], ordered_ids[(i + 1) % len(ordered_ids)])
                for i in range(len(ordered_ids))
            ]

        neighbor_count = min(3, len(ids) - 1)
        edge_set = set()
        for i, point_id in enumerate(ids):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            nearest_indices = np.argsort(dists)[1 : neighbor_count + 1]
            for j in nearest_indices:
                edge_set.add(tuple(sorted((point_id, ids[j]))))

        return list(edge_set)

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
