"""Matplotlib popup for arena simulation frustum and overlap visualization."""

from itertools import combinations
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
)
from scipy.spatial import ConvexHull, QhullError

import calipod.logger
from calipod.arena_simulation.arena_overlap import (
    get_camera_world_frustum_geometry,
    halfspaces_from_convex_mesh,
    intersection_vertices_from_halfspaces,
)

logger = calipod.logger.get(__name__)


class ArenaMatplotlibGraphWindow(QWidget):
    """Interactive matplotlib window for arena simulation geometry."""

    def __init__(self, arena_state: dict, arena_sim_dir: Path, parent=None):
        # Force this widget to be a standalone top-level window (not embedded in tabs).
        super().__init__(None)
        self.arena_state = arena_state
        self.arena_sim_dir = Path(arena_sim_dir)
        self.view_mode = "all"  # "all" or "intersection_only"
        self.setWindowFlag(Qt.Window, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Arena Simulation Graph")
        self.setGeometry(120, 120, 1200, 900)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from matplotlib.lines import Line2D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from matplotlib.patches import Patch
        except ImportError:
            logger.error("Matplotlib not available; cannot create arena graph popup.")
            return

        self.FigureCanvas = FigureCanvas
        self.Figure = Figure
        self.Line2D = Line2D
        self.Poly3DCollection = Poly3DCollection
        self.Patch = Patch

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.fig = Figure(figsize=(12, 9), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        view_mode_row = QHBoxLayout()
        view_mode_row.addStretch()
        self.view_mode_group = QButtonGroup()
        self.all_radio = QRadioButton("All")
        self.intersection_only_radio = QRadioButton("Intersection Only")
        self.all_radio.setChecked(True)
        self.view_mode_group.addButton(self.all_radio, 0)
        self.view_mode_group.addButton(self.intersection_only_radio, 1)
        self.view_mode_group.idClicked.connect(self._on_view_mode_changed)
        view_mode_row.addWidget(self.all_radio)
        view_mode_row.addWidget(self.intersection_only_radio)
        view_mode_row.addStretch()
        layout.addLayout(view_mode_row)

        button_row = QHBoxLayout()
        save_button = QPushButton("Save Current View as PNG")
        save_button.clicked.connect(self.save_current_view)
        button_row.addWidget(save_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.setLayout(layout)
        self.plot_arena_graph()

    def _state(self, key: str, default):
        return self.arena_state.get(key, default)

    def _on_view_mode_changed(self, mode_id: int):
        """Update view mode and redraw graph."""
        if mode_id == 0:
            self.view_mode = "all"
        else:
            self.view_mode = "intersection_only"
        self.plot_arena_graph()

    def _compute_overlap_polyhedra_mm(self):
        camera_count = int(self._state("camera_count", 0))
        overlap_mode = self._state("overlap_mode", "min_two")
        camera_frustum_angles_deg = self._state("camera_frustum_angles_deg", {})
        camera_translations_mm = self._state("camera_translations_mm", {})
        camera_rotations_deg = self._state("camera_rotations_deg", {})
        visualised_frustum_depth_cm = float(self._state("visualised_frustum_depth_cm", 100.0))
        mm_to_scene_scale = float(self._state("mm_to_scene_scale", 0.001))

        camera_indices = list(range(max(0, camera_count)))
        if len(camera_indices) < 2:
            return []

        frustum_halfspaces = {}
        for camera_index in camera_indices:
            verts_scene, faces = get_camera_world_frustum_geometry(
                camera_index=camera_index,
                camera_frustum_angles_deg=camera_frustum_angles_deg,
                camera_translations_mm=camera_translations_mm,
                camera_rotations_deg=camera_rotations_deg,
                visualised_frustum_depth_cm=visualised_frustum_depth_cm,
                mm_to_scene_scale=mm_to_scene_scale,
            )
            halfspaces = halfspaces_from_convex_mesh(verts_scene, faces)
            if halfspaces.size > 0:
                frustum_halfspaces[camera_index] = halfspaces

        if len(frustum_halfspaces) < 2:
            return []

        combined_sets = []
        ordered = sorted(frustum_halfspaces.keys())
        if overlap_mode == "max_all":
            combined_sets.append(np.vstack([frustum_halfspaces[idx] for idx in ordered]))
        else:
            for idx_a, idx_b in combinations(ordered, 2):
                combined_sets.append(np.vstack([frustum_halfspaces[idx_a], frustum_halfspaces[idx_b]]))

        overlap_vertices_mm = []
        for halfspaces in combined_sets:
            verts_scene = intersection_vertices_from_halfspaces(halfspaces)
            if verts_scene is None:
                continue
            verts_mm = verts_scene / max(mm_to_scene_scale, 1e-12)
            overlap_vertices_mm.append(verts_mm)

        return overlap_vertices_mm

    def _build_camera_legend_handles(self, view_mode: str):
        handles = []
        if view_mode == "all":
            camera_colors = self._state("camera_colors", {})
            for camera_index in range(int(self._state("camera_count", 0))):
                color = camera_colors.get(camera_index, (0.3, 0.3, 0.3, 1.0))
                rgb = tuple(color[:3])
                handles.append(
                    self.Patch(facecolor=rgb, edgecolor=rgb, alpha=0.35, label=f"Camera {camera_index + 1} Frustum")
                )
                handles.append(
                    self.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=rgb,
                        markeredgecolor=rgb,
                        markersize=7,
                        label=f"Camera {camera_index + 1} Optical Centre",
                    )
                )
            handles.append(self.Patch(facecolor=(1.0, 1.0, 1.0), edgecolor=(0.0, 0.0, 0.0), alpha=0.35, label="Intersection"))
        else:  # intersection_only
            handles.append(self.Patch(facecolor=(1.0, 1.0, 1.0), edgecolor=(0.0, 0.0, 0.0), alpha=0.35, label="Intersection Mesh"))
        return handles

    @staticmethod
    def _promote_poly_collection(poly_collection, sort_zpos: float):
        """Bias a 3D collection to draw later in mplot3d's painter-style ordering."""
        if hasattr(poly_collection, "set_zsort"):
            poly_collection.set_zsort("max")
        if hasattr(poly_collection, "set_sort_zpos"):
            poly_collection.set_sort_zpos(sort_zpos)

    def plot_arena_graph(self):
        self.ax.clear()

        camera_count = int(self._state("camera_count", 0))
        camera_frustum_angles_deg = self._state("camera_frustum_angles_deg", {})
        camera_translations_mm = self._state("camera_translations_mm", {})
        camera_rotations_deg = self._state("camera_rotations_deg", {})
        visualised_frustum_depth_cm = float(self._state("visualised_frustum_depth_cm", 100.0))
        mm_to_scene_scale = float(self._state("mm_to_scene_scale", 0.001))
        camera_colors = self._state("camera_colors", {})

        all_points = []
        scaling_points = []  # Points used for scaling even if not rendered
        
        # Compute camera frustum geometry (render only if "all" mode, always use for scaling)
        for camera_index in range(max(0, camera_count)):
            verts_scene, faces = get_camera_world_frustum_geometry(
                camera_index=camera_index,
                camera_frustum_angles_deg=camera_frustum_angles_deg,
                camera_translations_mm=camera_translations_mm,
                camera_rotations_deg=camera_rotations_deg,
                visualised_frustum_depth_cm=visualised_frustum_depth_cm,
                mm_to_scene_scale=mm_to_scene_scale,
            )
            verts_mm = verts_scene / max(mm_to_scene_scale, 1e-12)
            scaling_points.append(verts_mm)
            
            if self.view_mode == "all":
                all_points.append(verts_mm)
                tris = [[verts_mm[face[0]], verts_mm[face[1]], verts_mm[face[2]]] for face in faces]
                cam_color = camera_colors.get(camera_index, (0.4, 0.4, 0.4, 1.0))
                poly = self.Poly3DCollection(
                    tris,
                    alpha=0.22,
                    facecolor=tuple(cam_color[:3]),
                    edgecolor=tuple(cam_color[:3]),
                    linewidth=0.8,
                )
                self._promote_poly_collection(poly, sort_zpos=float(np.max(verts_mm[:, 2])))
                self.ax.add_collection3d(poly)

                origin = np.asarray(camera_translations_mm.get(camera_index, (0.0, 0.0, 0.0)), dtype=float)
                self.ax.scatter([origin[0]], [origin[1]], [origin[2]], color=tuple(cam_color[:3]), s=32)

        # Draw intersection polyhedra (shown in both modes)
        for overlap_verts_mm in self._compute_overlap_polyhedra_mm():
            try:
                hull = ConvexHull(overlap_verts_mm)
            except QhullError:
                continue

            hull_tris = [
                [
                    overlap_verts_mm[simplex[0]],
                    overlap_verts_mm[simplex[1]],
                    overlap_verts_mm[simplex[2]],
                ]
                for simplex in hull.simplices
            ]
            overlap_poly = self.Poly3DCollection(
                hull_tris,
                alpha=0.35,
                facecolor=(1.0, 1.0, 1.0),
                edgecolor=(0.0, 0.0, 0.0),
                linewidth=1.2,
            )
            self._promote_poly_collection(overlap_poly, sort_zpos=1e9)
            self.ax.add_collection3d(overlap_poly)
            all_points.append(overlap_verts_mm)

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        
        if self.view_mode == "all":
            self.ax.set_title("Arena Simulation Frustum Layout")
        else:
            self.ax.set_title("Arena Simulation - Intersection Mesh Only")

        legend_handles = self._build_camera_legend_handles(self.view_mode)
        self.ax.legend(handles=legend_handles, loc="upper right")

        # Use all_points if available (for normal content), otherwise use scaling_points for axis bounds
        points_for_scaling = all_points if all_points else scaling_points
        if points_for_scaling:
            points = np.vstack(points_for_scaling)
            center = points.mean(axis=0)
            max_range = max(points.max(axis=0) - points.min(axis=0)) * 0.6
            max_range = max(max_range, 100.0)
            self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
            self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
            self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
            self.ax.set_box_aspect([1, 1, 1])

        self.ax.view_init(elev=20, azim=-60)
        self.canvas.draw()

    def save_current_view(self):
        self.arena_sim_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(self.arena_sim_dir / "arena_sim_graph.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Arena Graph As",
            default_path,
            "PNG Images (*.png)",
        )
        if not file_path:
            return

        try:
            self.fig.savefig(file_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved arena graph image to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save arena graph image: {e}")
