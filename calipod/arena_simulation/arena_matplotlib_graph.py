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
    QCheckBox,
)
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog

from calipod.core import logger as calipod_logger
from calipod.arena_simulation.arena_overlap import (
    get_camera_world_frustum_geometry,
    halfspaces_from_convex_mesh,
    intersection_vertices_from_halfspaces,
)

logger = calipod_logger.get(__name__)


class ArenaMatplotlibGraphWindow(QWidget):
    """Interactive matplotlib window for arena simulation geometry."""

    def __init__(self, arena_state: dict, arena_sim_dir: Path, parent=None):
        # Force this widget to be a standalone top-level window (not embedded in tabs).
        super().__init__(None)
        self.arena_state = arena_state
        self.arena_sim_dir = Path(arena_sim_dir)
        self.view_mode = "all"  # "all", "intersection_only", or "largest_cube"
        self.show_intersection_measurements = False
        self._measurement_text_artist = None
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
        self.largest_cube_radio = QRadioButton("Largest Intersection Cube")
        self.all_radio.setChecked(True)
        self.view_mode_group.addButton(self.all_radio, 0)
        self.view_mode_group.addButton(self.intersection_only_radio, 1)
        self.view_mode_group.addButton(self.largest_cube_radio, 2)
        self.view_mode_group.idClicked.connect(self._on_view_mode_changed)
        view_mode_row.addWidget(self.all_radio)
        view_mode_row.addWidget(self.intersection_only_radio)
        view_mode_row.addWidget(self.largest_cube_radio)
        self.show_measurements_checkbox = QCheckBox("Show Intersection Measurements")
        self.show_measurements_checkbox.toggled.connect(self._on_measurements_toggled)
        view_mode_row.addWidget(self.show_measurements_checkbox)
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
        elif mode_id == 1:
            self.view_mode = "intersection_only"
        else:
            self.view_mode = "largest_cube"
        self.plot_arena_graph()

    def _on_measurements_toggled(self, checked: bool):
        """Toggle intersection measurement overlays."""
        self.show_intersection_measurements = bool(checked)
        self.plot_arena_graph()

    @staticmethod
    def _angle_deg_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute the angle in degrees between two vectors."""
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return float("nan")
        cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _intersection_measurements(self, hull, overlap_verts_mm: np.ndarray) -> tuple[list[dict], list[dict]]:
        """Build edge-length and vertex-angle measurements from a convex hull."""
        edge_set = set()
        for simplex in hull.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            edge_set.add(tuple(sorted((a, b))))
            edge_set.add(tuple(sorted((b, c))))
            edge_set.add(tuple(sorted((a, c))))

        sorted_edges = sorted(edge_set)
        edge_measurements = []
        neighbours: dict[int, set[int]] = {}
        for edge_id, (idx_a, idx_b) in enumerate(sorted_edges, start=1):
            point_a = overlap_verts_mm[idx_a]
            point_b = overlap_verts_mm[idx_b]
            midpoint = (point_a + point_b) * 0.5
            length_mm = float(np.linalg.norm(point_b - point_a))
            edge_measurements.append(
                {
                    "id": edge_id,
                    "a": idx_a,
                    "b": idx_b,
                    "midpoint": midpoint,
                    "length_mm": length_mm,
                }
            )
            neighbours.setdefault(idx_a, set()).add(idx_b)
            neighbours.setdefault(idx_b, set()).add(idx_a)

        vertex_measurements = []
        for vertex_id in sorted(neighbours.keys()):
            connected = sorted(neighbours[vertex_id])
            if len(connected) < 2:
                continue

            center = overlap_verts_mm[vertex_id]
            vectors = [overlap_verts_mm[idx] - center for idx in connected]
            min_angle = float("inf")
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    angle_deg = self._angle_deg_between(vectors[i], vectors[j])
                    if np.isfinite(angle_deg) and angle_deg < min_angle:
                        min_angle = angle_deg

            if np.isfinite(min_angle):
                vertex_measurements.append(
                    {
                        "id": vertex_id,
                        "point": center,
                        "angle_deg": float(min_angle),
                    }
                )

        return edge_measurements, vertex_measurements

    @staticmethod
    def _cube_vertices_from_center(center: np.ndarray, half_side_mm: float) -> np.ndarray:
        """Build 8 vertices for an axis-aligned cube centered at center."""
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        h = float(half_side_mm)
        return np.array(
            [
                [cx - h, cy - h, cz - h],
                [cx + h, cy - h, cz - h],
                [cx + h, cy + h, cz - h],
                [cx - h, cy + h, cz - h],
                [cx - h, cy - h, cz + h],
                [cx + h, cy - h, cz + h],
                [cx + h, cy + h, cz + h],
                [cx - h, cy + h, cz + h],
            ],
            dtype=float,
        )

    @staticmethod
    def _cube_triangle_indices() -> list[tuple[int, int, int]]:
        """Return 12 triangles for cube faces."""
        return [
            (0, 1, 2), (0, 2, 3),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 1, 5), (0, 5, 4),  # front
            (1, 2, 6), (1, 6, 5),  # right
            (2, 3, 7), (2, 7, 6),  # back
            (3, 0, 4), (3, 4, 7),  # left
        ]

    def _largest_axis_aligned_cube_from_hull(self, hull: ConvexHull) -> dict | None:
        """Find largest axis-aligned cube inside a convex hull using linear programming."""
        equations = np.asarray(hull.equations, dtype=float)
        if equations.ndim != 2 or equations.shape[1] != 4:
            return None

        normals = equations[:, :3]
        offsets = equations[:, 3]

        # Variables: [cx, cy, cz, s] where s is half-side length (mm).
        a_ub = np.hstack([normals, np.sum(np.abs(normals), axis=1, keepdims=True)])
        b_ub = -offsets

        c_obj = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)  # maximize s
        bounds = [(None, None), (None, None), (None, None), (0.0, None)]

        lp_result = linprog(c=c_obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not lp_result.success:
            return None

        center = np.asarray(lp_result.x[:3], dtype=float)
        half_side_mm = float(lp_result.x[3])
        if half_side_mm <= 1e-6:
            return None

        vertices = self._cube_vertices_from_center(center=center, half_side_mm=half_side_mm)
        return {
            "center": center,
            "half_side_mm": half_side_mm,
            "side_mm": 2.0 * half_side_mm,
            "vertices": vertices,
        }

    def _cube_measurements(self, cube_vertices: np.ndarray) -> tuple[list[dict], list[dict]]:
        """Build edge and vertex measurements for the axis-aligned cube."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        edge_measurements = []
        for edge_id, (idx_a, idx_b) in enumerate(edges, start=1):
            point_a = cube_vertices[idx_a]
            point_b = cube_vertices[idx_b]
            edge_measurements.append(
                {
                    "id": edge_id,
                    "a": idx_a,
                    "b": idx_b,
                    "midpoint": 0.5 * (point_a + point_b),
                    "length_mm": float(np.linalg.norm(point_b - point_a)),
                }
            )

        vertex_measurements = [
            {"id": vertex_id, "point": cube_vertices[vertex_id], "angle_deg": 90.0}
            for vertex_id in range(8)
        ]
        return edge_measurements, vertex_measurements

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
        elif view_mode == "intersection_only":
            handles.append(self.Patch(facecolor=(1.0, 1.0, 1.0), edgecolor=(0.0, 0.0, 0.0), alpha=0.35, label="Intersection Mesh"))
        else:
            handles.append(self.Patch(facecolor=(1.0, 1.0, 1.0), edgecolor=(0.0, 0.0, 0.0), alpha=0.30, label="Intersection Mesh"))
            handles.append(self.Patch(facecolor=(0.65, 0.90, 1.0), edgecolor=(0.1, 0.4, 0.6), alpha=0.45, label="Largest Fitting Cube"))
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
        if self._measurement_text_artist is not None:
            try:
                self._measurement_text_artist.remove()
            except Exception:
                pass
            self._measurement_text_artist = None

        self.fig.subplots_adjust(right=0.80 if self.show_intersection_measurements else 0.95)

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

        # Build overlap polyhedra once for all display/measurement modes.
        overlap_polyhedra = []
        for overlap_verts_mm in self._compute_overlap_polyhedra_mm():
            try:
                hull = ConvexHull(overlap_verts_mm)
            except QhullError:
                continue
            overlap_polyhedra.append((overlap_verts_mm, hull))

        # Draw intersection polyhedra (shown in all/intersection_only/largest_cube modes)
        measurement_lines: list[str] = []
        mesh_counter = 0
        if self.view_mode in {"all", "intersection_only", "largest_cube"}:
            for overlap_verts_mm, hull in overlap_polyhedra:
                mesh_counter += 1

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

                if self.show_intersection_measurements and self.view_mode != "largest_cube":
                    edge_measurements, vertex_measurements = self._intersection_measurements(hull, overlap_verts_mm)

                    measurement_lines.append(f"Mesh {mesh_counter}")
                    for edge in edge_measurements:
                        midpoint = edge["midpoint"]
                        edge_tag = f"E{mesh_counter}.{edge['id']}"
                        self.ax.text(
                            float(midpoint[0]),
                            float(midpoint[1]),
                            float(midpoint[2]),
                            edge_tag,
                            fontsize=7,
                            color="#1f77b4",
                        )
                        measurement_lines.append(
                            f"  {edge_tag}: {edge['length_mm']:.2f} mm"
                        )

                    for vertex in vertex_measurements:
                        point = vertex["point"]
                        vertex_tag = f"V{mesh_counter}.{int(vertex['id'])}"
                        self.ax.text(
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            vertex_tag,
                            fontsize=7,
                            color="#d62728",
                        )
                        measurement_lines.append(
                            f"  {vertex_tag}: {vertex['angle_deg']:.1f} deg"
                        )

        if self.view_mode == "largest_cube":
            best_cube = None
            for overlap_verts_mm, hull in overlap_polyhedra:
                candidate_cube = self._largest_axis_aligned_cube_from_hull(hull)
                if candidate_cube is None:
                    continue
                if best_cube is None or candidate_cube["side_mm"] > best_cube["side_mm"]:
                    best_cube = candidate_cube

            if best_cube is not None:
                cube_vertices = best_cube["vertices"]
                cube_tris = [
                    [cube_vertices[a], cube_vertices[b], cube_vertices[c]]
                    for a, b, c in self._cube_triangle_indices()
                ]
                cube_poly = self.Poly3DCollection(
                    cube_tris,
                    alpha=0.45,
                    facecolor=(0.65, 0.90, 1.0),
                    edgecolor=(0.1, 0.4, 0.6),
                    linewidth=1.4,
                )
                self._promote_poly_collection(cube_poly, sort_zpos=1e9)
                self.ax.add_collection3d(cube_poly)
                all_points.append(cube_vertices)

                if self.show_intersection_measurements:
                    edge_measurements, vertex_measurements = self._cube_measurements(cube_vertices)
                    measurement_lines.append("Largest Fitting Cube")
                    measurement_lines.append(f"  side length: {best_cube['side_mm']:.2f} mm")

                    for edge in edge_measurements:
                        midpoint = edge["midpoint"]
                        edge_tag = f"E1.{edge['id']}"
                        self.ax.text(
                            float(midpoint[0]),
                            float(midpoint[1]),
                            float(midpoint[2]),
                            edge_tag,
                            fontsize=7,
                            color="#1f77b4",
                        )
                        measurement_lines.append(
                            f"  {edge_tag}: {edge['length_mm']:.2f} mm"
                        )

                    for vertex in vertex_measurements:
                        point = vertex["point"]
                        vertex_tag = f"V1.{int(vertex['id'])}"
                        self.ax.text(
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            vertex_tag,
                            fontsize=7,
                            color="#d62728",
                        )
                        measurement_lines.append(
                            f"  {vertex_tag}: {vertex['angle_deg']:.1f} deg"
                        )

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        
        if self.view_mode == "all":
            self.ax.set_title("Arena Simulation Frustum Layout")
        elif self.view_mode == "intersection_only":
            self.ax.set_title("Arena Simulation - Intersection Mesh Only")
        else:
            self.ax.set_title("Arena Simulation - Largest Fitting Cube in Intersection")

        legend_handles = self._build_camera_legend_handles(self.view_mode)
        self.ax.legend(handles=legend_handles, loc="upper right")

        if self.show_intersection_measurements:
            if measurement_lines:
                self._measurement_text_artist = self.fig.text(
                    0.81,
                    0.96,
                    "\n".join(measurement_lines),
                    va="top",
                    ha="left",
                    fontsize=8,
                    family="monospace",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
                )
            else:
                self._measurement_text_artist = self.fig.text(
                    0.81,
                    0.96,
                    "No intersection geometry available",
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
                )

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


