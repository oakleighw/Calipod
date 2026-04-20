import numpy as np
import pyqtgraph.opengl as gl

from calipod.arena_simulation.arena_overlap import build_overlap_mesh_items
from calipod.core.camera_colours import (
    camera_rgba,
    darken_rgba,
    rgba_to_css,
    rgba_with_alpha,
)
from calipod.gui.utils.grids import (
    build_complete_grid_label_specs,
    build_plane_grid_lines,
    format_axis_label,
    nice_step,
)
from calipod.gui.vizualize.camera_mesh import (
    build_camera_frustum_item,
    build_camera_origin_cube_item,
)


class ArenaDesignerVisualizer:
    """Arena simulation visualizer with color-coded camera cubes."""

    def __init__(self, camera_count: int = 0):
        self.camera_count = camera_count
        self.arena_depth_cm = 100.0
        self.mm_to_scene_scale = 0.001
        self.visualised_frustum_depth_cm = 100.0
        self.grid_spacing_mm = 100.0
        self.grid_major_spacing_mm = 500.0
        self.show_scale_grid = True
        self.grid_labels = []
        self.camera_translations_mm = {}
        self.camera_rotations_deg = {}
        self.camera_frustum_angles_deg = {}
        self.camera_min_working_distance_mm = {}
        self.overlap_mode = "min_two"
        self.scene = gl.GLViewWidget()
        self.scene.setBackgroundColor("w")
        self.scene.setCameraPosition(distance=4)
        self.refresh_scene()

    def _init_empty_scene(self):
        """Create the default scene baseline before adding arena/camera items."""
        self.scene.clear()
        # Keep arena designer contrast consistent for dark grid lines.
        self.scene.setBackgroundColor("w")
        axis = gl.GLAxisItem()
        self.scene.addItem(axis)
        if self.show_scale_grid:
            self._add_scale_grids()

    def _add_scale_grids(self):
        """Render lightweight XY/XZ scale grids tied to mm-to-scene scale."""
        depth_mm = max(self.visualised_frustum_depth_cm, 0.1) * 10.0
        extent_mm = max(2000.0, depth_mm * 2.0)
        size_scene = extent_mm * self.mm_to_scene_scale
        self.grid_spacing_mm, self.grid_major_spacing_mm = self._get_adaptive_grid_spacing_mm()

        minor_spacing_scene = max(self.grid_spacing_mm * self.mm_to_scene_scale, 1e-4)
        major_spacing_scene = max(self.grid_major_spacing_mm * self.mm_to_scene_scale, 1e-4)
        half = size_scene * 0.5

        minor_color = (0.20, 0.20, 0.20, 0.30)
        major_color = (0.05, 0.05, 0.05, 0.75)
        self.xy_grid_minor = self._build_grid_item(
            "xy", minor_spacing_scene, half, minor_color, width=0.7, opaque=False
        )
        self.xz_grid_minor = self._build_grid_item(
            "xz", minor_spacing_scene, half, minor_color, width=0.7, opaque=False
        )
        self.xy_grid_major = self._build_grid_item("xy", major_spacing_scene, half, major_color, width=1.3, opaque=True)
        self.xz_grid_major = self._build_grid_item("xz", major_spacing_scene, half, major_color, width=1.3, opaque=True)

        for grid_item in (
            self.xy_grid_minor,
            self.xz_grid_minor,
            self.xy_grid_major,
            self.xz_grid_major,
        ):
            if grid_item is not None:
                self.scene.addItem(grid_item)

        self._add_scale_tick_labels(half=half, label_spacing_scene=major_spacing_scene)

    def _build_grid_item(self, plane: str, spacing_scene: float, half_scene: float, color, width: float, opaque: bool):
        """Create a grid item for the arena scene using shared line-building helpers."""
        lines = build_plane_grid_lines(plane, spacing_scene, half_scene)
        if not lines:
            return None
        item = gl.GLLinePlotItem(
            pos=np.asarray(lines, dtype=np.float32).reshape(-1, 3),
            color=color,
            width=width,
            mode="lines",
            antialias=True,
        )
        item.setGLOptions("opaque" if opaque else "translucent")
        return item

    @staticmethod
    def _nice_step_mm(raw_mm: float) -> float:
        """Round a raw spacing to a human-friendly 1/2/5*10^n step in mm."""
        raw_mm = max(float(raw_mm), 1.0)
        exponent = int(np.floor(np.log10(raw_mm)))
        base = 10.0**exponent
        for multiplier in (1.0, 2.0, 5.0, 10.0):
            candidate = multiplier * base
            if candidate >= raw_mm:
                return candidate
        return 10.0 ** (exponent + 1)

    def _get_adaptive_grid_spacing_mm(self) -> tuple[float, float]:
        """Pick adaptive minor/major grid spacing based on current scene scale.

        Targets roughly 8-12 major intervals across the visible grid width.
        """
        target_major_spacing_scene = 0.22
        raw_major_mm = target_major_spacing_scene / max(self.mm_to_scene_scale, 1e-12)
        major_mm = max(1.0, nice_step(raw_major_mm))

        raw_minor_mm = major_mm / 5.0
        minor_mm = max(1.0, nice_step(raw_minor_mm))
        if minor_mm >= major_mm:
            minor_mm = max(1.0, major_mm / 5.0)
        return minor_mm, major_mm

    def _clear_scale_tick_labels(self):
        for label in self.grid_labels:
            try:
                self.scene.removeItem(label)
            except Exception:
                pass
        self.grid_labels = []

    def _format_mm_label(self, scene_value: float) -> str:
        return format_axis_label(scene_value, self.mm_to_scene_scale)

    def _add_scale_tick_labels(self, half: float, label_spacing_scene: float):
        """Add edge-only tick labels and axis labels in mm, centered around 0 mm at the origin."""
        self._clear_scale_tick_labels()
        if label_spacing_scene <= 0:
            return

        for pos, text in build_complete_grid_label_specs(
            half_extent=half,
            label_interval=label_spacing_scene,
            label_formatter=self._format_mm_label,
        ):
            tick = gl.GLTextItem(pos=pos, text=text, color="black")
            self.scene.addItem(tick)
            self.grid_labels.append(tick)

    def _fallback_color(self, index: int, total: int) -> tuple[float, float, float, float]:
        """Generate deterministic colors when camera color metadata is unavailable."""
        return camera_rgba(index, total)

    def get_camera_color(self, camera_index: int) -> tuple[float, float, float, float]:
        """Return the color assigned to a camera index."""
        return self._fallback_color(camera_index, max(1, int(self.camera_count or 0)))

    @staticmethod
    def color_to_css(color: tuple[float, float, float, float]) -> str:
        """Convert an RGBA color tuple into a stylesheet color string."""
        return rgba_to_css(color)

    def _add_camera_cubes(self):
        self.camera_cubes = {}

        # Arena designer intentionally starts disconnected from calibration:
        # spawn camera_count cubes at origin for manual placement workflows.
        total = max(0, int(self.camera_count or 0))
        for i in range(total):
            self._ensure_camera_state(i)
            color = self._fallback_color(i, total)
            cube = build_camera_origin_cube_item(color=color, edge_color=(0, 0, 0, 1))
            self.camera_cubes[i] = cube
            self.scene.addItem(cube)
            self._apply_camera_transform(i)

    def _add_camera_frustums(self):
        self.camera_frustums = {}
        self.camera_min_distance_frustums = {}
        depth_mm = max(self.visualised_frustum_depth_cm, 0.1) * 10.0
        for camera_index in range(max(0, int(self.camera_count or 0))):
            self._ensure_camera_state(camera_index)
            horizontal_angle_deg, vertical_angle_deg = self.camera_frustum_angles_deg.get(camera_index, (60.0, 45.0))
            min_working_distance_mm = self.camera_min_working_distance_mm.get(camera_index, 1.0)
            color = self._fallback_color(camera_index, max(1, int(self.camera_count or 0)))
            frustum_color = rgba_with_alpha(color, 0.18)
            frustum = build_camera_frustum_item(
                horizontal_angle_deg=horizontal_angle_deg,
                vertical_angle_deg=vertical_angle_deg,
                depth=depth_mm * self.mm_to_scene_scale,
                color=frustum_color,
                edge_color=color,
            )
            self.camera_frustums[camera_index] = frustum
            self.scene.addItem(frustum)

            # Darker same-hue sub-frustum marks the minimum working distance.
            min_depth_mm = max(0.1, min(float(min_working_distance_mm), depth_mm))
            dark_color = darken_rgba(color, factor=0.45, alpha=0.80)
            dark_edge = darken_rgba(color, factor=0.35, alpha=1.0)
            min_distance_frustum = build_camera_frustum_item(
                horizontal_angle_deg=horizontal_angle_deg,
                vertical_angle_deg=vertical_angle_deg,
                depth=min_depth_mm * self.mm_to_scene_scale,
                color=dark_color,
                edge_color=dark_edge,
                gl_options="additive",
            )
            self.camera_min_distance_frustums[camera_index] = min_distance_frustum
            self.scene.addItem(min_distance_frustum)
            self._apply_camera_transform(camera_index)

    def _add_overlap_meshes(self):
        """Render overlap coverage mesh based on selected overlap mode."""
        self.overlap_meshes = build_overlap_mesh_items(
            camera_count=self.camera_count,
            overlap_mode=self.overlap_mode,
            camera_frustum_angles_deg=self.camera_frustum_angles_deg,
            camera_translations_mm=self.camera_translations_mm,
            camera_rotations_deg=self.camera_rotations_deg,
            visualised_frustum_depth_cm=self.visualised_frustum_depth_cm,
            mm_to_scene_scale=self.mm_to_scene_scale,
        )
        for overlap_mesh in self.overlap_meshes:
            self.scene.addItem(overlap_mesh)

    def _remove_overlap_meshes(self):
        """Remove currently rendered overlap items from the scene."""
        for overlap_item in getattr(self, "overlap_meshes", []):
            try:
                self.scene.removeItem(overlap_item)
            except Exception:
                pass
        self.overlap_meshes = []

    def _refresh_overlap_meshes(self):
        """Recompute and redraw overlap items without rebuilding all camera meshes."""
        self._remove_overlap_meshes()
        self._add_overlap_meshes()

    def _ensure_camera_state(self, camera_index: int):
        """Initialize stored state for a camera if it does not exist yet."""
        self.camera_translations_mm.setdefault(camera_index, (0.0, 0.0, 0.0))
        self.camera_rotations_deg.setdefault(camera_index, (0.0, 0.0, 0.0))
        self.camera_frustum_angles_deg.setdefault(camera_index, (60.0, 45.0))
        self.camera_min_working_distance_mm.setdefault(camera_index, 1.0)

    def _apply_camera_transform(self, camera_index: int):
        """Apply the current rotation and translation for a specific camera cube."""
        if (
            camera_index not in self.camera_cubes
            and camera_index not in getattr(self, "camera_frustums", {})
            and camera_index not in getattr(self, "camera_min_distance_frustums", {})
        ):
            return

        x_mm, y_mm, z_mm = self.camera_translations_mm.get(camera_index, (0.0, 0.0, 0.0))
        pan_deg, tilt_deg, roll_deg = self.camera_rotations_deg.get(camera_index, (0.0, 0.0, 0.0))
        x = x_mm * self.mm_to_scene_scale
        y = y_mm * self.mm_to_scene_scale
        z = z_mm * self.mm_to_scene_scale

        for item in (
            self.camera_cubes.get(camera_index),
            getattr(self, "camera_frustums", {}).get(camera_index),
            getattr(self, "camera_min_distance_frustums", {}).get(camera_index),
        ):
            if item is None:
                continue
            item.resetTransform()
            item.rotate(roll_deg, 1, 0, 0, local=True)
            item.rotate(tilt_deg, 0, 1, 0, local=True)
            item.rotate(pan_deg, 0, 0, 1, local=True)
            item.translate(x, y, z)

    def _apply_all_camera_translations(self):
        """Re-apply translations after scene scale changes."""
        for camera_index in self.camera_cubes:
            self._apply_camera_transform(camera_index)

    def refresh_scene(self):
        self._init_empty_scene()
        self._add_camera_cubes()
        self._add_camera_frustums()
        self._add_overlap_meshes()

    def update_camera_count(self, camera_count: int):
        self.camera_count = camera_count
        self.refresh_scene()

    def set_camera_translation(self, camera_index: int, x_mm: float, y_mm: float, z_mm: float):
        """Set camera cube translation in mm and update the corresponding mesh."""
        self.camera_translations_mm[camera_index] = (float(x_mm), float(y_mm), float(z_mm))
        self._apply_camera_transform(camera_index)
        self._refresh_overlap_meshes()

    def set_camera_rotation(self, camera_index: int, pan_deg: float, tilt_deg: float, roll_deg: float):
        """Set camera cube rotation in degrees and update the corresponding mesh."""
        self.camera_rotations_deg[camera_index] = (float(pan_deg), float(tilt_deg), float(roll_deg))
        self._apply_camera_transform(camera_index)
        self._refresh_overlap_meshes()

    def set_camera_frustum_angles(self, camera_index: int, horizontal_angle_deg: float, vertical_angle_deg: float):
        """Set lens angles for a camera frustum and rebuild the scene."""
        self.camera_frustum_angles_deg[camera_index] = (float(horizontal_angle_deg), float(vertical_angle_deg))
        self.refresh_scene()

    def set_camera_min_working_distance(self, camera_index: int, min_working_distance_mm: float):
        """Set minimum working distance for a camera and rebuild sub-frustum."""
        self.camera_min_working_distance_mm[camera_index] = max(float(min_working_distance_mm), 0.1)
        self.refresh_scene()

    def set_mm_to_scene_scale(self, mm_to_scene_scale: float):
        """Set conversion factor from mm to scene units and refresh transforms."""
        self.mm_to_scene_scale = float(mm_to_scene_scale)
        self.refresh_scene()

    def set_arena_depth_cm(self, depth_cm: float):
        """Map user arena depth in cm to scene scale, anchored at 100 cm -> 0.001."""
        depth_cm = max(float(depth_cm), 0.1)
        self.mm_to_scene_scale = 0.1 / depth_cm
        self.refresh_scene()

    def set_visualised_frustum_depth_cm(self, depth_cm: float):
        """Set the visualised frustum depth and rebuild the scene."""
        self.visualised_frustum_depth_cm = max(float(depth_cm), 0.1)
        self.refresh_scene()

    def set_overlap_mode(self, mode: str):
        """Set overlap visualization mode: 'min_two' or 'max_all'."""
        if mode not in {"min_two", "max_all"}:
            return
        self.overlap_mode = mode
        self.refresh_scene()

    def reset_scene(self):
        """Reset to the default empty scene baseline and rebuild camera cubes."""
        self.refresh_scene()
