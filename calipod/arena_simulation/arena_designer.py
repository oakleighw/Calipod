import colorsys

import pyqtgraph.opengl as gl

from calipod.gui.vizualize.camera_mesh import build_camera_frustum_item, build_camera_origin_cube_item


class ArenaDesignerVisualizer:
	"""Arena simulation visualizer with color-coded camera cubes."""

	def __init__(self, camera_count: int = 0):
		self.camera_count = camera_count
		self.mm_to_scene_scale = 0.001
		self.visualised_frustum_depth_cm = 100.0
		self.camera_translations_mm = {}
		self.camera_rotations_deg = {}
		self.camera_frustum_angles_deg = {}
		self.camera_min_working_distance_mm = {}
		self.scene = gl.GLViewWidget()
		self.scene.setBackgroundColor("w")
		self.scene.setCameraPosition(distance=4)
		self.refresh_scene()

	def _init_empty_scene(self):
		"""Create the default scene baseline before adding arena/camera items."""
		self.scene.clear()
		axis = gl.GLAxisItem()
		self.scene.addItem(axis)

	def _fallback_color(self, index: int, total: int) -> tuple[float, float, float, float]:
		"""Generate deterministic colors when camera color metadata is unavailable."""
		hue = 0 if total <= 0 else index / total
		r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.9)
		return (r, g, b, 1.0)

	def get_camera_color(self, camera_index: int) -> tuple[float, float, float, float]:
		"""Return the color assigned to a camera index."""
		return self._fallback_color(camera_index, max(1, int(self.camera_count or 0)))

	@staticmethod
	def color_to_css(color: tuple[float, float, float, float]) -> str:
		"""Convert an RGBA color tuple into a stylesheet color string."""
		r, g, b, _ = color
		return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

	@staticmethod
	def _darken_color(color: tuple[float, float, float, float], factor: float = 0.45, alpha: float = 0.75):
		"""Return a darker variant of the given RGBA color."""
		r, g, b, _ = color
		return (
			max(0.0, min(1.0, r * factor)),
			max(0.0, min(1.0, g * factor)),
			max(0.0, min(1.0, b * factor)),
			max(0.0, min(1.0, alpha)),
		)

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
			frustum_color = (color[0], color[1], color[2], 0.18)
			frustum = build_camera_frustum_item(
				horizontal_angle_deg=horizontal_angle_deg,
				vertical_angle_deg=vertical_angle_deg,
				depth=depth_mm * self.mm_to_scene_scale,
				color=frustum_color,
				edge_color=color,
			)
			self.camera_frustums[camera_index] = frustum
			self.scene.addItem(frustum)

			# Dark red sub-frustum marks the minimum working distance.
			min_depth_mm = max(0.1, min(float(min_working_distance_mm), depth_mm))
			dark_color = self._darken_color(color, factor=0.45, alpha=0.80)
			dark_edge = self._darken_color(color, factor=0.35, alpha=1.0)
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

	def update_camera_count(self, camera_count: int):
		self.camera_count = camera_count
		self.refresh_scene()

	def set_camera_translation(self, camera_index: int, x_mm: float, y_mm: float, z_mm: float):
		"""Set camera cube translation in mm and update the corresponding mesh."""
		self.camera_translations_mm[camera_index] = (float(x_mm), float(y_mm), float(z_mm))
		self._apply_camera_transform(camera_index)

	def set_camera_rotation(self, camera_index: int, pan_deg: float, tilt_deg: float, roll_deg: float):
		"""Set camera cube rotation in degrees and update the corresponding mesh."""
		self.camera_rotations_deg[camera_index] = (float(pan_deg), float(tilt_deg), float(roll_deg))
		self._apply_camera_transform(camera_index)

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

	def reset_scene(self):
		"""Reset to the default empty scene baseline and rebuild camera cubes."""
		self.refresh_scene()
