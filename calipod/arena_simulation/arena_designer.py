import colorsys

import pyqtgraph.opengl as gl

from calipod.gui.vizualize.camera_mesh import build_camera_origin_cube_item


class ArenaDesignerVisualizer:
	"""Arena simulation visualizer with color-coded camera cubes."""

	def __init__(self, camera_count: int = 0):
		self.camera_count = camera_count
		self.mm_to_scene_scale = 0.001
		self.camera_translations_mm = {}
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

	def _add_camera_cubes(self):
		self.camera_cubes = {}

		# Arena designer intentionally starts disconnected from calibration:
		# spawn camera_count cubes at origin for manual placement workflows.
		total = max(0, int(self.camera_count or 0))
		for i in range(total):
			color = self._fallback_color(i, total)
			cube = build_camera_origin_cube_item(color=color, edge_color=(0, 0, 0, 1))
			if i not in self.camera_translations_mm:
				self.camera_translations_mm[i] = (0.0, 0.0, 0.0)
			self.camera_cubes[i] = cube
			self.scene.addItem(cube)
			self._apply_camera_translation(i)

	def _apply_camera_translation(self, camera_index: int):
		"""Apply the current translation (in mm) for a specific camera cube."""
		if camera_index not in self.camera_cubes:
			return

		x_mm, y_mm, z_mm = self.camera_translations_mm.get(camera_index, (0.0, 0.0, 0.0))
		x = x_mm * self.mm_to_scene_scale
		y = y_mm * self.mm_to_scene_scale
		z = z_mm * self.mm_to_scene_scale

		cube = self.camera_cubes[camera_index]
		cube.resetTransform()
		cube.translate(x, y, z)

	def _apply_all_camera_translations(self):
		"""Re-apply translations after scene scale changes."""
		for camera_index in self.camera_cubes:
			self._apply_camera_translation(camera_index)

	def refresh_scene(self):
		self._init_empty_scene()
		self._add_camera_cubes()

	def update_camera_count(self, camera_count: int):
		self.camera_count = camera_count
		self.refresh_scene()

	def set_camera_translation(self, camera_index: int, x_mm: float, y_mm: float, z_mm: float):
		"""Set camera cube translation in mm and update the corresponding mesh."""
		self.camera_translations_mm[camera_index] = (float(x_mm), float(y_mm), float(z_mm))
		self._apply_camera_translation(camera_index)

	def set_mm_to_scene_scale(self, mm_to_scene_scale: float):
		"""Set conversion factor from mm to scene units and refresh transforms."""
		self.mm_to_scene_scale = float(mm_to_scene_scale)
		self._apply_all_camera_translations()

	def set_arena_depth_cm(self, depth_cm: float):
		"""Map user arena depth in cm to scene scale, anchored at 100 cm -> 0.001."""
		depth_cm = max(float(depth_cm), 0.1)
		self.mm_to_scene_scale = 0.1 / depth_cm
		self._apply_all_camera_translations()

	def reset_scene(self):
		"""Reset to the default empty scene baseline and rebuild camera cubes."""
		self.refresh_scene()
