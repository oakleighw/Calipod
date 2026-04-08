import colorsys

import pyqtgraph.opengl as gl

from calipod.gui.vizualize.camera_mesh import build_camera_origin_cube_item


class ArenaDesignerVisualizer:
	"""Arena simulation visualizer with color-coded camera cubes."""

	def __init__(self, camera_count: int = 0):
		self.camera_count = camera_count
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
			self.camera_cubes[i] = cube
			self.scene.addItem(cube)

	def refresh_scene(self):
		self._init_empty_scene()
		self._add_camera_cubes()

	def update_camera_count(self, camera_count: int):
		self.camera_count = camera_count
		self.refresh_scene()

	def reset_scene(self):
		"""Reset to the default empty scene baseline and rebuild camera cubes."""
		self.refresh_scene()
