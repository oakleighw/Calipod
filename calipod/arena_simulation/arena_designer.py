import pyqtgraph.opengl as gl


class ArenaDesignerVisualizer:
	"""Provides an initially empty 3D scene for arena design interactions."""

	def __init__(self):
		self.scene = gl.GLViewWidget()
		self.scene.setBackgroundColor("w")
		self.scene.setCameraPosition(distance=4)
		self._init_empty_scene()

	def _init_empty_scene(self):
		"""Create a blank scene baseline; arena_designer logic can add items later."""
		self.scene.clear()
		axis = gl.GLAxisItem()
		self.scene.addItem(axis)

	def reset_scene(self):
		"""Reset to the default empty scene baseline."""
		self._init_empty_scene()
