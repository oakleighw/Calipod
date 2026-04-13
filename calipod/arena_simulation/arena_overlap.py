"""Overlap mesh computation for arena simulation frustums."""

import itertools

import numpy as np
import pyqtgraph.opengl as gl
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from calipod.gui.vizualize.camera_mesh import build_camera_frustum_geometry, build_mesh_item_from_geometry


def _promote_always_visible(item):
    """Make a GL item draw over other scene geometry whenever possible."""
    item.setGLOptions("additive")
    if hasattr(item, "setDepthValue"):
        # Draw after most geometry in the scene graph.
        item.setDepthValue(1_000_000)


def rotation_matrix_from_euler_deg(pan_deg: float, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """Build rotation matrix matching local roll->tilt->pan transform order."""
    rx = np.radians(roll_deg)
    ry = np.radians(tilt_deg)
    rz = np.radians(pan_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # Visualizer applies local rotations in this exact sequence:
    # roll (x), then tilt (y), then pan (z).
    # For local/intrinsic updates, compose as Rx @ Ry @ Rz.
    return rx_m @ ry_m @ rz_m


def get_camera_world_frustum_geometry(
    camera_index: int,
    camera_frustum_angles_deg: dict,
    camera_translations_mm: dict,
    camera_rotations_deg: dict,
    visualised_frustum_depth_cm: float,
    mm_to_scene_scale: float,
):
    """Return transformed frustum vertices/faces for overlap computation."""
    horizontal_angle_deg, vertical_angle_deg = camera_frustum_angles_deg.get(camera_index, (60.0, 45.0))
    depth_mm = max(visualised_frustum_depth_cm, 0.1) * 10.0
    depth_scene = depth_mm * mm_to_scene_scale

    verts, faces = build_camera_frustum_geometry(horizontal_angle_deg, vertical_angle_deg, depth_scene)

    x_mm, y_mm, z_mm = camera_translations_mm.get(camera_index, (0.0, 0.0, 0.0))
    pan_deg, tilt_deg, roll_deg = camera_rotations_deg.get(camera_index, (0.0, 0.0, 0.0))
    translation = np.array([x_mm, y_mm, z_mm], dtype=float) * mm_to_scene_scale
    rotation = rotation_matrix_from_euler_deg(pan_deg, tilt_deg, roll_deg)

    world_verts = (rotation @ verts.T).T + translation
    return world_verts, faces


def halfspaces_from_convex_mesh(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Convert convex mesh triangles to halfspaces A x + b <= 0."""
    centroid = np.mean(verts, axis=0)
    halfspaces = []
    for face in faces:
        p0, p1, p2 = verts[face]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = -np.dot(normal, p0)
        if np.dot(normal, centroid) + offset > 0:
            normal = -normal
            offset = -offset
        halfspaces.append(np.append(normal, offset))
    return np.array(halfspaces)


def find_strict_interior_point(halfspaces: np.ndarray):
    """Find interior point by maximizing margin t for A x + b + t <= 0."""
    if halfspaces.size == 0:
        return None

    a = halfspaces[:, :3]
    b = halfspaces[:, 3]
    a_ub = np.hstack([a, np.ones((a.shape[0], 1))])
    b_ub = -b
    c = np.array([0.0, 0.0, 0.0, -1.0])
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None

    t = res.x[3]
    if t <= 1e-9:
        return None
    return res.x[:3]


def intersection_vertices_from_halfspaces(halfspaces: np.ndarray):
    """Compute intersection polyhedron vertices from combined halfspaces."""
    interior = find_strict_interior_point(halfspaces)
    if interior is None:
        return None

    try:
        intersection = HalfspaceIntersection(halfspaces, interior)
        verts = intersection.intersections
        if verts is None or len(verts) < 4:
            return None
        return verts
    except (QhullError, ValueError):
        return None


def build_overlap_mesh_items(
    camera_count: int,
    overlap_mode: str,
    camera_frustum_angles_deg: dict,
    camera_translations_mm: dict,
    camera_rotations_deg: dict,
    visualised_frustum_depth_cm: float,
    mm_to_scene_scale: float,
):
    """Build overlap meshes for either pairwise or all-camera frustum intersections."""
    overlap_meshes = []
    camera_indices = list(range(max(0, int(camera_count or 0))))
    if len(camera_indices) < 2:
        return overlap_meshes

    frustum_halfspaces = {}
    for camera_index in camera_indices:
        verts, faces = get_camera_world_frustum_geometry(
            camera_index,
            camera_frustum_angles_deg,
            camera_translations_mm,
            camera_rotations_deg,
            visualised_frustum_depth_cm,
            mm_to_scene_scale,
        )
        halfspaces = halfspaces_from_convex_mesh(verts, faces)
        if halfspaces.size > 0:
            frustum_halfspaces[camera_index] = halfspaces

    if len(frustum_halfspaces) < 2:
        return overlap_meshes

    intersection_sets = []
    if overlap_mode == "max_all":
        combined = np.vstack([frustum_halfspaces[idx] for idx in sorted(frustum_halfspaces.keys())])
        intersection_sets.append(combined)
    else:
        for idx_a, idx_b in itertools.combinations(sorted(frustum_halfspaces.keys()), 2):
            intersection_sets.append(np.vstack([frustum_halfspaces[idx_a], frustum_halfspaces[idx_b]]))

    for combined_halfspaces in intersection_sets:
        verts = intersection_vertices_from_halfspaces(combined_halfspaces)
        if verts is None:
            continue
        try:
            hull = ConvexHull(verts)
        except QhullError:
            continue

        overlap_mesh = build_mesh_item_from_geometry(
            vertexes=np.asarray(verts, dtype=np.float32),
            faces=np.asarray(hull.simplices, dtype=np.uint32),
            color=(1.0, 1.0, 1.0, 0.28),
            edge_color=(0.0, 0.0, 0.0, 1.0),
            gl_options="additive",
            smooth=False,
            draw_edges=False,
        )
        _promote_always_visible(overlap_mesh)
        overlap_meshes.append(overlap_mesh)

        # Add a thicker explicit hull outline so overlap boundaries stay visible
        # against translucent frustum meshes.
        unique_edges = set()
        for tri in hull.simplices:
            tri = [int(tri[0]), int(tri[1]), int(tri[2])]
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                a, b = sorted(edge)
                unique_edges.add((a, b))

        if unique_edges:
            line_points = []
            for a, b in sorted(unique_edges):
                line_points.append(verts[a])
                line_points.append(verts[b])

            outline = gl.GLLinePlotItem(
                pos=np.asarray(line_points, dtype=np.float32),
                color=(0.0, 0.0, 0.0, 1.0),
                width=5.0,
                mode="lines",
                antialias=True,
            )
            _promote_always_visible(outline)
            overlap_meshes.append(outline)

    return overlap_meshes

