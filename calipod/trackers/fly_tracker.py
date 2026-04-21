from pathlib import Path
from queue import Queue
from threading import Thread

# caliscope/trackers/fly_tracker.py
import numpy as np

from calipod.annotation_management.yolo_utils import load_yolo_file
from calipod.core import logger as calipod_logger
from calipod.core.packets import PointPacket
from calipod.tracker import Tracker

logger = calipod_logger.get(__name__)


class FlyTracker(Tracker):
    def __init__(self):
        super().__init__()
        self.in_queues = {}  # Dictionary to hold input queues for each port
        self.out_queues = {}  # Dictionary to hold output queues for each port
        self.threads = {}  # Dictionary to hold processing threads for each port

        # Instance attribute for annotations_dir, set via property
        self._annotations_dir: Path = None
        # You can define self.yolo_model_path here, which would be derived from annotations_dir
        # For live YOLO inference, you'd load a model (e.g., yolov8n.pt)
        self.yolo_model = None  # Placeholder for your loaded YOLO model
        self._annotations_only_mode = False  # Track if we're using pre-made annotations without video processing
        self.bbox_data = {}  # Store bounding box data by (port, frame_idx, point_id)

    @property
    def name(self):
        return "FLY"

    @property
    def annotations_only_mode(self) -> bool:
        """Returns True if operating in annotations-only mode (no video processing needed)"""
        return self._annotations_only_mode

    def check_annotations_available(self, recording_dir: Path, all_camera_data: dict) -> bool:
        """Check if all cameras have complete annotations for the recording.

        Returns True if annotations exist for all ports and frames.
        """
        if not self._annotations_dir or not self._annotations_dir.exists():
            return False

        logger.info("Checking if annotations are available for all cameras...")
        for camera_data in all_camera_data.values():
            port = camera_data.port
            annotations_path = Path(self._annotations_dir, f"port_{port}", "labels", "train")

            if not annotations_path.exists():
                logger.info(f"No annotations found for port {port}")
                return False

            # Check if any annotation files exist
            annotation_files = list(annotations_path.glob("frame_*.txt"))
            if not annotation_files:
                logger.info(f"No annotation files found in {annotations_path}")
                return False

        logger.info("Annotations available for all cameras - enabling annotations-only mode")
        self._annotations_only_mode = True
        return True

    @property
    def annotations_dir(self) -> Path:
        if self._annotations_dir is None:
            raise RuntimeError("FlyTracker's annotations_dir has not been set.")
        return self._annotations_dir

    @annotations_dir.setter
    def annotations_dir(self, path: Path):
        self._annotations_dir = path
        # TODO: IF NO ANNOTATIONS DIR, LOOK FOR MODEL WEIGHTS PATH & LOAD...
        try:
            # Placeholder: Load your YOLO model here or pass its path
            # Example using ultralytics: from ultralytics import YOLO; self.yolo_model = YOLO("yolov8n.pt")
            logger.info("FlyTracker: Initializing YOLO model (placeholder).")
            # This is where an actual YOLO model loading code would go
            # self.yolo_model = YourYOLOLibrary.load_model(self.yolo_model_path)
        except Exception as e:
            logger.error(f"No model to load YOLO model in FlyTracker: {e}")
            self.yolo_model = None  # Ensure it's None if loading fails

    @classmethod
    def metarig_mapped(cls) -> bool:
        return True

    def run_frame_processor(self, port: int, rotation_count: int):
        logger.info(f"FlyTracker thread for port {port} started (annotations_only_mode={self._annotations_only_mode}).")

        while True:
            # Always initialize an empty packet, in case an error prevents populating it
            point_packet = PointPacket(np.array([], dtype=int), np.array([], dtype=np.float32), None)

            try:
                # 1. Get frame and frame_idx from input queue (blocking, like other trackers)
                frame, frame_idx = self.in_queues[port].get()
                logger.debug(f"FlyTracker (Port {port}): Got frame {frame_idx} from queue.")

                point_ids = np.array([], dtype=int)
                landmark_xy = np.array([], dtype=np.float32)
                bboxes = np.array([], dtype=np.float32)

                if self.yolo_model is not None:
                    # --- LIVE YOLO INFERENCE PATH ---
                    logger.debug(f"FlyTracker (Port {port}): Processing frame for live detection...")
                    height, width, _ = frame.shape

                    # Assuming your YOLO model expects a BGR image and returns results
                    # You might need to adjust based on your actual YOLO library's requirements
                    results = self.yolo_model(frame, verbose=False)

                    for r in results:  # Each 'r' is a result object for an image (usually just one)
                        for i, box in enumerate(r.boxes):
                            class_id = int(box.cls[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            box_width = x2 - x1
                            box_height = y2 - y1

                            # Add center point
                            point_ids.append(class_id)
                            landmark_xy.append((x_center, y_center))
                            bboxes.append((box_width, box_height))

                            # For fruit (9) and leaves (10), add corner points
                            if class_id in [9, 10]:
                                # Corner positions: TL, TR, BR, BL
                                corners = [
                                    (x1, y1),  # Top-left (0)
                                    (x2, y1),  # Top-right (1)
                                    (x2, y2),  # Bottom-right (2)
                                    (x1, y2),  # Bottom-left (3)
                                ]

                                for corner_idx, (cx, cy) in enumerate(corners):
                                    corner_id = class_id * 1000 + corner_idx
                                    point_ids.append(corner_id)
                                    landmark_xy.append((cx, cy))
                                    bboxes.append((box_width, box_height))

                    point_ids = np.array(point_ids, dtype=int)
                    landmark_xy = np.array(landmark_xy, dtype=np.float32)
                    bboxes = np.array(bboxes, dtype=np.float32)

                else:
                    # --- STATIC FILE READING PATH (annotations from disk) ---
                    logger.debug(
                        f"FlyTracker (Port {port}): Processing frame {frame_idx} with offline detections from file..."
                    )

                    # Construct label file path using frame_idx
                    label_filename = f"frame_{frame_idx:06d}.txt"
                    label_file_path = Path(self.annotations_dir, f"port_{port}", "labels", "train", label_filename)

                    if label_file_path.exists():
                        # Always provide frame_shape for proper pixel coordinate scaling
                        # In annotations-only mode, frame may be a dummy zero array, but shape is still valid
                        frame_shape = frame.shape if frame is not None else None
                        point_ids, landmark_xy, bboxes = load_yolo_file(label_file_path, frame_shape)
                        logger.debug(
                            f"FlyTracker (Port {port}, Frame {frame_idx}): "
                            f"Found {len(point_ids)} points from {label_file_path}"
                        )
                    else:
                        logger.debug(
                            f"FlyTracker (Port {port}, Frame {frame_idx}): "
                            f"No label file found at {label_file_path}. "
                            "Returning empty points for this frame."
                        )

                # Store bounding box data for later use in 3D visualization
                for i, point_id in enumerate(point_ids):
                    key = (port, frame_idx, int(point_id))
                    if i < len(bboxes):
                        self.bbox_data[key] = bboxes[i]

                # Create the PointPacket with processed data
                point_packet = PointPacket(point_ids, landmark_xy, None)

            except Exception as e:  # Catch ANY other error during processing this frame
                logger.error(f"FlyTracker (Port {port}): Unhandled error during frame processing: {e}", exc_info=True)
                # If an error occurred, point_packet remains the default empty one
                point_packet = PointPacket(np.array([], dtype=int), np.array([], dtype=np.float32), None)

            # 4. Put the point_packet into the output queue (blocking, like other trackers)
            self.out_queues[port].put(point_packet)
            logger.debug(f"FlyTracker (Port {port}): Processed frame and put result in out_queue.")

    def get_points(self, frame: np.ndarray, port: int, rotation_count: int, frame_idx: int) -> PointPacket:
        # Initialize queues and thread for this port if not already done
        if port not in self.in_queues:
            self.in_queues[port] = Queue(1)  # Queue capacity 1 (store one frame)
            self.out_queues[port] = Queue(1)  # Queue capacity 1 (store one result)

            self.threads[port] = Thread(
                target=self.run_frame_processor,
                args=(port, rotation_count),
                daemon=True,
            )
            self.threads[port].start()
            logger.info(f"FlyTracker: Initialized thread for port {port}.")

        # Put the incoming (frame, frame_idx) tuple into the input queue
        self.in_queues[port].put((frame, frame_idx))
        logger.debug(f"FlyTracker (Port {port}): Frame {frame_idx} put into in_queue.")

        # Get the processed PointPacket from the output queue (blocking)
        point_packet = self.out_queues[port].get()
        logger.debug(f"FlyTracker (Port {port}): PointPacket retrieved from out_queue.")
        return point_packet

    # --- Other helper methods ---
    def get_point_name(self, point_id: int) -> str:
        return str(point_id)

    def get_point_id(self, point_name: str) -> int:
        return int(point_name)

    def get_connected_points(self):
        return set()  # FlyTracker doesn't use Charuco board connections

    def scatter_draw_instructions(self, point_id: int) -> dict:
        rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        return rules

    def get_bbox_for_point(self, port: int, frame_idx: int, point_id: int):
        """Retrieve bounding box dimensions for a specific point.

        Returns:
            tuple: (width, height) in pixels, or None if not found
        """
        key = (port, frame_idx, point_id)
        return self.bbox_data.get(key, None)

    def load_predictions_from_file(self, predictions_file_path: Path, frame_shape=None):
        """Load predictions from a YOLO predictions file.

        This method is useful for comparing predictions against ground truth.
        Uses the same parsing logic as load_yolo_file for consistency.

        Args:
            predictions_file_path: Path to YOLO predictions label file
            frame_shape: Optional tuple of (height, width, channels)

        Returns:
            ids: array of class IDs
            img_loc: array of (x, y) positions
            bboxes: array of (width, height) for each detection
        """
        if not predictions_file_path.exists():
            logger.debug(f"Predictions file not found: {predictions_file_path}")
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

        # Use shared load_yolo_file with highest_confidence_only=True
        # for predictions until better track handling is implemented.
        return load_yolo_file(predictions_file_path, frame_shape, highest_confidence_only=True)

    def get_average_bbox_by_point_id(self):
        """Calculate average bounding box dimensions for each point_id across all frames.

        Returns:
            dict: {point_id: (avg_width, avg_height)}
        """
        from collections import defaultdict

        bbox_by_id = defaultdict(list)

        for (port, frame_idx, point_id), (width, height) in self.bbox_data.items():
            bbox_by_id[point_id].append((width, height))

        avg_bbox = {}
        for point_id, bboxes in bbox_by_id.items():
            if bboxes:
                avg_width = np.mean([b[0] for b in bboxes])
                avg_height = np.mean([b[1] for b in bboxes])
                avg_bbox[point_id] = (avg_width, avg_height)

        return avg_bbox

    @staticmethod
    def create_flat_square_mesh(center_xyz, bbox_width, bbox_height, color=(0, 1, 0, 0.6)):
        """Create a flat square mesh for 'leaves' label.

        Args:
            center_xyz: (x, y, z) center position in 3D space
            bbox_width: width of bounding box in pixels (used for square dimensions)
            bbox_height: height of bounding box in pixels (used for square dimensions)
            color: RGBA color tuple (default green with transparency)

        Returns:
            tuple: (vertices, faces, colors) for GLMeshItem
        """
        cx, cy, cz = center_xyz

        # Depth-dependent scaling: larger z (farther away) means bbox pixels represent larger real-world size
        # Approximate: at 1 meter depth, 100 pixels ≈ 0.1 meters (for typical webcam FOV)
        # Scale factor: pixels * depth / reference_focal_length_approx
        depth = abs(cz)
        pixel_to_meter = depth / 1000.0  # Approximate: 1000 pixels at 1 meter = 1 meter

        # Use the average of width and height for square size
        avg_size = (bbox_width + bbox_height) / 2.0
        half_size = avg_size * pixel_to_meter / 2.0

        # Create 4 corners of a square in the XY plane (flat, perpendicular to Z axis)
        vertices = np.array(
            [
                [cx - half_size, cy - half_size, cz],  # Bottom-left
                [cx + half_size, cy - half_size, cz],  # Bottom-right
                [cx + half_size, cy + half_size, cz],  # Top-right
                [cx - half_size, cy + half_size, cz],  # Top-left
            ],
            dtype=np.float32,
        )

        # Two triangles to form the square
        faces = np.array(
            [
                [0, 1, 2],  # First triangle
                [0, 2, 3],  # Second triangle
            ],
            dtype=np.uint32,
        )

        # Color for both faces
        colors = np.array([color, color], dtype=np.float32)

        return vertices, faces, colors

    @staticmethod
    def create_hemisphere_mesh(center_xyz, bbox_width, bbox_height, color=(1, 0, 0, 0.6), segments=16):
        """Create a hemisphere mesh for 'fruit' label.

        Args:
            center_xyz: (x, y, z) center position in 3D space (will be the peak of hemisphere)
            bbox_width: width of bounding box in pixels
            bbox_height: height of bounding box in pixels
            color: RGBA color tuple (default red with transparency)
            segments: number of segments for hemisphere smoothness

        Returns:
            tuple: (vertices, faces, colors) for GLMeshItem
        """
        cx, cy, cz = center_xyz

        # Depth-dependent scaling
        depth = abs(cz)
        pixel_to_meter = depth / 1000.0  # Approximate conversion

        # The radius is the tightest circle that fits within the bounding box
        # Use min dimension and scale by 0.5 to make fruit smaller than leaves
        radius = min(bbox_width, bbox_height) * pixel_to_meter * 0.25  # Reduced from 0.5

        vertices = []
        faces = []

        # Add center peak point of hemisphere
        vertices.append([cx, cy, cz])

        # Generate hemisphere vertices
        # Latitude: 0 (equator) to pi/2 (north pole)
        # Longitude: 0 to 2*pi (full circle)
        for i in range(segments // 2 + 1):  # From equator to pole
            lat = i * (np.pi / 2) / (segments // 2)  # 0 to pi/2
            z_offset = -radius * np.cos(lat)  # Negative because hemisphere extends down from peak
            ring_radius = radius * np.sin(lat)

            for j in range(segments):
                lon = j * (2 * np.pi) / segments
                x_offset = ring_radius * np.cos(lon)
                y_offset = ring_radius * np.sin(lon)

                vertices.append([cx + x_offset, cy + y_offset, cz + z_offset])

        vertices = np.array(vertices, dtype=np.float32)

        # Create faces
        # Connect peak to first ring
        for j in range(segments):
            next_j = (j + 1) % segments
            faces.append([0, j + 1, next_j + 1])

        # Connect rings
        for i in range(segments // 2):
            for j in range(segments):
                next_j = (j + 1) % segments

                # Current ring starts at index: 1 + i * segments
                current_base = 1 + i * segments
                next_base = 1 + (i + 1) * segments

                v1 = current_base + j
                v2 = current_base + next_j
                v3 = next_base + next_j
                v4 = next_base + j

                # Two triangles per quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        faces = np.array(faces, dtype=np.uint32)

        # Create colors array for all faces
        colors = np.array([color] * len(faces), dtype=np.float32)

        return vertices, faces, colors
