
from pathlib import Path


# caliscope/trackers/fly_tracker.py

import cv2
import numpy as np
from pathlib import Path
from queue import Queue
from threading import Thread
import os

import caliscope.logger
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = caliscope.logger.get(__name__)


class FlyTracker(Tracker):
    def __init__(self):
        super().__init__()
        self.in_queues = {}  # Dictionary to hold input queues for each port
        self.out_queues = {} # Dictionary to hold output queues for each port
        self.threads = {}    # Dictionary to hold processing threads for each port

        # Instance attribute for annotations_dir, set via property
        self._annotations_dir: Path = None
        # You can define self.yolo_model_path here, which would be derived from annotations_dir
        # For live YOLO inference, you'd load a model (e.g., yolov8n.pt)
        self.yolo_model = None # Placeholder for your loaded YOLO model


    def yolo_to_idloc(self, frame_shape, text_file):
        ids = np.array([],dtype=int)
        img_loc = np.array([],dtype=float)
        try:
            with open(text_file, 'r') as txt:
                lines = txt.readlines()
                # initialize variables so none will be created if no points detected
                ids = []
                img_loc = []
                for i, line in enumerate(lines):
                    split_l = line.strip().split(' ')
                    # Extract class_id from the YOLO label (first value)
                    class_id = int(split_l[0])
                    x_centre = float(split_l[1]) * frame_shape[1]
                    y_centre = float(split_l[2]) * frame_shape[0]
                    ids.append(class_id)  # Use class_id instead of line index
                    img_loc.append((x_centre, y_centre))

                    logger.debug(f"Parsed point: class={class_id}, pos=({x_centre}, {y_centre}) from '{line.strip()}' in {text_file}")
                    logger.debug(f"Type of x_pixel: {type(x_centre)}, Type of y_pixel: {type(y_centre)}")

        except FileNotFoundError:
            logger.warning(f"YOLO label file not found (inside yolo_to_idloc): {text_file}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing YOLO label file {text_file}: {e}")

        return  np.array(ids, dtype=int), np.array(img_loc, dtype=np.float32)
    
    @property
    def name(self):
        return "FLY"

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
            logger.info(f"FlyTracker: Initializing YOLO model (placeholder).")
            # This is where an actual YOLO model loading code would go
            # self.yolo_model = YourYOLOLibrary.load_model(self.yolo_model_path)
        except Exception as e:
            logger.error(f"No model to load YOLO model in FlyTracker: {e}")
            self.yolo_model = None # Ensure it's None if loading fails

    @classmethod
    def metarig_mapped(cls) -> bool:
        return True

    def run_frame_processor(self, port: int, rotation_count: int):
        logger.info(f"FlyTracker thread for port {port} started.")

        while True:
            # Always initialize an empty packet, in case an error prevents populating it
            point_packet = PointPacket(np.array([], dtype=int), np.array([], dtype=np.float32), None) 

            try:
                # 1. Get frame and frame_idx from input queue (blocking, like other trackers)
                frame, frame_idx = self.in_queues[port].get()
                logger.debug(f"FlyTracker (Port {port}): Got frame {frame_idx} from queue.")

                height, width, _ = frame.shape
                
                point_ids = np.array([], dtype=int)
                landmark_xy = np.array([], dtype=np.float32)

                if self.yolo_model is not None:
                    # --- LIVE YOLO INFERENCE PATH ---
                    logger.debug(f"FlyTracker (Port {port}): Processing frame for live detection...")
                    
                    # Assuming your YOLO model expects a BGR image and returns results
                    # You might need to adjust based on your actual YOLO library's requirements
                    results = self.yolo_model(frame, verbose=False) 
                    
                    for r in results: # Each 'r' is a result object for an image (usually just one)
                        for i, box in enumerate(r.boxes):
                            class_id = int(box.cls[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2

                            point_ids.append(class_id)
                            landmark_xy.append((x_center, y_center))
                    
                    point_ids = np.array(point_ids, dtype=int)
                    landmark_xy = np.array(landmark_xy, dtype=np.float32)

                else:
                    # --- STATIC FILE READING PATH ---
                    logger.debug(f"FlyTracker (Port {port}): Processing frame {frame_idx} with offline detections from file...")

                    # Construct label file path using frame_idx
                    label_filename = f"frame_{frame_idx:06d}.txt" # CONFIRMED CORRECT with your file naming
                    label_file_path = Path(self.annotations_dir, f"port_{port}", "labels", "train", label_filename)
                    
                    if label_file_path.exists():
                        # Call yolo_to_idloc (the CORRECTED version)
                        point_ids, landmark_xy = self.yolo_to_idloc(frame.shape, label_file_path)
                        logger.debug(f"FlyTracker (Port {port}, Frame {frame_idx}): Found {len(point_ids)} points from {label_file_path}")
                    else:
                        logger.debug(f"FlyTracker (Port {port}, Frame {frame_idx}): No label file found at {label_file_path}. Returning empty points for this frame.")

                
                # Create the PointPacket with processed data
                point_packet = PointPacket(point_ids, landmark_xy, None)

            except Exception as e: # Catch ANY other error during processing this frame
                logger.error(f"FlyTracker (Port {port}): Unhandled error during frame processing: {e}", exc_info=True)
                # If an error occurred, point_packet remains the default empty one
                point_packet = PointPacket(np.array([], dtype=int), np.array([], dtype=np.float32), None) 
            
            # 4. Put the point_packet into the output queue (blocking, like other trackers)
            self.out_queues[port].put(point_packet)
            logger.debug(f"FlyTracker (Port {port}): Processed frame and put result in out_queue.")

    def get_points(self, frame: np.ndarray, port: int, rotation_count: int, frame_idx: int) -> PointPacket:
        # Initialize queues and thread for this port if not already done
        if port not in self.in_queues:
            self.in_queues[port] = Queue(1) # Queue capacity 1 (store one frame)
            self.out_queues[port] = Queue(1) # Queue capacity 1 (store one result)

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
        return set() # FlyTracker doesn't use Charuco board connections

 

    def scatter_draw_instructions(self, point_id: int) -> dict:
        rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        return rules
