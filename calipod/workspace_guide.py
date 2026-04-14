from pathlib import Path

from calipod.core import logger as calipod_logger
from calipod.core.configurator import Configurator
from calipod.core.annotation_checker import AnnotationFormatChecker

logger = calipod_logger.get(__name__)


class WorkspaceGuide:
    def __init__(self, workspace_dir, camera_count) -> None:
        self.workspace_dir = workspace_dir
        self.camera_count = camera_count
        self.intrinsic_dir = Path(workspace_dir, "calibration", "intrinsic")
        self.extrinsic_dir = Path(workspace_dir, "calibration", "extrinsic")
        self.recording_dir = Path(workspace_dir, "recordings")
        self.annotations_dir = Path(workspace_dir, "annotations")
        self.arena_sim_dir = Path(workspace_dir, "arena_sim")

    def get_ports_in_dir(self, directory: Path) -> list:
        """
        Returns a list of port indices that are currently exist in calibration/intrinsic
        in the correct file format (i.e. 'port_#.mp4')

        """
        all_ports = []
        for file in directory.iterdir():
            if file.stem[0:5] == "port_" and file.suffix == ".mp4":
                port = file.stem.split("_")[1]
                all_ports.append(int(port))
        return all_ports

    def all_instrinsic_mp4s_available(self):
        return self.missing_files_in_dir(self.intrinsic_dir) == "NONE"

    def all_extrinsic_mp4s_available(self):
        return self.missing_files_in_dir(self.extrinsic_dir) == "NONE"

    def missing_files_in_dir(self, directory: Path):
        files = []
        target_ports = [i for i in range(1, self.camera_count + 1)]
        current_ports = self.get_ports_in_dir(directory)

        missing_ports = [port for port in target_ports if port not in current_ports]
        for port in missing_ports:
            files.append(f"port_{port}.mp4")

        missing_files = ",".join(files)
        if len(missing_files) == 0:
            missing_files = "NONE"
        return missing_files

    def uncalibrated_cameras(self):
        uncalibrated = []
        for cam in self.camera_array.cameras.values():
            if cam.distortions is None and cam.matrix is None and cam.error is None:
                uncalibrated.append(str(cam.port))

        uncalibrated = ",".join(uncalibrated)
        if len(uncalibrated) == 0:
            uncalibrated = "NONE"
        return uncalibrated

    def intrinsic_calibration_status(self):
        if self.camera_array.all_intrinsics_calibrated() and self.all_instrinsic_mp4s_available():
            return "COMPLETE"
        else:
            return "INCOMPLETE"

    def extrinsic_calibration_status(self):
        if self.camera_array.all_extrinsics_calibrated() and self.all_extrinsic_mp4s_available():
            return "COMPLETE"
        else:
            return "INCOMPLETE"

    def valid_recording_dirs(self):
        dir_list = []
        for p in self.recording_dir.iterdir():
            if p.is_dir():
                if self.missing_files_in_dir(p) == "NONE":
                    dir_list.append(p.stem)

        return dir_list

    def valid_recording_dir_text(self) -> str:
        recording_dir_text = ",".join(self.valid_recording_dirs())

        if len(recording_dir_text) == 0:
            recording_dir_text = "NONE"
        return recording_dir_text
    
    def valid_annotation_dirs(self):
        """
        Check for valid annotation directories and return info about their contents.
        Returns list of dicts with annotation details:
        - dir_name: Directory name
        - has_ground_truth: Whether ground truth file exists
        - has_predictions: Whether predictions file exists
        - gt_format: Annotation format of ground truth (YOLO, COCO, Pascal_VOC, CSV, etc.)
        - pred_format: Annotation format of predictions
        - gt_has_bboxes: Whether ground truth has bounding boxes
        - pred_has_bboxes: Whether predictions have bounding boxes
        - gt_classes: List of unique classes in ground truth
        - pred_classes: List of unique classes in predictions
        """
        anno_list = []
        
        # Check if annotations directory exists
        if not self.annotations_dir.exists():
            logger.debug(f"Annotations directory does not exist: {self.annotations_dir}")
            return anno_list
        
        try:
            for p in self.annotations_dir.iterdir():
                if p.is_dir():
                    anno_info = {
                        "dir_name": p.stem,
                        "has_ground_truth": False,
                        "has_predictions": False,
                        "gt_format": None,
                        "pred_format": None,
                        "gt_has_bboxes": False,
                        "pred_has_bboxes": False,
                        "gt_classes": [],
                        "pred_classes": [],
                    }
                    
                    # Look for ground truth files (common patterns)
                    gt_file = None
                    for pattern in ["*ground_truth*", "*gt*", "annotations*"]:
                        matching_files = list(p.glob(pattern))
                        if matching_files:
                            gt_file = matching_files[0]
                            break
                    
                    # Look for prediction files
                    pred_file = None
                    for pattern in ["*predictions*", "*pred*", "*output*"]:
                        matching_files = list(p.glob(pattern))
                        if matching_files:
                            pred_file = matching_files[0]
                            break
                    
                    # Check ground truth file
                    if gt_file and gt_file.exists():
                        gt_info = AnnotationFormatChecker.detect_format(gt_file)
                        anno_info["has_ground_truth"] = True
                        anno_info["gt_format"] = gt_info["format"]
                        anno_info["gt_has_bboxes"] = gt_info["has_bboxes"]
                        anno_info["gt_classes"] = gt_info["classes"]
                    
                    # Check prediction file
                    if pred_file and pred_file.exists():
                        pred_info = AnnotationFormatChecker.detect_format(pred_file)
                        anno_info["has_predictions"] = True
                        anno_info["pred_format"] = pred_info["format"]
                        anno_info["pred_has_bboxes"] = pred_info["has_bboxes"]
                        anno_info["pred_classes"] = pred_info["classes"]
                    
                    # Only include if at least one annotation type exists
                    if anno_info["has_ground_truth"] or anno_info["has_predictions"]:
                        anno_list.append(anno_info)
        except Exception as e:
            logger.debug(f"Error reading annotation directories: {e}")
        
        return anno_list
    
    def valid_annotation_dir_text(self) -> str:
        annotation_dirs = self.valid_annotation_dirs()
        
        if not annotation_dirs:
            return "NONE"
        
        text_parts = []
        for anno in annotation_dirs:
            part = f"{anno['dir_name']}"
            if anno["has_ground_truth"]:
                bbox_str = " (bbox)" if anno["gt_has_bboxes"] else ""
                part += f" [GT: {anno['gt_format']}{bbox_str}"
                if anno['gt_classes']:
                    part += f" classes={anno['gt_classes']}"
                part += "]"
            if anno["has_predictions"]:
                bbox_str = " (bbox)" if anno["pred_has_bboxes"] else ""
                part += f" [Pred: {anno['pred_format']}{bbox_str}"
                if anno['pred_classes']:
                    part += f" classes={anno['pred_classes']}"
                part += "]"
            text_parts.append(part)
        
        return "; ".join(text_parts)

    def get_html_summary(self) -> str:
        """
        Provide granular summary of where the workspace is in the calibration process
        Note that the currently configured camera array is reloaded each time this
        is called to determine the state of the data that is currently saved out.
        """
        config = Configurator(self.workspace_dir)
        self.camera_array = config.get_camera_array()
        self.camera_count = config.get_camera_count()

        html = f"""
            <html>
                <head>
                    <style>
                        p {{
                            text-indent: 30px;
                        }}
                    </style>
                </head>
                <body>
                    <h4>Summary</h4>
                    <p>    Directory: {str(self.workspace_dir)}</p>
                    <p>    Camera Count: {self.camera_count}</p>
                    <h4>Intrinsic Calibration: {self.intrinsic_calibration_status()}</h4>
                    <p>    subdirectory: {str(self.intrinsic_dir)}</p>
                    <p>    missing files:{self.missing_files_in_dir(self.intrinsic_dir)}</p>
                    <p>    cameras needing calibration: {self.uncalibrated_cameras()}</p>
                    <h4>Extrinsic Calibration: {self.extrinsic_calibration_status()}</h4>
                    <p>    subdirectory: {str(self.extrinsic_dir)}</p>
                    <p>    missing files:{self.missing_files_in_dir(self.extrinsic_dir)}</p>
                    <h4>Recordings</h4>
                    <p>    valid directories: {self.valid_recording_dir_text()}</p>
                    <h4>Annotations</h4>
                    <p>    annotation directories: {self.valid_annotation_dir_text()}</p>
                    <p>
                </body>
            </html>
            """

        return html


if __name__ == "__main__":
    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\caliscope\4_cam_prerecorded_practice_working")
    camera_count = 4
    workflow_guide = WorkspaceGuide(workspace_dir, camera_count)

    logger.info(workflow_guide.get_html_summary())
