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
        Handles structure: annotations/port_n/labels/ (ground truth)
                         annotations/predictions/port_n/labels/ (predictions)
        Returns dict with:
        - has_ground_truth: Whether ground truth exists
        - has_predictions: Whether predictions exist
        - gt_subdirs: List of port directories with ground truth
        - gt_format: Annotation format of ground truth
        - gt_has_bboxes: Whether ground truth has bounding boxes
        - gt_classes: Aggregated list of unique classes across all GT files
        - pred_subdirs: List of port directories with predictions
        - pred_format: Annotation format of predictions
        - pred_has_bboxes: Whether predictions have bounding boxes
        - pred_classes: Aggregated list of unique classes across all pred files
        """
        anno_info = {
            "has_ground_truth": False,
            "has_predictions": False,
            "gt_subdirs": [],
            "gt_format": None,
            "gt_has_bboxes": False,
            "gt_classes": set(),
            "pred_subdirs": [],
            "pred_format": None,
            "pred_has_bboxes": False,
            "pred_classes": set(),
        }
        
        # Check if annotations directory exists
        if not self.annotations_dir.exists():
            logger.info(f"Annotations directory does not exist: {self.annotations_dir}")
            return anno_info
        else:
            logger.info(f"Found annotations directory: {self.annotations_dir}")
        
        try:
            # Look for ground truth files in annotations/port_n/**/ recursively
            for p in self.annotations_dir.iterdir():
                if p.is_dir() and p.name.startswith("port_"):
                    # Find first valid annotation file to detect format
                    format_info = None
                    all_files = []
                    
                    for pattern in ["**/*.txt", "**/*.json", "**/*.xml", "**/*.csv"]:
                        for file in p.rglob(pattern):
                            if file.is_file():
                                all_files.append(file)
                                # Get format from first valid file only
                                if format_info is None:
                                    gt_info = AnnotationFormatChecker.detect_format(file)
                                    if gt_info["format"] != "Unknown":
                                        format_info = gt_info
                    
                    # If we found a valid format, aggregate classes from all files
                    if format_info is not None:
                        anno_info["has_ground_truth"] = True
                        anno_info["gt_format"] = format_info["format"]
                        anno_info["gt_has_bboxes"] = format_info["has_bboxes"]
                        anno_info["gt_classes"].update(format_info["classes"])
                        
                        # Read remaining files for additional classes
                        for file in all_files:
                            gt_info = AnnotationFormatChecker.detect_format(file)
                            if gt_info["format"] != "Unknown":
                                anno_info["gt_classes"].update(gt_info["classes"])
                        
                        if p.name not in anno_info["gt_subdirs"]:
                            anno_info["gt_subdirs"].append(p.name)
            
            # Look for prediction files in annotations/predictions/port_n/**/ recursively
            pred_dir = self.annotations_dir / "predictions"
            if pred_dir.exists():
                for p in pred_dir.iterdir():
                    if p.is_dir() and p.name.startswith("port_"):
                        # Find first valid annotation file to detect format
                        format_info = None
                        all_files = []
                        
                        for pattern in ["**/*.txt", "**/*.json", "**/*.xml", "**/*.csv"]:
                            for file in p.rglob(pattern):
                                if file.is_file():
                                    all_files.append(file)
                                    # Get format from first valid file only
                                    if format_info is None:
                                        pred_info = AnnotationFormatChecker.detect_format(file)
                                        if pred_info["format"] != "Unknown":
                                            format_info = pred_info
                        
                        # If we found a valid format, aggregate classes from all files
                        if format_info is not None:
                            anno_info["has_predictions"] = True
                            anno_info["pred_format"] = format_info["format"]
                            anno_info["pred_has_bboxes"] = format_info["has_bboxes"]
                            anno_info["pred_classes"].update(format_info["classes"])
                            
                            # Read remaining files for additional classes
                            for file in all_files:
                                pred_info = AnnotationFormatChecker.detect_format(file)
                                if pred_info["format"] != "Unknown":
                                    anno_info["pred_classes"].update(pred_info["classes"])
                            
                            if p.name not in anno_info["pred_subdirs"]:
                                anno_info["pred_subdirs"].append(p.name)

            
            # Sort subdirectories and convert classes to sorted lists
            anno_info["gt_subdirs"] = sorted(anno_info["gt_subdirs"])
            anno_info["pred_subdirs"] = sorted(anno_info["pred_subdirs"])
            anno_info["gt_classes"] = sorted(list(anno_info["gt_classes"]))
            anno_info["pred_classes"] = sorted(list(anno_info["pred_classes"]))
        except Exception as e:
            logger.debug(f"Error reading annotation directories: {e}")
        
        return anno_info
    
    def valid_annotation_dir_text(self) -> str:
        annotation_info = self.valid_annotation_dirs()
        
        if not annotation_info["has_ground_truth"] and not annotation_info["has_predictions"]:
            return "NONE"
        
        text_parts = ["annotation directories:"]
        
        if annotation_info["has_ground_truth"]:
            text_parts.append("  ground truth:")
            text_parts.append(f"    subdirectories: {', '.join(annotation_info['gt_subdirs'])}")
            text_parts.append(f"    format: {annotation_info['gt_format']}")
            text_parts.append(f"    classes: {annotation_info['gt_classes']}")
        
        if annotation_info["has_predictions"]:
            text_parts.append("  predictions:")
            text_parts.append(f"    subdirectories: {', '.join(annotation_info['pred_subdirs'])}")
            text_parts.append(f"    format: {annotation_info['pred_format']}")
            text_parts.append(f"    classes: {annotation_info['pred_classes']}")
        
        return "\n".join(text_parts)

    def get_html_summary(self) -> str:
        """
        Provide granular summary of where the workspace is in the calibration process
        Note that the currently configured camera array is reloaded each time this
        is called to determine the state of the data that is currently saved out.
        """
        config = Configurator(self.workspace_dir)
        self.camera_array = config.get_camera_array()
        self.camera_count = config.get_camera_count()

        # Get annotation text and convert newlines to <br> for HTML display
        anno_text = self.valid_annotation_dir_text()
        anno_html = anno_text.replace("\n", "<br>")

        html = f"""
            <html>
                <head>
                    <style>
                        p {{
                            text-indent: 30px;
                        }}
                        .annotation-text {{
                            white-space: pre-wrap;
                            margin-left: 30px;
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
                    <div class="annotation-text">{anno_html}</div>
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
