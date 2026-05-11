from pathlib import Path

from calipod.annotation_management import AnnotationIndex
from calipod.core import logger as calipod_logger
from calipod.core.configurator import Configurator

logger = calipod_logger.get(__name__)


class WorkspaceGuide:
    def __init__(self, workspace_dir, camera_count) -> None:
        self.workspace_dir = workspace_dir
        self.camera_count = camera_count
        self.intrinsic_dir = Path(workspace_dir, "calibration", "intrinsic")
        self.extrinsic_dir = Path(workspace_dir, "calibration", "extrinsic")
        self.recording_dir = Path(workspace_dir, "recordings")
        self.annotations_dir = Path(workspace_dir, "annotations")
        self.ground_truth_dir = Path(workspace_dir, "annotations", "ground_truth")
        self.predictions_dir = Path(workspace_dir, "annotations", "predictions")
        self.focus_dir = Path(workspace_dir, "calibration", "focus")
        self.arena_sim_dir = Path(workspace_dir, "arena_sim")
        self.annotation_index = AnnotationIndex(workspace_dir)
        # Cache configurator to avoid repeated config reloads during periodic updates
        self._configurator = Configurator(workspace_dir)
        self._camera_array = None

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
        if self._camera_array is not None:
            for cam in self._camera_array.cameras.values():
                if cam.distortions is None and cam.matrix is None and cam.error is None:
                    uncalibrated.append(str(cam.port))

        uncalibrated = ",".join(uncalibrated)
        if len(uncalibrated) == 0:
            uncalibrated = "NONE"
        return uncalibrated

    def intrinsic_calibration_status(self):
        if self._camera_array is not None \
        and self._camera_array.all_intrinsics_calibrated() \
        and self.all_instrinsic_mp4s_available():
            return "COMPLETE"
        else:
            return "INCOMPLETE"

    def extrinsic_calibration_status(self):
        if self._camera_array is not None \
        and self._camera_array.all_extrinsics_calibrated() \
        and self.all_extrinsic_mp4s_available():
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
        annotation_info = self.annotation_index.get_cached_annotation_info()
        if annotation_info is None:
            return self.annotation_index._empty_annotation_info()
        return annotation_info

    def valid_annotation_dir_text(self) -> str:
        return self.annotation_index.get_cached_annotation_dir_text()

    def get_cached_annotation_dir_text(self) -> str:
        return self.annotation_index.get_cached_annotation_dir_text()

    def annotation_scan_in_progress(self) -> bool:
        return self.annotation_index.annotation_scan_in_progress()

    def get_cached_annotation_classes(self):
        return self.annotation_index.get_cached_annotation_classes()

    def get_cached_annotation_label_maps(self):
        return self.annotation_index.get_cached_label_maps()

    def invalidate_config_cache(self) -> None:
        """Invalidate cached configurator when workspace files change."""
        self._configurator = Configurator(self.workspace_dir)
        self._camera_array = None

    def get_html_summary(self) -> str:
        """
        Provide granular summary of where the workspace is in the calibration process.
        Uses cached camera array to avoid repeated config reloads.
        """
        # Reload camera array only if not cached (config.toml changes are less frequent)
        if self._camera_array is None:
            self._camera_array = self._configurator.get_camera_array()
        self.camera_count = self._configurator.get_camera_count()

        # Get annotation text and convert newlines to <br> for HTML display
        anno_text = self.get_cached_annotation_dir_text()
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
