import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import rtoml
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QTabWidget,
    QWidget,
)

from calipod import __log_dir__, __root__, __settings_path__
from calipod.cameras.camera_array import CameraArray
from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.annotation_widget import AnnotationWidget
from calipod.gui.arena_sim_widget import ArenaSimWidget
from calipod.gui.bgs_processing_widget import BGSProcessingWidget
from calipod.gui.camera_management.multiplayback_widget import (
    MultiIntrinsicPlaybackWidget,
)
from calipod.gui.capture_widget import CameraCaptureWidget
from calipod.gui.charuco_widget import CharucoWidget
from calipod.gui.circuit_management_widget import CircuitManagementWidget
from calipod.gui.detection_widget import DetectionWidget
from calipod.gui.focus_widget import FocusWidget
from calipod.gui.log_widget import LogWidget
from calipod.gui.organise_widget import OrganisationWidget
from calipod.gui.post_processing_widget import PostProcessingWidget
from calipod.gui.vizualize.calibration.capture_volume_visualizer import CaptureVolumeVisualizer
from calipod.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from calipod.gui.workspace_widget import WorkspaceSummaryWidget

logger = calipod_logger.get(__name__)


class TabTypes(Enum):
    Workspace = 1
    Charuco = 2
    Cameras = 3
    CaptureVolume = 4


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(__settings_path__)

        self.setWindowTitle("Calipod")
        self.setWindowIcon(QIcon(str(Path(__root__, "calipod/gui/icons/box3d-center.svg"))))
        self.setMinimumSize(500, 500)
        self.central_tab = QWidget(self)
        self.setCentralWidget(self.central_tab)

        self.build_menus()
        self.connect_menu_actions()
        self.build_docked_logger()

    def connect_menu_actions(self):
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.exit_pyxy3d_action.triggered.connect(QApplication.instance().quit)
        self.open_log_directory_action.triggered.connect(self.open_log_dir)

    def build_menus(self):
        # File Menu
        self.menu = self.menuBar()

        # CREATE FILE MENU
        self.file_menu = self.menu.addMenu("&File")
        self.open_project_action = QAction("New/Open Project", self)
        self.file_menu.addAction(self.open_project_action)

        ####################  Open Recent  ################################
        self.open_recent_project_submenu = QMenu("Recent Projects...", self)

        # Populate the submenu with recent project paths;
        # reverse so that last one appended is at the top of the list
        for project_path in reversed(self.app_settings["recent_projects"]):
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)
        ###################################################################

        self.open_log_directory_action = QAction("Open Log Directory")
        self.file_menu.addAction(self.open_log_directory_action)
        self.exit_pyxy3d_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_pyxy3d_action)

    def build_central_tabs(self):
        self.central_tab = QTabWidget(self)
        self.setCentralWidget(self.central_tab)

        logger.info("Building workspace summary")
        self.workspace_summary = WorkspaceSummaryWidget(self.controller)
        self.workspace_summary.reload_workspace_btn.clicked.connect(self.reload_workspace)
        self.central_tab.addTab(self.workspace_summary, "Workspace")

        if self.controller.all_extrinsic_mp4s_available() and self.controller.camera_array.all_intrinsics_calibrated():
            self.workspace_summary.calibrate_btn.setEnabled(True)
        else:
            self.workspace_summary.calibrate_btn.setEnabled(False)

        logger.info("Creating organisation widget")
        self.organisation_widget = OrganisationWidget(self.controller)
        self.central_tab.addTab(self.organisation_widget, "Organisation")

        logger.info("Building arena sim widget")
        self.arena_sim_widget = ArenaSimWidget(self.controller)
        self.central_tab.addTab(self.arena_sim_widget, "Arena Sim")

        logger.info("Circuit Management widget")
        self.circuit_management_widget = CircuitManagementWidget(self.controller)
        self.central_tab.addTab(self.circuit_management_widget, "Circuit Management")

        logger.info("Focus widget")
        self.focus_widget = FocusWidget(self.controller)
        self.central_tab.addTab(self.focus_widget, "Focus")

        logger.info("Building camera capture widget")
        self.camera_capture_widget = CameraCaptureWidget(self.controller)
        self.central_tab.addTab(self.camera_capture_widget, "Camera Capture")

        logger.info("Building Charuco widget")
        self.charuco_widget = CharucoWidget(self.controller)
        self.central_tab.addTab(self.charuco_widget, "Charuco")

        logger.info("About to load Camera tab")
        if self.controller.cameras_loaded:
            logger.info("Creating MultiIntrinsic Playback Widget")
            self.intrinsic_cal_widget = MultiIntrinsicPlaybackWidget(self.controller)
            logger.info("MultiIntrinsic Playback Widget created")
        else:
            self.intrinsic_cal_widget = QWidget()

        logger.info("finished loading camera tab")
        self.central_tab.addTab(self.intrinsic_cal_widget, "Calibration")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Calibration"), self.controller.cameras_loaded)
        logger.info("Camera tab enabled")

        logger.info("About to load capture volume tab")
        if self.controller.capture_volume_loaded:
            logger.info("Creating capture Volume Widget")
            self.capture_volume_widget = CaptureVolumeWidget(self.controller)
        else:
            logger.info("Creating dummy widget")
            self.capture_volume_widget = QWidget()
        self.central_tab.addTab(self.capture_volume_widget, "Capture Volume")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Capture Volume"), self.controller.capture_volume_loaded
        )

        logger.info("Building annotation widget")
        self.annotation_widget = AnnotationWidget(self.controller)
        self.central_tab.addTab(self.annotation_widget, "Annotation")

        logger.info("About to load BGS detection tab")
        if self.controller.capture_volume_loaded and self.controller.recordings_available():
            logger.info("Creating BGS detection widget")
            self.bgs_processing_widget = BGSProcessingWidget(self.controller)
            bgs_processing_enabled = True
        else:
            logger.info("Creating dummy widget")
            self.bgs_processing_widget = QWidget()
            bgs_processing_enabled = False
        self.central_tab.addTab(self.bgs_processing_widget, "BGS Detection")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("BGS Detection"), bgs_processing_enabled)

        logger.info("Building DL Detection widget")
        self.detection_widget = DetectionWidget(self.controller)
        self.central_tab.addTab(self.detection_widget, "DL Detection")

        logger.info("About to load post-processing tab")
        if self.controller.capture_volume_loaded and self.controller.recordings_available():
            logger.info("Creating post processing widget")
            self.post_processing_widget = PostProcessingWidget(self.controller)
            self.controller.capture_volume_shifted.connect(self.post_processing_widget.refresh_visualizer)
            post_processing_enabled = True
        else:
            logger.info("Creating dummy widget")
            self.post_processing_widget = QWidget()
            post_processing_enabled = False
        self.central_tab.addTab(self.post_processing_widget, "Post Processing")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Post Processing"), post_processing_enabled)


    def build_docked_logger(self):
        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)
        self.resizeDocks([self.docked_logger], [180], Qt.Orientation.Vertical)

    def launch_workspace(self, path_to_workspace: str):
        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        self.controller = Controller(Path(path_to_workspace))
        self.controller.load_workspace_thread.finished.connect(self.build_central_tabs)
        logger.info("Initiate controller loading")
        self.controller.load_workspace()

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)

    def find_tab_index_by_title(self, title):
        # Iterate through tabs to find the index of the tab with the given title
        for index in range(self.central_tab.count()):
            if self.central_tab.tabText(index) == title:
                return index
        return -1  # Return -1 if the tab is not found

    def reload_workspace(self):
        # Clear all existing tabs
        logger.info("Clearing workspace")
        # Iterate backwards through the tabs and remove them
        for index in range(self.central_tab.count() - 1, -1, -1):
            widget_to_remove = self.central_tab.widget(index)
            logger.info(f"Removing tab with index {index}")
            self.central_tab.removeTab(index)
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()

            self.central_tab.clear()

        workspace = self.controller.workspace
        del self.controller
        self.controller = Controller(workspace_dir=workspace)
        self.controller.load_workspace()
        self.controller.load_workspace_thread.finished.connect(self.build_central_tabs)

    def add_to_recent_project(self, project_path: str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        project_path = action.text()
        logger.info(f"Opening recent session stored at {project_path}")
        self.launch_workspace(project_path)

    def open_log_dir(self):
        logger.info(f"Opening logging directory within File Explorer...  located at {__log_dir__}")
        if sys.platform == "win32":
            os.startfile(__log_dir__)
        elif sys.platform == "darwin":
            subprocess.run(["open", __log_dir__])
        else:  # Linux and Unix-like systems
            subprocess.run(["xdg-open", __log_dir__])
        pass

    def create_new_project_folder(self):
        default_folder = Path(self.app_settings["last_project_parent"])
        dialog = QFileDialog()
        path_to_folder = dialog.getExistingDirectory(
            parent=None,
            caption="Open Previous or Create New Project Directory",
            dir=str(default_folder),
            options=QFileDialog.Option.ShowDirsOnly,
        )

        if path_to_folder:
            logger.info(("Creating new project in :", path_to_folder))
            self.add_project_to_recent(path_to_folder)
            self.launch_workspace(path_to_folder)

    def add_project_to_recent(self, folder_path):
        if str(folder_path) in self.app_settings["recent_projects"]:
            pass
        else:
            self.app_settings["recent_projects"].append(str(folder_path))
            self.app_settings["last_project_parent"] = str(Path(folder_path).parent)
            self.update_app_settings()
            self.add_to_recent_project(folder_path)

    def update_app_settings(self):
        logger.info(f"Saving out app settings to {__settings_path__}")
        with open(__settings_path__, "w") as f:
            rtoml.dump(self.app_settings, f)

    def closeEvent(self, event):
        """Handle window close event to clean up resources"""
        logger.info("MainWindow closeEvent triggered - cleaning up resources")

        # Clean up controller resources FIRST (before closing streams)
        if hasattr(self, "controller"):
            logger.info("Cleaning up controller resources")

            # Stop intrinsic stream manager and its threads
            if hasattr(self.controller, "intrinsic_stream_manager"):
                logger.info("Closing intrinsic stream manager")
                try:
                    self.controller.intrinsic_stream_manager.close_stream_tools()
                except Exception as e:
                    logger.error(f"Error closing intrinsic stream manager: {e}")

            # Stop extrinsic stream manager if it exists
            if hasattr(self.controller, "extrinsic_stream_manager"):
                logger.info("Stopping extrinsic stream manager")
                try:
                    if hasattr(self.controller.extrinsic_stream_manager, "streams"):
                        for port, stream in self.controller.extrinsic_stream_manager.streams.items():
                            stream.stop_event.set()
                            if hasattr(stream, "thread"):
                                stream.thread.join(timeout=1.0)
                    if hasattr(self.controller.extrinsic_stream_manager, "recorder"):
                        self.controller.extrinsic_stream_manager.recorder.stop_recording()
                except Exception as e:
                    logger.error(f"Error closing extrinsic stream manager: {e}")

            # Stop synchronizer
            if hasattr(self.controller, "synchronizer"):
                logger.info("Stopping synchronizer")
                try:
                    self.controller.synchronizer.stop()
                except Exception as e:
                    logger.error(f"Error stopping synchronizer: {e}")

        # Stop any QThread emitters if they exist
        if hasattr(self, "post_processing_widget") and hasattr(self.post_processing_widget, "thumbnail_emitter"):
            logger.info("Stopping post processing thumbnail emitter")
            try:
                self.post_processing_widget.thumbnail_emitter.stop()
                self.post_processing_widget.thumbnail_emitter.wait(1000)
            except Exception as e:
                logger.error(f"Error stopping post processing emitter: {e}")

        # Stop intrinsic calibration widget threads if they exist
        if hasattr(self, "intrinsic_cal_widget") and hasattr(self.intrinsic_cal_widget, "thumbnail_emitter"):
            logger.info("Stopping intrinsic calibration thumbnail emitter")
            try:
                self.intrinsic_cal_widget.thumbnail_emitter.stop()
                self.intrinsic_cal_widget.thumbnail_emitter.wait(1000)
            except Exception as e:
                logger.error(f"Error stopping intrinsic emitter: {e}")

        logger.info("MainWindow cleanup complete")
        event.accept()


def launch_main():
    # import qdarktheme

    app = QApplication(sys.argv)
    dummy_widget = CaptureVolumeVisualizer(camera_array=CameraArray({}))  #  try to force "blinking to initial main"
    del dummy_widget
    # qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_main()
    # pass
