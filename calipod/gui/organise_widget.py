"""This widget will allow user to link videos/annotations to the project if not already placed in the project folder. Aids transparent project management and organisation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

from calipod.gui.utils.file_tree import create_project_file_tree_view
from calipod.gui.utils.path_url_entry import create_path_url_entry
from calipod.gui.utils.styles import (
    create_styled_groupbox,
    create_subsection_title,
    create_subsubsection_title,
    resolve_camera_title_color,
)

logger = calipod_logger.get(__name__)

class OrganisationWidget(QWidget):
    def __init__(self, controller: Controller):
        super(OrganisationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_vbox = QVBoxLayout()
        self.bottom_vbox = QVBoxLayout()

        self.bottom_container = QWidget()
        self.bottom_container.setLayout(self.bottom_vbox)
        self.bottom_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.file_tree_scroll = QScrollArea()
        self.file_tree_scroll.setWidgetResizable(True)
        self.file_tree_scroll.setWidget(self.bottom_container)
        self.file_tree_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_tree_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.file_tree_scroll.setMinimumHeight(0)

        self.url_widget()
        self.project_file_tree_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addWidget(self.file_tree_scroll, stretch=1)

    # This section will have sub-headings for each pipeline stage that requires data, which each having a url to the file in use.
    # Next to each url is a browse button that changes the url to a different path.
    def url_widget(self):
        url_group, url_layout = create_styled_groupbox("Data Locations")
        camera_count = self.controller.get_camera_count()

        url_content = QWidget(self)
        url_content_layout = QVBoxLayout(url_content)
        url_content_layout.setContentsMargins(0, 0, 0, 0)
        url_content_layout.setSpacing(6)

        self.url_content_scroll = QScrollArea(self)
        self.url_content_scroll.setWidgetResizable(True)
        self.url_content_scroll.setWidget(url_content)
        self.url_content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.url_content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.url_content_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.url_content_scroll.setMinimumHeight(0)

        url_layout.addWidget(self.url_content_scroll)

        ### SUBSECTIONS ###

        # Calibration video URLS
        self.calibration_video_url_label = create_subsection_title("Calibration Video URLs", color="black")
        url_content_layout.addWidget(self.calibration_video_url_label)
        self.calibration_path_rows = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                url_content_layout,
                camera_index,
                self.calibration_path_rows,
                "C:/data/project",  # set this to current intrinsic calibration paths if found in project config.
                context= "Intrinsic"
            )
            self._add_camera_path_row(
                url_content_layout,
                camera_index,
                self.calibration_path_rows,
                "C:/data/project",  # set this to current extrinsiccalibration paths if found in project config.
                context="Extrinsic",
            )

        # Action video "recordings" URLS
        self.action_video_url_label = create_subsection_title("Action Video URLs", color="black")
        url_content_layout.addWidget(self.action_video_url_label)
        self.action_recordings_path_rows = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                url_content_layout,
                camera_index,
                self.action_recordings_path_rows,
                "C:/data/project",  # set this to current behaviour recordings paths if found in project config.
            )

        # Annotation URLS
        self.annotation_url_label = create_subsection_title("Annotation URLs", color="black")
        url_content_layout.addWidget(self.annotation_url_label)
        self.annotation_path_rows = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                url_content_layout,
                camera_index,
                self.annotation_path_rows,
                "C:/data/project",  # set this to current annotation paths if found in project config.
            )

        url_content_layout.addStretch(1)

        self.top_vbox.addWidget(url_group)

    def _add_camera_path_row(self, layout, camera_index: int, path_rows: list, initial_path: str, context: str = None):
        camera_data = self.controller.camera_array.cameras.get(camera_index)
        camera_label_color = resolve_camera_title_color(
            camera_index=camera_index,
            camera_count=self.controller.get_camera_count(),
            camera_data=camera_data,
        )
        if context:
            camera_label = camera_label = create_subsubsection_title(f"Camera {camera_index} {context}", color=camera_label_color)
        else:
            camera_label = create_subsubsection_title(f"Camera {camera_index}", color=camera_label_color)

        camera_path_row = create_path_url_entry(
            initial_path=initial_path,
            parent=self,
            dialog_caption="Select Data Folder",
            select_directory=False,
        )
        path_rows.append(camera_path_row)
        layout.addWidget(camera_label)
        layout.addWidget(camera_path_row.container)

    # This section will have a file tree of the project folder,
    # with the option to add files to the project folder by dragging and dropping.
    def project_file_tree_widget(self):
        file_tree_group, file_tree_layout = create_styled_groupbox("Project File Tree")
        project_dir_label = create_subsubsection_title("Project Directory", color="black")
        project_dir_path_label = QLabel(str(self.controller.workspace), self)
        project_dir_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        project_dir_path_label.setWordWrap(True)
        project_dir_path_label.setStyleSheet("color: #444;")

        file_tree_layout.addWidget(project_dir_label)
        file_tree_layout.addWidget(project_dir_path_label)
        self.project_file_tree, self.project_file_tree_model = create_project_file_tree_view(
            root_path=self.controller.workspace,
            parent=self,
        )
        file_tree_layout.addWidget(self.project_file_tree)
        self.bottom_vbox.addWidget(file_tree_group)
