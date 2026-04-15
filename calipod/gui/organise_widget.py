"""This widget will allow user to link videos/annotations to the project if not already placed in the project folder. Aids transparent project management and organisation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

from calipod.gui.utils.styles import create_styled_groupbox, create_subsection_title

logger = calipod_logger.get(__name__)

class OrganisationWidget(QWidget):
    def __init__(self, controller: Controller):
        super(OrganisationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.top_vbox = QVBoxLayout()
        self.bottom_vbox = QVBoxLayout()

        self.url_widget()
        self.project_file_tree_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addLayout(self.bottom_vbox, stretch=1)

    # This section will have sub-headings for each pipeline stage that requires data, which each having a url to the file in use.
    # Next to each url is a browse button that changes the url to a different path.
    def url_widget(self):
        url_group, url_layout = create_styled_groupbox("Data Locations")

        ### SUBSECTIONS ###

        # Calibration video URLS
        self.calibration_video_url_label = create_subsection_title("Calibration Video URLs", color="blue")

        # Action video "recordings" URLS
        self.action_video_url_label = create_subsection_title("Action Video URLs", color="blue")

        # Annotation URLS
        self.annotation_url_label = create_subsection_title("Annotation URLs", color="blue")

        url_layout.addWidget(self.calibration_video_url_label)
        url_layout.addWidget(self.action_video_url_label)
        url_layout.addWidget(self.annotation_url_label)
        
        self.top_vbox.addWidget(url_group)

    # This section will have a file tree of the project folder,
    # with the option to add files to the project folder by dragging and dropping.
    def project_file_tree_widget(self):
        file_tree_group, file_tree_layout = create_styled_groupbox("Project File Tree")
        self.bottom_vbox.addWidget(file_tree_group)
