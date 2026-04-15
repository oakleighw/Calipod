"""This widget will allow user to link videos/annotations to the project if not already placed in the project folder. Aids transparent project management and organisation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller

logger = calipod_logger.get(__name__)

class OrganisationWidget(QWidget):
    def __init__(self, controller: Controller):
        super(OrganisationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        temp_to_do_widget = QLabel("To do: organisation widget")
        temp_to_do_widget.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(temp_to_do_widget)
        self.setLayout(layout)

    # This section will have sub-headings for each pipeline stage that requires data, which each having a url to the file in use.
    # Next to each url is a browse button that changes the url to a different path.
    def url_widget(self):
        pass

    # This section will have a file tree of the project folder,
    # with the option to add files to the project folder by dragging and dropping.
    def project_file_tree_widget(self):
        pass
