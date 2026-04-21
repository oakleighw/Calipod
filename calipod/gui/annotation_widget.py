"""This widget will supply a basic annotation tool and also point to CVAT for extended functionality."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from calipod.annotation_management import LabelEditorWidget
from calipod.core import logger as calipod_logger
from calipod.core.controller import Controller
from calipod.gui.utils.styles import create_styled_groupbox

logger = calipod_logger.get(__name__)


class AnnotationWidget(QWidget):
    def __init__(self, controller: Controller):
        super(AnnotationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_vbox = QVBoxLayout()
        self.centre_vbox = QVBoxLayout()
        self.bottom_vbox = QVBoxLayout()

        self.anno_labelling_widget()
        self.annotation_checker_widget()
        self.external_links_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addLayout(self.centre_vbox, stretch=1)
        self.layout().addLayout(self.bottom_vbox, stretch=1)

    def anno_labelling_widget(self):
        """This widget allows the user to log string labels for annotation class index labels."""
        anno_label_group, anno_label_layout = create_styled_groupbox("Labels")

        # Create label editor widget
        label_editor = LabelEditorWidget(str(self.controller.workspace))
        anno_label_layout.addWidget(label_editor)

        self.top_vbox.addWidget(anno_label_group)

    def annotation_checker_widget(self):
        """This widget checks if annotations are correct by showing zoomed bounding boxes etc. overlaid on frames."""
        anno_check_group, anno_check_layout = create_styled_groupbox("Annotation Checker")
        self.centre_vbox.addWidget(anno_check_group)

    def external_links_widget(self):
        """This widget points to external links for annotations e.g. CVAT."""
        ext_links_group, ext_links_layout = create_styled_groupbox("Links")

        # Create a clickable link to CVAT documentation
        cvat_link = QLabel()
        cvat_link.setText(
            '<a href="https://docs.cvat.ai/docs/administration/community/basics/installation/">'
            'CVAT Installation Guide</a>'
        )
        cvat_link.setOpenExternalLinks(True)
        cvat_link.setAlignment(Qt.AlignCenter)

        ext_links_layout.addWidget(cvat_link)

        self.bottom_vbox.addWidget(ext_links_group)

