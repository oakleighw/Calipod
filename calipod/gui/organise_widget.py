"""Allow linking videos/annotations to the project when files are stored
outside the project folder to support transparent project organization."""

import html
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calipod.core.controller import Controller
from calipod.gui.utils.collapsible_container import create_collapsible_container
from calipod.gui.utils.external_path_linker import link_external_path_selection
from calipod.gui.utils.file_tree import create_project_file_tree_view
from calipod.gui.utils.path_url_entry import create_path_url_entry
from calipod.gui.utils.styles import (
    create_styled_groupbox,
    create_subsubsection_title,
    resolve_camera_title_color,
)
from calipod.gui.utils.video_metadata import MetadataWorker


class OrganisationWidget(QWidget):


    def __init__(self, controller: Controller):
        super(OrganisationWidget, self).__init__()
        self.controller = controller
        self.place_widgets()
        self.MetadataWorker = MetadataWorker

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.top_vbox = QVBoxLayout()
        self.middle_vbox = QVBoxLayout()
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
        self.check_video_info_widget()
        self.project_file_tree_widget()

        self.layout().addLayout(self.top_vbox, stretch=1)
        self.layout().addLayout(self.middle_vbox, stretch=1)
        self.layout().addWidget(self.file_tree_scroll, stretch=1)

    # This section has sub-headings for each pipeline stage requiring data,
    # each with a URL to the file in use.
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
        calibration_layout = self._add_collapsible_section(
            parent_layout=url_content_layout,
            title="Calibration Video URLs",
            expanded=False,
        )
        self.calibration_path_rows = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                calibration_layout,
                camera_index,
                self.calibration_path_rows,
                self._path_display_for_camera(
                    self.controller.workspace_guide.intrinsic_dir,
                    camera_index,
                    fallback_name=f"port_{camera_index}.mp4",
                ),
                context="Intrinsic",
                link_destination=self.controller.workspace_guide.intrinsic_dir / f"port_{camera_index}.mp4",
                link_label=f"Camera {camera_index} intrinsic calibration video",
            )
            self._add_camera_path_row(
                calibration_layout,
                camera_index,
                self.calibration_path_rows,
                self._path_display_for_camera(
                    self.controller.workspace_guide.extrinsic_dir,
                    camera_index,
                    fallback_name=f"port_{camera_index}.mp4",
                ),
                context="Extrinsic",
                link_destination=self.controller.workspace_guide.extrinsic_dir / f"port_{camera_index}.mp4",
                link_label=f"Camera {camera_index} extrinsic calibration video",
            )

        # Action video "recordings" URLS
        action_layout = self._add_collapsible_section(
            parent_layout=url_content_layout,
            title="Action Video URLs",
            expanded=False,
        )
        self.action_recordings_path_rows = {}
        recording_dirs = self._recording_directories()
        if not recording_dirs:
            recording_dirs = [self.controller.workspace_guide.recording_dir / "recording_1"]

        for index, recording_dir in enumerate(recording_dirs):
            self._add_recording_section(
                parent_layout=action_layout,
                recording_dir=recording_dir,
                camera_count=camera_count,
                expanded=False,
            )

        # Annotation URLS
        annotation_layout = self._add_collapsible_section(
            parent_layout=url_content_layout,
            title="Annotation URLs",
            expanded=False,
        )
        self.annotation_path_rows = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                annotation_layout,
                camera_index,
                self.annotation_path_rows,
                self._annotation_path_display(camera_index, is_prediction=False),
                context="Ground Truth",
                select_directory=True,
                link_destination=self.controller.workspace_guide.ground_truth_dir / f"port_{camera_index}",
                link_label=f"Camera {camera_index} ground truth annotations directory",
            )
            self._add_camera_path_row(
                annotation_layout,
                camera_index,
                self.annotation_path_rows,
                self._annotation_path_display(camera_index, is_prediction=True),
                context="Ext. Predicted Detections",
                select_directory=True,
                link_destination=self.controller.workspace_guide.predictions_dir / f"port_{camera_index}",
                link_label=f"Camera {camera_index} predicted detections directory",
            )

        url_content_layout.addStretch(1)

        self.top_vbox.addWidget(url_group)

    def _add_camera_path_row(
        self,
        layout,
        camera_index: int,
        path_rows: list,
        initial_path: str,
        context: str = None,
        select_directory: bool = False,
        link_destination: Path | None = None,
        link_label: str | None = None,
    ):
        camera_data = self.controller.camera_array.cameras.get(camera_index)
        camera_label_color = resolve_camera_title_color(
            camera_index=camera_index,
            camera_count=self.controller.get_camera_count(),
            camera_data=camera_data,
        )
        if context:
            camera_label = create_subsubsection_title(
                f'<span style="color: {camera_label_color};">Camera {camera_index}</span> '
                f'<span style="color: black;">{html.escape(context)}</span>'
            )
        else:
            camera_label = create_subsubsection_title(f"Camera {camera_index}", color=camera_label_color)

        camera_path_row = create_path_url_entry(
            initial_path=initial_path,
            parent=self,
            dialog_caption="Select Data Folder",
            select_directory=select_directory,
            on_path_selected=(
                lambda selected_path, destination=link_destination, is_dir=select_directory,
                label=link_label: link_external_path_selection(
                    parent=self,
                    selected_path=selected_path,
                    link_destination=destination,
                    workspace_root=Path(self.controller.workspace),
                    expect_directory=is_dir,
                    target_label=label,
                    on_link_created=self.refresh_project_file_tree,
                )
            ),
        )
        path_rows.append(camera_path_row)
        layout.addWidget(camera_label)
        layout.addWidget(camera_path_row.container)

    def _path_display_for_camera(self, root_dir: Path, camera_index: int, fallback_name: str) -> str:
        candidate = root_dir / fallback_name
        if candidate.exists():
            return str(candidate)
        return f"{candidate} (not uploaded/found yet)"

    def _recording_directories(self) -> list[Path]:
        recording_root = self.controller.workspace_guide.recording_dir
        if not recording_root.exists():
            return []
        return sorted([p for p in recording_root.iterdir() if p.is_dir()])

    def _recording_path_display_for_camera(self, recording_dir: Path, camera_index: int) -> str:
        return self._path_display_for_camera(
            recording_dir,
            camera_index,
            fallback_name=f"port_{camera_index}.mp4",
        )

    def _add_recording_section(self, parent_layout, recording_dir: Path, camera_count: int, expanded: bool):
        content_layout = create_collapsible_container(
            parent=self,
            parent_layout=parent_layout,
            title=recording_dir.name,
            expanded=expanded,
            font_weight=600,
            content_left_indent=12,
        )

        rows_for_recording = []
        for camera_index in range(1, camera_count + 1):
            self._add_camera_path_row(
                content_layout,
                camera_index,
                rows_for_recording,
                self._recording_path_display_for_camera(recording_dir, camera_index),
                link_destination=recording_dir / f"port_{camera_index}.mp4",
                link_label=f"{recording_dir.name} camera {camera_index} action recording video",
            )
        self.action_recordings_path_rows[recording_dir.name] = rows_for_recording

    def _add_collapsible_section(self, parent_layout, title: str, expanded: bool = False):
        return create_collapsible_container(
            parent=self,
            parent_layout=parent_layout,
            title=title,
            expanded=expanded,
            font_weight=700,
            content_left_indent=8,
        )

    def _annotation_path_display(self, camera_index: int, is_prediction: bool) -> str:
        base_dir = (
            self.controller.workspace_guide.predictions_dir
            if is_prediction
            else self.controller.workspace_guide.ground_truth_dir
        )
        camera_dir = base_dir / f"port_{camera_index}"
        existing_file = self._first_file_in_tree(camera_dir)
        if existing_file is not None:
            return str(existing_file.parent)

        sample_dir = camera_dir / "labels" / "train"
        return f"{sample_dir} (not uploaded/found yet)"

    def _first_file_in_tree(self, root_dir: Path) -> Path | None:
        if not root_dir.exists():
            return None

        for file_path in sorted(root_dir.rglob("*")):
            if file_path.is_file():
                return file_path
        return None

    def check_video_info_widget(self):
        """
        Video info widget: dropdown for video selection,
        'External video' option, path/browse controls, and details popup.
        """

        video_info_group, video_info_layout = create_styled_groupbox("Video Information")
        # Reduce vertical space usage
        video_info_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        video_info_group.setMaximumHeight(160)
        video_info_group.setMinimumHeight(0)
        video_info_group.setContentsMargins(8, 4, 8, 4)

        # Gather all video info (path, label, exists) from calibration and action recordings
        video_infos = []
        calib_contexts = ["Intrinsic", "Extrinsic"]
        if hasattr(self, 'calibration_path_rows'):
            for idx, row in enumerate(self.calibration_path_rows):
                path = row.line_edit.text().strip()
                context = calib_contexts[idx % 2]
                p = Path(path)
                parent = p.parent.name if p.parent.name else str(p.parent)
                label = f"{context} calibration / {parent} / {p.name}"
                exists = Path(path).exists()
                video_infos.append({"path": path, "label": label, "exists": exists})
        for rec_rows in getattr(self, 'action_recordings_path_rows', {}).values():
            for row in rec_rows:
                path = row.line_edit.text().strip()
                p = Path(path)
                parent = p.parent.name if p.parent.name else str(p.parent)
                label = f"Action recording / {parent} / {p.name}"
                exists = Path(path).exists()
                video_infos.append({"path": path, "label": label, "exists": exists})

        # Dropdown: first item is 'External video', then all found video labels (only if exists)
        dropdown = QComboBox(self)
        dropdown.addItem("External video")
        for info in video_infos:
            if info["exists"]:
                dropdown.addItem(info["label"])

        # Path input and browse button (hidden unless 'External video' is selected)
        external_path_row = create_path_url_entry(
            initial_path="",
            parent=self,
            dialog_caption="Select Video File",
            select_directory=False,
            file_filter="Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv *.flv *.webm);;All Files (*)",
        )
        external_path_row.container.setVisible(True)  # Show by default if 'External video' is selected

        # Details button
        details_btn = QPushButton("Details", self)

        # Layout
        video_info_layout.addWidget(dropdown)
        video_info_layout.addWidget(external_path_row.container)
        video_info_layout.addWidget(details_btn)

        self.middle_vbox.addWidget(video_info_group)

        def update_external_path_visibility():
            external_path_row.container.setVisible(dropdown.currentIndex() == 0)

        dropdown.currentIndexChanged.connect(update_external_path_visibility)
        update_external_path_visibility()

        def get_selected_video_path():
            if dropdown.currentIndex() == 0:
                return external_path_row.line_edit.text().strip()
            else:
                idx = dropdown.currentIndex() - 1
                # Only include videos that exist in the dropdown
                existing_infos = [info for info in video_infos if info["exists"]]
                if 0 <= idx < len(existing_infos):
                    return existing_infos[idx]["path"]
                return None



        def show_video_details():
            import logging

            from PySide6.QtCore import QObject, Signal
            path = get_selected_video_path()
            if not path or not Path(path).exists():
                dlg = QDialog(self)
                dlg.setWindowTitle("Video Details")
                layout = QVBoxLayout(dlg)
                layout.addWidget(QLabel("Video file not found or not selected."))
                dlg.exec()
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Video Details: {Path(path).name}")
            layout = QVBoxLayout(dlg)
            text = QTextEdit(dlg)
            text.setReadOnly(True)
            text.setText("Loading video metadata...")
            layout.addWidget(text)
            dlg.resize(400, 250)

            # Signal object to safely update GUI from worker thread
            class MetadataSignalEmitter(QObject):
                meta_ready = Signal(dict)

            signal_emitter = MetadataSignalEmitter()

            # Worker thread for metadata extraction
            thread = QThread()
            worker = self.MetadataWorker(path)
            worker.moveToThread(thread)

            def on_finished(meta):
                signal_emitter.meta_ready.emit(meta)

            def update_ui(meta):
                if "error" in meta:
                    err = meta["error"]
                    if "ffprobe" in err or "No such file or directory" in err or "not found" in err:
                        text.setText("Error: ffprobe (from ffmpeg) is not available on this system.\n" \
                        "\nPlease install ffmpeg and ensure it is in your system PATH.")
                        logging.error("ffprobe (from ffmpeg) is not available for video metadata extraction." \
                        " Please install ffmpeg and ensure it is in your system PATH.")
                    else:
                        text.setText(f"Error: {err}")
                        logging.error(f"Video metadata extraction error: {err}")
                else:
                    lines = [
                        f"Path: {path}",
                        f"Codec: {meta.get('codec', '?')}",
                        f"FPS: {meta.get('fps', '?')}",
                        f"Keyframe count: {meta.get('keyframe_count', '?')}",
                        f"Image format: {meta.get('pix_fmt', '?')}",
                    ]
                    text.setText("\n".join(lines))
                thread.quit()
                thread.wait()

            signal_emitter.meta_ready.connect(update_ui)
            worker.finished.connect(on_finished, Qt.QueuedConnection)
            thread.started.connect(worker.run)
            thread.start()
            dlg.exec()
            thread.quit()
            thread.wait()

        details_btn.clicked.connect(show_video_details)

    # This section will have a file tree of the project folder,
    # with the option to add files to the project folder by dragging and dropping.
    def project_file_tree_widget(self):
        file_tree_group, file_tree_layout = create_styled_groupbox("Project File Tree")
        project_dir_label = create_subsubsection_title("Project Directory", color="black")
        project_dir_path_label = QLabel(str(self.controller.workspace), self)
        project_dir_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        project_dir_path_label.setWordWrap(True)
        project_dir_path_label.setStyleSheet("color: #444;")
        self.refresh_file_tree_btn = QPushButton("Refresh", self)
        self.refresh_file_tree_btn.clicked.connect(self.refresh_project_file_tree)

        project_dir_header_layout = QHBoxLayout()
        project_dir_header_layout.addWidget(project_dir_label)
        project_dir_header_layout.addStretch(1)
        project_dir_header_layout.addWidget(self.refresh_file_tree_btn)

        file_tree_layout.addLayout(project_dir_header_layout)
        file_tree_layout.addWidget(project_dir_path_label)
        self.project_file_tree, self.project_file_tree_model = create_project_file_tree_view(
            root_path=self.controller.workspace,
            parent=self,
        )
        file_tree_layout.addWidget(self.project_file_tree)
        self.bottom_vbox.addWidget(file_tree_group)

    def refresh_project_file_tree(self):
        root_str = str(self.controller.workspace)
        self.project_file_tree_model.setRootPath(root_str)
        self.project_file_tree.setRootIndex(self.project_file_tree_model.index(root_str))
