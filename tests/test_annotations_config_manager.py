from calipod.annotation_management.annotations_config_manager import (
    STRUCTURE_GEOMETRY_FLAT,
    STRUCTURE_GEOMETRY_SEMI_SPHERE,
    AnnotationsConfigManager,
)


def test_structure_defaults_and_updates(tmp_path):
    manager = AnnotationsConfigManager(tmp_path)

    manager.save_annotations_config(
        ground_truth_labels={9: "fruit", 10: "leaves", 11: "custom"},
        ground_truth_format="YOLO",
    )

    fruit_structure = manager.get_label_structure_metadata(9, is_ground_truth=True)
    leaves_structure = manager.get_label_structure_metadata(10, is_ground_truth=True)
    custom_structure = manager.get_label_structure_metadata(11, is_ground_truth=True)

    assert fruit_structure["enabled"] is False
    assert fruit_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT
    assert leaves_structure["enabled"] is False
    assert leaves_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT
    assert custom_structure["enabled"] is False
    assert custom_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT

    manager.update_label_category(11, "frame roi", is_ground_truth=True)
    frame_roi_structure = manager.get_label_structure_metadata(11, is_ground_truth=True)
    assert frame_roi_structure["enabled"] is False
    assert frame_roi_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT

    manager.update_label_structure_metadata(
        11,
        is_ground_truth=True,
        enabled=True,
        geometry=STRUCTURE_GEOMETRY_SEMI_SPHERE,
    )

    refreshed = AnnotationsConfigManager(tmp_path)
    updated_structure = refreshed.get_label_structure_metadata(11, is_ground_truth=True)
    assert updated_structure["enabled"] is True
    assert updated_structure["geometry"] == STRUCTURE_GEOMETRY_SEMI_SPHERE


def test_get_frame_roi_structure_labels(tmp_path):
    manager = AnnotationsConfigManager(tmp_path)
    manager.save_annotations_config(
        ground_truth_labels={9: "fruit", 12: "leaf patch"},
        ground_truth_format="YOLO",
    )

    manager.update_label_category(9, "frame roi", is_ground_truth=True)
    manager.update_label_category(12, "frame roi", is_ground_truth=True)
    manager.update_label_structure_metadata(
        12,
        is_ground_truth=True,
        enabled=True,
        geometry=STRUCTURE_GEOMETRY_FLAT,
    )

    roi_labels = manager.get_frame_roi_structure_labels(is_ground_truth=True)
    assert 9 in roi_labels
    assert 12 in roi_labels
    assert roi_labels[9]["structure"]["geometry"] == STRUCTURE_GEOMETRY_FLAT
    assert roi_labels[12]["structure"]["enabled"] is True


def test_frame_roi_defaults_enabled_without_legacy_id_rules(tmp_path):
    manager = AnnotationsConfigManager(tmp_path)
    manager.save_annotations_config(
        ground_truth_labels={9: "fruit", 10: "leaves"},
        ground_truth_format="YOLO",
    )

    manager.update_label_category(9, "frame roi", is_ground_truth=True)
    manager.update_label_category(10, "frame roi", is_ground_truth=True)

    fruit_structure = manager.get_label_structure_metadata(9, is_ground_truth=True)
    leaves_structure = manager.get_label_structure_metadata(10, is_ground_truth=True)

    assert fruit_structure["enabled"] is False
    assert fruit_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT
    assert leaves_structure["enabled"] is False
    assert leaves_structure["geometry"] == STRUCTURE_GEOMETRY_FLAT


def test_arena_vertex_add_to_floor_metadata(tmp_path):
    manager = AnnotationsConfigManager(tmp_path)
    manager.save_annotations_config(
        ground_truth_labels={21: "arena_a", 22: "arena_b"},
        ground_truth_format="YOLO",
    )

    manager.update_label_category(21, "arena vertex", is_ground_truth=True)
    manager.update_label_category(22, "arena vertex", is_ground_truth=True)

    manager.update_label_structure_metadata(21, is_ground_truth=True, add_to_floor=True)

    arena = manager.get_arena_vertex_structure_labels(is_ground_truth=True)
    assert 21 in arena
    assert 22 in arena
    assert arena[21]["structure"]["add_to_floor"] is True
    assert arena[22]["structure"]["add_to_floor"] is False
