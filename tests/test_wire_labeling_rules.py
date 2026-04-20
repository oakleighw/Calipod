from circuit_management.wire_labeling_rules import available_wire_type_options, normalize_wire_type_selections


def test_normalize_wire_type_selections_enforces_max_counts_for_hirose_6():
    selected_values = [
        "General purpose I/O (GPIO) line",
        "General purpose I/O (GPIO) line",
        "General purpose I/O (GPIO) line",
        "Octo-coupled Output",
    ]

    canonical, used_counts = normalize_wire_type_selections("Hirose 6 Pin", selected_values)

    assert canonical == [
        "General purpose I/O (GPIO) line",
        "General purpose I/O (GPIO) line",
        "",
        "Octo-coupled Output",
    ]
    assert used_counts["General purpose I/O (GPIO) line"] == 2
    assert used_counts["Octo-coupled Output"] == 1


def test_available_wire_type_options_blocks_saturated_choices():
    used_counts = {
        "Octo-coupled Output": 1,
        "General purpose I/O (GPIO) line": 2,
    }

    options = available_wire_type_options("Hirose 6 Pin", "", used_counts)

    assert "Octo-coupled Output" not in options
    assert "General purpose I/O (GPIO) line" not in options
    assert "GPIO Ground" in options
    assert "Octo-coupled Input" in options


def test_available_wire_type_options_keeps_current_value_available():
    used_counts = {
        "Octo-coupled Output": 1,
    }

    options = available_wire_type_options("Hirose 6 Pin", "Octo-coupled Output", used_counts)

    assert "Octo-coupled Output" in options
