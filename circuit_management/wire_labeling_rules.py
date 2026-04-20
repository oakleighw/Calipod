"""Shared rules and options for circuit wire-labeling workflows."""

# Number of physical connector ports available by connector model.
CONNECTOR_PIN_COUNTS: dict[str, int] = {
    "Hirose 6 Pin": 6,
    "Hirose 4 Pin": 4,
}

# Valid wire-function labels that can be assigned for each connector model.
WIRE_TYPE_OPTIONS: dict[str, list[str]] = {
    "Hirose 4 Pin": [
        "External Ground",
        "Octo-coupled Output",
        "Octo-coupled Ground",
        "Octo-coupled Input",
    ],
    "Hirose 6 Pin": [
        "General purpose I/O (GPIO) line",
        "Octo-coupled Output",
        "Octo-coupled Input",
        "GPIO Ground",
        "Octo-coupled Ground",
    ],
}

# Maximum allowed uses of each wire-function label per connector model.
WIRE_TYPE_MAX_COUNTS: dict[str, dict[str, int]] = {
    "Hirose 4 Pin": {
        "External Ground": 1,
        "Octo-coupled Output": 1,
        "Octo-coupled Ground": 1,
        "Octo-coupled Input": 1,
    },
    "Hirose 6 Pin": {
        "General purpose I/O (GPIO) line": 2,
        "Octo-coupled Output": 1,
        "Octo-coupled Input": 1,
        "GPIO Ground": 1,
        "Octo-coupled Ground": 1,
    },
}

# Human-readable colour names and their display hex values.
WIRE_COLOUR_OPTIONS: list[tuple[str, str]] = [
    ("black", "#000000"),
    ("white", "#6E6E6E"),
    ("red", "#D32F2F"),
    ("green", "#2E7D32"),
    ("brown", "#795548"),
    ("blue", "#1976D2"),
    ("orange", "#EF6C00"),
    ("yellow", "#C9A200"),
    ("violet", "#7B1FA2"),
    ("grey", "#616161"),
    ("pink", "#C2185B"),
    ("light blue", "#2A9DDF"),
]


def normalize_wire_type_selections(
    connector_name: str,
    selected_values: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Return valid wire-type selections constrained by connector rules.

    Invalid or over-subscribed selections are replaced with empty values.

    Returns canonicalized selections plus the resulting usage counts.
    """
    options = WIRE_TYPE_OPTIONS.get(connector_name, [])
    max_counts = WIRE_TYPE_MAX_COUNTS.get(connector_name, {})

    canonical_values: list[str] = []
    used_counts: dict[str, int] = {}

    for value in selected_values:
        normalized = value.strip()
        max_allowed = max_counts.get(normalized, 1)
        if normalized in options and used_counts.get(normalized, 0) < max_allowed:
            canonical_values.append(normalized)
            used_counts[normalized] = used_counts.get(normalized, 0) + 1
        else:
            canonical_values.append("")

    return canonical_values, used_counts


def available_wire_type_options(
    connector_name: str,
    current_value: str,
    used_counts: dict[str, int],
) -> list[str]:
    """Return currently valid wire-type options for a dropdown.

    The current row's selection is allowed to remain visible even if the
    option is globally saturated by max-count limits.
    """
    options = WIRE_TYPE_OPTIONS.get(connector_name, [])
    max_counts = WIRE_TYPE_MAX_COUNTS.get(connector_name, {})

    allowed_options: list[str] = []
    for option in options:
        used_elsewhere = used_counts.get(option, 0) - (1 if current_value == option else 0)
        max_allowed = max_counts.get(option, 1)
        if used_elsewhere < max_allowed:
            allowed_options.append(option)

    return allowed_options
