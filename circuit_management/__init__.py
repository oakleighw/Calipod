"""Circuit management configuration module."""

from circuit_management.config_manager import CircuitManagementConfigManager
from circuit_management.timing_sync_planner import (
    TimingCheckpoint,
    TimingSyncPlan,
    build_timing_sync_plan,
    format_ms_as_clock,
    render_timing_sync_table,
)
from circuit_management.wire_labeling_rules import (
    CONNECTOR_PIN_COUNTS,
    WIRE_COLOUR_OPTIONS,
    WIRE_TYPE_MAX_COUNTS,
    WIRE_TYPE_OPTIONS,
    available_wire_type_options,
    normalize_wire_type_selections,
)

__all__ = [
    "CircuitManagementConfigManager",
    "TimingCheckpoint",
    "TimingSyncPlan",
    "build_timing_sync_plan",
    "format_ms_as_clock",
    "render_timing_sync_table",
    "CONNECTOR_PIN_COUNTS",
    "WIRE_COLOUR_OPTIONS",
    "WIRE_TYPE_MAX_COUNTS",
    "WIRE_TYPE_OPTIONS",
    "available_wire_type_options",
    "normalize_wire_type_selections",
]
