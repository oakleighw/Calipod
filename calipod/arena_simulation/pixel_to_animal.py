"""Pixel-to-animal computations and validation for arena simulation UI."""

from dataclasses import dataclass


PIXEL_TO_ANIMAL_CONFLICT_MESSAGE = "Too many values entered, submit only furthest distance or pixel size in pixels"
PIXEL_TO_ANIMAL_RESULT_DECIMALS = 4


@dataclass
class PixelToAnimalInputs:
    min_insect_size_mm: float
    pixel_size_on_sensor_um: float
    furthest_distance_mm: float
    animal_pixel_size: float
    focal_length_mm: float
    input_mode: str | None


@dataclass
class PixelToAnimalResult:
    insect_label_text: str
    distance_label_text: str
    computed_insect_pixel_count: float | None
    computed_furthest_distance_mm: float | None
    set_animal_pixel_size: float | None = None
    set_furthest_distance_mm: float | None = None


def compute_pixel_distance_coverage(
    animal_size_mm: float,
    pixel_size_pixels: float,
    distance_to_animal_mm: float,
    focal_length_mm: float,
    sensor_width_um: float,
) -> float | None:
    """Compute pixel coverage from distance or distance from pixel coverage."""
    sensor_width_mm = sensor_width_um / 1000.0

    if distance_to_animal_mm:
        pixel_size_pixels = (animal_size_mm * focal_length_mm) / (distance_to_animal_mm * sensor_width_mm)
        return pixel_size_pixels

    if pixel_size_pixels:
        distance_to_animal_mm = (animal_size_mm * focal_length_mm) / (pixel_size_pixels * sensor_width_mm)
        return distance_to_animal_mm

    return None


def resolve_input_mode_for_distance_change(current_mode: str | None, animal_pixel_size: float) -> str | None:
    """Return next input mode after distance value changes."""
    if current_mode == "pixel_size" and animal_pixel_size > 0:
        return current_mode
    return "distance"


def resolve_input_mode_for_pixel_size_change(current_mode: str | None, furthest_distance_mm: float) -> str | None:
    """Return next input mode after pixel-size value changes."""
    if current_mode == "distance" and furthest_distance_mm > 0:
        return current_mode
    return "pixel_size"


def compute_pixel_to_animal_result(inputs: PixelToAnimalInputs) -> PixelToAnimalResult:
    """Compute UI-facing pixel-to-animal outputs from current inputs and mode."""
    if (
        inputs.min_insect_size_mm <= 0
        or inputs.pixel_size_on_sensor_um <= 0
        or inputs.focal_length_mm <= 0
    ):
        return PixelToAnimalResult(
            insect_label_text="Missing valid lens/animal parameters",
            distance_label_text="Missing valid lens/animal parameters",
            computed_insect_pixel_count=None,
            computed_furthest_distance_mm=None,
        )

    if inputs.input_mode == "distance":
        if inputs.furthest_distance_mm <= 0:
            return PixelToAnimalResult(
                insect_label_text="Enter furthest distance (mm)",
                distance_label_text="Enter furthest distance (mm)",
                computed_insect_pixel_count=None,
                computed_furthest_distance_mm=None,
                set_animal_pixel_size=0.0,
            )

        computed_pixel_size = compute_pixel_distance_coverage(
            animal_size_mm=inputs.min_insect_size_mm,
            pixel_size_pixels=0.0,
            distance_to_animal_mm=inputs.furthest_distance_mm,
            focal_length_mm=inputs.focal_length_mm,
            sensor_width_um=inputs.pixel_size_on_sensor_um,
        )
        if computed_pixel_size is None:
            return PixelToAnimalResult(
                insect_label_text="Unable to compute animal pixel size",
                distance_label_text="Unable to compute animal pixel size",
                computed_insect_pixel_count=None,
                computed_furthest_distance_mm=None,
            )

        computed_pixel_size = float(computed_pixel_size)
        computed_furthest_distance_mm = float(inputs.furthest_distance_mm)
        return PixelToAnimalResult(
            insect_label_text=f"{computed_pixel_size:.{PIXEL_TO_ANIMAL_RESULT_DECIMALS}f} pixels @ furthest distance",
            distance_label_text=f"{computed_furthest_distance_mm:.{PIXEL_TO_ANIMAL_RESULT_DECIMALS}f} mm",
            computed_insect_pixel_count=computed_pixel_size,
            computed_furthest_distance_mm=computed_furthest_distance_mm,
            set_animal_pixel_size=computed_pixel_size,
        )

    if inputs.input_mode == "pixel_size":
        if inputs.animal_pixel_size <= 0:
            return PixelToAnimalResult(
                insect_label_text="Enter animal pixel size (pixels)",
                distance_label_text="Enter animal pixel size (pixels)",
                computed_insect_pixel_count=None,
                computed_furthest_distance_mm=None,
                set_furthest_distance_mm=0.0,
            )

        computed_furthest_distance_mm = compute_pixel_distance_coverage(
            animal_size_mm=inputs.min_insect_size_mm,
            pixel_size_pixels=inputs.animal_pixel_size,
            distance_to_animal_mm=0.0,
            focal_length_mm=inputs.focal_length_mm,
            sensor_width_um=inputs.pixel_size_on_sensor_um,
        )
        if computed_furthest_distance_mm is None:
            return PixelToAnimalResult(
                insect_label_text="Unable to compute furthest distance",
                distance_label_text="Unable to compute furthest distance",
                computed_insect_pixel_count=None,
                computed_furthest_distance_mm=None,
            )

        computed_insect_pixel_count = float(inputs.animal_pixel_size)
        computed_furthest_distance_mm = float(computed_furthest_distance_mm)
        return PixelToAnimalResult(
            insect_label_text=f"{computed_insect_pixel_count:.{PIXEL_TO_ANIMAL_RESULT_DECIMALS}f} pixels @ furthest distance",
            distance_label_text=f"{computed_furthest_distance_mm:.{PIXEL_TO_ANIMAL_RESULT_DECIMALS}f} mm",
            computed_insect_pixel_count=computed_insect_pixel_count,
            computed_furthest_distance_mm=computed_furthest_distance_mm,
            set_furthest_distance_mm=computed_furthest_distance_mm,
        )

    if inputs.furthest_distance_mm > 0 and inputs.animal_pixel_size > 0:
        return PixelToAnimalResult(
            insect_label_text=PIXEL_TO_ANIMAL_CONFLICT_MESSAGE,
            distance_label_text=PIXEL_TO_ANIMAL_CONFLICT_MESSAGE,
            computed_insect_pixel_count=None,
            computed_furthest_distance_mm=None,
        )

    return PixelToAnimalResult(
        insect_label_text="Enter either furthest distance or animal pixel size",
        distance_label_text="Enter either furthest distance or animal pixel size",
        computed_insect_pixel_count=None,
        computed_furthest_distance_mm=None,
    )

