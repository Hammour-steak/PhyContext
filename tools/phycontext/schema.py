import json
import math
from pathlib import Path
from typing import Iterable


SAMPLE_SCHEMA = "physweep.video_condition_sample.v4"
MANIFEST_SCHEMA = "physweep.video_condition_manifest.v4"
SWEEP_AXES = ("mass_kg", "contact_friction", "contact_restitution")
SPLITS = {"train", "validation", "test"}


class SchemaError(ValueError):
    pass


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise SchemaError(f"{context} is missing {key}")
    return mapping[key]


def _require_exact_keys(mapping: dict, keys: set[str], context: str) -> None:
    actual = set(mapping)
    if actual != keys:
        missing = sorted(keys - actual)
        unexpected = sorted(actual - keys)
        raise SchemaError(
            f"{context} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _finite_number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise SchemaError(f"{context} must be finite")
    return value


def _finite_vector(value, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{context} must contain {length} numbers")
    return tuple(
        _finite_number(component, f"{context}[{index}]")
        for index, component in enumerate(value)
    )


def _positive_definite_tensor(value, context: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise SchemaError(f"{context} must be a 3 x 3 matrix")
    matrix = tuple(
        _finite_vector(row, 3, f"{context}[{index}]")
        for index, row in enumerate(value)
    )
    scale = max(abs(component) for row in matrix for component in row)
    tolerance = max(scale * 1e-7, 1e-15)
    for row in range(3):
        for column in range(row + 1, 3):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise SchemaError(f"{context} must be symmetric")
    first_minor = matrix[0][0]
    second_minor = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if first_minor <= 0.0 or second_minor <= 0.0 or determinant <= 0.0:
        raise SchemaError(f"{context} must be positive definite")
    return matrix


def _relative_path(value: str, context: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaError(f"{context} must be a project-relative path")
    return path


def validate_training_record(
    record: dict,
    project_root: Path | None = None,
    check_files: bool = False,
) -> None:
    if record.get("schema") != SAMPLE_SCHEMA:
        raise SchemaError("unexpected training sample schema")
    sample_id = _require(record, "sample_id", "record")
    base_scene_id = _require(record, "base_scene_id", "record")
    if not isinstance(sample_id, str) or not sample_id:
        raise SchemaError("sample_id must be a non-empty string")
    if not isinstance(base_scene_id, str) or not base_scene_id:
        raise SchemaError("base_scene_id must be a non-empty string")
    if record.get("split") not in SPLITS:
        raise SchemaError("split must be train, validation, or test")

    conditioning = _require(record, "conditioning", "record")
    _require_exact_keys(
        conditioning,
        {"first_frame", "scene", "scene_source", "text", "physics"},
        "conditioning",
    )
    if conditioning["scene_source"] not in {"simulation_gt", "image_reconstruction"}:
        raise SchemaError("conditioning.scene_source is invalid")
    if not isinstance(conditioning["text"], str) or not conditioning["text"].strip():
        raise SchemaError("conditioning.text must be non-empty")
    paths = [
        (conditioning.get("first_frame"), "conditioning.first_frame"),
        (conditioning.get("scene"), "conditioning.scene"),
        (record.get("target", {}).get("video"), "target.video"),
        (record.get("target", {}).get("metadata"), "target.metadata"),
    ]
    resolved_paths = []
    for value, context in paths:
        if not isinstance(value, str) or not value:
            raise SchemaError(f"{context} must be a non-empty string")
        relative = _relative_path(value, context)
        resolved_paths.append((relative, context))
    if check_files:
        if project_root is None:
            raise SchemaError("project_root is required when check_files is enabled")
        root = project_root.resolve()
        for relative, context in resolved_paths:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise SchemaError(f"{context} escapes the project root")
            if not path.is_file():
                raise SchemaError(f"{context} does not exist: {relative}")

    physics = _require(conditioning, "physics", "conditioning")
    _require_exact_keys(
        physics,
        {"source", "coordinate_frame", "object", "world"},
        "physics",
    )
    if physics.get("source") not in {"simulation_gt", "user"}:
        raise SchemaError("physics.source must be simulation_gt or user")
    if physics.get("coordinate_frame") != "camera_right_up_forward":
        raise SchemaError("physics vectors must use camera_right_up_forward")
    object_control = _require(physics, "object", "physics")
    _require_exact_keys(
        object_control,
        {
            "object_id",
            "mass_kg",
            "inertia_tensor_camera_kg_m2",
            "friction",
            "restitution",
            "rolling_friction",
            "spinning_friction",
            "linear_damping",
            "angular_damping",
            "initial_state",
        },
        "physics.object",
    )
    object_id = object_control.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise SchemaError("physics.object.object_id must be a non-empty string")
    mass = _finite_number(object_control.get("mass_kg"), "mass_kg")
    _positive_definite_tensor(
        object_control.get("inertia_tensor_camera_kg_m2"),
        "inertia_tensor_camera_kg_m2",
    )
    friction = _finite_number(object_control.get("friction"), "friction")
    restitution = _finite_number(object_control.get("restitution"), "restitution")
    if mass <= 0:
        raise SchemaError("mass_kg must be positive")
    if friction < 0:
        raise SchemaError("friction must be non-negative")
    if not 0 <= restitution <= 1:
        raise SchemaError("restitution must be in [0, 1]")
    for key in (
        "rolling_friction",
        "spinning_friction",
        "linear_damping",
        "angular_damping",
    ):
        if _finite_number(object_control.get(key), key) < 0.0:
            raise SchemaError(f"{key} must be non-negative")
    initial_state = _require(object_control, "initial_state", "physics.object")
    _require_exact_keys(
        initial_state,
        {"linear_velocity_camera_m_s", "angular_velocity_camera_rad_s"},
        "physics.object.initial_state",
    )
    _finite_vector(
        initial_state.get("linear_velocity_camera_m_s"),
        3,
        "initial_state.linear_velocity_camera_m_s",
    )
    _finite_vector(
        initial_state.get("angular_velocity_camera_rad_s"),
        3,
        "initial_state.angular_velocity_camera_rad_s",
    )
    world = _require(physics, "world", "physics")
    _require_exact_keys(world, {"gravity_camera_m_s2"}, "physics.world")
    _finite_vector(
        world.get("gravity_camera_m_s2"),
        3,
        "physics.world.gravity_camera_m_s2",
    )

    sweep = _require(record, "sweep", "record")
    mode = sweep.get("mode")
    if mode not in {"base", "one_factor"}:
        raise SchemaError("sweep.mode must be base or one_factor")
    axis = sweep.get("axis")
    if mode == "base" and axis is not None:
        raise SchemaError("base samples cannot declare a sweep axis")
    if mode == "one_factor" and axis not in SWEEP_AXES:
        raise SchemaError("one-factor sample has an unsupported axis")
    level_count = sweep.get("level_count")
    level_index = sweep.get("level_index")
    base_level_index = sweep.get("base_level_index")
    if level_count != 5:
        raise SchemaError("the current sweep contract requires five levels")
    if not isinstance(level_index, int) or not 0 <= level_index < level_count:
        raise SchemaError("sweep.level_index is invalid")
    if not isinstance(base_level_index, int) or not 0 <= base_level_index < level_count:
        raise SchemaError("sweep.base_level_index is invalid")
    base_level_indices = sweep.get("base_level_indices")
    if not isinstance(base_level_indices, dict) or set(base_level_indices) != set(SWEEP_AXES):
        raise SchemaError("sweep.base_level_indices must cover every axis")
    if any(
        not isinstance(value, int) or not 0 <= value < level_count
        for value in base_level_indices.values()
    ):
        raise SchemaError("sweep.base_level_indices contains an invalid level")
    source_axis = sweep.get("source_axis")
    if source_axis not in SWEEP_AXES:
        raise SchemaError("sweep.source_axis is invalid")
    if base_level_index != base_level_indices[source_axis]:
        raise SchemaError("base_level_index does not match source_axis")
    if mode == "base" and level_index != base_level_index:
        raise SchemaError("base sample must use its source axis base level")
    if mode == "one_factor" and (
        source_axis != axis or level_index == base_level_indices[axis]
    ):
        raise SchemaError("one-factor sample must use a non-base level of its axis")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"invalid JSON at line {line_number}: {error}") from error


def validate_manifest(
    path: Path,
    project_root: Path | None = None,
    check_files: bool = False,
) -> dict:
    sample_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    split_counts = {name: 0 for name in sorted(SPLITS)}
    mode_counts = {"base": 0, **{axis: 0 for axis in SWEEP_AXES}}
    count = 0
    for record in iter_jsonl(path):
        validate_training_record(record, project_root, check_files)
        sample_id = record["sample_id"]
        if sample_id in sample_ids:
            raise SchemaError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        base_scene_id = record["base_scene_id"]
        split = record["split"]
        previous = group_splits.setdefault(base_scene_id, split)
        if previous != split:
            raise SchemaError(f"base scene crosses splits: {base_scene_id}")
        split_counts[split] += 1
        sweep = record["sweep"]
        mode_counts["base" if sweep["mode"] == "base" else sweep["axis"]] += 1
        count += 1
    return {
        "sample_count": count,
        "base_scene_count": len(group_splits),
        "split_counts": split_counts,
        "mode_counts": mode_counts,
    }
