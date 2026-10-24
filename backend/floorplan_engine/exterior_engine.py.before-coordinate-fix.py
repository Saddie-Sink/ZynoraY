from __future__ import annotations

from typing import Any


VERSION = "1.0"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _box(
    x: float,
    y: float,
    width: float,
    depth: float,
    height: float,
    name: str,
    role: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "x": round(x, 3),
        "y": round(y, 3),
        "width": round(width, 3),
        "depth": round(depth, 3),
        "height": round(height, 3),
    }


# ---------------------------------------------------------------------------
# Design-spec normalization
# ---------------------------------------------------------------------------

def _read_design_spec(project: dict[str, Any] | None) -> dict[str, Any]:
    """
    Read optional design-spec information without requiring it.

    The existing StructuredDesignSpec is intentionally optional, so the
    exterior engine must also work when no design_spec is supplied.
    """
    project = project or {}

    raw = project.get("design_spec")

    if raw is None:
        return {
            "style_tags": [],
            "wants_terrace": False,
            "wants_porch": False,
            "wants_double_height_volume": False,
            "material_preference": None,
        }

    # Pydantic model support.
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()

    if not isinstance(raw, dict):
        raw = {}

    massing = raw.get("massing_hints") or {}

    if hasattr(massing, "model_dump"):
        massing = massing.model_dump()

    if not isinstance(massing, dict):
        massing = {}

    return {
        "style_tags": list(raw.get("style_tags") or []),
        "wants_terrace": bool(
            massing.get("wants_terrace", False)
        ),
        "wants_porch": bool(
            massing.get("wants_porch", False)
        ),
        "wants_double_height_volume": bool(
            massing.get("wants_double_height_volume", False)
        ),
        "material_preference": raw.get(
            "material_preference"
        ),
    }


# ---------------------------------------------------------------------------
# Floor detection
# ---------------------------------------------------------------------------

def _get_floors(plan: dict[str, Any]) -> list[dict[str, Any]]:
    floors = plan.get("floor_plans")

    if isinstance(floors, list) and floors:
        return floors

    floors = plan.get("floors_data")

    if isinstance(floors, list) and floors:
        return floors

    # Fallback for a single-floor canonical-ish object.
    return [
        {
            "level": 0,
            "rooms": plan.get("rooms", []),
        }
    ]


# ---------------------------------------------------------------------------
# Exterior room analysis
# ---------------------------------------------------------------------------

def _room_type(room: dict[str, Any]) -> str:
    return str(room.get("type", "")).lower().strip()


def _front_room_candidates(
    plan: dict[str, Any],
    front: str = "bottom",
) -> list[dict[str, Any]]:
    """
    Find rooms touching the chosen front boundary.

    Current structural_generator convention:
        y = 0
        y + height = building height

    We default to the bottom edge because the canonical building boundary
    is rectangular and the actual Blender camera/front direction can later
    be mapped independently.
    """
    width = _float(plan.get("width"))
    height = _float(plan.get("height"))

    candidates: list[dict[str, Any]] = []

    for floor in _get_floors(plan):
        for room in floor.get("rooms", []):
            x = _float(room.get("x"))
            y = _float(room.get("y"))
            w = _float(room.get("width"))
            h = _float(room.get("height"))

            touches = False

            if front == "bottom":
                touches = abs(y + h - height) <= 0.10
            elif front == "top":
                touches = abs(y) <= 0.10
            elif front == "left":
                touches = abs(x) <= 0.10
            elif front == "right":
                touches = abs(x + w - width) <= 0.10

            if touches:
                candidates.append(room)

    return candidates


def _best_front_span(
    plan: dict[str, Any],
) -> tuple[float, float]:
    """
    Determine a useful front-facing span from rooms.

    Returns:
        (center_x, span_width)
    """
    width = _float(plan.get("width"), 30.0)

    candidates = _front_room_candidates(plan)

    if not candidates:
        return width / 2.0, width * 0.55

    preferred = [
        room
        for room in candidates
        if _room_type(room) in {
            "living",
            "dining",
            "foyer",
            "kitchen",
        }
    ]

    if not preferred:
        preferred = candidates

    min_x = min(
        _float(room.get("x"))
        for room in preferred
    )

    max_x = max(
        _float(room.get("x"))
        + _float(room.get("width"))
        for room in preferred
    )

    span = max(4.0, max_x - min_x)

    return (
        (min_x + max_x) / 2.0,
        min(span, width),
    )


# ---------------------------------------------------------------------------
# Massing generation
# ---------------------------------------------------------------------------

def _main_mass(
    width: float,
    depth: float,
    floors: int,
) -> dict[str, Any]:
    """
    Full ground-level building mass.

    This is intentionally conservative. The existing floorplan remains the
    source of truth.
    """
    floor_height = 3.2

    return _box(
        0.0,
        0.0,
        width,
        depth,
        floor_height,
        "main-ground-mass",
        "main_mass",
    )


def _upper_mass(
    width: float,
    depth: float,
    floors: int,
    style_tags: list[str],
) -> dict[str, Any] | None:
    """
    Create a visually lighter upper volume.

    We inset the upper floor rather than changing the underlying floorplan.
    The inset is an exterior articulation decision only.
    """
    if floors < 2:
        return None

    tags = {
        str(tag).lower().replace("_", "-")
        for tag in style_tags
    }

    # Conservative inset.
    inset_x = _clamp(width * 0.06, 1.0, 2.5)
    inset_y = _clamp(depth * 0.05, 0.8, 2.0)

    # Contemporary / modern designs benefit from a stronger step.
    if {
        "contemporary",
        "modern",
        "modern-contemporary",
    } & tags:
        inset_x = _clamp(width * 0.08, 1.2, 3.0)

    return _box(
        inset_x,
        inset_y,
        max(4.0, width - inset_x * 2.0),
        max(4.0, depth - inset_y),
        3.2,
        "upper-primary-mass",
        "upper_mass",
    )


def _front_projection(
    plan: dict[str, Any],
    floors: int,
) -> dict[str, Any] | None:
    """
    Generate a modest front architectural bay.

    The bay is derived from the front-facing room span instead of being
    randomly positioned.
    """
    if floors < 1:
        return None

    width = _float(plan.get("width"), 30.0)
    depth = _float(plan.get("height"), 30.0)

    center_x, span = _best_front_span(plan)

    projection_width = _clamp(
        span * 0.42,
        5.0,
        width * 0.42,
    )

    projection_depth = _clamp(
        depth * 0.045,
        0.8,
        1.8,
    )

    x = center_x - projection_width / 2.0

    return _box(
        max(0.0, x),
        depth,
        projection_width,
        projection_depth,
        3.2,
        "front-architectural-bay",
        "front_projection",
    )


# ---------------------------------------------------------------------------
# Porch
# ---------------------------------------------------------------------------

def _porch(
    plan: dict[str, Any],
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    width = _float(plan.get("width"), 30.0)
    depth = _float(plan.get("height"), 30.0)

    center_x, span = _best_front_span(plan)

    porch_width = _clamp(
        span * 0.45,
        4.5,
        width * 0.38,
    )

    porch_depth = _clamp(
        width * 0.055,
        2.0,
        3.5,
    )

    return {
        "name": "front-porch",
        "role": "porch",
        "x": round(center_x - porch_width / 2.0, 3),
        "y": round(depth, 3),
        "width": round(porch_width, 3),
        "depth": round(porch_depth, 3),
        "roof_height": 3.0,
        "columns": [
            {
                "x": round(center_x - porch_width / 2.0 + 0.25, 3),
                "y": round(depth + porch_depth * 0.85, 3),
            },
            {
                "x": round(center_x + porch_width / 2.0 - 0.25, 3),
                "y": round(depth + porch_depth * 0.85, 3),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Balcony
# ---------------------------------------------------------------------------

def _balcony(
    plan: dict[str, Any],
    floors: int,
) -> dict[str, Any] | None:
    if floors < 2:
        return None

    width = _float(plan.get("width"), 30.0)
    depth = _float(plan.get("height"), 30.0)

    center_x, span = _best_front_span(plan)

    balcony_width = _clamp(
        span * 0.58,
        5.5,
        width * 0.55,
    )

    balcony_depth = _clamp(
        width * 0.075,
        2.0,
        3.0,
    )

    return {
        "name": "front-balcony",
        "role": "balcony",
        "level": 1,
        "x": round(center_x - balcony_width / 2.0, 3),
        "y": round(depth, 3),
        "width": round(balcony_width, 3),
        "depth": round(balcony_depth, 3),
        "slab_thickness": 0.22,
        "railing_height": 1.05,
        "railing": {
            "type": "glass-metal",
            "post_spacing": 1.2,
        },
    }


# ---------------------------------------------------------------------------
# Terrace
# ---------------------------------------------------------------------------

def _terrace(
    plan: dict[str, Any],
    floors: int,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled or floors < 1:
        return None

    width = _float(plan.get("width"), 30.0)
    depth = _float(plan.get("height"), 30.0)

    terrace_inset = _clamp(
        min(width, depth) * 0.06,
        1.0,
        2.0,
    )

    return {
        "name": "roof-terrace",
        "role": "terrace",
        "x": round(terrace_inset, 3),
        "y": round(terrace_inset, 3),
        "width": round(max(4.0, width - terrace_inset * 2.0), 3),
        "depth": round(max(4.0, depth - terrace_inset * 2.0), 3),
        "parapet_height": 1.05,
    }


# ---------------------------------------------------------------------------
# Roof
# ---------------------------------------------------------------------------

def _roof(
    plan: dict[str, Any],
    floors: int,
    style_tags: list[str],
) -> dict[str, Any]:
    tags = {
        str(tag).lower().replace("_", "-")
        for tag in style_tags
    }

    roof_type = "flat"

    if "gable" in tags or "pitched-roof" in tags:
        roof_type = "gable"

    return {
        "type": roof_type,
        "level": max(0, floors - 1),
        "overhang": 0.35 if roof_type == "flat" else 0.55,
        "parapet": roof_type == "flat",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_exterior_massing(
    plan: dict[str, Any],
    project: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate an exterior-only architectural description.

    Input:
        Existing structural/canonical floorplan dictionary.

    Output:
        Exterior massing dictionary.

    No interior room geometry is changed.
    """
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")

    width = _float(plan.get("width"))
    depth = _float(plan.get("height"))

    if width <= 0 or depth <= 0:
        raise ValueError(
            "plan.width and plan.height must be positive"
        )

    floors_data = _get_floors(plan)
    floor_count = max(
        1,
        _float(plan.get("floors"), len(floors_data)),
    )
    floor_count = int(floor_count)

    spec = _read_design_spec(project)

    masses = [
        _main_mass(
            width,
            depth,
            floor_count,
        )
    ]

    upper = _upper_mass(
        width,
        depth,
        floor_count,
        spec["style_tags"],
    )

    if upper:
        masses.append(upper)

    projection = _front_projection(
        plan,
        floor_count,
    )

    if projection:
        masses.append(projection)

    porch = _porch(
        plan,
        spec["wants_porch"],
    )

    balcony = _balcony(
        plan,
        floor_count,
    )

    terrace = _terrace(
        plan,
        floor_count,
        spec["wants_terrace"],
    )

    roof = _roof(
        plan,
        floor_count,
        spec["style_tags"],
    )

    return {
        "schema": "zynora.exterior.massing.v1",
        "version": VERSION,

        "source_plan": {
            "id": plan.get("id"),
            "width": round(width, 3),
            "depth": round(depth, 3),
            "floors": floor_count,
        },

        "design": {
            "style_tags": spec["style_tags"],
            "material_preference": spec[
                "material_preference"
            ],
            "double_height_requested": spec[
                "wants_double_height_volume"
            ],
        },

        "massing": {
            "strategy": "stepped-modern",
            "masses": masses,
        },

        "porch": porch,
        "balcony": balcony,
        "terrace": terrace,
        "roof": roof,

        "interior_policy": {
            "modify_rooms": False,
            "modify_internal_walls": False,
            "generate_furniture": False,
        },
    }


def generate_exterior_from_document(
    document: dict[str, Any],
    project: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Adapter for the canonical ZYNORA floorplan document.

    Keeps the canonical floorplan untouched and converts it into the
    simpler structure expected by generate_exterior_massing().
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dictionary")

    if document.get("schemaVersion") != "zynora.floorplan.v1":
        raise ValueError(
            "Expected schemaVersion 'zynora.floorplan.v1'"
        )

    floors = [
        floor
        for floor in (document.get("floors") or [])
        if isinstance(floor, dict)
    ]

    if not floors:
        raise ValueError(
            "FloorPlanJSON does not contain any floors."
        )

    # Canonical ZYNORA floors use their outline as the source
    # of the building footprint.
    ground = floors[0]
    outline = ground.get("outline") or []

    if not outline:
        raise ValueError(
            "Ground floor does not contain an outline."
        )

    xs = []
    ys = []

    for point in outline:
        if not isinstance(point, dict):
            continue

        try:
            xs.append(float(point["x"]))
            ys.append(float(point["y"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not xs or not ys:
        raise ValueError(
            "Ground-floor outline contains no valid points."
        )

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    plan = {
        "id": document.get("id"),
        "width": max_x - min_x,
        "height": max_y - min_y,
        "floors": len(floors),
        "floor_plans": [],
    }

    for index, floor in enumerate(floors):
        rooms = []

        for room in floor.get("rooms") or []:
            if not isinstance(room, dict):
                continue

            rooms.append({
                "id": room.get("id"),
                "type": room.get("type"),
                "x": room.get("x", 0),
                "y": room.get("y", 0),
                "width": room.get("width", 0),
                "height": room.get("height", 0),
            })

        plan["floor_plans"].append({
            "level": index,
            "rooms": rooms,
        })

    return generate_exterior_massing(
        plan,
        project,
    )


__all__ = [
    "generate_exterior_massing",
    "generate_exterior_from_document",
]