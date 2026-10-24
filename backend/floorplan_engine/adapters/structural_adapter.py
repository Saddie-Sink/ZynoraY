"""Structural plan -> canonical document adapter.

Adapts generated structural candidates (from structural_generator.py) into
AdaptedFloorplan and canonical zynora.floorplan.v1 documents.
"""

from __future__ import annotations

import math
from typing import Any

from floorplan_engine.adapters import AdaptedFloorplan
from floorplan_engine.canonical_builder import (
    create_floor_plan_document,
    process_wall_topology,
)
from floorplan_engine.geometry_validator import validate_floor_plan_document
from floorplan_engine.schemas import FloorPlanDocument

DEFAULT_WALL_HEIGHT = 2.8
DEFAULT_EXTERIOR_THICKNESS = 0.20
DEFAULT_INTERIOR_THICKNESS = 0.15
MIN_WALL_LENGTH = 0.04
MIN_OPENING_WIDTH = 0.08

DEFAULT_OPENING_VERTICAL = {
    "door": {"bottom": 0.0, "height": 2.1},
    "window": {"bottom": 0.9, "height": 1.2},
}


def _room_to_outline(room: dict[str, Any]) -> list[dict[str, float]]:
    x = float(room.get("x", 0.0))
    z = float(room.get("y", 0.0))
    w = float(room.get("width", 0.0))
    h = float(room.get("height", 0.0))

    return [
        {"x": round(x, 5), "z": round(z, 5)},
        {"x": round(x + w, 5), "z": round(z, 5)},
        {"x": round(x + w, 5), "z": round(z + h, 5)},
        {"x": round(x, 5), "z": round(z + h, 5)},
    ]


def _room_to_canonical(room: dict[str, Any]) -> dict[str, Any]:
    w = float(room.get("width", 0.0))
    h = float(room.get("height", 0.0))
    area = float(room.get("area", w * h))

    return {
        "id": str(room.get("id", "room")),
        "type": str(room.get("type") or room.get("roomType") or "Room"),
        "outline": _room_to_outline(room),
        "area": round(area, 4),
    }


def _boundary_to_outline(
    boundary: list[Any] | None,
    fallback_w: float,
    fallback_h: float,
) -> list[dict[str, float]]:
    if boundary and len(boundary) >= 3:
        outline: list[dict[str, float]] = []
        for pt in boundary:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                outline.append(
                    {"x": round(float(pt[0]), 5), "z": round(float(pt[1]), 5)}
                )
            elif isinstance(pt, dict) and "x" in pt:
                z_val = pt.get("z", pt.get("y", 0.0))
                outline.append(
                    {"x": round(float(pt["x"]), 5), "z": round(float(z_val), 5)}
                )
        if len(outline) >= 3:
            return outline

    return [
        {"x": 0.0, "z": 0.0},
        {"x": round(fallback_w, 5), "z": 0.0},
        {"x": round(fallback_w, 5), "z": round(fallback_h, 5)},
        {"x": 0.0, "z": round(fallback_h, 5)},
    ]


def _extract_perimeter_walls(
    outline: list[dict[str, float]],
    floor_level: int,
    height: float,
    thickness: float,
) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    n = len(outline)

    for i in range(n):
        p1 = outline[i]
        p2 = outline[(i + 1) % n]
        length = math.hypot(p2["x"] - p1["x"], p2["z"] - p1["z"])
        if length < MIN_WALL_LENGTH:
            continue

        walls.append(
            {
                "id": f"ext-wall-{floor_level}-{i}",
                "x1": p1["x"],
                "z1": p1["z"],
                "x2": p2["x"],
                "z2": p2["z"],
                "height": height,
                "thickness": thickness,
                "isExterior": True,
                "color": "#eee9e1",
                "openings": [],
                "metadata": {"kind": "perimeter", "index": i},
            }
        )

    return walls


def _extract_interior_walls(
    rooms: list[dict[str, Any]],
    floor_level: int,
    height: float,
    thickness: float,
) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    wall_index = 0
    n = len(rooms)

    for i in range(n):
        r1 = rooms[i]
        x1, y1 = float(r1.get("x", 0.0)), float(r1.get("y", 0.0))
        w1, h1 = float(r1.get("width", 0.0)), float(r1.get("height", 0.0))

        for j in range(i + 1, n):
            r2 = rooms[j]
            x2, y2 = float(r2.get("x", 0.0)), float(r2.get("y", 0.0))
            w2, h2 = float(r2.get("width", 0.0)), float(r2.get("height", 0.0))

            # 1. Vertical boundary: r1 right edge touches r2 left edge
            if abs((x1 + w1) - x2) < 0.01:
                ov_start = max(y1, y2)
                ov_end = min(y1 + h1, y2 + h2)
                if ov_end - ov_start >= MIN_WALL_LENGTH:
                    wall_index += 1
                    walls.append(
                        {
                            "id": f"int-wall-{floor_level}-{wall_index}",
                            "x1": round(x2, 5),
                            "z1": round(ov_start, 5),
                            "x2": round(x2, 5),
                            "z2": round(ov_end, 5),
                            "height": height,
                            "thickness": thickness,
                            "isExterior": False,
                            "color": "#eee9e1",
                            "openings": [],
                            "metadata": {
                                "room1": str(r1.get("id")),
                                "room2": str(r2.get("id")),
                                "rooms": {str(r1.get("id")), str(r2.get("id"))},
                            },
                        }
                    )

            # 2. Vertical boundary: r2 right edge touches r1 left edge
            elif abs((x2 + w2) - x1) < 0.01:
                ov_start = max(y1, y2)
                ov_end = min(y1 + h1, y2 + h2)
                if ov_end - ov_start >= MIN_WALL_LENGTH:
                    wall_index += 1
                    walls.append(
                        {
                            "id": f"int-wall-{floor_level}-{wall_index}",
                            "x1": round(x1, 5),
                            "z1": round(ov_start, 5),
                            "x2": round(x1, 5),
                            "z2": round(ov_end, 5),
                            "height": height,
                            "thickness": thickness,
                            "isExterior": False,
                            "color": "#eee9e1",
                            "openings": [],
                            "metadata": {
                                "room1": str(r1.get("id")),
                                "room2": str(r2.get("id")),
                                "rooms": {str(r1.get("id")), str(r2.get("id"))},
                            },
                        }
                    )

            # 3. Horizontal boundary: r1 bottom touches r2 top
            if abs((y1 + h1) - y2) < 0.01:
                ov_start = max(x1, x2)
                ov_end = min(x1 + w1, x2 + w2)
                if ov_end - ov_start >= MIN_WALL_LENGTH:
                    wall_index += 1
                    walls.append(
                        {
                            "id": f"int-wall-{floor_level}-{wall_index}",
                            "x1": round(ov_start, 5),
                            "z1": round(y2, 5),
                            "x2": round(ov_end, 5),
                            "z2": round(y2, 5),
                            "height": height,
                            "thickness": thickness,
                            "isExterior": False,
                            "color": "#eee9e1",
                            "openings": [],
                            "metadata": {
                                "room1": str(r1.get("id")),
                                "room2": str(r2.get("id")),
                                "rooms": {str(r1.get("id")), str(r2.get("id"))},
                            },
                        }
                    )

            # 4. Horizontal boundary: r2 bottom touches r1 top
            elif abs((y2 + h2) - y1) < 0.01:
                ov_start = max(x1, x2)
                ov_end = min(x1 + w1, x2 + w2)
                if ov_end - ov_start >= MIN_WALL_LENGTH:
                    wall_index += 1
                    walls.append(
                        {
                            "id": f"int-wall-{floor_level}-{wall_index}",
                            "x1": round(ov_start, 5),
                            "z1": round(y1, 5),
                            "x2": round(ov_end, 5),
                            "z2": round(y1, 5),
                            "height": height,
                            "thickness": thickness,
                            "isExterior": False,
                            "color": "#eee9e1",
                            "openings": [],
                            "metadata": {
                                "room1": str(r1.get("id")),
                                "room2": str(r2.get("id")),
                                "rooms": {str(r1.get("id")), str(r2.get("id"))},
                            },
                        }
                    )

    return walls


def _project_openings_onto_walls(
    walls: list[dict[str, Any]],
    doors: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    seen_opening_ids: set[str] = set()

    all_openings = [("door", d) for d in doors] + [("window", w) for w in windows]

    for op_type, raw_op in all_openings:
        op_id = str(raw_op.get("id", f"{op_type}-{len(seen_opening_ids) + 1}"))
        if op_id in seen_opening_ids:
            op_id = f"{op_id}-dup"
        seen_opening_ids.add(op_id)

        is_horizontal = raw_op.get("orientation") == "horizontal"
        op_x = float(raw_op.get("x", 0.0))
        op_y = float(raw_op.get("y", 0.0))
        op_w = float(raw_op.get("width", 0.0))
        op_h = float(raw_op.get("height", 0.0))

        if is_horizontal:
            op_span_start = op_x
            op_span_end = op_x + op_w
            op_center_normal = op_y + op_h / 2.0
        else:
            op_span_start = op_y
            op_span_end = op_y + op_h
            op_center_normal = op_x + op_w / 2.0

        best_wall: dict[str, Any] | None = None
        best_score = -1e9
        best_start = 0.0
        best_width = 0.0

        from_room = str(raw_op.get("from_room_id") or "")
        to_room = str(raw_op.get("to_room_id") or "")
        room_id = str(raw_op.get("room_id") or "")

        for wall in walls:
            w_dx = wall["x2"] - wall["x1"]
            w_dz = wall["z2"] - wall["z1"]
            w_len = math.hypot(w_dx, w_dz)
            if w_len < MIN_WALL_LENGTH:
                continue

            wall_is_horiz = abs(w_dz) <= 0.01
            wall_is_vert = abs(w_dx) <= 0.01

            if is_horizontal and not wall_is_horiz:
                continue
            if not is_horizontal and not wall_is_vert:
                continue

            if is_horizontal:
                w_norm = (wall["z1"] + wall["z2"]) / 2.0
                dist = min(
                    abs(w_norm - op_center_normal),
                    abs(w_norm - op_y),
                    abs(w_norm - (op_y + op_h)),
                )
                w_start = min(wall["x1"], wall["x2"])
                w_end = max(wall["x1"], wall["x2"])
            else:
                w_norm = (wall["x1"] + wall["x2"]) / 2.0
                dist = min(
                    abs(w_norm - op_center_normal),
                    abs(w_norm - op_x),
                    abs(w_norm - (op_x + op_w)),
                )
                w_start = min(wall["z1"], wall["z2"])
                w_end = max(wall["z1"], wall["z2"])

            if dist > 0.40:
                continue

            overlap = min(w_end, op_span_end) - max(w_start, op_span_start)
            if overlap < 0.02:
                continue

            ux = w_dx / w_len
            uz = w_dz / w_len

            if is_horizontal:
                pt1 = (max(w_start, op_span_start), w_norm)
                pt2 = (min(w_end, op_span_end), w_norm)
            else:
                pt1 = (w_norm, max(w_start, op_span_start))
                pt2 = (w_norm, min(w_end, op_span_end))

            t1 = (pt1[0] - wall["x1"]) * ux + (pt1[1] - wall["z1"]) * uz
            t2 = (pt2[0] - wall["x1"]) * ux + (pt2[1] - wall["z1"]) * uz
            proj_start = max(0.0, min(t1, t2))
            proj_end = min(w_len, max(t1, t2))
            proj_width = proj_end - proj_start

            if proj_width < MIN_OPENING_WIDTH:
                continue

            score = overlap * 10.0 - dist * 20.0
            wall_rooms = (wall.get("metadata") or {}).get("rooms", set())

            if from_room and to_room:
                if from_room in wall_rooms and to_room in wall_rooms:
                    score += 1000.0
                elif from_room in wall_rooms or to_room in wall_rooms:
                    score += 100.0
            elif room_id:
                if wall.get("isExterior"):
                    score += 500.0
                elif room_id in wall_rooms:
                    score += 100.0

            if score > best_score:
                best_score = score
                best_wall = wall
                best_start = proj_start
                best_width = proj_width

        if best_wall is None:
            warnings.append(
                f"Opening {op_id!r} could not be associated with any wall; skipped."
            )
            continue

        vertical = DEFAULT_OPENING_VERTICAL[op_type]
        bottom = vertical["bottom"]
        height = vertical["height"]

        if bottom + height > best_wall["height"]:
            height = max(0.2, best_wall["height"] - bottom)

        best_wall["openings"].append(
            {
                "id": op_id,
                "type": op_type,
                "start": round(best_start, 5),
                "offset": round(best_start, 5),
                "width": round(best_width, 5),
                "bottom": round(bottom, 5),
                "height": round(height, 5),
            }
        )

    return warnings


class StructuralAdapter:
    """Adapts structural_generator candidates into AdaptedFloorplan and canonical documents."""

    def __init__(
        self,
        default_wall_height: float = DEFAULT_WALL_HEIGHT,
        exterior_thickness: float = DEFAULT_EXTERIOR_THICKNESS,
        interior_thickness: float = DEFAULT_INTERIOR_THICKNESS,
    ) -> None:
        self.default_wall_height = float(default_wall_height)
        self.exterior_thickness = float(exterior_thickness)
        self.interior_thickness = float(interior_thickness)

    def adapt(
        self,
        raw_input: dict[str, Any],
    ) -> AdaptedFloorplan:
        """Adapt a candidate or single floor dictionary into AdaptedFloorplan."""
        if not isinstance(raw_input, dict):
            raise ValueError(
                "raw_input must be a dict representing a candidate or floor plan."
            )

        if "floor_plans" in raw_input and raw_input["floor_plans"]:
            floor = raw_input["floor_plans"][0]
            candidate = raw_input
        else:
            floor = raw_input
            candidate = raw_input

        width = float(candidate.get("width", 30.0))
        height = float(candidate.get("height", 40.0))
        boundary = candidate.get("boundary")
        outline = _boundary_to_outline(boundary, width, height)

        floor_level = int(floor.get("level", 0))
        rooms_raw = floor.get("rooms") or [
            r
            for r in candidate.get("rooms", [])
            if r.get("floor_level", 0) == floor_level
        ]
        doors_raw = floor.get("doors") or [
            d
            for d in candidate.get("doors", [])
            if d.get("floor_level", 0) == floor_level
        ]
        windows_raw = floor.get("windows") or [
            w
            for w in candidate.get("windows", [])
            if w.get("floor_level", 0) == floor_level
        ]

        canonical_rooms = [_room_to_canonical(r) for r in rooms_raw]

        ext_walls = _extract_perimeter_walls(
            outline, floor_level, self.default_wall_height, self.exterior_thickness
        )
        int_walls = _extract_interior_walls(
            rooms_raw, floor_level, self.default_wall_height, self.interior_thickness
        )
        all_walls = ext_walls + int_walls

        warnings = _project_openings_onto_walls(all_walls, doors_raw, windows_raw)

        return AdaptedFloorplan(
            walls=all_walls,
            outline=outline,
            rooms=canonical_rooms,
            source_type="generated",
            warnings=warnings,
        )

    def adapt_candidate(
        self,
        candidate: dict[str, Any],
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        """Convert an entire multi-floor structural candidate into a verified canonical document."""
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be a dictionary.")

        width = float(candidate.get("width", 30.0))
        height = float(candidate.get("height", 40.0))
        boundary = candidate.get("boundary")
        outline = _boundary_to_outline(boundary, width, height)

        floor_plans = candidate.get("floor_plans") or []
        if not floor_plans:
            floor_plans = [
                {
                    "level": 0,
                    "name": "Ground Floor",
                    "rooms": candidate.get("rooms", []),
                    "doors": candidate.get("doors", []),
                    "windows": candidate.get("windows", []),
                }
            ]

        floor_payloads: list[dict[str, Any]] = []

        for floor_idx, floor in enumerate(floor_plans):
            level = int(floor.get("level", floor_idx))
            floor_id = f"floor-{level}" if level > 0 else "ground-floor"

            rooms_raw = floor.get("rooms") or [
                r
                for r in candidate.get("rooms", [])
                if r.get("floor_level", 0) == level
            ]
            doors_raw = floor.get("doors") or [
                d
                for d in candidate.get("doors", [])
                if d.get("floor_level", 0) == level
            ]
            windows_raw = floor.get("windows") or [
                w
                for w in candidate.get("windows", [])
                if w.get("floor_level", 0) == level
            ]

            canonical_rooms = [_room_to_canonical(r) for r in rooms_raw]

            ext_walls = _extract_perimeter_walls(
                outline, level, self.default_wall_height, self.exterior_thickness
            )
            int_walls = _extract_interior_walls(
                rooms_raw, level, self.default_wall_height, self.interior_thickness
            )
            all_walls = ext_walls + int_walls

            _project_openings_onto_walls(all_walls, doors_raw, windows_raw)

            topology = process_wall_topology(all_walls, outline, canonical_rooms)

            elevation = round(level * self.default_wall_height, 5)

            floor_payload = {
                "floorId": floor_id,
                "floorIndex": level,
                "level": level,
                "elevation": elevation,
                "height": self.default_wall_height,
                "outline": topology["exteriorOutline"],
                "exteriorOutline": topology["exteriorOutline"],
                "rooms": canonical_rooms,
                "walls": topology["walls"],
                "shellWalls": topology["shellWalls"],
                "slab": {"elevation": -0.16, "thickness": 0.18},
                "roof": {
                    "type": "flat",
                    "elevation": self.default_wall_height,
                    "thickness": 0.22,
                    "parapetHeight": 0.35,
                },
            }
            floor_payloads.append(floor_payload)

        doc_id = str(plan_id or candidate.get("id", "zynora-floor-plan"))
        active_floor_id = (
            floor_payloads[0]["floorId"] if floor_payloads else "ground-floor"
        )

        plan_payload = {
            "id": doc_id,
            "sourceType": "generated",
            "classifierVersion": "v5",
            "activeFloorId": active_floor_id,
            "floors": floor_payloads,
        }

        document = create_floor_plan_document(plan_payload)

        doc_model = FloorPlanDocument.model_validate(document)
        val_result = validate_floor_plan_document(doc_model)
        document["validation"] = val_result

        return document


def structural_candidate_to_canonical(
    candidate: dict[str, Any],
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Convenience entry point converting a candidate dict directly to canonical document."""
    adapter = StructuralAdapter()
    return adapter.adapt_candidate(candidate=candidate, plan_id=plan_id)
