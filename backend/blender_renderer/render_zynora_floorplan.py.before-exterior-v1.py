from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import bpy
from mathutils import Vector


EPSILON = 1e-5
RENDERER_VERSION = "phase4.1-visual-calibration"


def blender_arguments() -> list[str]:
    """Return only the arguments placed after Blender's ``--`` marker."""
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render five exterior views from zynora.floorplan.v1 JSON."
    )
    parser.add_argument("--input", required=True, help="FloorPlanJSON input file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--engine",
        choices=("eevee", "cycles"),
        default="eevee",
        help="Eevee is the recommended first test on a 4 GB RTX 3050.",
    )
    parser.add_argument(
        "--quality",
        choices=("preview", "final"),
        default="preview",
    )
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument(
        "--style",
        choices=("warm-modern", "graphite-white", "sandstone"),
        default="warm-modern",
    )
    parser.add_argument(
        "--hdri",
        default="",
        help="Optional local .hdr or .exr environment. If omitted, assets/environment.* is detected automatically.",
    )
    return parser.parse_args(blender_arguments())


def finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def hex_to_linear_rgb(value: str) -> tuple[float, float, float]:
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        return (0.8, 0.8, 0.8)

    srgb = tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))

    def convert(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    return tuple(convert(channel) for channel in srgb)


STYLE_PRESETS: dict[str, dict[str, str]] = {
    "warm-modern": {
        "wall": "#f3eee5",
        "wall_secondary": "#d9cbb8",
        "wood": "#ad7047",
        "wood_dark": "#704326",
        "charcoal": "#252c31",
        "glass": "#78aec2",
        "roof": "#dcd8cf",
        "roof_cap": "#f0ece4",
        "concrete": "#b8b4ad",
        "paving": "#9b9994",
        "road": "#343a40",
        "grass": "#527548",
        "leaf": "#326238",
    },
    "graphite-white": {
        "wall": "#ecebe7",
        "wall_secondary": "#c8c9c7",
        "wood": "#9a6844",
        "wood_dark": "#62412d",
        "charcoal": "#1d2429",
        "glass": "#87b7ca",
        "roof": "#d4d4d0",
        "roof_cap": "#f3f2ee",
        "concrete": "#b4b6b6",
        "paving": "#989b9c",
        "road": "#30363d",
        "grass": "#58774e",
        "leaf": "#3e6841",
    },
    "sandstone": {
        "wall": "#e4d8c6",
        "wall_secondary": "#c9b89f",
        "wood": "#87583b",
        "wood_dark": "#5a3828",
        "charcoal": "#292d2f",
        "glass": "#8bb9c6",
        "roof": "#d5c9b8",
        "roof_cap": "#eee5d8",
        "concrete": "#b9ad9d",
        "paving": "#a99d8d",
        "road": "#3a3d3f",
        "grass": "#687d4d",
        "leaf": "#49633b",
    },
}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material(
    name: str,
    color: str,
    *,
    metallic: float = 0.0,
    roughness: float = 0.6,
    alpha: float = 1.0,
    transmission: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    rgb = hex_to_linear_rgb(color)
    material.diffuse_color = (*rgb, alpha)

    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*rgb, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness

    alpha_input = shader.inputs.get("Alpha")
    if alpha_input is not None:
        alpha_input.default_value = alpha

    transmission_input = shader.inputs.get("Transmission Weight") or shader.inputs.get("Transmission")
    if transmission_input is not None:
        transmission_input.default_value = transmission

    if transmission > 0.0:
        coat_input = shader.inputs.get("Coat Weight")
        coat_roughness = shader.inputs.get("Coat Roughness")
        if coat_input is not None:
            coat_input.default_value = 0.24
        if coat_roughness is not None:
            coat_roughness.default_value = 0.10

    ior_input = shader.inputs.get("IOR")
    if ior_input is not None:
        ior_input.default_value = 1.46

    if alpha < 1.0:
        try:
            material.surface_render_method = "DITHERED"
        except (AttributeError, TypeError, ValueError):
            pass

    return material


def make_emissive_material(
    name: str,
    color: str,
    strength: float,
) -> bpy.types.Material:
    """Create a restrained warm emitter for exterior architectural lights."""
    material = make_material(name, color, roughness=0.34)
    shader = material.node_tree.nodes.get("Principled BSDF")
    if not shader:
        return material
    rgb = hex_to_linear_rgb(color)
    set_node_input(shader, "Emission Color", (*rgb, 1.0))
    set_node_input(shader, "Emission", (*rgb, 1.0))
    set_node_input(shader, "Emission Strength", strength)
    return material


def set_node_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def add_noise_surface(
    material: bpy.types.Material,
    color: str,
    *,
    scale: float,
    detail: float,
    variation: float,
    bump_strength: float,
    bump_distance: float,
) -> None:
    """Add a lightweight procedural PBR surface with no external texture files."""
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    if not shader:
        return

    coordinate = nodes.new(type="ShaderNodeTexCoord")
    coordinate.name = f"{material.name} coordinates"
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.name = f"{material.name} micro texture"
    set_node_input(noise, "Scale", scale)
    set_node_input(noise, "Detail", detail)
    set_node_input(noise, "Roughness", 0.68)

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.name = f"{material.name} color variation"
    base = hex_to_linear_rgb(color)
    dark = tuple(max(channel * (1.0 - variation), 0.0) for channel in base)
    light = tuple(min(channel * (1.0 + variation), 1.0) for channel in base)
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (*light, 1.0)

    bump = nodes.new(type="ShaderNodeBump")
    bump.name = f"{material.name} micro bump"
    set_node_input(bump, "Strength", bump_strength)
    set_node_input(bump, "Distance", bump_distance)

    links.new(coordinate.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])


def add_wood_surface(material: bpy.types.Material, dark_color: str, light_color: str) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    if not shader:
        return

    coordinate = nodes.new(type="ShaderNodeTexCoord")
    wave = nodes.new(type="ShaderNodeTexWave")
    wave.name = f"{material.name} vertical grain"
    wave.wave_type = "BANDS"
    # Vary across local X so doors and cladding read as restrained vertical
    # timber grain.  The previous Z bands produced distracting horizontal waves.
    wave.bands_direction = "X"
    set_node_input(wave, "Scale", 13.0)
    set_node_input(wave, "Distortion", 1.35)
    set_node_input(wave, "Detail", 2.5)
    set_node_input(wave, "Detail Scale", 2.0)
    set_node_input(wave, "Detail Roughness", 0.55)

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.name = f"{material.name} timber tones"
    ramp.color_ramp.elements[0].position = 0.34
    ramp.color_ramp.elements[0].color = (*hex_to_linear_rgb(dark_color), 1.0)
    ramp.color_ramp.elements[1].position = 0.66
    ramp.color_ramp.elements[1].color = (*hex_to_linear_rgb(light_color), 1.0)

    bump = nodes.new(type="ShaderNodeBump")
    set_node_input(bump, "Strength", 0.09)
    set_node_input(bump, "Distance", 0.018)

    links.new(coordinate.outputs["Generated"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])


def create_materials(style_name: str) -> dict[str, bpy.types.Material]:
    style = STYLE_PRESETS[style_name]
    materials = {
        "wall": make_material("Cream textured plaster", style["wall"], roughness=0.82),
        "wall_secondary": make_material(
            "Secondary textured plaster", style["wall_secondary"], roughness=0.86
        ),
        "wood": make_material("Warm natural wood", style["wood"], roughness=0.52),
        "wood_dark": make_material("Dark natural wood", style["wood_dark"], roughness=0.58),
        "charcoal": make_material(
            "Charcoal frames", style["charcoal"], metallic=0.5, roughness=0.28
        ),
        "glass": make_material(
            "Architectural glass",
            style["glass"],
            roughness=0.075,
            alpha=0.66,
            transmission=0.52,
        ),
        "roof": make_material("Roof", style["roof"], roughness=0.78),
        "roof_cap": make_material("Roof cap", style["roof_cap"], roughness=0.72),
        "concrete": make_material("Concrete", style["concrete"], roughness=0.86),
        "paving": make_material("Neutral paving", style["paving"], roughness=0.9),
        "road": make_material("Road", style["road"], roughness=0.96),
        "grass": make_material("Grass", style["grass"], roughness=1.0),
        "leaf": make_material("Shrub leaves", style["leaf"], roughness=0.94),
        "leaf_light": make_material("Sunlit leaves", "#416d35", roughness=0.92),
        "trunk": make_material("Tree bark", "#684632", roughness=0.92),
        "curb": make_material("Pale stone curb", "#d7d2c8", roughness=0.86),
        "road_line": make_material("Road markings", "#eee8d8", roughness=0.72),
        "accent_band": make_material("Warm mineral facade band", "#a4603c", roughness=0.76),
        "interior_dark": make_material("Window interior depth", "#151c1f", roughness=0.88),
        "curtain": make_material(
            "Warm translucent curtains",
            "#e9dfcf",
            roughness=0.82,
            alpha=0.76,
            transmission=0.12,
        ),
        "hardware": make_material(
            "Brushed dark hardware", "#343638", metallic=0.72, roughness=0.24
        ),
        "soil": make_material("Landscape soil", "#49392c", roughness=0.98),
        "warm_light": make_emissive_material("Warm exterior light", "#ffd19a", 5.0),
    }

    add_noise_surface(
        materials["wall"],
        style["wall"],
        scale=34.0,
        detail=4.0,
        variation=0.055,
        bump_strength=0.12,
        bump_distance=0.018,
    )
    add_noise_surface(
        materials["wall_secondary"],
        style["wall_secondary"],
        scale=28.0,
        detail=4.0,
        variation=0.075,
        bump_strength=0.16,
        bump_distance=0.024,
    )
    add_wood_surface(materials["wood"], "#55331f", "#ba8054")
    add_wood_surface(materials["wood_dark"], "#321d13", "#7a4b30")
    add_noise_surface(
        materials["concrete"],
        style["concrete"],
        scale=19.0,
        detail=5.0,
        variation=0.08,
        bump_strength=0.18,
        bump_distance=0.035,
    )
    add_noise_surface(
        materials["paving"],
        style["paving"],
        scale=24.0,
        detail=3.0,
        variation=0.10,
        bump_strength=0.10,
        bump_distance=0.018,
    )
    add_noise_surface(
        materials["road"],
        style["road"],
        scale=46.0,
        detail=3.0,
        variation=0.12,
        bump_strength=0.18,
        bump_distance=0.022,
    )
    add_noise_surface(
        materials["grass"],
        style["grass"],
        scale=18.0,
        detail=4.0,
        variation=0.22,
        bump_strength=0.22,
        bump_distance=0.025,
    )
    add_noise_surface(
        materials["accent_band"],
        "#a4603c",
        scale=22.0,
        detail=3.0,
        variation=0.065,
        bump_strength=0.10,
        bump_distance=0.016,
    )
    add_noise_surface(
        materials["leaf"],
        style["leaf"],
        scale=7.0,
        detail=3.0,
        variation=0.18,
        bump_strength=0.12,
        bump_distance=0.018,
    )
    add_noise_surface(
        materials["soil"],
        "#49392c",
        scale=21.0,
        detail=4.0,
        variation=0.20,
        bump_strength=0.24,
        bump_distance=0.026,
    )
    return materials


def add_box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation_z: float = 0.0,
    bevel: float = 0.012,
) -> bpy.types.Object:
    clean_size = tuple(max(abs(value), 0.001) for value in size)
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=(0.0, 0.0, rotation_z))
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(value / 2.0 for value in clean_size)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)

    if bevel > 0:
        modifier = obj.modifiers.new(name="Small edge bevel", type="BEVEL")
        modifier.width = min(bevel, min(clean_size) * 0.18)
        modifier.segments = 2

    return obj


def unit_scale(document: dict[str, Any]) -> float:
    unit = str(document.get("unit", "m")).strip().lower()
    return {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "cm": 0.01,
        "mm": 0.001,
    }.get(unit, 1.0)


def point(value: dict[str, Any], scale: float) -> tuple[float, float]:
    return (finite(value.get("x")) * scale, finite(value.get("z", value.get("y"))) * scale)


def clean_outline(values: Iterable[dict[str, Any]], scale: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for value in values or []:
        candidate = point(value, scale)
        if not result or math.dist(result[-1], candidate) > EPSILON:
            result.append(candidate)
    if len(result) > 2 and math.dist(result[0], result[-1]) <= EPSILON:
        result.pop()
    return result


def polygon_area(outline: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        outline[index][0] * outline[(index + 1) % len(outline)][1]
        - outline[(index + 1) % len(outline)][0] * outline[index][1]
        for index in range(len(outline))
    )


def triangle_cross(
    left: tuple[float, float],
    middle: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return (
        (middle[0] - left[0]) * (right[1] - middle[1])
        - (middle[1] - left[1]) * (right[0] - middle[0])
    )


def inside_triangle(
    point_value: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    def sign(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point_value, a, b)
    d2 = sign(point_value, b, c)
    d3 = sign(point_value, c, a)
    return not ((d1 < -EPSILON or d2 < -EPSILON or d3 < -EPSILON) and (d1 > EPSILON or d2 > EPSILON or d3 > EPSILON))


def triangulate(outline: Sequence[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation for a simple concave floor outline."""
    if len(outline) < 3:
        return []

    indices = list(range(len(outline)))
    if polygon_area(outline) < 0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = len(indices) * len(indices) * 2

    while len(indices) > 3 and guard > 0:
        guard -= 1
        ear_found = False
        for position, middle_index in enumerate(indices):
            left_index = indices[position - 1]
            right_index = indices[(position + 1) % len(indices)]
            a, b, c = outline[left_index], outline[middle_index], outline[right_index]
            if triangle_cross(a, b, c) <= EPSILON:
                continue
            if any(
                inside_triangle(outline[index], a, b, c)
                for index in indices
                if index not in (left_index, middle_index, right_index)
            ):
                continue
            triangles.append((left_index, middle_index, right_index))
            del indices[position]
            ear_found = True
            break
        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append(tuple(indices))

    if not triangles:
        triangles = [(0, index, index + 1) for index in range(1, len(outline) - 1)]
    return triangles


def add_prism(
    name: str,
    outline: Sequence[tuple[float, float]],
    bottom: float,
    top: float,
    material: bpy.types.Material,
) -> bpy.types.Object | None:
    if len(outline) < 3 or top - bottom < EPSILON:
        return None

    triangles = triangulate(outline)
    count = len(outline)
    vertices = [(x, y, bottom) for x, y in outline] + [(x, y, top) for x, y in outline]
    faces: list[tuple[int, ...]] = []

    for a, b, c in triangles:
        faces.append((c, b, a))
        faces.append((a + count, b + count, c + count))

    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, next_index + count, index + count))

    mesh = bpy.data.meshes.new(f"{name}-mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    bevel = obj.modifiers.new(name="Slab edge bevel", type="BEVEL")
    bevel.width = min(0.018, (top - bottom) * 0.12)
    bevel.segments = 2
    return obj


def wall_data(wall: dict[str, Any], scale: float) -> dict[str, Any] | None:
    start_value = wall.get("start") or wall.get("from")
    end_value = wall.get("end") or wall.get("to")
    if not isinstance(start_value, dict) or not isinstance(end_value, dict):
        return None
    x1, y1 = point(start_value, scale)
    x2, y2 = point(end_value, scale)
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 0.02:
        return None
    return {
        "source": wall,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length": length,
        "tx": (x2 - x1) / length,
        "ty": (y2 - y1) / length,
        "angle": math.atan2(y2 - y1, x2 - x1),
        "thickness": max(finite(wall.get("thickness"), 0.2) * scale, 0.05),
    }


def point_in_polygon(candidate: tuple[float, float], outline: Sequence[tuple[float, float]]) -> bool:
    x, y = candidate
    inside = False
    for index in range(len(outline)):
        x1, y1 = outline[index]
        x2, y2 = outline[(index + 1) % len(outline)]
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-20) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def outward_normal(wall: dict[str, Any], outline: Sequence[tuple[float, float]]) -> tuple[float, float]:
    left = (-wall["ty"], wall["tx"])
    right = (wall["ty"], -wall["tx"])
    middle = ((wall["x1"] + wall["x2"]) / 2, (wall["y1"] + wall["y2"]) / 2)
    probe = max(wall["thickness"] * 1.8, 0.28)
    left_inside = point_in_polygon((middle[0] + left[0] * probe, middle[1] + left[1] * probe), outline)
    right_inside = point_in_polygon((middle[0] + right[0] * probe, middle[1] + right[1] * probe), outline)
    if left_inside and not right_inside:
        return right
    if right_inside and not left_inside:
        return left

    center_x = sum(value[0] for value in outline) / max(len(outline), 1)
    center_y = sum(value[1] for value in outline) / max(len(outline), 1)
    radial = (middle[0] - center_x, middle[1] - center_y)
    return left if left[0] * radial[0] + left[1] * radial[1] >= 0 else right


def opening_values(
    opening: dict[str, Any],
    wall_length: float,
    wall_height: float,
    scale: float,
) -> dict[str, Any]:
    kind = "window" if "window" in str(opening.get("type", "")).lower() else "door"
    width = max(finite(opening.get("width"), 1.35 if kind == "window" else 0.9) * scale, 0.18)
    width = min(width, max(wall_length - 0.08, 0.18))
    center = finite(opening.get("offset", opening.get("center")), wall_length / (2 * scale)) * scale
    center = min(max(center, width / 2), wall_length - width / 2)
    bottom = max(finite(opening.get("bottom"), 0.9 if kind == "window" else 0.0) * scale, 0.0)
    height = max(finite(opening.get("height"), 1.15 if kind == "window" else 2.1) * scale, 0.2)
    height = min(height, max(wall_height - bottom - 0.05, 0.2))
    return {
        "source": opening,
        "type": kind,
        "width": width,
        "center": center,
        "start": center - width / 2,
        "end": center + width / 2,
        "bottom": bottom,
        "top": bottom + height,
        "height": height,
    }


def add_wall_box(
    name: str,
    wall: dict[str, Any],
    distance: float,
    vertical_center: float,
    width: float,
    height: float,
    depth: float,
    base_elevation: float,
    material: bpy.types.Material,
    *,
    normal_offset: float = 0.0,
    normal: tuple[float, float] | None = None,
    bevel: float = 0.01,
) -> bpy.types.Object:
    nx, ny = normal or (-wall["ty"], wall["tx"])
    x = wall["x1"] + wall["tx"] * distance + nx * normal_offset
    y = wall["y1"] + wall["ty"] * distance + ny * normal_offset
    return add_box(
        name,
        (width, depth, height),
        (x, y, base_elevation + vertical_center),
        material,
        rotation_z=wall["angle"],
        bevel=bevel,
    )


def merge_intervals(values: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted((min(a, b), max(a, b)) for a, b in values):
        if not merged or start > merged[-1][1] + EPSILON:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(value[0], value[1]) for value in merged]


def add_opening_architecture(
    name: str,
    wall: dict[str, Any],
    opening: dict[str, Any],
    elevation: float,
    outline: Sequence[tuple[float, float]],
    materials: dict[str, bpy.types.Material],
    main_entrance: bool,
) -> None:
    normal = outward_normal(wall, outline)
    outside = wall["thickness"] / 2 + 0.018
    frame_width = min(0.075, opening["width"] * 0.08)
    frame_depth = 0.075
    panel_depth = 0.04

    if opening["type"] == "window":
        # A dark inner plane and softly coloured curtains stop exported windows
        # from reading as flat cyan stickers while preserving the opening size.
        add_wall_box(
            f"{name}-interior-depth",
            wall,
            opening["center"],
            opening["bottom"] + opening["height"] / 2,
            max(opening["width"] - frame_width * 1.2, 0.05),
            max(opening["height"] - frame_width * 1.2, 0.05),
            0.025,
            elevation,
            materials["interior_dark"],
            normal_offset=-wall["thickness"] / 2 - 0.065,
            normal=normal,
            bevel=0.002,
        )
        curtain_width = max(opening["width"] * 0.17, 0.08)
        for curtain_index, curtain_center in enumerate(
            (
                opening["start"] + curtain_width * 0.66,
                opening["end"] - curtain_width * 0.66,
            )
        ):
            add_wall_box(
                f"{name}-curtain-{curtain_index}",
                wall,
                curtain_center,
                opening["bottom"] + opening["height"] / 2,
                curtain_width,
                max(opening["height"] - 0.11, 0.08),
                0.018,
                elevation,
                materials["curtain"],
                normal_offset=-wall["thickness"] / 2 - 0.045,
                normal=normal,
                bevel=0.002,
            )
        add_wall_box(
            f"{name}-glass",
            wall,
            opening["center"],
            opening["bottom"] + opening["height"] / 2,
            max(opening["width"] - frame_width * 2, 0.05),
            max(opening["height"] - frame_width * 2, 0.05),
            panel_depth,
            elevation,
            materials["glass"],
            normal_offset=outside,
            normal=normal,
            bevel=0.004,
        )
        pieces = [
            (opening["start"] + frame_width / 2, opening["bottom"] + opening["height"] / 2, frame_width, opening["height"]),
            (opening["end"] - frame_width / 2, opening["bottom"] + opening["height"] / 2, frame_width, opening["height"]),
            (opening["center"], opening["bottom"] + frame_width / 2, opening["width"], frame_width),
            (opening["center"], opening["top"] - frame_width / 2, opening["width"], frame_width),
            (opening["center"], opening["bottom"] + opening["height"] / 2, frame_width * 0.72, opening["height"]),
        ]
        for index, (distance, center_z, width, height) in enumerate(pieces):
            add_wall_box(
                f"{name}-frame-{index}",
                wall,
                distance,
                center_z,
                width,
                height,
                frame_depth,
                elevation,
                materials["charcoal"],
                normal_offset=outside + 0.018,
                normal=normal,
                bevel=0.006,
            )
        add_wall_box(
            f"{name}-outer-sill",
            wall,
            opening["center"],
            max(opening["bottom"] - 0.035, 0.03),
            opening["width"] + 0.15,
            0.055,
            0.16,
            elevation,
            materials["roof_cap"],
            normal_offset=wall["thickness"] / 2 + 0.078,
            normal=normal,
            bevel=0.008,
        )
        return

    add_wall_box(
        f"{name}-door",
        wall,
        opening["center"],
        opening["bottom"] + opening["height"] / 2,
        max(opening["width"] - frame_width * 2, 0.12),
        max(opening["height"] - frame_width, 0.3),
        panel_depth,
        elevation,
        materials["wood" if main_entrance else "wood_dark"],
        normal_offset=outside,
        normal=normal,
        bevel=0.01,
    )
    pieces = [
        (opening["start"] + frame_width / 2, opening["bottom"] + opening["height"] / 2, frame_width, opening["height"]),
        (opening["end"] - frame_width / 2, opening["bottom"] + opening["height"] / 2, frame_width, opening["height"]),
        (opening["center"], opening["top"] - frame_width / 2, opening["width"], frame_width),
    ]
    for index, (distance, center_z, width, height) in enumerate(pieces):
        add_wall_box(
            f"{name}-frame-{index}",
            wall,
            distance,
            center_z,
            width,
            height,
            frame_depth,
            elevation,
            materials["charcoal"],
            normal_offset=outside + 0.02,
            normal=normal,
            bevel=0.006,
        )

    # Brushed handle and a small warm wall light give the entrance scale and
    # depth without altering the exported door opening.
    handle_side = min(opening["end"] - frame_width * 1.8, opening["center"] + opening["width"] * 0.26)
    add_wall_box(
        f"{name}-handle",
        wall,
        handle_side,
        opening["bottom"] + opening["height"] * 0.48,
        0.035,
        0.27,
        0.045,
        elevation,
        materials["hardware"],
        normal_offset=outside + 0.052,
        normal=normal,
        bevel=0.008,
    )

    if main_entrance:
        canopy_depth = 0.95
        add_wall_box(
            f"{name}-canopy",
            wall,
            opening["center"],
            min(opening["top"] + 0.24, 2.65),
            opening["width"] + 0.72,
            0.13,
            canopy_depth,
            elevation,
            materials["roof_cap"],
            normal_offset=wall["thickness"] / 2 + canopy_depth / 2,
            normal=normal,
            bevel=0.018,
        )
        for step_index in range(3):
            step_depth = 0.28 * (step_index + 1)
            step_height = 0.055 * (3 - step_index)
            add_wall_box(
                f"{name}-step-{step_index}",
                wall,
                opening["center"],
                step_height / 2 - 0.01,
                opening["width"] + 0.5 + step_index * 0.12,
                step_height,
                step_depth,
                elevation,
                materials["concrete"],
                normal_offset=wall["thickness"] / 2 + step_depth / 2,
                normal=normal,
                bevel=0.01,
            )
        lamp_distance = max(opening["start"] - 0.24, 0.10)
        add_wall_box(
            f"{name}-warm-sconce",
            wall,
            lamp_distance,
            min(opening["top"] * 0.66, 1.55),
            0.09,
            0.20,
            0.07,
            elevation,
            materials["warm_light"],
            normal_offset=outside + 0.07,
            normal=normal,
            bevel=0.016,
        )


def add_wall(
    wall_source: dict[str, Any],
    floor: dict[str, Any],
    floor_index: int,
    wall_index: int,
    outline: Sequence[tuple[float, float]],
    scale: float,
    materials: dict[str, bpy.types.Material],
    entrance_key: tuple[int, int, int] | None,
) -> None:
    wall = wall_data(wall_source, scale)
    if not wall:
        return
    elevation = finite(floor.get("elevation")) * scale
    wall_height = max(finite(wall_source.get("height"), finite(floor.get("height"), 2.8)) * scale, 0.4)
    openings = [
        opening_values(value, wall["length"], wall_height, scale)
        for value in (wall_source.get("openings") or [])
        if isinstance(value, dict)
    ]
    openings.sort(key=lambda value: value["start"])

    cursor = 0.0
    for opening_index, opening in enumerate(openings):
        start = max(cursor, opening["start"])
        if start - cursor > 0.015:
            add_wall_box(
                f"floor-{floor_index}-wall-{wall_index}-left-{opening_index}",
                wall,
                (cursor + start) / 2,
                wall_height / 2,
                start - cursor,
                wall_height,
                wall["thickness"],
                elevation,
                materials["wall" if wall_index % 5 else "wall_secondary"],
            )

        if opening["bottom"] > 0.015:
            add_wall_box(
                f"floor-{floor_index}-wall-{wall_index}-sill-{opening_index}",
                wall,
                opening["center"],
                opening["bottom"] / 2,
                opening["width"],
                opening["bottom"],
                wall["thickness"],
                elevation,
                materials["wall"],
            )
        if wall_height - opening["top"] > 0.015:
            add_wall_box(
                f"floor-{floor_index}-wall-{wall_index}-head-{opening_index}",
                wall,
                opening["center"],
                (opening["top"] + wall_height) / 2,
                opening["width"],
                wall_height - opening["top"],
                wall["thickness"],
                elevation,
                materials["wall"],
            )

        add_opening_architecture(
            f"floor-{floor_index}-wall-{wall_index}-opening-{opening_index}",
            wall,
            opening,
            elevation,
            outline,
            materials,
            entrance_key == (floor_index, wall_index, opening_index),
        )
        cursor = max(cursor, opening["end"])

    if wall["length"] - cursor > 0.015:
        add_wall_box(
            f"floor-{floor_index}-wall-{wall_index}-right",
            wall,
            (cursor + wall["length"]) / 2,
            wall_height / 2,
            wall["length"] - cursor,
            wall_height,
            wall["thickness"],
            elevation,
            materials["wall" if wall_index % 5 else "wall_secondary"],
        )


def exterior_walls(floor: dict[str, Any]) -> list[dict[str, Any]]:
    values = floor.get("exteriorWalls") or floor.get("shellWalls")
    if not values:
        values = [wall for wall in (floor.get("walls") or []) if wall.get("isExterior", True)]
    return [value for value in values if isinstance(value, dict)]


def opening_score(wall: dict[str, Any], opening: dict[str, Any], scale: float) -> float:
    wall_value = wall_data(wall, scale)
    if not wall_value:
        return -math.inf
    label = str(opening.get("name", opening.get("label", opening.get("type", "")))).lower()
    width = finite(opening.get("width"), 0.9) * scale
    explicit = 20.0 if any(value in label for value in ("main", "front", "entrance", "entry")) else 0.0
    garage = -12.0 if any(value in label for value in ("garage", "vehicle", "shutter")) or width > 2.25 else 0.0
    window_count = sum(
        "window" in str(value.get("type", "")).lower() for value in (wall.get("openings") or [])
    )
    return explicit + garage + wall_value["length"] + width * 5.0 + window_count * 1.4


def find_main_entrance(
    floors: Sequence[dict[str, Any]], scale: float
) -> tuple[tuple[int, int, int] | None, dict[str, Any] | None]:
    if not floors:
        return None, None
    best: tuple[float, tuple[int, int, int], dict[str, Any]] | None = None
    for wall_index, wall in enumerate(exterior_walls(floors[0])):
        for opening_index, opening in enumerate(wall.get("openings") or []):
            if "door" not in str(opening.get("type", "door")).lower():
                continue
            score = opening_score(wall, opening, scale)
            if best is None or score > best[0]:
                best = (score, (0, wall_index, opening_index), wall)
    if best:
        return best[1], best[2]

    walls = exterior_walls(floors[0])
    longest = max(walls, key=lambda value: (wall_data(value, scale) or {}).get("length", 0.0), default=None)
    return None, longest


def add_roof_and_parapet(
    floor: dict[str, Any],
    scale: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    roof = floor.get("roof") or {}
    outline = clean_outline(roof.get("outline") or floor.get("outline") or [], scale)
    if len(outline) < 3:
        return
    elevation = finite(roof.get("elevation"), finite(floor.get("elevation")) + finite(floor.get("height"), 2.8)) * scale
    thickness = max(finite(roof.get("thickness"), 0.22) * scale, 0.08)
    add_prism("top-roof", outline, elevation, elevation + thickness, materials["roof"])

    parapet_height = max(finite(roof.get("parapetHeight"), 0.35) * scale, 0.18)
    parapet_thickness = 0.15
    for index in range(len(outline)):
        x1, y1 = outline[index]
        x2, y2 = outline[(index + 1) % len(outline)]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 0.06:
            continue
        add_box(
            f"parapet-{index}",
            (length + parapet_thickness, parapet_thickness, parapet_height),
            ((x1 + x2) / 2, (y1 + y2) / 2, elevation + thickness + parapet_height / 2),
            materials["roof_cap"],
            rotation_z=math.atan2(y2 - y1, x2 - x1),
            bevel=0.01,
        )


def add_balcony(
    floors: Sequence[dict[str, Any]],
    front_wall_source: dict[str, Any] | None,
    scale: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    if len(floors) < 2 or not front_wall_source:
        return
    ground_outline = clean_outline(floors[0].get("outline") or [], scale)
    front_wall = wall_data(front_wall_source, scale)
    if not front_wall:
        return
    front_normal = outward_normal(front_wall, ground_outline)

    upper_walls = exterior_walls(floors[1])
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for source in upper_walls:
        value = wall_data(source, scale)
        if not value:
            continue
        normal = outward_normal(value, clean_outline(floors[1].get("outline") or [], scale))
        alignment = normal[0] * front_normal[0] + normal[1] * front_normal[1]
        opening_bonus = len(source.get("openings") or []) * 0.65
        candidates.append((alignment * 10 + value["length"] + opening_bonus, source, value))
    if not candidates:
        return

    _, source, wall = max(candidates, key=lambda value: value[0])

    # Keep the balcony as a deliberate entrance feature instead of stretching it
    # across most of a long facade.  If the upper floor has a door, centre the
    # balcony on that opening so the result remains useful for irregular plans.
    upper_height = finite(floors[1].get("height"), 2.8) * scale
    upper_openings = [
        opening_values(value, wall["length"], upper_height, scale)
        for value in (source.get("openings") or [])
    ]
    balcony_doors = [value for value in upper_openings if value["type"] == "door"]
    if balcony_doors:
        anchor = min(
            balcony_doors,
            key=lambda value: abs(value["center"] - wall["length"] / 2),
        )
        center = anchor["center"]
    else:
        center = wall["length"] / 2

    maximum_width = min(6.2, wall["length"] - 0.6)
    balcony_width = min(max(wall["length"] * 0.34, 3.6), maximum_width)
    if balcony_width < 1.4:
        return
    edge_clearance = 0.25
    center = min(
        max(center, balcony_width / 2 + edge_clearance),
        wall["length"] - balcony_width / 2 - edge_clearance,
    )
    normal = outward_normal(wall, clean_outline(floors[1].get("outline") or [], scale))
    base = finite(floors[1].get("elevation")) * scale
    depth = 1.15

    add_wall_box(
        "balcony-slab",
        wall,
        center,
        0.0,
        balcony_width,
        0.16,
        depth,
        base,
        materials["concrete"],
        normal_offset=wall["thickness"] / 2 + depth / 2,
        normal=normal,
        bevel=0.016,
    )

    rail_distance = wall["thickness"] / 2 + depth - 0.06
    add_wall_box(
        "balcony-glass-front",
        wall,
        center,
        0.55,
        balcony_width,
        0.94,
        0.035,
        base,
        materials["glass"],
        normal_offset=rail_distance,
        normal=normal,
        bevel=0.004,
    )
    add_wall_box(
        "balcony-top-rail",
        wall,
        center,
        1.04,
        balcony_width + 0.08,
        0.06,
        0.07,
        base,
        materials["charcoal"],
        normal_offset=rail_distance + 0.018,
        normal=normal,
        bevel=0.008,
    )
    post_count = max(4, int(balcony_width / 1.0) + 1)
    for index in range(post_count):
        distance = center - balcony_width / 2 + balcony_width * index / (post_count - 1)
        add_wall_box(
            f"balcony-post-{index}",
            wall,
            distance,
            0.54,
            0.055,
            1.02,
            0.065,
            base,
            materials["charcoal"],
            normal_offset=rail_distance + 0.02,
            normal=normal,
            bevel=0.006,
        )


def add_facade_fins(
    floors: Sequence[dict[str, Any]],
    front_wall_source: dict[str, Any] | None,
    scale: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    if not floors or not front_wall_source:
        return
    wall = wall_data(front_wall_source, scale)
    if not wall:
        return
    outline = clean_outline(floors[0].get("outline") or [], scale)
    normal = outward_normal(wall, outline)
    openings = [opening_values(value, wall["length"], finite(floors[0].get("height"), 2.8) * scale, scale) for value in (front_wall_source.get("openings") or [])]
    blocked = merge_intervals([(max(0, value["start"] - 0.16), min(wall["length"], value["end"] + 0.16)) for value in openings])
    gaps: list[tuple[float, float]] = []
    cursor = 0.18
    for start, end in blocked:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < wall["length"] - 0.18:
        gaps.append((cursor, wall["length"] - 0.18))
    if not gaps:
        return
    start, end = max(gaps, key=lambda value: value[1] - value[0])
    width = min(end - start - 0.04, 1.05)
    if width < 0.42:
        return
    center = (start + end) / 2
    height = min(finite(floors[0].get("height"), 2.8) * scale - 0.35, 2.45)
    slat_count = max(4, int(width / 0.12))
    for index in range(slat_count):
        distance = center - width / 2 + width * (index + 0.5) / slat_count
        add_wall_box(
            f"front-wood-fin-{index}",
            wall,
            distance,
            0.22 + height / 2,
            width / slat_count * 0.62,
            height,
            0.055,
            finite(floors[0].get("elevation")) * scale,
            materials["wood" if index % 2 else "wood_dark"],
            normal_offset=wall["thickness"] / 2 + 0.052,
            normal=normal,
            bevel=0.007,
        )


def add_facade_band(
    floors: Sequence[dict[str, Any]],
    front_wall_source: dict[str, Any] | None,
    scale: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    """Add the warm horizontal datum used by the approved exterior direction."""
    if len(floors) < 2 or not front_wall_source:
        return
    wall = wall_data(front_wall_source, scale)
    if not wall or wall["length"] < 1.0:
        return
    outline = clean_outline(floors[0].get("outline") or [], scale)
    normal = outward_normal(wall, outline)
    ground_elevation = finite(floors[0].get("elevation")) * scale
    upper_elevation = finite(floors[1].get("elevation")) * scale
    band_height = 0.24
    add_wall_box(
        "front-warm-horizontal-band",
        wall,
        wall["length"] / 2,
        upper_elevation - ground_elevation - band_height / 2 + 0.04,
        max(wall["length"] - 0.12, 0.5),
        band_height,
        0.065,
        ground_elevation,
        materials["accent_band"],
        normal_offset=wall["thickness"] / 2 + 0.045,
        normal=normal,
        bevel=0.012,
    )


def bounds_from_floors(floors: Sequence[dict[str, Any]], scale: float) -> dict[str, float]:
    points = [
        value
        for floor in floors
        for value in clean_outline(floor.get("outline") or [], scale)
    ]
    if not points:
        points = [(-4.0, -4.0), (4.0, 4.0)]
    min_x = min(value[0] for value in points)
    max_x = max(value[0] for value in points)
    min_y = min(value[1] for value in points)
    max_y = max(value[1] for value in points)
    top = max(
        (finite(floor.get("elevation")) + finite(floor.get("height"), 2.8)) * scale
        for floor in floors
    )
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center_x": (min_x + max_x) / 2,
        "center_y": (min_y + max_y) / 2,
        "width": max_x - min_x,
        "depth": max_y - min_y,
        "height": top,
    }


def add_shrub(
    name: str,
    location: tuple[float, float, float],
    materials: dict[str, bpy.types.Material],
) -> None:
    """Build a compact layered shrub instead of a single low-poly sphere."""
    add_box(
        f"{name}-soil",
        (0.92, 0.70, 0.035),
        (location[0], location[1], 0.006),
        materials["soil"],
        bevel=0.16,
    )
    clusters = (
        (-0.24, -0.03, 0.18, 0.27),
        (0.22, -0.04, 0.19, 0.28),
        (-0.10, 0.15, 0.24, 0.30),
        (0.11, 0.13, 0.25, 0.29),
        (0.00, -0.13, 0.29, 0.31),
        (-0.02, 0.02, 0.38, 0.32),
    )
    for index, (offset_x, offset_y, offset_z, radius) in enumerate(clusters):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=3,
            radius=radius,
            location=(
                location[0] + offset_x,
                location[1] + offset_y,
                offset_z,
            ),
        )
        shrub = bpy.context.object
        shrub.name = f"{name}-leaf-{index + 1}"
        shrub.scale = (1.08, 0.92, 0.88)
        shrub.data.materials.append(
            materials["leaf_light" if index in (2, 5) else "leaf"]
        )
        for polygon in shrub.data.polygons:
            polygon.use_smooth = True


def add_tree(
    name: str,
    location: tuple[float, float],
    size: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    generator = random.Random(sum(ord(character) for character in name) + int(size * 100))
    trunk_height = 2.9 * size
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=0.16 * size,
        depth=trunk_height,
        location=(location[0], location[1], trunk_height / 2 - 0.01),
    )
    trunk = bpy.context.object
    trunk.name = f"{name}-trunk"
    trunk.data.materials.append(materials["trunk"])

    # Overlapping irregular clusters remain inexpensive but read as a leafy
    # canopy at architectural camera distances.
    crowns: list[tuple[tuple[float, float, float], float]] = []
    for ring_index, (ring_radius, ring_height, count) in enumerate(
        ((0.62, 2.72, 7), (0.48, 3.27, 6), (0.25, 3.72, 4))
    ):
        phase = generator.uniform(0.0, math.tau)
        for item_index in range(count):
            angle = phase + math.tau * item_index / count
            radial = ring_radius * generator.uniform(0.72, 1.12)
            crowns.append(
                (
                    (
                        math.cos(angle) * radial,
                        math.sin(angle) * radial * 0.78,
                        ring_height + generator.uniform(-0.18, 0.18),
                    ),
                    generator.uniform(0.55, 0.78),
                )
            )
    crowns.append(((0.0, 0.0, 3.58), 0.78))
    for index, ((offset_x, offset_y, offset_z), radius) in enumerate(crowns):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=3,
            radius=radius * size,
            location=(
                location[0] + offset_x * size,
                location[1] + offset_y * size,
                offset_z * size,
            ),
        )
        crown = bpy.context.object
        crown.name = f"{name}-crown-{index + 1}"
        crown.scale = (
            generator.uniform(0.86, 1.14),
            generator.uniform(0.78, 1.04),
            generator.uniform(0.82, 1.10),
        )
        crown.data.materials.append(
            materials["leaf_light" if index % 4 == 1 else "leaf"]
        )
        for polygon in crown.data.polygons:
            polygon.use_smooth = True


def add_site(
    bounds: dict[str, float],
    front_wall_source: dict[str, Any] | None,
    ground_outline: Sequence[tuple[float, float]],
    scale: float,
    materials: dict[str, bpy.types.Material],
) -> None:
    span = max(bounds["width"], bounds["depth"], 8.0)
    add_box(
        "landscaped-site",
        (span * 3.1, span * 3.1, 0.12),
        (bounds["center_x"], bounds["center_y"], -0.11),
        materials["grass"],
        bevel=0.0,
    )
    add_box(
        "building-paving",
        (bounds["width"] + 2.2, bounds["depth"] + 2.2, 0.10),
        (bounds["center_x"], bounds["center_y"], -0.035),
        materials["paving"],
        bevel=0.0,
    )

    if not front_wall_source:
        return
    wall = wall_data(front_wall_source, scale)
    if not wall:
        return
    normal = outward_normal(wall, ground_outline)
    tangent = (wall["tx"], wall["ty"])
    middle = ((wall["x1"] + wall["x2"]) / 2, (wall["y1"] + wall["y2"]) / 2)
    road_depth = 4.8
    road_center = (
        middle[0] + normal[0] * (road_depth / 2 + 3.0),
        middle[1] + normal[1] * (road_depth / 2 + 3.0),
    )
    add_box(
        "front-road",
        (span * 2.8, road_depth, 0.08),
        (road_center[0], road_center[1], -0.02),
        materials["road"],
        rotation_z=math.atan2(tangent[1], tangent[0]),
        bevel=0.0,
    )

    sidewalk_center = (
        middle[0] + normal[0] * 2.28,
        middle[1] + normal[1] * 2.28,
    )
    add_box(
        "front-sidewalk",
        (span * 2.8, 1.18, 0.10),
        (sidewalk_center[0], sidewalk_center[1], 0.015),
        materials["curb"],
        rotation_z=math.atan2(tangent[1], tangent[0]),
        bevel=0.012,
    )
    curb_center = (
        middle[0] + normal[0] * 2.94,
        middle[1] + normal[1] * 2.94,
    )
    add_box(
        "front-curb",
        (span * 2.8, 0.18, 0.18),
        (curb_center[0], curb_center[1], 0.07),
        materials["curb"],
        rotation_z=math.atan2(tangent[1], tangent[0]),
        bevel=0.018,
    )

    # A restrained dashed centre line makes the plot read as an architectural
    # streetscape while keeping the foreground lightweight for the RTX 3050.
    road_angle = math.atan2(tangent[1], tangent[0])
    marking_count = max(5, int(span / 1.8))
    for index in range(-marking_count, marking_count + 1):
        along = index * 2.7
        line_x = road_center[0] + tangent[0] * along
        line_y = road_center[1] + tangent[1] * along
        add_box(
            f"road-centre-marking-{index + marking_count}",
            (1.35, 0.085, 0.018),
            (line_x, line_y, 0.032),
            materials["road_line"],
            rotation_z=road_angle,
            bevel=0.004,
        )
    wall_height = 2.8 * scale
    door_values = [
        value
        for value in (
            opening_values(opening, wall["length"], wall_height, scale)
            for opening in (front_wall_source.get("openings") or [])
        )
        if value["type"] == "door"
    ]
    door_centres = [value["center"] for value in door_values]

    # Connect normal-sized entry doors to the public sidewalk. Wide garage or
    # shutter openings deliberately do not receive a pedestrian path.
    pedestrian_doors = [value for value in door_values if value["width"] <= 1.8]
    if pedestrian_doors:
        entry = min(pedestrian_doors, key=lambda value: abs(value["center"] - wall["length"] / 2))
        path_length = 2.05
        path_x = wall["x1"] + tangent[0] * entry["center"] + normal[0] * (path_length / 2 + wall["thickness"] / 2)
        path_y = wall["y1"] + tangent[1] * entry["center"] + normal[1] * (path_length / 2 + wall["thickness"] / 2)
        add_box(
            "front-entry-walkway",
            (max(entry["width"] + 0.50, 1.30), path_length, 0.055),
            (path_x, path_y, 0.035),
            materials["paving"],
            rotation_z=math.atan2(tangent[1], tangent[0]),
            bevel=0.018,
        )

    shrub_distances = (0.09, 0.20, 0.78, 0.91)
    for index, ratio in enumerate(shrub_distances):
        distance = wall["length"] * ratio
        # Never place a decorative shrub in front of an exported doorway.
        if any(abs(distance - door_center) < 1.35 for door_center in door_centres):
            continue
        setback = 1.18 if index % 2 == 0 else 1.48
        shrub_x = wall["x1"] + tangent[0] * distance + normal[0] * setback
        shrub_y = wall["y1"] + tangent[1] * distance + normal[1] * setback
        add_shrub(f"front-shrub-{index + 1}", (shrub_x, shrub_y, 0.0), materials)

    rear_distance = bounds["depth"] * 0.76 + 4.2
    rear_center = (
        middle[0] - normal[0] * rear_distance,
        middle[1] - normal[1] * rear_distance,
    )
    tree_layout = (
        (-span * 0.92, 0.30, 1.22),
        (-span * 0.62, -0.10, 1.34),
        (-span * 0.31, 0.36, 1.18),
        (0.0, -0.55, 1.28),
        (span * 0.32, 0.22, 1.20),
        (span * 0.61, -0.14, 1.32),
        (span * 0.91, 0.38, 1.22),
    )
    for index, (along, setback, size_value) in enumerate(tree_layout):
        tree_x = rear_center[0] + tangent[0] * along - normal[0] * setback
        tree_y = rear_center[1] + tangent[1] * along - normal[1] * setback
        add_tree(f"background-tree-{index + 1}", (tree_x, tree_y), size_value, materials)


def point_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def resolve_hdri_path(args: argparse.Namespace) -> Path | None:
    if args.hdri:
        candidate = Path(args.hdri).expanduser().resolve()
        if candidate.is_file():
            return candidate
        print(f"ZYNORA_WARNING=HDRI file was not found; using procedural sky: {candidate}")

    assets = Path(__file__).resolve().parent / "assets"
    for name in ("environment.exr", "environment.hdr", "zynora-environment.exr", "zynora-environment.hdr"):
        candidate = assets / name
        if candidate.is_file():
            return candidate
    return None


def configure_world_and_lighting(bounds: dict[str, float], args: argparse.Namespace) -> None:
    center = (bounds["center_x"], bounds["center_y"], bounds["height"] * 0.45)
    bpy.ops.object.light_add(
        type="SUN",
        location=(center[0] - 6.0, center[1] - 8.0, bounds["height"] + 10.0),
    )
    sun = bpy.context.object
    sun.name = "Architectural daylight"
    sun.data.energy = 1.48
    sun.data.angle = math.radians(2.8)
    sun.data.color = (1.0, 0.94, 0.86)
    sun.rotation_euler = (math.radians(31), math.radians(-18), math.radians(-38))

    bpy.ops.object.light_add(
        type="AREA",
        location=(center[0] - 7.0, center[1] - 9.0, bounds["height"] + 5.0),
    )
    fill = bpy.context.object
    fill.name = "Soft sky fill"
    fill.data.energy = 82.0
    fill.data.color = (0.68, 0.80, 1.0)
    fill.data.shape = "DISK"
    fill.data.size = max(bounds["width"], bounds["depth"], 8.0)
    point_at(fill, center)

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    hdri_path = resolve_hdri_path(args)
    if hdri_path:
        coordinate = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        environment = nodes.new(type="ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
        mapping.inputs["Rotation"].default_value[2] = math.radians(18.0)
        background.inputs["Strength"].default_value = 0.34
        links.new(coordinate.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        links.new(environment.outputs["Color"], background.inputs["Color"])
        print(f"ZYNORA_HDRI={hdri_path}")
    else:
        sky = nodes.new(type="ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(34.0)
        sky.sun_rotation = math.radians(132.0)
        sky.altitude = 0.10
        if hasattr(sky, "air_density"):
            sky.air_density = 0.82
        if hasattr(sky, "dust_density"):
            sky.dust_density = 0.12
        if hasattr(sky, "ozone_density"):
            sky.ozone_density = 1.10
        background.inputs["Strength"].default_value = 0.19
        links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def configure_renderer(args: argparse.Namespace) -> tuple[int, int, int]:
    scene = bpy.context.scene
    preview = args.quality == "preview"
    width = args.width or (960 if preview else 1600)
    height = args.height or (540 if preview else 900)
    samples = args.samples or (16 if preview else 48)

    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "GPU"
        scene.cycles.max_bounces = 6
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 3
        scene.cycles.transmission_bounces = 4
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
        if hasattr(scene.cycles, "adaptive_threshold"):
            scene.cycles.adaptive_threshold = 0.028 if preview else 0.015
        if hasattr(scene.cycles, "sample_clamp_indirect"):
            scene.cycles.sample_clamp_indirect = 2.2
        if hasattr(scene.cycles, "blur_glossy"):
            scene.cycles.blur_glossy = 0.35
        if hasattr(scene.cycles, "use_light_tree"):
            scene.cycles.use_light_tree = True
        if hasattr(scene.cycles, "use_guiding"):
            scene.cycles.use_guiding = False
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            for compute_type in ("OPTIX", "CUDA"):
                try:
                    preferences.compute_device_type = compute_type
                    preferences.get_devices()
                    break
                except (AttributeError, TypeError):
                    continue
            for device in preferences.devices:
                device.use = device.type in {"OPTIX", "CUDA"}
        except (KeyError, AttributeError):
            print("ZYNORA_WARNING=Cycles GPU selection was unavailable; Blender will choose the device.")
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_samples"):
            scene.eevee.taa_samples = samples
        try:
            scene.render.image_settings.color_depth = "8"
        except (AttributeError, TypeError):
            pass

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 22
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        pass
    scene.view_settings.exposure = -0.30
    scene.view_settings.gamma = 1.0
    return width, height, samples


def create_cameras(
    bounds: dict[str, float],
    front_wall_source: dict[str, Any] | None,
    ground_outline: Sequence[tuple[float, float]],
    scale: float,
) -> list[tuple[str, bpy.types.Object]]:
    if front_wall_source:
        front_wall = wall_data(front_wall_source, scale)
    else:
        front_wall = None

    if front_wall:
        front = outward_normal(front_wall, ground_outline)
        tangent = (front_wall["tx"], front_wall["ty"])
    else:
        front = (0.0, -1.0)
        tangent = (1.0, 0.0)

    diagonal = math.hypot(bounds["width"], bounds["depth"])
    distance = max(
        diagonal * 1.51,
        (front_wall["length"] if front_wall else bounds["width"]) * 1.58,
        bounds["height"] * 3.2,
        16.0,
    )
    overall_target = (
        bounds["center_x"],
        bounds["center_y"],
        max(bounds["height"] * 0.44, 1.5),
    )

    if front_wall:
        front_midpoint = (
            (front_wall["x1"] + front_wall["x2"]) / 2,
            (front_wall["y1"] + front_wall["y2"]) / 2,
        )
        inward_offset = min(max(bounds["depth"] * 0.06, 0.35), 1.1)
        front_target = (
            front_midpoint[0] - front[0] * inward_offset,
            front_midpoint[1] - front[1] * inward_offset,
            max(bounds["height"] * 0.44, 1.5),
        )
    else:
        front_target = overall_target

    camera_height = max(bounds["height"] * 0.64, 3.35)

    directions = [
        ("01-front-hero", 1.00, 0.46, camera_height, front_target, 0.83),
        ("02-front-left", 1.00, -0.52, camera_height * 0.96, front_target, 0.86),
        ("03-front-straight", 1.00, 0.00, camera_height * 0.90, front_target, 0.84),
        ("04-right-side", 0.70, 0.88, camera_height * 0.94, overall_target, 0.94),
        ("05-left-side", 0.70, -0.88, camera_height * 0.94, overall_target, 0.94),
    ]
    cameras: list[tuple[str, bpy.types.Object]] = []

    for name, front_amount, tangent_amount, height, target, distance_multiplier in directions:
        horizontal_x = front[0] * front_amount + tangent[0] * tangent_amount
        horizontal_y = front[1] * front_amount + tangent[1] * tangent_amount
        magnitude = max(math.hypot(horizontal_x, horizontal_y), EPSILON)
        horizontal_x /= magnitude
        horizontal_y /= magnitude
        bpy.ops.object.camera_add(
            location=(
                target[0] + horizontal_x * distance * distance_multiplier,
                target[1] + horizontal_y * distance * distance_multiplier,
                height,
            )
        )
        camera = bpy.context.object
        camera.name = name
        camera.data.lens = 44.0
        camera.data.sensor_width = 36.0
        camera.data.dof.use_dof = False
        point_at(camera, target)
        cameras.append((name, camera))
    return cameras


def build_house(
    document: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, bpy.types.Object]], dict[str, Any]]:
    if document.get("schemaVersion") != "zynora.floorplan.v1":
        raise ValueError(
            "Expected schemaVersion 'zynora.floorplan.v1'. Download FloorPlanJSON from the ZYNORA 3D page."
        )
    floors = sorted(
        [value for value in (document.get("floors") or []) if isinstance(value, dict)],
        key=lambda value: finite(value.get("elevation")),
    )
    if not floors:
        raise ValueError("FloorPlanJSON does not contain any floors.")

    scale = unit_scale(document)
    materials = create_materials(args.style)
    bounds = bounds_from_floors(floors, scale)
    entrance_key, front_wall_source = find_main_entrance(floors, scale)
    ground_outline = clean_outline(floors[0].get("outline") or [], scale)

    add_site(bounds, front_wall_source, ground_outline, scale, materials)

    for floor_index, floor in enumerate(floors):
        outline = clean_outline(floor.get("outline") or [], scale)
        slabs = floor.get("slabs") or []
        if slabs:
            slab = slabs[0]
            slab_outline = clean_outline(slab.get("outline") or floor.get("outline") or [], scale)
            slab_bottom = finite(slab.get("elevation"), finite(floor.get("elevation")) - 0.16) * scale
            slab_thickness = max(finite(slab.get("thickness"), 0.18) * scale, 0.06)
        else:
            slab_outline = outline
            slab_bottom = (finite(floor.get("elevation")) - 0.16) * scale
            slab_thickness = 0.18 * scale
        add_prism(
            f"floor-{floor_index}-slab",
            slab_outline,
            slab_bottom,
            slab_bottom + slab_thickness,
            materials["concrete"],
        )

        for wall_index, wall in enumerate(exterior_walls(floor)):
            add_wall(
                wall,
                floor,
                floor_index,
                wall_index,
                outline,
                scale,
                materials,
                entrance_key,
            )

    add_roof_and_parapet(floors[-1], scale, materials)
    add_facade_band(floors, front_wall_source, scale, materials)
    add_balcony(floors, front_wall_source, scale, materials)
    add_facade_fins(floors, front_wall_source, scale, materials)
    configure_world_and_lighting(bounds, args)
    cameras = create_cameras(bounds, front_wall_source, ground_outline, scale)
    return cameras, {
        "schemaVersion": document.get("schemaVersion"),
        "rendererVersion": RENDERER_VERSION,
        "floorCount": len(floors),
        "wallCount": sum(len(exterior_walls(floor)) for floor in floors),
        "bounds": bounds,
        "style": args.style,
    }


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.input).expanduser().resolve()
    output_directory = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"FloorPlanJSON was not found: {input_path}")
    output_directory.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as handle:
        document = json.load(handle)

    clear_scene()
    cameras, summary = build_house(document, args)
    width, height, samples = configure_renderer(args)
    scene = bpy.context.scene
    scene.render.use_persistent_data = True

    blend_path = output_directory / "zynora-floorplan-five-views.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    images: list[str] = []
    for name, camera in cameras:
        scene.camera = camera
        image_path = output_directory / f"{name}.png"
        scene.render.filepath = str(image_path)
        print(f"ZYNORA_RENDERING={name}")
        bpy.ops.render.render(write_still=True)
        images.append(str(image_path))

    manifest = {
        **summary,
        "input": str(input_path),
        "output": str(output_directory),
        "engine": args.engine,
        "quality": args.quality,
        "resolution": [width, height],
        "samples": samples,
        "blendFile": str(blend_path),
        "images": images,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"ZYNORA_FIVE_VIEW_OUTPUT={output_directory}")
    print(f"ZYNORA_BLEND_FILE={blend_path}")
    print(f"ZYNORA_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
