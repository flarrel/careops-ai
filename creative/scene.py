"""Build the editable Blender shots for the explainer."""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy


COLORS = {
    "back": "#0B1820", "panel": "#182F39", "panel_light": "#203D48",
    "muted": "#ABC0C6", "white": "#F1F8F7", "mint": "#65E7C0",
    "dark_mint": "#123C35", "amber": "#FFD18B", "dark_amber": "#483626",
}


def linear(channel):
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def material(name, color):
    rgb = tuple(linear(int(color[i:i + 2], 16) / 255) for i in (1, 3, 5))
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*rgb, 1)
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return mat


class Shot:
    def __init__(self, scene, index, start, spec, palette):
        self.offset = 20 * index
        self.start = start
        self.palette = palette
        data = bpy.data.cameras.new(spec["title"])
        data.type = "ORTHO"
        data.ortho_scale = 12.8
        camera = bpy.data.objects.new(spec["title"], data)
        scene.collection.objects.link(camera)
        camera.location = (self.offset, 0, 15)
        scene.timeline_markers.new(spec["title"], frame=start).camera = camera
        if index == 0:
            scene.camera = camera

        self.card("Background", 0, 0, 13, 7.4, "back", -0.1)
        self.card("Header rule", 0, 2.62, 11.1, .02, "panel_light", .01)
        self.label("CAREOPS AI", -5.54, 2.94, .22, "mint", left=True)
        self.label(f"{index + 1:02d} / 03", 5.10, 2.94, .20, "muted")
        self.label(spec["title"], -5.55, 1.85, .59, "white", left=True)
        self.label(spec["subtitle"], -5.53, -2.78, .29, "muted", left=True)

    def link(self, name, data, x, y, z, color):
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = (self.offset + x, y, z)
        obj.data.materials.append(self.palette[color])
        return obj

    def card(self, name, x, y, width, height, color, z, radius=.12):
        radius = min(radius, width / 2, height / 2)
        vertices = []
        for cx, cy, angle in (
            (width / 2 - radius, height / 2 - radius, 0),
            (-width / 2 + radius, height / 2 - radius, 90),
            (-width / 2 + radius, -height / 2 + radius, 180),
            (width / 2 - radius, -height / 2 + radius, 270),
        ):
            for step in range(7):
                rad = math.radians(angle + 15 * step)
                vertices.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad), 0))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], [tuple(range(len(vertices)))])
        return self.link(name, mesh, x, y, z, color)

    def label(self, body, x, y, size, color, *, left=False):
        font = bpy.data.curves.new(body, "FONT")
        font.body = body
        font.align_x = "LEFT" if left else "CENTER"
        font.align_y = "CENTER"
        font.size = size
        return self.link(body, font, x, y, .20, color)

    def animate(self, obj, channel, keys):
        for frame, value in keys:
            setattr(obj, channel, value)
            obj.keyframe_insert(data_path=channel, frame=self.start + frame)

    def reveal(self, objects, first, last):
        for obj in objects:
            self.animate(obj, "scale", [(first, (0, 0, 0)), (last, (1, 1, 1))])


def trace(shot, revised):
    shot.card("Source panel", -3.4, -.43, 4.25, 2.65, "panel", .02)
    shot.card("Excerpt panel", 3.4, -.43, 4.25, 2.65, "panel", .02)
    shot.label("POLICY DOCUMENT", -3.4, .40, .23, "muted")
    for y, width in [(-.15, 2.7), (-.48, 2.7), (-1.17, 2.7)]:
        shot.card("Source line", -3.4, y, width, .08, "panel_light", .08)
    shot.card("Route line", 0, -.76, 2.56, .045, "panel_light", .05)
    cursor = shot.card("Citation marker", -1.25, -.76, .19, .19, "mint", .12)
    shot.animate(cursor, "location", [
        (4, (shot.offset - 1.25, -.76, .12)), (26, (shot.offset + 1.25, -.76, .12)),
    ])
    evidence = [
        shot.label("CITED EXCERPT", 3.4, .40, .23, "mint"),
        shot.card("Excerpt highlight", 3.4, -.76, 2.75, .28, "dark_mint", .08),
        shot.card("Excerpt line", 3.4, -.76, 2.38, .06, "mint", .10),
    ]
    if revised:
        evidence.extend([
            shot.card("Source tab", -3.4, -.76, 2.75, .28, "dark_mint", .08),
            shot.label("SOURCE PAGE", -3.4, -.76, .19, "mint"),
        ])
    shot.reveal(evidence, 12, 27)


def abstain(shot, revised):
    shot.card("Evidence panel", -3.4, -.43, 4.25, 2.65, "panel", .02)
    shot.label("EVIDENCE CHECK", -3.4, .40, .23, "muted")
    for x in (-4.37, -3.52, -2.45):
        shot.card("Incomplete passage", x, -.52, .58, .12, "panel_light", .08)
    for x in (-.98, -.54, .55, .99):
        shot.card("Interrupted route", x, -.52, .26, .045, "muted", .08)
    marker = shot.label("×", 0, -.52, .44, "amber")
    shot.reveal([marker], 7, 18)
    panel = shot.card("Abstention panel", 3.4, -.43, 4.25, 2.65,
                      "dark_amber" if revised else "panel", .02)
    title = shot.label("ABSTAIN", 3.4, -.38, .55, "amber")
    reason = shot.label("Evidence insufficient", 3.4, -.98, .22, "muted")
    shot.reveal([panel, title, reason], 18 if revised else 8, 30)


def review(shot, revised):
    shot.card("Reviewer panel", -3.4, -.43, 4.25, 2.65, "panel", .02)
    shot.card("Lookup panel", 3.4, -.43, 4.25, 2.65, "panel", .02)
    shot.label("REVIEWER", -3.4, .61, .23, "muted")
    shot.label("SYNTHETIC FIXTURE", 3.4, .40, .23, "muted")
    shot.card("Reviewer torso", -4.45, -1.13, .75, .70, "panel_light", .11, .28)
    head = shot.card("Reviewer head", -4.45, -.48, .57, .57, "mint", .13, .28)
    for x in (-4.54, -4.36):
        eye = shot.card("Reviewer eye", x, -.46, .057, .057, "back", .16)
        shot.animate(eye, "location", [
            (8, (shot.offset + x, -.46, .16)), (21, (shot.offset + x, -.53, .16)),
        ])
    shot.animate(head, "location", [
        (8, (shot.offset - 4.45, -.48, .13)), (21, (shot.offset - 4.45, -.55, .13)),
    ])
    button = shot.card("Allow button", -2.72, -.62, 1.84, .52, "mint", .09)
    shot.label("ALLOW LOOKUP", -2.72, -.62, .19, "back")
    arm = shot.card("Reviewer forearm", -4.15, -.94, .81, .105, "mint", .12)
    shot.animate(arm, "rotation_euler", [
        (5, (0, 0, -.45)), (22, (0, 0, .38)), (34, (0, 0, -.18)),
    ])
    shot.animate(arm, "location", [
        (5, (shot.offset - 4.15, -.94, .12)),
        (22, (shot.offset - 3.91, -.76, .12)),
        (34, (shot.offset - 4.13, -.92, .12)),
    ])
    shot.animate(button, "scale", [
        (19, (1, 1, 1)), (22, (.92, .90, 1)), (27, (1, 1, 1)),
    ])
    result = [
        shot.card("Read status", 3.4, -.63, 3.32, .62, "dark_mint", .08),
        shot.label("READ-ONLY LOOKUP", 3.4, -.63, .25, "mint"),
    ]
    if revised:
        shot.card("Approval route", 0, -.62, 2.5, .04, "panel_light", .07)
        cursor = shot.card("Approved read marker", -1.29, -.62, .16, .16, "mint", .11)
        shot.reveal([cursor], 22, 25)
        shot.animate(cursor, "location", [
            (22, (shot.offset - 1.29, -.62, .11)), (33, (shot.offset + 1.29, -.62, .11)),
        ])
    shot.reveal(result, 33 if revised else 22, 39 if revised else 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("baseline", "revised"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    plan = json.loads(Path(__file__).with_name("storyboard.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.fps = plan["fps"]
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = "//frames/frame"
    span = plan["fps"] * plan["seconds_per_shot"]
    scene.frame_start = 1
    scene.frame_end = span * len(plan["shots"])
    palette = {name: material(name, color) for name, color in COLORS.items()}
    for index, (draw, spec) in enumerate(zip((trace, abstain, review), plan["shots"], strict=True)):
        shot = Shot(scene, index, span * index + 1, spec, palette)
        draw(shot, args.version == "revised")
    markers = scene.timeline_markers
    markers.new("Reviewer presses Allow lookup", frame=2 * span + 23)
    markers.new("Read becomes visible", frame=2 * span + (35 if args.version == "revised" else 24))
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()), compress=True)


if __name__ == "__main__":
    main()
