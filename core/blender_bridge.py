"""Write a temporary Blender Python script and invoke headless Blender.

Used when a real .glb/.obj template exists and blender is on PATH.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from loguru import logger

from config import SETTINGS, CACHE
from core.models import ProjectData
from core.utils import which


def blender_available() -> bool:
    return which(SETTINGS.blender_bin) is not None or which("blender") is not None


def write_and_run(
    mesh_path: Path,
    texture_path: Path,
    out_mp4: Path,
    project: ProjectData,
    progress_cb=None,
) -> Path:
    bin_ = which(SETTINGS.blender_bin) or which("blender")
    if not bin_:
        raise RuntimeError("Blender not found")

    script = CACHE / "blender_job.py"
    script.write_text(_script(mesh_path, texture_path, out_mp4, project))
    cmd = [bin_, "-b", "-P", str(script)]
    logger.info("Running {}", " ".join(cmd))
    if progress_cb:
        progress_cb(0.1, "Launching Blender")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender failed:\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}")
    if progress_cb:
        progress_cb(1.0, "Blender finished")
    if not out_mp4.exists():
        raise RuntimeError("Blender finished but output video is missing")
    return out_mp4


def _script(mesh: Path, tex: Path, out: Path, project: ProjectData) -> str:
    anim = project.animation
    rend = project.render
    res = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}[rend.resolution]
    samples = 32 if rend.quality == "preview" else max(64, rend.samples)
    engine = "BLENDER_EEVEE_NEXT" if rend.quality == "preview" else "CYCLES"
    return textwrap.dedent(
        f"""
        import bpy
        from math import radians
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=r"{mesh}") if r"{mesh}".lower().endswith((".glb",".gltf")) else bpy.ops.wm.obj_import(filepath=r"{mesh}")
        objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not objs:
            raise RuntimeError("No mesh imported")
        head = objs[0]
        mat = bpy.data.materials.new("Skin")
        mat.use_nodes = True
        nt = mat.node_tree
        img = bpy.data.images.load(r"{tex}")
        texn = nt.nodes.new("ShaderNodeTexImage")
        texn.image = img
        bsdf = nt.nodes.get("Principled BSDF")
        nt.links.new(texn.outputs["Color"], bsdf.inputs["Base Color"])
        if head.data.materials:
            head.data.materials[0] = mat
        else:
            head.data.materials.append(mat)
        cam = bpy.data.cameras.new("Cam")
        cam.lens = 85
        cam_obj = bpy.data.objects.new("Cam", cam)
        bpy.context.scene.collection.objects.link(cam_obj)
        cam_obj.location = (0, -2.4, 0.08)
        cam_obj.rotation_euler = (radians(90), 0, 0)
        bpy.context.scene.camera = cam_obj
        light = bpy.data.lights.new("Key", "AREA")
        light.energy = 250
        light.size = 1.2
        lo = bpy.data.objects.new("Key", light)
        lo.location = (0.4, -1.4, 0.8)
        bpy.context.scene.collection.objects.link(lo)
        fill = bpy.data.lights.new("Fill", "AREA")
        fill.energy = 80
        fill.size = 2.0
        fo = bpy.data.objects.new("Fill", fill)
        fo.location = (-0.8, -1.2, 0.3)
        bpy.context.scene.collection.objects.link(fo)
        scene = bpy.context.scene
        scene.render.engine = "{engine}"
        scene.render.resolution_x = {res[0]}
        scene.render.resolution_y = {res[1]}
        scene.render.fps = {anim.fps}
        scene.frame_start = 1
        n = max(8, int({anim.duration_sec} * {anim.fps}))
        scene.frame_end = n
        scene.render.filepath = r"{out.with_suffix('')}_"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        if hasattr(scene, "cycles"):
            scene.cycles.samples = {samples}
        yaw = {anim.yaw_range_deg}
        for i in range(n):
            t = i / max(n-1, 1)
            # simple ping-pong yaw
            phase = t * 4
            if phase < 1:
                y = -yaw * phase
            elif phase < 2:
                y = -yaw + yaw * (phase-1)
            elif phase < 3:
                y = yaw * (phase-2)
            else:
                y = yaw - yaw * (phase-3)
            head.rotation_euler = (0, 0, radians(y))
            head.keyframe_insert("rotation_euler", frame=i+1)
        bpy.ops.render.render(animation=True)
        """
    )
