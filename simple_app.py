#!/usr/bin/env python3
"""FaceScan — one screen. Pick a preset head, drop a photo, tweak, render."""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from config import CACHE, OUTPUT, SETTINGS, ensure_dirs
from core.animator import sample_clip
from core.head_mesh import make_template
from core.models import AnimationSettings, FaceMorphSettings, ProjectData, TextureSettings
from core.rasterizer import render_head
from core.renderer import render_project
from core.texture_engine import build_face_plate
from core.utils import new_id, setup_logging

ensure_dirs()
setup_logging()

PRESETS = [
    "male_average",
    "female_average",
    "male_athletic",
    "androgynous_average",
]


def _photo_path(photo):
    if photo is None:
        return None
    if isinstance(photo, dict):
        return photo.get("path") or photo.get("name")
    return str(photo)


def preview(photo, preset, scale, offx, offy, rot, jaw, width, yaw_view):
    p = _photo_path(photo)
    if not p:
        raise gr.Error("Drop a frontal face photo first.")
    tex = TextureSettings(
        resolution=768,
        photo_scale=float(scale),
        photo_offset_x=float(offx),
        photo_offset_y=float(offy),
        photo_rotation=float(rot),
        color_match=True,
    )
    plate = build_face_plate(Path(p), tex, size=768)
    rgb = plate.image[:, :, ::-1]
    im = Image.fromarray(rgb)
    w, h = im.size
    nw = max(8, int(w * (1.0 + 0.18 * float(width))))
    nh = max(8, int(h * (1.0 + 0.10 * float(jaw))))
    stretched = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (120, 110, 100))
    canvas.paste(stretched, ((w - nw) // 2, (h - nh) // 2))
    rgb = np.array(canvas)
    head = make_template(preset)
    frame = render_head(head, rgb, yaw=float(yaw_view), pitch=4.0, width=640, height=640)
    out = CACHE / f"prev_{new_id()}.png"
    Image.fromarray(frame).save(out)
    return str(out)


def generate(photo, preset, scale, offx, offy, rot, jaw, width, progress=gr.Progress()):
    p = _photo_path(photo)
    if not p:
        raise gr.Error("Drop a frontal face photo first.")
    dest = CACHE / f"photo_{new_id()}{Path(p).suffix or '.png'}"
    shutil.copy2(p, dest)
    project = ProjectData(
        name="scan",
        template_id=preset,
        photo_path=str(dest),
        texture=TextureSettings(
            resolution=768,
            photo_scale=float(scale),
            photo_offset_x=float(offx),
            photo_offset_y=float(offy),
            photo_rotation=float(rot),
        ),
        morph=FaceMorphSettings(jaw_width=float(jaw), face_width=float(width)),
        animation=AnimationSettings(
            preset="2k_official",
            duration_sec=12.0,
            fps=24,
            yaw_range_deg=40.0,
            pitch_range_deg=6.0,
        ),
    )

    def cb(frac, msg):
        progress(frac, desc=msg)

    path = render_project(project, progress_cb=cb)
    return str(path)


def build():
    with gr.Blocks(title="FaceScan") as demo:
        gr.Markdown(
            "# FaceScan\n"
            "Pick a preset 3D head. Drop a straight-on photo. It maps onto the head. "
            "Nudge sliders if needed. Make scan video. Play that video fullscreen for 2K HQ."
        )
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="filepath", label="Your photo (front, even light)", height=280)
                preset = gr.Dropdown(choices=PRESETS, value="male_average", label="Preset 3D head")
                scale = gr.Slider(0.80, 1.25, 1.00, step=0.01, label="Photo fit")
                offx = gr.Slider(-20, 20, 0, step=0.5, label="Slide left / right")
                offy = gr.Slider(-20, 20, 0, step=0.5, label="Slide up / down")
                rot = gr.Slider(-12, 12, 0, step=0.5, label="Tilt")
                jaw = gr.Slider(-1, 1, 0, step=0.05, label="Jaw")
                width = gr.Slider(-1, 1, 0, step=0.05, label="Face width")
                yaw_view = gr.Slider(-40, 40, 18, step=1, label="Preview turn")
                btn_prev = gr.Button("Map photo onto head")
                btn_go = gr.Button("Make scan video", variant="primary")
            with gr.Column():
                preview_img = gr.Image(label="3D head", height=420)
                video = gr.Video(label="Scan video")
        btn_prev.click(
            preview,
            inputs=[photo, preset, scale, offx, offy, rot, jaw, width, yaw_view],
            outputs=[preview_img],
        )
        btn_go.click(
            generate,
            inputs=[photo, preset, scale, offx, offy, rot, jaw, width],
            outputs=[video],
        )
        gr.Markdown("Videos save in the output folder next to the app.")
    return demo


if __name__ == "__main__":
    build().queue().launch(
        server_name=SETTINGS.host,
        server_port=SETTINGS.port,
        share=False,
        allowed_paths=[str(OUTPUT), str(CACHE)],
    )
