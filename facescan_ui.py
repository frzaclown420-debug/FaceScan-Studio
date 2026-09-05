#!/usr/bin/env python3
"""FaceScan — 2K HQ scan editor. Sliders always work. Landmark snap is optional."""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from config import CACHE, MESHES, OUTPUT, SETTINGS, ensure_dirs
from core.glb_export import compose_texture, write_glb
from core.head_mesh import PRESET_FILES, make_template
from core.models import AnimationSettings, FaceMorphSettings, ProjectData, TextureSettings
from config import CACHE, MESHES, OUTPUT, SETTINGS, ensure_dirs
from core.renderer import render_scan_texture
from core.texture_engine import build_face_plate
from core.utils import new_id, setup_logging

ensure_dirs()
setup_logging()

PRESETS = ["male_average", "female_average", "male_athletic", "androgynous_average"]
MARK_ORDER = ["left_eye", "right_eye", "nose", "mouth"]
# Where those features sit on the 3D head after front projection (texture 0-1, top = forehead)
MESH_TARGETS = {
    "left_eye": (0.37, 0.42),
    "right_eye": (0.63, 0.42),
    "nose": (0.50, 0.55),
    "mouth": (0.50, 0.70),
}


def preset_obj(preset):
    path = MESHES / PRESET_FILES.get(preset, f"{preset}.obj")
    return str(path) if path.exists() else None


def _photo_path(photo):
    if photo is None:
        return None
    if isinstance(photo, dict):
        return photo.get("path") or photo.get("name")
    return str(photo)


def _params(scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness=0.0, contrast=0.0):
    return dict(
        scale=float(scale),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        offset_x=float(offx),
        offset_y=float(offy),
        rotation=float(rot),
        zoom=float(zoom),
        skin_hex=str(skin_hex or "#c6987a"),
        shade=float(shade),
        brightness=float(brightness),
        contrast=float(contrast),
    )


def _rgb(photo):
    p = _photo_path(photo)
    if not p:
        return None
    plate = build_face_plate(Path(p), TextureSettings(resolution=1024, color_match=True), size=1024)
    return plate.image[:, :, ::-1]


def update_head(photo, preset, scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast):
    rgb = _rgb(photo)
    if rgb is None:
        raise gr.Error("Drop a frontal photo first.")
    head = make_template(preset)
    p = _params(scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast)
    glb = write_glb(head, rgb, CACHE / f"head_{new_id()}.glb", tex_params=p)
    tex = compose_texture(rgb, size=640, **p)
    preview = CACHE / f"tex_{new_id()}.png"
    Image.fromarray(tex).save(preview)
    return str(glb), str(preview)


def generate(
    photo, preset, scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast,
    progress=gr.Progress(),
):
    p = _photo_path(photo)
    if not p:
        raise gr.Error("Drop a frontal photo first.")
    rgb = _rgb(photo)
    if rgb is None:
        raise gr.Error("Drop a frontal photo first.")
    tex = compose_texture(
        rgb,
        size=1024,
        **_params(scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast),
    )
    anim = AnimationSettings(
        preset="2k_official",
        duration_sec=12.0,
        fps=20,
        yaw_range_deg=45.0,
        pitch_range_deg=4.0,
        blink_per_minute=8.0,
        blink_duration_ms=140.0,
        eye_saccades=False,
    )
    out = OUTPUT / f"{new_id('scan')}.mp4"
    path = render_scan_texture(tex, preset, anim, out, progress_cb=lambda f, m: progress(f, desc=m))
    return str(path)


def reset():
    return 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "#c6987a", 0.45, 0.08, 0.05


def _empty_marks():
    return {k: None for k in MARK_ORDER}


def click_mark(evt: gr.SelectData, marks: dict, photo):
    marks = dict(marks or _empty_marks())
    nxt = next((k for k in MARK_ORDER if marks.get(k) is None), None)
    if nxt is None:
        marks = _empty_marks()
        nxt = MARK_ORDER[0]
    # evt.index is pixel xy on the displayed image
    x, y = evt.index
    p = _photo_path(photo)
    if p:
        im = Image.open(p)
        w, h = im.size
        marks[nxt] = (x / max(w, 1), y / max(h, 1))
    else:
        marks[nxt] = (0.5, 0.5)
    left = [k for k in MARK_ORDER if marks.get(k) is None]
    msg = "Marked " + nxt.replace("_", " ")
    if left:
        msg += " — next click: " + left[0].replace("_", " ")
    else:
        msg += " — all four set. Hit Snap to 3D eyes."
    return marks, msg, _draw_marks(photo, marks)


def _draw_marks(photo, marks):
    p = _photo_path(photo)
    if not p:
        return None
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    colors = {"left_eye": (0, 255, 80), "right_eye": (0, 255, 80), "nose": (255, 200, 0), "mouth": (80, 180, 255)}
    for k, pt in (marks or {}).items():
        if not pt:
            continue
        x, y = int(pt[0] * w), int(pt[1] * h)
        r = max(6, w // 80)
        d.ellipse((x - r, y - r, x + r, y + r), outline=colors.get(k, (255, 255, 255)), width=3)
        d.text((x + r + 2, y - r), k.replace("_", " "), fill=colors.get(k, (255, 255, 255)))
    out = CACHE / f"marks_{new_id()}.png"
    im.save(out)
    return str(out)


def snap_marks(marks, scale, scale_x, scale_y, offx, offy, rot, zoom):
    marks = marks or {}
    le, re = marks.get("left_eye"), marks.get("right_eye")
    if not le or not re:
        raise gr.Error("Click left eye then right eye on the photo first. Nose and mouth help but are optional.")
    # photo eye mid and width
    mx = (le[0] + re[0]) / 2.0
    my = (le[1] + re[1]) / 2.0
    pw = max(abs(re[0] - le[0]), 0.05)
    ph = pw * 1.15
    tx = (MESH_TARGETS["left_eye"][0] + MESH_TARGETS["right_eye"][0]) / 2.0
    ty = (MESH_TARGETS["left_eye"][1] + MESH_TARGETS["right_eye"][1]) / 2.0
    tw = abs(MESH_TARGETS["right_eye"][0] - MESH_TARGETS["left_eye"][0])
    # total scale so photo eye width matches mesh eye width
    tot_x = tw / pw
    tot_y = tot_x
    if marks.get("mouth"):
        # vertical span eyes -> mouth
        pv = max(abs(marks["mouth"][1] - my), 0.05)
        tv = abs(MESH_TARGETS["mouth"][1] - ty)
        tot_y = tv / pv
    # sliders: zoom stays, pack into scale * scale_x
    new_scale = float(np_clip(tot_x, 0.45, 1.9))
    new_sx = 1.0
    new_sy = float(np_clip(tot_y / max(new_scale, 0.2), 0.72, 1.38))
    # offsets so photo mid-eyes land on mesh mid-eyes
    # u = 0.5 + offx/100 + (mx-0.5)*new_scale*new_sx*zoom
    z = float(zoom) if zoom else 1.0
    new_offx = (tx - 0.5 - (mx - 0.5) * new_scale * new_sx * z) * 100.0
    new_offy = -((ty - 0.5 - (my - 0.5) * new_scale * new_sy * z) * 100.0)
    new_offx = float(np_clip(new_offx, -40, 40))
    new_offy = float(np_clip(new_offy, -40, 40))
    return new_scale, new_sx, new_sy, new_offx, new_offy, rot, zoom


def np_clip(v, a, b):
    return max(a, min(b, v))


def clear_marks():
    return _empty_marks(), "Clicks cleared. Current sliders still work.", None


def build():
    css = """
    .gradio-container {max-width: 1600px !important;}
    """
    with gr.Blocks(title="FaceScan", css=css) as demo:
        gr.Markdown(
            "# FaceScan — 2K HQ\n"
            "Default path is unchanged: drop photo, place with sliders, update the 3D head, make the scan video. "
            "Optional: click eyes / nose / mouth on the photo, then Snap, to lock features to the mesh. "
            "Scan video is built to 2K HQ rules — even front light, slow 45° head turn, eyes forward, plain backdrop."
        )
        marks = gr.State(_empty_marks())
        with gr.Row():
            with gr.Column(scale=4):
                photo = gr.Image(type="filepath", label="Photo — click to mark eyes / nose / mouth (optional)", height=420)
                mark_guide = gr.Image(label="Your marks", height=220)
                mark_msg = gr.Textbox(label="Landmark status", value="Optional. Click left eye, right eye, nose, mouth on the photo.", interactive=False)
                with gr.Row():
                    btn_snap = gr.Button("Snap marks to 3D eyes")
                    btn_clr = gr.Button("Clear marks")
                preset = gr.Dropdown(choices=PRESETS, value="male_average", label="Preset 3D head")
            with gr.Column(scale=6):
                spin = gr.Model3D(value=preset_obj("male_average"), label="3D head — drag to spin")
                tex_prev = gr.Image(label="Skin on the mesh", height=320)
                video = gr.Video(label="2K HQ scan video")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Place")
                scale = gr.Slider(0.40, 2.00, 1.00, step=0.005, label="Size")
                zoom = gr.Slider(0.60, 1.80, 1.00, step=0.005, label="Face zoom / crop")
                scale_x = gr.Slider(0.70, 1.45, 1.00, step=0.005, label="Width stretch")
                scale_y = gr.Slider(0.70, 1.45, 1.00, step=0.005, label="Height stretch")
                offx = gr.Slider(-45, 45, 0, step=0.1, label="Move left / right")
                offy = gr.Slider(-45, 45, 0, step=0.1, label="Move up / down")
                rot = gr.Slider(-30, 30, 0, step=0.1, label="Tilt")
            with gr.Column():
                gr.Markdown("### Look (2K cares about even light)")
                skin_hex = gr.ColorPicker(value="#c6987a", label="Head skin color")
                shade = gr.Slider(0.0, 1.0, 0.45, step=0.05, label="Shadow / form")
                brightness = gr.Slider(-0.4, 0.4, 0.08, step=0.01, label="Brightness")
                contrast = gr.Slider(-0.4, 0.4, 0.05, step=0.01, label="Contrast")
                with gr.Row():
                    btn_upd = gr.Button("Update 3D head", variant="primary")
                    btn_rst = gr.Button("Reset place")
                btn_go = gr.Button("Make 2K HQ scan video")
        sliders = [scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast]
        photo.select(click_mark, inputs=[marks, photo], outputs=[marks, mark_msg, mark_guide])
        btn_snap.click(
            snap_marks,
            inputs=[marks, scale, scale_x, scale_y, offx, offy, rot, zoom],
            outputs=[scale, scale_x, scale_y, offx, offy, rot, zoom],
        )
        btn_clr.click(clear_marks, outputs=[marks, mark_msg, mark_guide])
        btn_upd.click(update_head, inputs=[photo, preset, *sliders], outputs=[spin, tex_prev])
        btn_rst.click(reset, outputs=sliders)
        btn_go.click(generate, inputs=[photo, preset, *sliders], outputs=[video])
    return demo


if __name__ == "__main__":
    build().queue().launch(
        server_name=SETTINGS.host,
        server_port=SETTINGS.port,
        share=False,
        allowed_paths=[str(OUTPUT), str(CACHE), str(MESHES)],
    )
