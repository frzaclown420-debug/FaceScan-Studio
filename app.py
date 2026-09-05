#!/usr/bin/env python3
"""FaceScan — place face on 3D head, fullscreen that head. No video."""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from config import CACHE, MESHES, OUTPUT, SETTINGS, ensure_dirs
from core.glb_export import compose_texture, write_glb
from core.head_mesh import PRESET_FILES, make_template
from core.models import TextureSettings
from core.texture_engine import build_face_plate
from core.utils import new_id, setup_logging
from serve_view import PORT, start_viewer_server

ensure_dirs()
setup_logging()
start_viewer_server()

PRESETS = ["male_average", "female_average", "male_athletic", "androgynous_average"]
MARK_ORDER = ["left_eye", "right_eye", "nose", "mouth"]
MESH_TARGETS = {
    "left_eye": (0.37, 0.42),
    "right_eye": (0.63, 0.42),
    "nose": (0.50, 0.55),
    "mouth": (0.50, 0.70),
}
HEAD_GLB = OUTPUT / "head.glb"
VIEW_URL = f"http://127.0.0.1:{PORT}/viewer.html"


def _iframe(bust: str = "0") -> str:
    return (
        f'<iframe src="{VIEW_URL}?glb=head.glb&t={bust}" '
        f'style="width:100%;height:78vh;border:0;background:#222" allow="fullscreen"></iframe>'
        f'<p style="margin:6px 0"><a href="{VIEW_URL}?glb=head.glb" target="_blank" '
        f'style="color:#9cf">Open this 3D head in its own tab — then press F11 for true fullscreen</a></p>'
    )


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
        scale=float(scale), scale_x=float(scale_x), scale_y=float(scale_y),
        offset_x=float(offx), offset_y=float(offy), rotation=float(rot), zoom=float(zoom),
        skin_hex=str(skin_hex or "#c6987a"), shade=float(shade),
        brightness=float(brightness), contrast=float(contrast),
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
    shutil.copy2(glb, HEAD_GLB)
    tex = compose_texture(rgb, size=512, **p)
    preview = CACHE / f"tex_{new_id()}.png"
    Image.fromarray(tex).save(preview)
    return _iframe(new_id()), str(preview)


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
        msg += " — next: " + left[0].replace("_", " ")
    else:
        msg += " — hit Snap."
    return marks, msg, _draw_marks(photo, marks)


def _draw_marks(photo, marks):
    p = _photo_path(photo)
    if not p:
        return None
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for k, pt in (marks or {}).items():
        if not pt:
            continue
        x, y = int(pt[0] * w), int(pt[1] * h)
        r = max(6, w // 80)
        d.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 80), width=3)
    out = CACHE / f"marks_{new_id()}.png"
    im.save(out)
    return str(out)


def snap_marks(marks, scale, scale_x, scale_y, offx, offy, rot, zoom):
    marks = marks or {}
    le, re = marks.get("left_eye"), marks.get("right_eye")
    if not le or not re:
        raise gr.Error("Click left eye then right eye first.")
    mx, my = (le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0
    pw = max(abs(re[0] - le[0]), 0.05)
    tx = (MESH_TARGETS["left_eye"][0] + MESH_TARGETS["right_eye"][0]) / 2.0
    ty = (MESH_TARGETS["left_eye"][1] + MESH_TARGETS["right_eye"][1]) / 2.0
    tw = abs(MESH_TARGETS["right_eye"][0] - MESH_TARGETS["left_eye"][0])
    tot_x = tw / pw
    tot_y = tot_x
    if marks.get("mouth"):
        pv = max(abs(marks["mouth"][1] - my), 0.05)
        tot_y = abs(MESH_TARGETS["mouth"][1] - ty) / pv
    new_scale = max(0.3, min(3.0, tot_x))
    new_sy = max(0.3, min(3.0, tot_y / max(new_scale, 0.2)))
    z = float(zoom) or 1.0
    new_offx = max(-40, min(40, (tx - 0.5 - (mx - 0.5) * new_scale * z) * 100))
    new_offy = max(-40, min(40, -((ty - 0.5 - (my - 0.5) * new_scale * new_sy * z) * 100)))
    return new_scale, 1.0, new_sy, new_offx, new_offy, rot, zoom


def clear_marks():
    return _empty_marks(), "Cleared.", None


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --bg:#0d1016; --card:#161b24; --line:#2a3344; --text:#e8edf5; --muted:#8b97a8; --acc:#3d8bfd; }
.gradio-container { max-width: 1680px !important; font-family: Inter, system-ui, sans-serif !important; }
body, .gradio-container, .main { background: var(--bg) !important; color: var(--text) !important; }
footer, .footer { display:none !important; }
.hero { padding: 8px 4px 14px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
.hero h1 { font-size: 28px; letter-spacing: -0.03em; margin: 0 0 6px; }
.hero p { color: var(--muted); margin: 0; font-size: 14px; line-height: 1.45; }
.hero a { color: #7eb3ff; }
button.primary, .primary { background: linear-gradient(180deg,#4c96ff,#2f74e6) !important; border: 0 !important; }
"""


def build():
    with gr.Blocks(title="FaceScan", css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.HTML(
            f"""<div class="hero">
            <h1>FaceScan</h1>
            <p>Drop a photo. Place it on the 3D head. Update. Then open the live head fullscreen and scan or record it.<br>
            Fullscreen head: <a href="{VIEW_URL}" target="_blank">{VIEW_URL}</a> — press F11. Drag to turn. Scroll to zoom. Backdrop is in that window.</p>
            </div>"""
        )
        marks = gr.State(_empty_marks())
        stage = gr.HTML(_iframe())
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(type="filepath", label="Photo", height=300)
                mark_guide = gr.Image(label="Marks", height=140)
                mark_msg = gr.Textbox(interactive=False)
                with gr.Row():
                    btn_snap = gr.Button("Snap eyes")
                    btn_clr = gr.Button("Clear marks")
                preset = gr.Dropdown(choices=PRESETS, value="male_average", label="Preset head")
                tex_prev = gr.Image(label="Skin", height=160)
            with gr.Column(scale=1):
                scale = gr.Slider(0.20, 3.00, 1.00, step=0.01, label="Size")
                zoom = gr.Slider(0.20, 3.00, 1.00, step=0.01, label="Zoom")
                scale_x = gr.Slider(0.20, 3.00, 1.00, step=0.01, label="Width")
                scale_y = gr.Slider(0.20, 3.00, 1.00, step=0.01, label="Height")
                offx = gr.Slider(-45, 45, 0, step=0.1, label="Left / right")
                offy = gr.Slider(-45, 45, 0, step=0.1, label="Up / down")
                rot = gr.Slider(-30, 30, 0, step=0.1, label="Tilt")
                skin_hex = gr.ColorPicker(value="#c6987a", label="Skin fill")
                shade = gr.Slider(0, 1, 0.45, step=0.05, label="Shadow")
                brightness = gr.Slider(-0.4, 0.4, 0.08, step=0.01, label="Brightness")
                contrast = gr.Slider(-0.4, 0.4, 0.05, step=0.01, label="Contrast")
                btn_upd = gr.Button("Update 3D head", variant="primary")
                btn_rst = gr.Button("Reset")
        sliders = [scale, scale_x, scale_y, offx, offy, rot, zoom, skin_hex, shade, brightness, contrast]
        photo.select(click_mark, inputs=[marks, photo], outputs=[marks, mark_msg, mark_guide])
        btn_snap.click(snap_marks, inputs=[marks, scale, scale_x, scale_y, offx, offy, rot, zoom], outputs=[scale, scale_x, scale_y, offx, offy, rot, zoom])
        btn_clr.click(clear_marks, outputs=[marks, mark_msg, mark_guide])
        btn_upd.click(update_head, inputs=[photo, preset, *sliders], outputs=[stage, tex_prev])
        btn_rst.click(reset, outputs=sliders)
    return demo


if __name__ == "__main__":
    print(f"FULLSCREEN 3D HEAD: {VIEW_URL}")
    print("Open that URL in Firefox and press F11.")
    build().queue().launch(
        server_name=SETTINGS.host,
        server_port=SETTINGS.port,
        share=False,
        allowed_paths=[str(OUTPUT), str(CACHE), str(MESHES)],
    )
