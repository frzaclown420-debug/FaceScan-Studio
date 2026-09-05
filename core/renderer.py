"""Render a 2K-HQ-ready scan video.

Primary backend: OpenCV 2.5D (works without Blender or a 3D mesh).
Optional backend: Blender headless when a real mesh + blender exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from config import CACHE, OUTPUT
from core.animator import sample_clip
from core.blender_bridge import blender_available, write_and_run
from core.lighting import apply_lighting
from core.mask_system import apply_layers
from core.mesh_manager import get_template, template_file
from core.models import ProjectData
from core.texture_engine import FacePlate, build_face_plate, save_plate
from core.utils import RESOLUTION_MAP, cleanup_dir, encode_frames_to_mp4, new_id


def _bg(h: int, w: int) -> np.ndarray:
    """Soft gray-green studio backdrop — uncluttered for the 2K app."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = yy / max(h - 1, 1)
    col = np.stack(
        [
            118 + t * 18,
            124 + t * 16,
            128 + t * 12,
        ],
        axis=-1,
    )
    return np.clip(col, 0, 255).astype(np.uint8)


def _perspective_warp(img: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    # yaw positive = turn right (subject's left cheek toward camera)
    y = np.deg2rad(yaw)
    p = np.deg2rad(pitch)
    # compress the receding side, shift center
    k = np.sin(y) * 0.28
    q = np.sin(p) * 0.16
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # left edge and right edge vertical scale
    left_shrink = max(0.72, 1.0 + k)
    right_shrink = max(0.72, 1.0 - k)
    top_shift = q * h
    dst = np.float32(
        [
            [0 + abs(k) * w * 0.15, (1 - left_shrink) * h * 0.5 + top_shift],
            [w - abs(k) * w * 0.15, (1 - right_shrink) * h * 0.5 + top_shift],
            [w - abs(k) * w * 0.08, h - (1 - right_shrink) * h * 0.5 - top_shift * 0.3],
            [0 + abs(k) * w * 0.08, h - (1 - left_shrink) * h * 0.5 - top_shift * 0.3],
        ]
    )
    if yaw < 0:
        dst[[0, 3], 0] += abs(k) * w * 0.22
    else:
        dst[[1, 2], 0] -= abs(k) * w * 0.22
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def _apply_blink(img: np.ndarray, amount: float, landmarks: np.ndarray | None) -> np.ndarray:
    if amount <= 0.02:
        return img
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    # default eye boxes if no landmarks
    boxes = [
        (int(w * 0.27), int(h * 0.36), int(w * 0.18), int(h * 0.09)),
        (int(w * 0.55), int(h * 0.36), int(w * 0.18), int(h * 0.09)),
    ]
    if landmarks is not None and len(landmarks) >= 468:
        # MediaPipe iris / eyelid clusters (approx)
        left_idx = [33, 133, 159, 145]
        right_idx = [362, 263, 386, 374]
        for idxs in (left_idx, right_idx):
            pts = landmarks[idxs]
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            pad_x = (x1 - x0) * 0.55
            pad_y = max(8, (y1 - y0) * 1.6)
            boxes.append(
                (
                    int(x0 - pad_x),
                    int(y0 - pad_y),
                    int((x1 - x0) + 2 * pad_x),
                    int((y1 - y0) + 2 * pad_y),
                )
            )
        boxes = boxes[2:]

    for x, y, bw, bh in boxes:
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        roi = out[y:y2, x:x2]
        if roi.size == 0:
            continue
        closed = cv2.resize(roi, (roi.shape[1], max(1, int(roi.shape[0] * (1.0 - 0.82 * amount)))))
        pad = roi.shape[0] - closed.shape[0]
        top = pad // 2
        canvas = np.zeros_like(roi)
        # fill with neighboring skin
        skin = np.median(roi.reshape(-1, 3), axis=0)
        canvas[:, :] = skin
        canvas[top : top + closed.shape[0]] = closed
        # dark crease
        crease_y = top + closed.shape[0] // 2
        if 0 <= crease_y < canvas.shape[0]:
            canvas[max(0, crease_y - 1) : crease_y + 2] = (canvas[max(0, crease_y - 1) : crease_y + 2] * 0.55).astype(np.uint8)
        mix = amount
        out[y:y2, x:x2] = (roi.astype(np.float32) * (1 - mix) + canvas.astype(np.float32) * mix).astype(np.uint8)
    return out


def _letterbox(fg: np.ndarray, bg: np.ndarray, scale: float = 0.72) -> np.ndarray:
    import cv2

    H, W = bg.shape[:2]
    h, w = fg.shape[:2]
    target = int(min(H, W) * scale)
    fg2 = cv2.resize(fg, (target, target), interpolation=cv2.INTER_AREA)
    y0 = (H - target) // 2 - int(H * 0.02)
    x0 = (W - target) // 2
    out = bg.copy()
    y1 = min(H, y0 + target)
    x1 = min(W, x0 + target)
    fy0 = 0 if y0 >= 0 else -y0
    fx0 = 0 if x0 >= 0 else -x0
    y0 = max(0, y0)
    x0 = max(0, x0)
    sl = fg2[fy0 : fy0 + (y1 - y0), fx0 : fx0 + (x1 - x0)]
    # soft circular alpha so plate isn't a hard square
    hh, ww = sl.shape[:2]
    yy, xx = np.ogrid[:hh, :ww]
    cy, cx = hh / 2, ww / 2
    r = min(hh, ww) * 0.48
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    alpha = np.clip((r + 18 - dist) / 18.0, 0, 1)[..., None]
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1 - alpha) + sl.astype(np.float32) * alpha).astype(np.uint8)
    return out


def render_opencv(plate: FacePlate, project: ProjectData, out_path: Path, progress_cb=None) -> Path:
    from PIL import Image as PImage

    from core.head_mesh import make_template
    from core.rasterizer import render_head

    poses = sample_clip(project.animation)
    w, h = RESOLUTION_MAP[project.render.resolution]
    work = CACHE / new_id("frames")
    cleanup_dir(work)
    base = apply_layers(plate.image, project.masks)
    try:
        from core.morph_engine import apply_morphs

        base = apply_morphs(base, project.morph)
    except Exception as e:
        logger.warning("morph skip: {}", e)
    tex_rgb = base[:, :, ::-1].copy()
    head = make_template(project.template_id)
    n = len(poses)
    logger.info("3D raster {} frames {}x{} template={}", n, w, h, project.template_id)
    # Square 1080 framed like the 2K green box: head fills the frame.
    box = 1080
    bg = np.array([118, 118, 120], dtype=np.float32)
    for i, pose in enumerate(poses):
        rgb = render_head(head, tex_rgb, pose.yaw, pose.pitch, width=900, height=900, blink=pose.blink, bg=bg)
        frame = PImage.fromarray(rgb).resize((box, box), PImage.LANCZOS)
        canvas = PImage.new("RGB", (w, h), (118, 118, 120))
        canvas.paste(frame, ((w - box) // 2, (h - box) // 2))
        canvas.save(work / f"f_{i:05d}.png")
        if progress_cb and i % 3 == 0:
            progress_cb((i + 1) / n * 0.85, f"3D frame {i+1}/{n}")
    if progress_cb:
        progress_cb(0.9, "Encoding video")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode_frames_to_mp4(str(work / "f_%05d.png"), out_path, fps=project.animation.fps, codec=project.render.codec)
    if progress_cb:
        progress_cb(1.0, "Done")
    return out_path


def render_scan_texture(tex_rgb, template_id: str, animation, out_path: Path, progress_cb=None) -> Path:
    """HQ path: already-composed skin on the real head, filling a 1080p frame."""
    from PIL import Image as PImage

    from core.head_mesh import make_template
    from core.rasterizer import render_head

    poses = sample_clip(animation)
    w, h = 1920, 1080
    work = CACHE / new_id("frames")
    cleanup_dir(work)
    head = make_template(template_id)
    n = len(poses)
    bg = np.array([118, 118, 120], dtype=np.float32)
    box = 1080
    logger.info("HQ scan {} frames 1920x1080", n)
    for i, pose in enumerate(poses):
        rgb = render_head(head, tex_rgb, pose.yaw, pose.pitch, width=960, height=960, blink=pose.blink, bg=bg)
        frame = PImage.fromarray(rgb).resize((box, box), PImage.LANCZOS)
        canvas = PImage.new("RGB", (w, h), (118, 118, 120))
        canvas.paste(frame, ((w - box) // 2, 0))
        canvas.save(work / f"f_{i:05d}.png")
        if progress_cb and i % 2 == 0:
            progress_cb((i + 1) / n * 0.88, f"HQ frame {i+1}/{n}")
    if progress_cb:
        progress_cb(0.92, "Encoding 1080p")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode_frames_to_mp4(str(work / "f_%05d.png"), out_path, fps=animation.fps, codec="h264")
    if progress_cb:
        progress_cb(1.0, "Done")
    return out_path


def render_project(project: ProjectData, progress_cb=None) -> Path:
    if not project.photo_path:
        raise ValueError("No photo selected")
    photo = Path(project.photo_path)
    if not photo.exists():
        raise FileNotFoundError(photo)

    if progress_cb:
        progress_cb(0.02, "Building face plate")
    plate = build_face_plate(photo, project.texture, size=project.texture.resolution)
    plate_path = save_plate(plate, CACHE / f"{new_id('plate')}.png")

    out = OUTPUT / f"{new_id('scan')}.mp4"
    tmpl = get_template(project.template_id)
    mesh = template_file(tmpl)

    use_blender = (
        project.render.backend in ("auto", "blender")
        and blender_available()
        and mesh is not None
        and mesh.suffix.lower() in {".glb", ".gltf", ".obj", ".fbx"}
    )
    if project.render.backend == "opencv":
        use_blender = False
    if project.render.backend == "blender" and not use_blender:
        logger.warning("Blender backend requested but unavailable; using OpenCV 2.5D")

    if use_blender:
        try:
            if progress_cb:
                progress_cb(0.08, "Rendering with Blender")
            return write_and_run(mesh, plate_path, out, project, progress_cb)
        except Exception as e:
            logger.exception("Blender path failed, falling back: {}", e)

    return render_opencv(plate, project, out, progress_cb)
