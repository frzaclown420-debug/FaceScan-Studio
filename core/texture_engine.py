"""Build a square face plate from a photo. PIL + numpy only. OpenCV optional."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance

from core.models import TextureSettings


@dataclass
class FacePlate:
    image: np.ndarray  # BGR uint8, square
    landmarks: Optional[np.ndarray]
    box: tuple[int, int, int, int]
    aligned: bool
    message: str


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import ImageOps

    im = Image.open(path)
    im = ImageOps.exif_transpose(im) or im
    return np.array(im.convert("RGB"))


def _detect_mediapipe(rgb: np.ndarray):
    try:
        import mediapipe as mp
    except Exception:
        return None, None
    if not hasattr(mp, "solutions"):
        return None, None
    try:
        h, w = rgb.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as mesh:
            res = mesh.process(rgb)
        if not res or not res.multi_face_landmarks:
            return None, None
        lms = res.multi_face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lms.landmark], dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]
        return pts, (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    except Exception as e:
        logger.warning("mediapipe skipped: {}", e)
        return None, None


def _detect_haar(rgb: np.ndarray):
    try:
        import cv2
        if not hasattr(cv2, "CascadeClassifier"):
            return None, None
        gray = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        path = getattr(getattr(cv2, "data", None), "haarcascades", "") + "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(path)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if faces is None or len(faces) == 0:
            return None, None
        faces = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)
        x, y, w, h = [int(v) for v in faces[0]]
        return None, (x, y, w, h)
    except Exception as e:
        logger.warning("haar skipped: {}", e)
        return None, None


def _expand_box(box, shape, margin=0.55):
    x, y, w, h = box
    H, W = shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1.0 + margin)
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2))
    y1 = int(min(H, cy + side / 2))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def build_face_plate(photo_path: Path, settings: TextureSettings, size: int = 1024) -> FacePlate:
    rgb = _read_rgb(photo_path)
    lms, box = _detect_mediapipe(rgb)
    aligned = False
    msg = "mediapipe"
    if box is None:
        lms, box = _detect_haar(rgb)
        msg = "haar"
    if box is None:
        h, w = rgb.shape[:2]
        side = min(h, w)
        box = ((w - side) // 2, (h - side) // 2, side, side)
        msg = "center crop"
        logger.info("No detector available; center crop")

    x, y, w, h = _expand_box(box, rgb.shape, margin=0.55)
    crop = rgb[y : y + h, x : x + w].copy()
    im = Image.fromarray(crop)
    if abs(settings.photo_rotation) > 0.01:
        im = im.rotate(-settings.photo_rotation, resample=Image.BICUBIC, expand=False)
    if abs(settings.photo_scale - 1.0) > 0.01:
        nw, nh = im.size
        im = im.resize((max(1, int(nw * settings.photo_scale)), max(1, int(nh * settings.photo_scale))), Image.BICUBIC)
        # center pad/crop back
        canvas = Image.new("RGB", (nw, nh), (120, 110, 100))
        canvas.paste(im, ((nw - im.size[0]) // 2, (nh - im.size[1]) // 2))
        im = canvas
    im = im.resize((size, size), Image.LANCZOS)
    if settings.color_match:
        im = ImageEnhance.Contrast(im).enhance(1.08)
        im = ImageEnhance.Color(im).enhance(0.96)
    plate_rgb = np.array(im)
    plate_bgr = plate_rgb[:, :, ::-1].copy()
    plate_lms = None
    if lms is not None:
        plate_lms = lms.copy()
        plate_lms[:, 0] = (lms[:, 0] - x) / max(w, 1) * size
        plate_lms[:, 1] = (lms[:, 1] - y) / max(h, 1) * size
        aligned = True
    logger.info("Face plate via {} {}px", msg, size)
    return FacePlate(image=plate_bgr, landmarks=plate_lms, box=(x, y, w, h), aligned=aligned, message=msg)


def save_plate(plate: FacePlate, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(plate.image[:, :, ::-1]).save(path)
    return path
