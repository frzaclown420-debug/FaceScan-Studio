"""2.5D face-parameter warps applied to the aligned plate.

These are not a 3D morphable model. They are spatially localized grid
warps that let you push jaw / nose / eyes / width on the textured plate
before the look-around render. Useful when the photo doesn't match the
template proportions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models import FaceMorphSettings


@dataclass
class MorphField:
    name: str
    # normalized [0,1] peak location and radii
    cx: float
    cy: float
    rx: float
    ry: float
    dx: float  # pixel direction at +1 strength (as fraction of width)
    dy: float


FIELDS = [
    MorphField("jaw_width", 0.50, 0.78, 0.42, 0.22, 0.06, 0.00),
    MorphField("jaw_length", 0.50, 0.86, 0.28, 0.16, 0.00, 0.07),
    MorphField("chin", 0.50, 0.90, 0.16, 0.12, 0.00, 0.06),
    MorphField("cheek", 0.50, 0.58, 0.48, 0.16, 0.05, 0.00),
    MorphField("nose_width", 0.50, 0.52, 0.14, 0.12, 0.04, 0.00),
    MorphField("nose_length", 0.50, 0.50, 0.12, 0.16, 0.00, 0.05),
    MorphField("eye_size", 0.50, 0.40, 0.38, 0.10, 0.00, 0.00),  # scale handled separately
    MorphField("eye_sep", 0.50, 0.40, 0.36, 0.10, 0.04, 0.00),
    MorphField("brow_height", 0.50, 0.34, 0.38, 0.08, 0.00, -0.04),
    MorphField("face_width", 0.50, 0.55, 0.50, 0.40, 0.05, 0.00),
]


def _gaussian_mask(h: int, w: int, f: MorphField) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(w - 1, 1)
    ny = yy / max(h - 1, 1)
    val = np.exp(-(((nx - f.cx) / max(f.rx, 1e-4)) ** 2 + ((ny - f.cy) / max(f.ry, 1e-4)) ** 2))
    return val


def apply_morphs(bgr: np.ndarray, morph: FaceMorphSettings) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return bgr
    h, w = bgr.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))

    strengths = {
        "jaw_width": morph.jaw_width,
        "jaw_length": morph.jaw_length,
        "chin": morph.chin,
        "cheek": morph.cheek,
        "nose_width": morph.nose_width,
        "nose_length": morph.nose_length,
        "eye_sep": morph.eye_sep,
        "brow_height": morph.brow_height,
        "face_width": morph.face_width,
    }
    for f in FIELDS:
        s = float(strengths.get(f.name, 0.0))
        if abs(s) < 1e-4 and f.name != "eye_size":
            continue
        mask = _gaussian_mask(h, w, f)
        # outward from center for width-style fields
        nx = (np.linspace(0, 1, w, dtype=np.float32) - 0.5)
        sign_x = np.sign(nx).reshape(1, -1)
        map_x += mask * s * f.dx * w * sign_x
        map_y += mask * s * f.dy * h

    # eye size: local scale around each eye
    if abs(morph.eye_size) > 1e-4:
        for cx in (0.34, 0.66):
            mask = _gaussian_mask(h, w, MorphField("e", cx, 0.40, 0.12, 0.08, 0, 0))
            map_x += (map_x - cx * w) * mask * (-0.18 * morph.eye_size)
            map_y += (map_y - 0.40 * h) * mask * (-0.18 * morph.eye_size)

    return cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
