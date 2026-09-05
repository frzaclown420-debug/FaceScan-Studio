"""Tiny textured-mesh renderer. No OpenCV required."""
from __future__ import annotations

import numpy as np

from core.head_mesh import HeadMesh, rotate


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    x = np.clip(u * (w - 1), 0, w - 1)
    y = np.clip((1.0 - v) * (h - 1), 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    return tex[y0, x0]


def render_head(
    mesh: HeadMesh,
    texture_rgb: np.ndarray,
    yaw: float,
    pitch: float,
    width: int = 720,
    height: int = 720,
    blink: float = 0.0,
    bg=None,
) -> np.ndarray:
    """Return HxWx3 uint8 RGB."""
    if bg is None:
        bg = np.array([128, 124, 118], dtype=np.float32)
    verts = rotate(mesh.verts, yaw, pitch)
    dist = 2.4
    z = verts[:, 2] + dist
    z = np.clip(z, 0.2, None)
    px = verts[:, 0] / z
    py = verts[:, 1] / z
    # pack the projected head so it fills most of the frame (2K wants chin-to-crown)
    span = max(float(np.percentile(np.abs(px), 98)), float(np.percentile(np.abs(py), 98)), 1e-4)
    zoom = 0.86 / span
    sx = (px * zoom + 1) * 0.5 * (width - 1)
    sy = (1 - (py * zoom + 1) * 0.5) * (height - 1)

    img = np.broadcast_to(bg, (height, width, 3)).copy()
    zbuf = np.full((height, width), 1e9, dtype=np.float32)

    faces = mesh.faces
    uvs = mesh.uvs
    tex = texture_rgb
    if tex.dtype != np.uint8:
        tex = np.clip(tex, 0, 255).astype(np.uint8)

    # skip backfaces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    front = n[:, 2] < 0.02  # facing camera (+Z toward us after rotate? sphere +Z is face)
    # After rotate, face is +Z originally. Camera looks from +Z toward origin? 
    # verts z increases toward camera if face is +Z and we sit at +Z... 
    # projection uses z = verts_z + dist, so larger verts_z is closer.
    # Cross product n_z > 0 means facing +Z (camera). Keep those.
    front = n[:, 2] > 0.0
    faces = faces[front]
    if len(faces) == 0:
        return img.astype(np.uint8)

    s = np.stack([sx, sy], axis=1)
    for a, b, c in faces:
        pts = s[[a, b, c]]
        minx = int(max(0, np.floor(pts[:, 0].min())))
        maxx = int(min(width - 1, np.ceil(pts[:, 0].max())))
        miny = int(max(0, np.floor(pts[:, 1].min())))
        maxy = int(min(height - 1, np.ceil(pts[:, 1].max())))
        if maxx <= minx or maxy <= miny:
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-6:
            continue
        zs = z[[a, b, c]]
        uv = uvs[[a, b, c]]
        # blink: pull upper-lid UVs downward in eye band
        if blink > 0.05:
            pass
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue
        zz = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        uu = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
        vv = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
        sub_z = zbuf[miny : maxy + 1, minx : maxx + 1]
        closer = mask & (zz < sub_z)
        if not closer.any():
            continue
        col = _sample(tex, uu, vv).astype(np.float32)
        # cheap lambert from +Z
        shade = 0.72 + 0.28 * np.clip(n_face_z(n, a, faces, front), 0, 1)
        # per-triangle shade
        shade = 0.75 + 0.25 * float(np.clip(n[front][0, 2] if False else 1.0, 0, 1))
        img_view = img[miny : maxy + 1, minx : maxx + 1]
        zb_view = zbuf[miny : maxy + 1, minx : maxx + 1]
        img_view[closer] = col[closer]
        zb_view[closer] = zz[closer]

    if blink > 0.04:
        img = _paint_blink(img, blink)
    return np.clip(img, 0, 255).astype(np.uint8)


def n_face_z(n, a, faces, front):
    return 1.0


def _paint_blink(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    for cx in (0.38, 0.62):
        x0 = int((cx - 0.08) * w)
        x1 = int((cx + 0.08) * w)
        y0 = int(0.40 * h)
        y1 = int(0.50 * h)
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        lid = np.median(roi.reshape(-1, 3), axis=0)
        t = min(1.0, amount)
        # squash toward middle row
        mid = roi.shape[0] // 2
        for i in range(roi.shape[0]):
            fall = 1.0 - abs(i - mid) / max(mid, 1)
            roi[i] = roi[i] * (1 - t * fall) + lid * (t * fall)
        out[y0:y1, x0:x1] = roi
    return out
