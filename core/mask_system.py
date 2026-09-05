"""Mask library + layer compositing onto a BGR face plate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from config import MASKS
from core.models import MaskInfo, MaskLayer


METADATA = MASKS / "metadata.json"

DEFAULT_MASKS = [
    MaskInfo(id="stubble", name="Light Stubble", category="beards", path="beards/stubble.png", tags=["beard", "short"]),
    MaskInfo(id="full_beard", name="Full Beard", category="beards", path="beards/full_beard.png", tags=["beard"]),
    MaskInfo(id="goatee", name="Goatee", category="beards", path="beards/goatee.png", tags=["beard"]),
    MaskInfo(id="brows_thick", name="Thick Brows", category="eyebrows", path="eyebrows/thick.png", tags=["brows"]),
    MaskInfo(id="scar_brow", name="Brow Scar", category="scars", path="scars/brow.png", tags=["scar"]),
    MaskInfo(id="freckles", name="Freckles", category="makeup", path="makeup/freckles.png", tags=["skin"]),
]


def seed_masks() -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    if not METADATA.exists():
        METADATA.write_text(json.dumps({"masks": [m.model_dump() for m in DEFAULT_MASKS]}, indent=2))
    # Generate simple placeholder PNGs if missing
    for m in DEFAULT_MASKS:
        p = MASKS / m.path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            _write_placeholder_mask(p, m.id)


def _write_placeholder_mask(path: Path, kind: str) -> None:
    """Soft-alpha placeholders so the layer system is usable before custom art."""
    w = h = 1024
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    if kind in ("stubble", "full_beard", "goatee"):
        # lower face band
        band = (yy > h * (0.58 if kind != "goatee" else 0.62)) & (yy < h * 0.92)
        if kind == "goatee":
            band &= np.abs(xx - cx) < w * 0.16
        if kind == "stubble":
            alpha = np.where(band, 70, 0)
        else:
            alpha = np.where(band, 140, 0)
        arr[..., 0:3] = (40, 30, 22)
        arr[..., 3] = alpha.astype(np.uint8)
    elif kind == "brows_thick":
        for sign in (-1, 1):
            bx = cx + sign * w * 0.16
            by = h * 0.38
            ell = ((xx - bx) / (w * 0.10)) ** 2 + ((yy - by) / (h * 0.035)) ** 2 < 1
            arr[ell, 0:3] = (30, 22, 16)
            arr[ell, 3] = 180
    elif kind == "scar_brow":
        line = (np.abs((yy - h * 0.36) - 0.15 * (xx - cx)) < 3) & (xx > cx + w * 0.05) & (xx < cx + w * 0.22)
        arr[line, 0:3] = (150, 90, 80)
        arr[line, 3] = 120
    elif kind == "freckles":
        rng = np.random.default_rng(7)
        for _ in range(180):
            x = int(rng.integers(int(w * 0.25), int(w * 0.75)))
            y = int(rng.integers(int(h * 0.28), int(h * 0.62)))
            r = int(rng.integers(1, 3))
            arr[max(0, y - r) : y + r, max(0, x - r) : x + r, 0:3] = (90, 55, 40)
            arr[max(0, y - r) : y + r, max(0, x - r) : x + r, 3] = 90
    Image.fromarray(arr, "RGBA").save(path)
    logger.info("Wrote placeholder mask {}", path)


def list_masks() -> list[MaskInfo]:
    seed_masks()
    data = json.loads(METADATA.read_text())
    masks = [MaskInfo(**m) for m in data.get("masks", [])]
    # discover extra user PNGs
    known = {m.path for m in masks}
    for cat in ("beards", "eyebrows", "scars", "makeup", "hair", "accessories"):
        folder = MASKS / cat
        folder.mkdir(exist_ok=True)
        for p in folder.glob("*.png"):
            rel = f"{cat}/{p.name}"
            if rel in known:
                continue
            masks.append(MaskInfo(id=p.stem, name=p.stem.replace("_", " ").title(), category=cat, path=rel))
    return masks


def get_mask(mask_id: str) -> MaskInfo | None:
    for m in list_masks():
        if m.id == mask_id:
            return m
    return None


def _blend(base: np.ndarray, overlay_rgb: np.ndarray, alpha: np.ndarray, mode: str) -> np.ndarray:
    a = alpha[..., None]
    if mode == "multiply":
        mix = (base.astype(np.float32) * overlay_rgb.astype(np.float32) / 255.0)
    elif mode == "overlay":
        b = base.astype(np.float32) / 255.0
        o = overlay_rgb.astype(np.float32) / 255.0
        mix = np.where(b < 0.5, 2 * b * o, 1 - 2 * (1 - b) * (1 - o)) * 255.0
    elif mode == "soft_light":
        b = base.astype(np.float32) / 255.0
        o = overlay_rgb.astype(np.float32) / 255.0
        mix = ((1 - 2 * o) * b * b + 2 * o * b) * 255.0
    else:
        mix = overlay_rgb.astype(np.float32)
    out = base.astype(np.float32) * (1 - a) + mix * a
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_layers(plate_bgr: np.ndarray, layers: list[MaskLayer]) -> np.ndarray:
    out = plate_bgr.copy()
    h, w = out.shape[:2]
    for layer in layers:
        if not layer.visible:
            continue
        info = get_mask(layer.mask_id)
        if info is None:
            continue
        path = MASKS / info.path
        if not path.exists():
            continue
        im = Image.open(path).convert("RGBA").resize((w, h), Image.BILINEAR)
        if abs(layer.rotation) > 0.01:
            im = im.rotate(-layer.rotation, resample=Image.BILINEAR, expand=False)
        rgba = np.array(im)
        rgb = rgba[:, :, :3][:, :, ::-1]
        alpha = (rgba[:, :, 3].astype(np.float32) / 255.0) * float(layer.opacity)
        out = _blend(out, rgb, alpha, layer.blend_mode)
    return out
