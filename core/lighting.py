"""Lighting presets applied as 2D gradient overlays and as Blender rigs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import PRESETS
from core.models import LightingSettings


LIGHTING_PRESETS = {
    "neutral_scan": {
        "label": "Neutral Scan (recommended)",
        "key": (1.0, 0.02, -0.05),
        "fill": 0.42,
        "rim": 0.08,
        "warmth": 0.0,
        "notes": "Even front light, mild form. Best for 2K HQ.",
    },
    "studio_even": {
        "label": "Studio Even",
        "key": (0.85, 0.0, 0.0),
        "fill": 0.55,
        "rim": 0.05,
        "warmth": 0.0,
        "notes": "Flat commercial beauty lighting.",
    },
    "soft_key_fill": {
        "label": "Soft Key + Fill",
        "key": (1.05, 0.12, -0.04),
        "fill": 0.38,
        "rim": 0.18,
        "warmth": 0.04,
        "notes": "Dimensional but still scan-safe.",
    },
    "outdoor_soft": {
        "label": "Outdoor Soft",
        "key": (1.1, 0.08, 0.1),
        "fill": 0.5,
        "rim": 0.22,
        "warmth": 0.08,
        "notes": "Slightly cooler fill, open-sky feel.",
    },
    "warm_studio": {
        "label": "Warm Studio",
        "key": (0.95, 0.06, 0.0),
        "fill": 0.4,
        "rim": 0.12,
        "warmth": 0.12,
        "notes": "Warmer key. Use if photo is cold.",
    },
}


def list_presets() -> dict:
    return LIGHTING_PRESETS


def seed_presets() -> None:
    d = PRESETS / "lighting"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "presets.json"
    if not p.exists():
        p.write_text(json.dumps(LIGHTING_PRESETS, indent=2))


def lighting_overlay(h: int, w: int, settings: LightingSettings) -> np.ndarray:
    """Return float32 HxWx3 multiplier map centered around 1.0."""
    preset = LIGHTING_PRESETS.get(settings.preset, LIGHTING_PRESETS["neutral_scan"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2 - 1
    ny = (yy / max(h - 1, 1)) * 2 - 1
    kx, ky, kz = preset["key"]
    # lambertian-ish 2D stand-in: brighter toward key direction
    key = np.clip(1.0 + (nx * kx * 0.18 + ny * ky * 0.12) * settings.key_intensity, 0.55, 1.45)
    fill = 1.0 + settings.fill_ratio * preset["fill"] * 0.15
    rim = 1.0 + np.clip(np.abs(nx) - 0.55, 0, 1) * preset["rim"] * settings.rim_strength * 0.8
    warmth = (preset["warmth"] + (settings.temperature_k - 5600) / 8000.0) * 0.25
    overlay = np.stack(
        [
            key * fill * rim * (1.0 - warmth * 0.15),  # B
            key * fill * rim,
            key * fill * rim * (1.0 + warmth),  # R-ish in BGR later
        ],
        axis=-1,
    )
    # convert intended RGB multipliers into BGR order
    overlay = overlay[..., ::-1]
    exp = 2.0 ** settings.exposure
    overlay *= exp
    return overlay.astype(np.float32)


def apply_lighting(bgr: np.ndarray, settings: LightingSettings) -> np.ndarray:
    h, w = bgr.shape[:2]
    mul = lighting_overlay(h, w, settings)
    out = bgr.astype(np.float32) * mul
    return np.clip(out, 0, 255).astype(np.uint8)
