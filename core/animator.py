"""Look-around + blink tracks sampled per frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models import AnimationSettings
from core.utils import apply_easing, clamp


@dataclass
class FramePose:
    yaw: float
    pitch: float
    blink: float  # 0 open, 1 closed
    saccade_x: float
    saccade_y: float


def _look_curve(n: int, yaw: float, pitch: float, hold: float, fps: int, easing: str) -> tuple[np.ndarray, np.ndarray]:
    """Center -> left -> center -> right -> center with holds at extremes."""
    hold_f = max(1, int(hold * fps))
    # segment weights
    segs = np.array([1.0, 0.15, 1.0, 0.15, 1.0, 0.15, 1.0])
    # map last 3 holds+moves; total 7 pieces: moveL, holdL, moveC, holdC, moveR, holdR, moveC
    raw = np.array([1.6, hold, 1.4, hold * 0.6, 1.6, hold, 1.6])
    raw = raw / raw.sum()
    counts = np.maximum(1, (raw * n).astype(int))
    counts[-1] += n - int(counts.sum())
    yaw_keys = [0, -yaw, -yaw, 0, yaw, yaw, 0]
    pit_keys = [0, pitch * 0.25, pitch * 0.25, 0, -pitch * 0.15, -pitch * 0.15, 0]
    ys, ps = [], []
    for i in range(6):
        c = counts[i]
        for k in range(c):
            t = apply_easing(k / max(c - 1, 1), easing)
            # holds: first and last of hold segments stay put
            if i in (1, 3, 5):
                t = 0.0 if k < c else 1.0
                ys.append(yaw_keys[i])
                ps.append(pit_keys[i])
            else:
                ys.append(yaw_keys[i] + (yaw_keys[i + 1] - yaw_keys[i]) * t)
                ps.append(pit_keys[i] + (pit_keys[i + 1] - pit_keys[i]) * t)
    # pad / trim
    ys = np.array(ys[:n], dtype=np.float32)
    ps = np.array(ps[:n], dtype=np.float32)
    if len(ys) < n:
        ys = np.pad(ys, (0, n - len(ys)), constant_values=0)
        ps = np.pad(ps, (0, n - len(ps)), constant_values=0)
    return ys, ps


def _blink_track(n: int, fps: int, per_min: float, duration_ms: float, rand: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    blink = np.zeros(n, dtype=np.float32)
    span = max(3, int((duration_ms / 1000.0) * fps))
    interval = max(span * 3, int((60.0 / max(per_min, 1e-3)) * fps))
    t = int(fps * 1.2)
    while t < n - span - 2:
        jitter = int((rng.random() * 2 - 1) * rand * interval)
        start = clamp(t + jitter, 0, n - span - 1)
        start = int(start)
        for i in range(span):
            # triangle open-close-open
            u = i / max(span - 1, 1)
            val = u * 2 if u < 0.45 else (1 - u) / 0.55
            val = clamp(val, 0, 1)
            idx = start + i
            if 0 <= idx < n:
                blink[idx] = max(blink[idx], val)
        t += interval
    return blink


def sample_clip(settings: AnimationSettings) -> list[FramePose]:
    n = max(8, int(settings.duration_sec * settings.fps))
    yaw, pitch = _look_curve(
        n,
        settings.yaw_range_deg,
        settings.pitch_range_deg,
        settings.hold_at_extreme_sec,
        settings.fps,
        settings.easing,
    )
    blink = _blink_track(
        n,
        settings.fps,
        settings.blink_per_minute,
        settings.blink_duration_ms,
        settings.blink_randomness,
    )
    rng = np.random.default_rng(9)
    sx = np.zeros(n)
    sy = np.zeros(n)
    if settings.eye_saccades:
        sx = np.cumsum((rng.normal(0, 0.015, n))).astype(np.float32)
        sy = np.cumsum((rng.normal(0, 0.01, n))).astype(np.float32)
        sx -= sx.mean()
        sy -= sy.mean()
    return [
        FramePose(float(yaw[i]), float(pitch[i]), float(blink[i]), float(sx[i]), float(sy[i]))
        for i in range(n)
    ]
