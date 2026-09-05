"""Helpers: logging, IDs, image IO, easing, video encode."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Iterable

from loguru import logger

from config import LOGS, CACHE, OUTPUT


def setup_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOGS / "facescan.log",
        rotation="10 MB",
        retention="14 days",
        level="INFO",
        enqueue=True,
    )


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoothstep(t: float) -> float:
    return ease_in_out(t)


def apply_easing(t: float, mode: str) -> float:
    if mode == "linear":
        return clamp(t, 0.0, 1.0)
    if mode == "smoothstep":
        s = clamp(t, 0.0, 1.0)
        return s * s * s * (s * (s * 6 - 15) + 10)
    return ease_in_out(t)


RESOLUTION_MAP = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def run_ffmpeg(args: list[str]) -> None:
    bin_ = which("ffmpeg")
    if not bin_:
        raise RuntimeError("ffmpeg not found. Install ffmpeg.")
    cmd = [bin_, "-y", *args]
    logger.info("ffmpeg {}", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")


def encode_frames_to_mp4(
    frame_pattern: str,
    out_path: Path,
    fps: int = 30,
    codec: str = "h264",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vcodec = "libx265" if codec == "h265" else "libx264"
    pix = "yuv420p"
    extra = ["-crf", "16", "-preset", "slow"]
    if vcodec == "libx264":
        extra += ["-pix_fmt", pix]
    run_ffmpeg(
        [
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-c:v",
            vcodec,
            *extra,
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return out_path


def cleanup_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
