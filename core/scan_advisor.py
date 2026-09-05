"""Heuristic checklist for whether a photo/video will scan cleanly."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def analyze_photo(path: Path) -> dict:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    issues = []
    score = 100

    if min(h, w) < 600:
        issues.append("Resolution is low. Use at least ~1000px on the short side.")
        score -= 25
    gray = arr.mean(axis=2)
    if gray.mean() < 55:
        issues.append("Image is dark. Reshoot with even front light.")
        score -= 20
    if gray.mean() > 210:
        issues.append("Image is blown out. Lower exposure.")
        score -= 12
    # simple contrast
    if gray.std() < 18:
        issues.append("Very flat contrast — possible filter or haze.")
        score -= 8
    # color cast
    means = arr.reshape(-1, 3).mean(axis=0)
    if abs(float(means[0] - means[2])) > 35:
        issues.append("Strong color cast. Disable filters / white-balance.")
        score -= 10
    aspect = w / max(h, 1)
    if aspect > 1.6 or aspect < 0.55:
        issues.append("Odd aspect ratio. A tighter head-and-shoulders crop is better.")
        score -= 6

    if not issues:
        issues.append("Photo looks usable. Neutral expression + no glasses still required.")

    return {
        "width": w,
        "height": h,
        "score": max(0, score),
        "issues": issues,
        "verdict": "GOOD" if score >= 75 else ("OK" if score >= 55 else "RESHOOT"),
    }


def format_report(report: dict) -> str:
    lines = [
        f"Scan advisor: {report['verdict']}  ({report['score']}/100)",
        f"Size: {report['width']}x{report['height']}",
        "",
    ]
    for i in report["issues"]:
        lines.append(f"- {i}")
    lines.append("")
    lines.append("2K HQ wants: even FRONT light, ~18 inches, slow full-head turn, eyes forward, plain background.")
    return "\n".join(lines)
