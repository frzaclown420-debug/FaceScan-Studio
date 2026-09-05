#!/usr/bin/env python3
"""CLI: python cli.py --photo face.jpg --out output/scan.mp4"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import OUTPUT, ensure_dirs
from core.models import AnimationSettings, LightingSettings, ProjectData, RenderSettings, TextureSettings
from core.renderer import render_project
from core.utils import setup_logging


def main() -> None:
    ensure_dirs()
    setup_logging()
    p = argparse.ArgumentParser(description="FaceScan Studio CLI")
    p.add_argument("--photo", required=True)
    p.add_argument("--template", default="male_average")
    p.add_argument("--out", default="")
    p.add_argument("--yaw", type=float, default=40)
    p.add_argument("--duration", type=float, default=16)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--resolution", default="1080p")
    p.add_argument("--backend", default="opencv")
    args = p.parse_args()

    project = ProjectData(
        name="cli",
        template_id=args.template,
        photo_path=str(Path(args.photo).resolve()),
        texture=TextureSettings(),
        animation=AnimationSettings(duration_sec=args.duration, fps=args.fps, yaw_range_deg=args.yaw),
        lighting=LightingSettings(),
        render=RenderSettings(backend=args.backend, resolution=args.resolution, fps=args.fps),
    )
    path = render_project(project)
    dest = Path(args.out) if args.out else path
    if dest != path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        path = dest
    print(path)


if __name__ == "__main__":
    main()
