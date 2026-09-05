#!/usr/bin/env python3
"""Download public-domain / CC head bases into assets/meshes/.

Default source: Blender Studio Human Base Meshes bundle (CC0), if the
published URL is reachable. Also accepts a direct file URL.

This does NOT scrape copyrighted 2K scans or celebrity faces.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "meshes"

# Official Blender demo bundle (CC0 human bases). URL may move; override with --url.
DEFAULT_URL = "https://www.blender.org/download/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.2.0.zip"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceScanStudio/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def unpack(zpath: Path) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        keep = [n for n in names if n.lower().endswith((".glb", ".gltf", ".obj", ".fbx", ".blend"))]
        print(f"archive has {len(names)} files, {len(keep)} mesh-like")
        for n in keep:
            target = DEST / Path(n).name
            with z.open(n) as src, target.open("wb") as out:
                out.write(src.read())
            print(" extracted", target.name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--zip", default=str(DEST / "_bundle.zip"))
    args = p.parse_args()
    try:
        z = download(args.url, Path(args.zip))
        unpack(z)
    except Exception as e:
        print("Download failed:", e, file=sys.stderr)
        print("Get Blender Human Base Meshes manually and drop .glb/.obj into assets/meshes/", file=sys.stderr)
        return 1
    print("Done. Restart FaceScan Studio to pick up new templates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
