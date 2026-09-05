#!/usr/bin/env bash
# FaceScan Studio installer for Kali / Debian / Ubuntu
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> FaceScan Studio installer"
echo "    root: $ROOT"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential
else
  echo "apt-get not found; install python3, venv, pip, ffmpeg manually."
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

mkdir -p assets/meshes assets/masks/{beards,eyebrows,scars,makeup,hair,accessories} \
         assets/textures/{eyes,skin_details,hdri} assets/presets/{lighting,animation,materials} \
         data/{projects,cache,thumbnails,logs} output

python3 - <<'PY'
from core.mesh_manager import seed_metadata
from core.mask_system import seed_masks
from core.lighting import seed_presets
from config import ensure_dirs
ensure_dirs()
seed_metadata()
seed_masks()
seed_presets()
print("Seeded templates, masks, presets.")
PY

chmod +x run.sh || true
echo
echo "Install complete."
echo "  source .venv/bin/activate"
echo "  ./run.sh"
echo "Then open http://127.0.0.1:7860"
echo
echo "Optional: install Blender and drop .glb templates into assets/meshes/ for the 3D backend."
