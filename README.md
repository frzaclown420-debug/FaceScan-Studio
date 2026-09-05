# FaceScan Studio

Local tool for turning a **single frontal photo** into a **2K HQ face-scan video**.

Template head + photo projection + masks + look-around + blinks + scan-safe lighting → MP4 you play fullscreen and scan with the NBA 2K HQ app.

## Preview

[Watch the FaceScan Studio preview video on Jumpshare](https://www.image2url.com/r2/default/gifs/1788594294910-71ea09f8-d9bc-4739-8d6e-4a48b303cd62.gif)

## What you get

- Gradio web UI on `http://127.0.0.1:7860`
- Multiple average-face templates (2.5D built-in; drop `.glb` for real 3D)
- Custom face builder (detect → align → color-match)
- Mask library (beards, brows, scars, freckles — add your own PNGs)
- Look-around + blink animation tuned for 2K HQ
- Lighting presets (Neutral Scan recommended)
- Two backends:
  - **OpenCV 2.5D** (works with only a photo — default)
  - **Blender headless** (if `blender` is installed and a `.glb/.obj` is in `assets/meshes/`)

## Kali / Debian install

```bash
cd facescan-studio
chmod +x install.sh run.sh
./install.sh
./run.sh
```

Open http://127.0.0.1:7860

## CLI

```bash
source .venv/bin/activate
export PYTHONPATH=.
python cli.py --photo /path/to/face.jpg --resolution 1080p --duration 16
```

## Scan workflow

1. Use a straight-on photo, even front light, no glasses/hat, neutral expression.
2. Preview the plate, add masks if needed.
3. Keep yaw ≤ 45°, slow duration (~16s), Neutral Scan lighting.
4. Generate MP4.
5. Play **fullscreen** on a monitor. Point the 2K HQ app at the screen. Wait for green frame. Let the video turn the head.
6. In-game: MyPLAYER → Appearance → Check for Head Scan Data.

## Adding assets

| Type | Where |
|------|--------|
| 3D templates | `assets/meshes/*.glb` + `assets/meshes/metadata.json` |
| Texture masks | `assets/masks/<category>/*.png` (RGBA) |
| Lighting | `assets/presets/lighting/presets.json` |

Placeholder masks are generated on first run so the layer stack works immediately.

## Layout

See `docs/` and the full specification in `docs/FaceScan_Studio_Full_Blueprint.txt`.

## Notes

- Results depend on photo quality and how well the 2K app locks onto the screen.
- The OpenCV backend is a 2.5D approximation (perspective warp + blink + lighting), not a full 3D morphable model. For higher geometric fidelity, add a clean head mesh and use the Blender backend.
- Use only photos you have rights to. This is a local lab / personal pipeline tool.
