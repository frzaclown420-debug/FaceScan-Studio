"""Write a textured GLB so Gradio Model3D shows the photo ON the mesh."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from core.head_mesh import HeadMesh


def _align4(b: bytes) -> bytes:
    pad = (4 - (len(b) % 4)) % 4
    return b + (b" " * pad if False else b"\x00" * pad)


def _normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-8)
    return (n / lens).astype(np.float32)


def _hex_rgb(s: str, fallback=(198, 152, 122)):
    try:
        s = (s or "").strip().lstrip("#")
        if len(s) == 6:
            return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return fallback


def compose_texture(
    photo_rgb: np.ndarray,
    size: int = 1024,
    scale: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation: float = 0.0,
    zoom: float = 1.0,
    skin_hex: str = "#c6987a",
    shade: float = 0.65,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    """Place the photo in texture space. Upright. Skin fill + baked light."""
    src = Image.fromarray(photo_rgb).convert("RGB")
    fill = _hex_rgb(skin_hex)
    canvas = Image.new("RGB", (size, size), fill)
    sx = max(0.15, float(scale) * float(scale_x) * float(zoom))
    sy = max(0.15, float(scale) * float(scale_y) * float(zoom))
    nw = max(8, int(size * sx))
    nh = max(8, int(size * sy))
    face = src.resize((nw, nh), Image.LANCZOS)
    if abs(rotation) > 0.05:
        face = face.rotate(-float(rotation), resample=Image.BICUBIC, expand=True)
    cx = size // 2 + int(float(offset_x) / 100.0 * size)
    cy = size // 2 - int(float(offset_y) / 100.0 * size)
    canvas.paste(face, (cx - face.size[0] // 2, cy - face.size[1] // 2))
    arr = np.array(canvas).astype(np.float32)
    if abs(brightness) > 0.001 or abs(contrast) > 0.001:
        arr = (arr - 127.5) * (1.0 + float(contrast)) + 127.5 + float(brightness) * 80.0
        arr = np.clip(arr, 0, 255)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xx / size - 0.5) * 2.0
    ny = (yy / size - 0.5) * 2.0
    # key light upper-left, rim, cavity under brow/nose band
    lambert = np.clip(0.52 + 0.38 * (-0.35 * nx + 0.55 * -ny + 0.55), 0.28, 1.0)
    ao = np.clip(1.0 - 0.18 * (nx * nx + ny * ny), 0.62, 1.0)
    under = 1.0 - 0.16 * np.clip(ny - 0.15, 0, 1)  # slightly darker toward chin/neck
    amt = float(np.clip(shade, 0.0, 1.0))
    lit = lambert * ao * under
    lit = (1.0 - amt) + amt * lit
    arr *= lit[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def _front_uvs(verts: np.ndarray, uv_scale=1.0, uv_shift=(0.0, 0.0)) -> np.ndarray:
    """Project the photo straight onto the front of the head. Ignore file UVs."""
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    rx = max(float(np.percentile(np.abs(x), 96)), 1e-4)
    ry = max(float(np.percentile(np.abs(y), 96)), 1e-4)
    u = 0.5 + (x / (2.05 * rx)) * float(uv_scale) + float(uv_shift[0])
    # Keep the photo upright on the head. V increases toward the crown.
    v = 0.5 - (y / (2.05 * ry)) * float(uv_scale) + float(uv_shift[1])
    uvs = np.stack([u, v], axis=1).astype(np.float32)
    # keep back-of-head from wrapping a second face
    back = z < -0.12
    if back.any():
        uvs[back, 0] = np.clip(uvs[back, 0], 0.02, 0.98)
        uvs[back, 1] = np.clip(uvs[back, 1], 0.02, 0.20)
    return np.clip(uvs, 0.0, 1.0)


def write_glb(
    mesh: HeadMesh,
    photo_rgb: np.ndarray,
    out_path: Path,
    uv_shift=(0.0, 0.0),
    uv_scale=1.0,
    tex_params: dict | None = None,
) -> Path:
    if tex_params:
        tex = compose_texture(photo_rgb, size=1024, **tex_params)
    else:
        tex = compose_texture(photo_rgb, size=1024)
    png_path = out_path.with_suffix(".png")
    Image.fromarray(tex).save(png_path, "PNG")
    png_bytes = png_path.read_bytes()

    verts = mesh.verts.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    uvs = _front_uvs(verts, uv_scale=uv_scale, uv_shift=uv_shift)
    norms = _normals(verts, faces)

    pos = verts.tobytes()
    uvb = uvs.tobytes()
    nrm = norms.tobytes()
    idx = faces.reshape(-1).astype(np.uint32).tobytes()

    chunks = [pos, uvb, nrm, idx, png_bytes]
    views = []
    offset = 0
    aligned = []
    for c in chunks:
        if offset % 4:
            pad = 4 - (offset % 4)
            aligned.append(b"\x00" * pad)
            offset += pad
        views.append((offset, len(c)))
        aligned.append(c)
        offset += len(c)
    if offset % 4:
        pad = 4 - (offset % 4)
        aligned.append(b"\x00" * pad)
        offset += pad
    blob = b"".join(aligned)

    mn, mx = verts.min(0).tolist(), verts.max(0).tolist()
    j = {
        "asset": {"version": "2.0", "generator": "FaceScan"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.65,
                },
                "doubleSided": True,
            }
        ],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": views[0][0], "byteLength": views[0][1], "target": 34962},
            {"buffer": 0, "byteOffset": views[1][0], "byteLength": views[1][1], "target": 34962},
            {"buffer": 0, "byteOffset": views[2][0], "byteLength": views[2][1], "target": 34962},
            {"buffer": 0, "byteOffset": views[3][0], "byteLength": views[3][1], "target": 34963},
            {"buffer": 0, "byteOffset": views[4][0], "byteLength": views[4][1]},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
                "min": mn,
                "max": mx,
            },
            {"bufferView": 1, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5126, "count": len(norms), "type": "VEC3"},
            {"bufferView": 3, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"},
        ],
    }
    js = json.dumps(j, separators=(",", ":")).encode("utf-8")
    if len(js) % 4:
        js += b" " * (4 - len(js) % 4)

    length = 12 + 8 + len(js) + 8 + len(blob)
    header = struct.pack("<4sII", b"glTF", 2, length)
    json_chunk = struct.pack("<II", len(js), 0x4E4F534A) + js
    bin_chunk = struct.pack("<II", len(blob), 0x004E4942) + blob
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + json_chunk + bin_chunk)
    return out_path
