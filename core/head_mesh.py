"""Load real head OBJs from assets/meshes, or fall back to a parametric head."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import MESHES


@dataclass
class HeadMesh:
    verts: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    name: str
    source: str = "generated"


PRESET_FILES = {
    "male_average": "male_average.obj",
    "female_average": "female_average.obj",
    "male_athletic": "male_athletic.obj",
    "androgynous_average": "androgynous_average.obj",
}


def _front_uvs(verts: np.ndarray) -> np.ndarray:
    x = verts[:, 0]
    y = verts[:, 1]
    u = 0.5 + (x - x.mean()) / (max(x.max() - x.min(), 1e-6) * 1.05)
    v = 0.5 + (y - y.mean()) / (max(y.max() - y.min(), 1e-6) * 1.05)
    return np.stack([np.clip(u, 0, 1), np.clip(v, 0, 1)], axis=1).astype(np.float32)


def _normalize(verts: np.ndarray) -> np.ndarray:
    v = verts.astype(np.float32).copy()
    v -= v.mean(axis=0)
    # keep head-ish region if this is a full body (Y span huge)
    span = v.max(axis=0) - v.min(axis=0)
    if span[1] > 2.5 * max(span[0], span[2]):
        ymin = v[:, 1].max() - span[1] * 0.22
        keep = v[:, 1] >= ymin
        if keep.sum() > 50:
            v = v[keep]
            # faces remapped later — this path only used if we filter verts before faces
    scale = float(np.abs(v).max())
    if scale > 0:
        v /= scale
    return v


def load_obj(path: Path) -> HeadMesh:
    verts = []
    faces = []
    raw_uvs = []
    face_uvs = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("vt "):
            p = line.split()
            raw_uvs.append((float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
        elif line.startswith("f "):
            bits = line.split()[1:]
            idxs = []
            uidxs = []
            for b in bits:
                parts = b.split("/")
                idxs.append(int(parts[0]) - 1)
                if len(parts) > 1 and parts[1]:
                    uidxs.append(int(parts[1]) - 1)
            # triangulate fan
            for i in range(1, len(idxs) - 1):
                faces.append((idxs[0], idxs[i], idxs[i + 1]))
                if len(uidxs) == len(idxs):
                    face_uvs.append((uidxs[0], uidxs[i], uidxs[i + 1]))
    v = np.array(verts, dtype=np.float32)
    f = np.array(faces, dtype=np.int32)
    # drop unused verts after optional body crop
    span = v.max(axis=0) - v.min(axis=0)
    if span[1] > 2.5 * max(span[0], span[2]):
        ymin = v[:, 1].max() - span[1] * 0.22
        keep = np.where(v[:, 1] >= ymin)[0]
        remap = -np.ones(len(v), dtype=np.int32)
        remap[keep] = np.arange(len(keep), dtype=np.int32)
        mask = (remap[f[:, 0]] >= 0) & (remap[f[:, 1]] >= 0) & (remap[f[:, 2]] >= 0)
        f = remap[f[mask]]
        v = v[keep]
    v = v - v.mean(axis=0)
    scale = float(np.abs(v).max()) or 1.0
    v = v / scale
    uv = _front_uvs(v)
    if raw_uvs and face_uvs and len(face_uvs) == len(f):
        acc = np.zeros((len(v), 2), dtype=np.float32)
        cnt = np.zeros((len(v), 1), dtype=np.float32)
        ruv = np.array(raw_uvs, dtype=np.float32)
        for fi, (a, b, c) in enumerate(f):
            ua, ub, uc = face_uvs[fi]
            if max(ua, ub, uc) < len(ruv):
                acc[a] += ruv[ua]
                acc[b] += ruv[ub]
                acc[c] += ruv[uc]
                cnt[[a, b, c]] += 1
        good = cnt[:, 0] > 0
        uv[good] = acc[good] / np.maximum(cnt[good], 1)
    return HeadMesh(verts=v, faces=f, uvs=uv, name=path.stem, source=str(path.name))


def _sphere(n_lat=16, n_lon=20):
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    verts, uvs, faces = [], [], []
    for la in lats:
        for lo in lons:
            verts.append((np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)))
            uvs.append((0.5 + lo / (2 * np.pi), 0.5 - la / np.pi))
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.array(verts, np.float32), np.array(faces, np.int32), np.array(uvs, np.float32)


def make_template(template_id: str = "male_average") -> HeadMesh:
    fname = PRESET_FILES.get(template_id, f"{template_id}.obj")
    path = MESHES / fname
    if path.exists():
        mesh = load_obj(path)
        mesh.name = template_id
        if "female" in template_id:
            mesh.verts[:, 0] *= 0.93
            mesh.verts[:, 1] *= 1.02
        elif "athletic" in template_id:
            mesh.verts[:, 0] *= 1.04
            mesh.verts[:, 2] *= 1.03
        return mesh
    v, f, uv = _sphere()
    v[:, 0] *= 0.78
    v[:, 1] *= 1.05
    v[:, 2] *= 0.92
    back = np.clip(-v[:, 2], 0, None)
    v[:, 2] = v[:, 2] + back * 0.08
    uv = _front_uvs(v)
    return HeadMesh(verts=v, faces=f, uvs=uv, name=template_id, source="generated")


def rotate(verts: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return verts @ rx.T @ ry.T


def write_obj(mesh: HeadMesh, obj_path: Path, tex_path: Path | None = None) -> Path:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = obj_path.with_suffix(".mtl").name
    lines = []
    if tex_path is not None:
        lines.append(f"mtllib {mtl_name}")
        lines.append("usemtl skin")
        mtl = obj_path.with_suffix(".mtl")
        mtl.write_text(
            "newmtl skin\nKd 1 1 1\nmap_Kd "
            + tex_path.name
            + "\n"
        )
    for x, y, z in mesh.verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for u, v in mesh.uvs:
        lines.append(f"vt {u:.6f} {v:.6f}")
    for a, b, c in mesh.faces:
        lines.append(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}")
    obj_path.write_text("\n".join(lines) + "\n")
    return obj_path
