"""Discover and load face templates."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from config import MESHES
from core.models import TemplateInfo


METADATA = MESHES / "metadata.json"

DEFAULT_TEMPLATES = [
    TemplateInfo(
        id="male_average",
        name="Male Average (Neutral)",
        gender="male",
        ethnicity="multi",
        age_range="20-35",
        path="male_average.json",
        has_blink_keys=True,
        notes="Default average male template used by the OpenCV 2.5D pipeline.",
    ),
    TemplateInfo(
        id="female_average",
        name="Female Average (Neutral)",
        gender="female",
        ethnicity="multi",
        age_range="20-35",
        path="female_average.json",
        has_blink_keys=True,
        notes="Default average female template used by the OpenCV 2.5D pipeline.",
    ),
    TemplateInfo(
        id="male_athletic",
        name="Male Athletic",
        gender="male",
        ethnicity="multi",
        age_range="20-32",
        path="male_athletic.json",
        has_blink_keys=True,
        notes="Slightly longer face, stronger jaw bias in 2.5D warp.",
    ),
    TemplateInfo(
        id="androgynous_average",
        name="Androgynous Average",
        gender="androgynous",
        ethnicity="multi",
        age_range="18-30",
        path="androgynous_average.json",
        has_blink_keys=True,
        notes="Neutral proportions for custom identities.",
    ),
]


def seed_metadata() -> None:
    MESHES.mkdir(parents=True, exist_ok=True)
    if not METADATA.exists():
        payload = {"templates": [t.model_dump() for t in DEFAULT_TEMPLATES]}
        METADATA.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote default template metadata")


def list_templates() -> list[TemplateInfo]:
    seed_metadata()
    data = json.loads(METADATA.read_text())
    templates = [TemplateInfo(**t) for t in data.get("templates", [])]
    # Discover extra .glb/.obj dropped by the user
    known = {t.path for t in templates}
    for ext in ("*.glb", "*.gltf", "*.obj"):
        for p in MESHES.glob(ext):
            if p.name in known:
                continue
            templates.append(
                TemplateInfo(
                    id=p.stem,
                    name=p.stem.replace("_", " ").title(),
                    path=p.name,
                    notes="User-added mesh (Blender backend).",
                )
            )
    return templates


def get_template(template_id: str) -> TemplateInfo:
    for t in list_templates():
        if t.id == template_id:
            return t
    return DEFAULT_TEMPLATES[0]


def template_file(info: TemplateInfo) -> Path | None:
    p = MESHES / info.path
    return p if p.exists() else None
