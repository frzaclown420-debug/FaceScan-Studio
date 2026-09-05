"""Save / load .fss project files."""
from __future__ import annotations

import json
from pathlib import Path

from config import PROJECTS
from core.models import ProjectData
from core.utils import new_id


def project_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_") or "untitled"
    return PROJECTS / f"{safe}.fss"


def save_project(data: ProjectData, path: Path | None = None) -> Path:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    path = path or project_path(data.name or new_id("project"))
    path.write_text(data.model_dump_json(indent=2))
    return path


def load_project(path: Path) -> ProjectData:
    return ProjectData.model_validate_json(path.read_text())


def list_projects() -> list[Path]:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    return sorted(PROJECTS.glob("*.fss"), key=lambda p: p.stat().st_mtime, reverse=True)
