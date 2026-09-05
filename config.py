"""Global configuration for FaceScan Studio."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
import json
import os


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
CACHE = DATA / "cache"
PROJECTS = DATA / "projects"
LOGS = DATA / "logs"
THUMBS = DATA / "thumbnails"

MESHES = ASSETS / "meshes"
MASKS = ASSETS / "masks"
TEXTURES = ASSETS / "textures"
PRESETS = ASSETS / "presets"

USER_CONFIG = DATA / "user_config.json"


@dataclass
class Settings:
    blender_bin: str = os.environ.get("BLENDER_BIN", "blender")
    default_resolution: str = "1080p"
    default_fps: int = 30
    default_duration: float = 16.0
    preview_width: int = 720
    cache_keep_hours: int = 48
    host: str = "127.0.0.1"
    port: int = 7860
    theme: str = "dark"
    prefer_blender: bool = True
    texture_resolution: int = 2048
    yaw_range_deg: float = 40.0
    pitch_range_deg: float = 8.0

    def save(self, path: Path = USER_CONFIG) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path = USER_CONFIG) -> "Settings":
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


def ensure_dirs() -> None:
    for p in (ASSETS, DATA, OUTPUT, CACHE, PROJECTS, LOGS, THUMBS, MESHES, MASKS, TEXTURES, PRESETS):
        p.mkdir(parents=True, exist_ok=True)
    for sub in ("beards", "eyebrows", "scars", "makeup", "hair", "accessories"):
        (MASKS / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("lighting", "animation", "materials"):
        (PRESETS / sub).mkdir(parents=True, exist_ok=True)


SETTINGS = Settings.load()
ensure_dirs()
