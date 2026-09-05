"""Pydantic data models for FaceScan Studio."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TemplateInfo(BaseModel):
    id: str
    name: str
    gender: Literal["male", "female", "androgynous"] = "male"
    ethnicity: str = "average"
    age_range: str = "20-35"
    path: str
    thumbnail: Optional[str] = None
    polycount: int = 0
    has_blink_keys: bool = False
    notes: str = ""


class MaskInfo(BaseModel):
    id: str
    name: str
    category: str
    path: str
    type: Literal["texture", "geometry"] = "texture"
    tags: list[str] = Field(default_factory=list)
    thumbnail: Optional[str] = None


class MaskLayer(BaseModel):
    id: str
    mask_id: str
    name: str
    opacity: float = 1.0
    blend_mode: Literal["normal", "multiply", "overlay", "soft_light"] = "normal"
    visible: bool = True
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0


class FaceMorphSettings(BaseModel):
    """Signed strengths in [-1, 1]. 0 = no change."""
    jaw_width: float = 0.0
    jaw_length: float = 0.0
    chin: float = 0.0
    cheek: float = 0.0
    nose_width: float = 0.0
    nose_length: float = 0.0
    eye_size: float = 0.0
    eye_sep: float = 0.0
    brow_height: float = 0.0
    face_width: float = 0.0


class ScanProfile(BaseModel):
    id: str
    label: str
    yaw_range_deg: float = 40.0
    pitch_range_deg: float = 8.0
    duration_sec: float = 16.0
    fps: int = 30
    lighting: str = "neutral_scan"
    notes: str = ""


class TextureSettings(BaseModel):
    resolution: int = 2048
    projection: Literal["front_view", "cylindrical", "smart"] = "front_view"
    auto_align: bool = True
    photo_offset_x: float = 0.0
    photo_offset_y: float = 0.0
    photo_scale: float = 1.0
    photo_rotation: float = 0.0
    color_match: bool = True
    seam_dilate: int = 8


class AnimationSettings(BaseModel):
    preset: str = "2k_official"
    duration_sec: float = 16.0
    fps: int = 30
    yaw_range_deg: float = 40.0
    pitch_range_deg: float = 8.0
    easing: Literal["ease_in_out", "linear", "smoothstep"] = "ease_in_out"
    hold_at_extreme_sec: float = 0.35
    blink_per_minute: float = 14.0
    blink_duration_ms: float = 180.0
    blink_randomness: float = 0.35
    eye_saccades: bool = True


class LightingSettings(BaseModel):
    preset: str = "neutral_scan"
    key_intensity: float = 1.0
    fill_ratio: float = 0.45
    rim_strength: float = 0.15
    temperature_k: int = 5600
    exposure: float = 0.0
    sss_strength: float = 0.35
    shadow_softness: float = 0.7


class RenderSettings(BaseModel):
    backend: Literal["auto", "blender", "opencv"] = "auto"
    quality: Literal["preview", "final"] = "final"
    resolution: Literal["720p", "1080p", "1440p", "4k"] = "1080p"
    fps: int = 30
    codec: Literal["h264", "h265"] = "h264"
    samples: int = 128


class ProjectData(BaseModel):
    version: str = "1.0"
    name: str = "Untitled"
    template_id: str = "male_average"
    photo_path: Optional[str] = None
    texture: TextureSettings = Field(default_factory=TextureSettings)
    morph: FaceMorphSettings = Field(default_factory=FaceMorphSettings)
    masks: list[MaskLayer] = Field(default_factory=list)
    animation: AnimationSettings = Field(default_factory=AnimationSettings)
    lighting: LightingSettings = Field(default_factory=LightingSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    notes: str = ""
