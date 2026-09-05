"""Official-style scan profiles for different 2K HQ capture habits."""
from __future__ import annotations

from core.models import ScanProfile


PROFILES = [
    ScanProfile(
        id="2k_hq_safe",
        label="2K HQ Safe (recommended)",
        yaw_range_deg=40,
        pitch_range_deg=8,
        duration_sec=16,
        lighting="neutral_scan",
        notes="Slow yaw, mild pitch, even light. Highest lock rate on the official app.",
    ),
    ScanProfile(
        id="2k_hq_wide",
        label="2K HQ Wide Turn",
        yaw_range_deg=48,
        pitch_range_deg=6,
        duration_sec=18,
        lighting="studio_even",
        notes="Pushes toward the 45° cap. Use if the app keeps asking for more side.",
    ),
    ScanProfile(
        id="2k_hq_short",
        label="2K HQ Short",
        yaw_range_deg=36,
        pitch_range_deg=6,
        duration_sec=10,
        lighting="neutral_scan",
        notes="Faster capture. Riskier if the phone drops tracking.",
    ),
    ScanProfile(
        id="beauty_dim",
        label="Beauty Dimensional",
        yaw_range_deg=38,
        pitch_range_deg=10,
        duration_sec=18,
        lighting="soft_key_fill",
        notes="More form. Only if Neutral Scan looks too flat in-game.",
    ),
]


def list_profiles() -> list[ScanProfile]:
    return PROFILES


def get_profile(pid: str) -> ScanProfile:
    for p in PROFILES:
        if p.id == pid:
            return p
    return PROFILES[0]
