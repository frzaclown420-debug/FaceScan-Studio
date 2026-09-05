from core.animator import sample_clip
from core.models import AnimationSettings


def test_clip_length():
    s = AnimationSettings(duration_sec=2.0, fps=30)
    clip = sample_clip(s)
    assert len(clip) == 60
    assert all(-60 <= p.yaw <= 60 for p in clip)
    assert all(0 <= p.blink <= 1 for p in clip)
