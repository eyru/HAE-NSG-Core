"""HAE-NSG Core —— 降噪与增益前端。

    Hearizon Audiology Engine — Noise-Suppression Gain Core

一句话：把一段「听不清」的录音变成 ASR 听得清的录音，且不损伤已经清楚的录音。

    import hae_nsg
    audio, rate = hae_nsg.read_wav("meeting.wav")
    out = hae_nsg.process(audio, "far", rate=rate)
    hae_nsg.write_wav("meeting_clean.wav", out, rate)

设计取舍见 docs/ALGORITHM.md，为什么不做谱减也写在那里。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AGGRESSIVE",
    "DENOISE_ONLY",
    "FAR",
    "NEAR",
    "NoiseFloor",
    "Profile",
    "frame_energies",
    "get_profile",
    "noise_gate",
    "normalize_gain",
    "peak",
    "percentile",
    "pre_emphasis",
    "process",
    "profile_names",
    "read_info",
    "read_wav",
    "rms",
    "soft_limit",
    "to_db",
    "write_wav",
    "write_wav_stereo_compare",
]

from .core import (
    AGGRESSIVE,
    DENOISE_ONLY,
    FAR,
    NEAR,
    Profile,
    get_profile,
    noise_gate,
    normalize_gain,
    peak,
    percentile,
    pre_emphasis,
    process,
    profile_names,
    rms,
    soft_limit,
    to_db,
)
from .noise import NoiseFloor, frame_energies
from .wav import read_info, read_wav, write_wav, write_wav_stereo_compare
