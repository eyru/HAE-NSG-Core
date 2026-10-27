"""参数预设。

同一套算法在不同场景下需要的参数差别很大，所以把「一整组参数」封装成
Profile，调用方只需要挑一个名字，不需要理解每个参数。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """一组完整的处理参数。

    字段含义见 docs/ALGORITHM.md，这里只强调两件事：

    1. target_rms 是「目标电平」，不是「必须达到」。只放大不缩小，
       已经够响的音频原样返回，所以近场录音走远场档也不会被压小。
    2. max_gain 是硬上限。放大倍数超过它就截断，避免把底噪一起抬成噪声墙。
    """

    name: str
    target_rms: float
    max_gain: float
    pre_emph: float          # 预加重系数，0 表示不做高频补偿
    pre_emph_mix: float      # 补偿量与原始信号的混合比例
    gate_enabled: bool       # 是否启用帧级噪声门
    gate_frame_ms: float     # 噪声门的分析帧长
    gate_attenuation: float  # 判决为噪声的帧保留多少能量
    limit_knee: float        # 软限幅拐点，超过它对峰值做 tanh 压缩


#: 近场：贴着电脑说话。只做轻度归一 + 噪声门，不做高频补偿。
NEAR = Profile(
    name="near",
    target_rms=0.12,
    max_gain=6.0,
    pre_emph=0.0,
    pre_emph_mix=0.0,
    gate_enabled=True,
    gate_frame_ms=20.0,
    gate_attenuation=0.30,
    limit_knee=0.97,
)

#: 远场：四五米以上。补高频 + 强增益 + 软限幅。
FAR = Profile(
    name="far",
    target_rms=0.22,
    max_gain=40.0,
    pre_emph=0.55,
    pre_emph_mix=0.38,
    gate_enabled=True,
    gate_frame_ms=20.0,
    gate_attenuation=0.30,
    limit_knee=0.85,
)

#: 强制放大：麦克风很远、说话很轻。增益上限放到 60 倍，代价是底噪一起变大。
AGGRESSIVE = Profile(
    name="aggressive",
    target_rms=0.38,
    max_gain=120.0,
    pre_emph=0.65,
    pre_emph_mix=0.38,
    gate_enabled=True,
    gate_frame_ms=20.0,
    gate_attenuation=0.30,
    limit_knee=0.85,
)

#: 只做降噪，不改音量。适合已经在做后级增益的场景。
DENOISE_ONLY = Profile(
    name="denoise",
    target_rms=0.0,
    max_gain=1.0,
    pre_emph=0.0,
    pre_emph_mix=0.0,
    gate_enabled=True,
    gate_frame_ms=20.0,
    gate_attenuation=0.30,
    limit_knee=0.97,
)

_PRESETS = {
    "near": NEAR,
    "far": FAR,
    "aggressive": AGGRESSIVE,
    "denoise": DENOISE_ONLY,
}


def get(name):
    """按名字取预设，名字不认识就抛 ValueError（不静默回退）。"""
    try:
        return _PRESETS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            "未知的 profile：%r，可选 %s" % (name, ", ".join(sorted(_PRESETS)))
        ) from None


def names():
    """返回所有预设名（排序后，便于 CLI 展示）。"""
    return sorted(_PRESETS)
