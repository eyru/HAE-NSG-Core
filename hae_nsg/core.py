"""HAE-NSG 核心算法。

设计原则（也是和 C++ 版对齐的前提）：
    1. 中间计算一律用 float64，只在最后一步转回 float32。
    2. 不依赖任何「库之间可能有差异」的函数。分位数自己算，
       排序用稳定排序，插值公式写死，这样 numpy 版、纯 Python 版、
       C++ 版三方拿到的是同一个数。
    3. 所有函数对脏输入（空、None、NaN、超短、全零）都不抛异常，
       原样返回或安全降级——音频处理不该因为一帧坏数据把整条流打断。

算法本身见 docs/ALGORITHM.md。
"""

from __future__ import annotations

import math

import numpy as np

from .profiles import (
    AGGRESSIVE,
    DENOISE_ONLY,
    FAR,
    NEAR,
    Profile,
    get as get_profile,
    names as profile_names,
)

__all__ = [
    "AGGRESSIVE",
    "DENOISE_ONLY",
    "FAR",
    "NEAR",
    "Profile",
    "get_profile",
    "profile_names",
    "rms",
    "peak",
    "to_db",
    "percentile",
    "pre_emphasis",
    "soft_limit",
    "noise_gate",
    "normalize_gain",
    "process",
]

_EPS = 1e-12

#: 空输入的统一返回值。返回空数组而不是原对象，是为了保证
#: 无论输入是 list、None 还是空数组，出口类型始终是 float32 ndarray。
_EMPTY = np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- 基础度量
def _as_f64(audio):
    """把任意输入转成 float64 一维数组；转不动就返回 None。"""
    if audio is None:
        return None
    try:
        a = np.asarray(audio, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if a.size == 0:
        return None
    if not np.all(np.isfinite(a)):
        # 含 NaN/Inf：先把坏点归零，进度比报错重要
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def rms(audio):
    """均方根。空输入返回 0.0。"""
    a = _as_f64(audio)
    if a is None:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def peak(audio):
    """绝对峰值。空输入返回 0.0。"""
    a = _as_f64(audio)
    if a is None:
        return 0.0
    return float(np.max(np.abs(a)))


def to_db(level, floor_db=-120.0):
    """幅度比转 dB。0 或负数返回地板值，不做 -inf。"""
    try:
        x = float(level)
    except (TypeError, ValueError):
        return floor_db
    if x <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(x))


# ---------------------------------------------------------------- 分位数
def percentile(values, p):
    """线性插值分位数，与 numpy 的默认方式一致，但公式写死。

    pos = (n - 1) * p / 100
    lo = floor(pos)，hi = ceil(pos)，结果 = v[lo] + (v[hi] - v[lo]) * (pos - lo)

    C++ 版用同一公式，所以两边逐位相同。
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(a[0])
    q = min(100.0, max(0.0, float(p)))
    s = np.sort(a)                      # 稳定排序，重复值不改变结果
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


# ---------------------------------------------------------------- 处理单元
def pre_emphasis(audio, coeff=0.55, mix=0.38):
    """一阶高通与原信号按比例混合，用来补回远场丢失的辅音。

    空气对 2 kHz 以上的吸收随距离明显增加，/s/ /f/ /θ/ 这几个音最先糊掉。
    纯高通会把低频掏空、声音发尖，所以只取 38% 的补偿量混回去。

    coeff=0 或 mix=0 时原样返回（近场档就走这条路）。

    注意：出口会把电平缩回输入的电平。预加重的职责是改频谱形状，不是改音量；
    如果它顺手把电平降了，后面 normalize_gain 就得用更大的倍数去补，
    很容易一头撞到 max_gain 上、白白浪费动态范围。
    """
    a = _as_f64(audio)
    if a is None:
        return _EMPTY.copy()
    c = float(coeff)
    m = float(mix)
    if c <= 0.0 or m <= 0.0:
        return a.astype(np.float32)
    hi = a.copy()
    hi[1:] = a[1:] - c * a[:-1]         # hi[0] 保持原样
    y = (1.0 - m) * a + m * hi

    r0 = math.sqrt(float(np.mean(a * a)))
    r1 = math.sqrt(float(np.mean(y * y)))
    if r1 > _EPS and r0 > _EPS:
        y = y * (r0 / r1)               # 只改频谱，不改电平
    return y.astype(np.float32)


def soft_limit(audio, knee=0.85):
    """峰值软限幅。超过拐点用 tanh 压缩，不产生硬削波的方波谐波。"""
    a = _as_f64(audio)
    if a is None:
        return _EMPTY.copy()
    k = float(knee)
    if k <= 0.0:
        return a.astype(np.float32)
    pk = float(np.max(np.abs(a)))
    if pk <= k:
        return a.astype(np.float32)
    y = np.tanh(a / k) * k
    return y.astype(np.float32)


def normalize_gain(audio, target=0.12, max_gain=6.0):
    """把整段拉到目标电平。只放大不缩小，已经够响的原样返回。

    只放大是刻意的：近场录音走这一档不该被压小；远场本来就要放大。
    缩小交给调用方自己决定，库不替它做主。
    """
    a = _as_f64(audio)
    if a is None:
        return _EMPTY.copy()
    cur = float(np.sqrt(np.mean(a * a)))
    if cur <= _EPS:
        return a.astype(np.float32)
    gain = float(target) / cur
    if gain <= 1.0:
        return a.astype(np.float32)
    gain = min(gain, float(max_gain))
    y = a * gain
    pk = float(np.max(np.abs(y)))
    if pk > 0.97:
        y = y * (0.97 / pk)
    return y.astype(np.float32)


def noise_gate(audio, rate=16000, frame_ms=20.0, attenuation=0.30):
    """帧级噪声门：只压静音段与稳定底噪，语音帧一个样本都不动。

    为什么不用谱减法：谱减在低 SNR 下会产生音乐噪声（musical noise），
    听感上比原来的底噪更难受，而且会吃掉句尾的气声。帧级门限虽然粗糙，
    但它对语音是零损伤的——这一点在 ASR 前端比「降得干净」更重要。

    判决用两簇聚类而不是固定分位数，理由见 _split_energy。
    """
    a = _as_f64(audio)
    if a is None:
        return _EMPTY.copy()
    frame = int(round(float(rate) * float(frame_ms) / 1000.0))
    if frame < 32:
        frame = 32
    n = a.size // frame
    if n < 2:
        return a.astype(np.float32)

    body = a[: n * frame].reshape(n, frame)
    energy = np.sqrt(np.mean(body * body, axis=1))
    split = _split_energy(energy)
    if split is None:
        return a.astype(np.float32)
    thr, _, _ = split

    quiet = energy < thr
    gain = np.ones(n, dtype=np.float64)
    gain[quiet] = float(attenuation)

    out = body * gain[:, None]
    tail = a[n * frame:]
    y = np.concatenate([out.ravel(), tail])
    # 不做峰值保护：门控只会把样本乘小，输入峰值多大输出就多大。
    # 在这里统一缩放会把语音帧也改掉，「语音零损伤」的承诺就破了。
    return y.astype(np.float32)



def _split_energy(energy):
    """把帧能量分成「静音簇」和「语音簇」，返回 (阈值, 静音均值, 语音均值)。

    为什么不用固定分位数：第 10 分位落在静音帧自身的分布里，门限会横切
    在静音段中间，一半静音帧被判静音、一半漏掉。漏掉的那部分随后被增益
    放大几十倍，底噪反而比处理前更响——这正是「测量上降了 10 dB、
    听起来更吵」的成因。

    两簇一维 k-means 只需要几轮迭代，比调分位数可靠：
    找到的阈值落在两簇之间，而不是某一簇的内部。
    """
    e = np.asarray(energy, dtype=np.float64).ravel()
    if e.size < 3:
        return None
    lo = float(np.percentile(e, 10))
    hi = float(np.percentile(e, 90))
    if hi <= lo * 1.5:
        # 两簇分不开：整段能量本来就均匀，没有可判的静音段
        return None
    thr = (lo + hi) / 2.0
    c_lo, c_hi = lo, hi
    for _ in range(12):
        quiet = e < thr
        if not np.any(quiet) or np.all(quiet):
            return None
        c_lo = float(np.mean(e[quiet]))
        c_hi = float(np.mean(e[~quiet]))
        new_thr = (c_lo + c_hi) / 2.0
        if abs(new_thr - thr) < 1e-9:
            thr = new_thr
            break
        thr = new_thr
    # 两簇中心必须真的分得开，否则「降噪」只是在锯信号
    if c_hi < c_lo * 2.0:
        return None
    return thr, c_lo, c_hi


def _quiet_mask(audio, rate, frame_ms, enabled):
    """标出「静音帧」，返回样本级 bool 掩码；不该做门控时返回 None。

    判决逻辑与 noise_gate 完全一致，只是不在这里动手，留给 process 统一结算：
    增益要跳过这些帧，降噪也要针对这些帧，两件事必须用同一份判决，
    否则会出现「语音按整体电平算了增益、底噪又按别处电平压」的错位。
    """
    if not enabled:
        return None
    a = _as_f64(audio)
    if a is None:
        return None
    frame = int(round(float(rate) * float(frame_ms) / 1000.0))
    if frame < 32:
        frame = 32
    n = a.size // frame
    if n < 2:
        return None

    body = a[: n * frame].reshape(n, frame)
    energy = np.sqrt(np.mean(body * body, axis=1))
    split = _split_energy(energy)
    if split is None:
        return None
    thr, _, _ = split

    quiet = energy < thr
    mask = np.zeros(a.size, dtype=bool)
    for i in np.nonzero(quiet)[0]:
        mask[i * frame:(i + 1) * frame] = True
    return mask


def process(audio, profile=FAR, rate=16000):
    """完整处理链：高频补偿 → 增益归一 → 静音段压制 → 软限幅。

    参数可以是 Profile 实例，也可以是预设名字（"near"/"far"/...）。

    为什么把增益和降噪放在一起算，而不是「先降噪再整体放大」：

        先降噪再放大，底噪会被放大倍数原样抬回去。20 倍增益下，先把底噪压到
        0.3 倍、再整体乘 20，净效果是底噪变响 6 倍——测量上「降了 10 dB」，
        实际听起来更吵。这是上一版引擎的真实毛病。

        所以这里让增益只作用在语音帧上，静音帧走独立路径：
        静音样本先按 gate_attenuation/gain 缩放，随后随整段再乘 gain，
        两者抵消，最终倍数恒为 gate_attenuation，与增益多大无关。
        这才是「放大人声、不放大噪声」。
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    a = _as_f64(audio)
    if a is None:
        return _EMPTY.copy()

    y = pre_emphasis(a.astype(np.float32),
                     coeff=profile.pre_emph, mix=profile.pre_emph_mix)

    quiet = _quiet_mask(y, rate, profile.gate_frame_ms, profile.gate_enabled)

    # 增益按「语音部分」的电平算，不算静音段，否则静音越多增益算得越大
    gain = 1.0
    if profile.target_rms > 0.0:
        if quiet is not None and not quiet.all():
            cur = float(np.sqrt(np.mean(y[~quiet].astype(np.float64) ** 2)))
        else:
            cur = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if cur > _EPS:
            g = float(profile.target_rms) / cur
            if g > 1.0:
                gain = min(g, float(profile.max_gain))

    out = y.astype(np.float64) * gain

    if quiet is not None:
        # 静音段的「最终倍数」固定为 gate_attenuation，与增益无关。
        # 语音抬多少倍都不影响底噪；底噪永远净降 10 dB 左右。
        # 若只封顶到 1/gain（净倍数 1），底噪顶多回到原来的水平，
        # 那只是「没更吵」，谈不上降噪——这正是上一版的毛病。
        out[quiet] *= float(profile.gate_attenuation) / max(gain, 1.0)

    y = soft_limit(out, knee=profile.limit_knee)
    return y
