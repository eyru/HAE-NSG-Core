"""环境噪声底跟踪。

VAD 用固定门限在两种场景下都会失效：安静房间里 0.008 太高（把气声当静音），
嘈杂环境里 0.008 又太低（把噪声当语音，一句话永远切不完）。

所以门限跟着环境走：持续记录短时能量，取低分位数作为「当前环境的底」，
再乘一个倍数得到门限。用分位数而不是均值，是为了不被偶发的响声带偏。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .core import percentile, rms

__all__ = ["NoiseFloor"]

_DEFAULT_BASE = 0.008
_DEFAULT_FACTOR = 2.4


class NoiseFloor:
    """滑窗噪声底估计。

    size 是窗口长度（单位：帧）。按 20 ms 一帧、16000 Hz 算，
    180 帧 ≈ 3.6 秒，足够覆盖一次换气停顿，又不会慢到跟不上环境变化。
    """

    def __init__(self, size=180, percentile_=20.0, factor=_DEFAULT_FACTOR,
                 base=_DEFAULT_BASE):
        if size < 2:
            raise ValueError("size 至少为 2")
        self._hist = deque(maxlen=int(size))
        self._pct = float(percentile_)
        self._factor = float(factor)
        self._base = float(base)

    # ------------------------------------------------------------ 更新
    def update(self, value):
        """塞进一个短时能量值，返回更新后的噪声底。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.floor()
        if not np.isfinite(v) or v < 0.0:
            return self.floor()
        self._hist.append(v)
        return self.floor()

    def update_audio(self, audio, rate=16000, frame_ms=20.0):
        """直接塞一段音频，按帧算能量后逐帧更新。"""
        a = np.asarray(audio, dtype=np.float64).ravel() if audio is not None else None
        if a is None or a.size == 0:
            return self.floor()
        frame = max(32, int(round(rate * frame_ms / 1000.0)))
        n = a.size // frame
        if n == 0:
            return self.floor()
        body = a[: n * frame].reshape(n, frame)
        energy = np.sqrt(np.mean(body * body, axis=1))
        for e in energy:
            self.update(float(e))
        return self.floor()

    # ------------------------------------------------------------ 查询
    def floor(self):
        """当前噪声底。窗口为空时返回 0.0。"""
        if not self._hist:
            return 0.0
        return percentile(np.fromiter(self._hist, dtype=np.float64), self._pct)

    def threshold(self, base=None, factor=None):
        """动态门限：不低于 base，噪声高时按倍数抬高。

        base 是「绝对下限」——再安静的房间也不该低于它，否则风扇声、
        键盘声都会被当成语音。factor 是「相对倍数」——底噪抬高时门限
        跟着抬，避免一整段噪声被切成无数个假句子。
        """
        b = self._base if base is None else float(base)
        f = self._factor if factor is None else float(factor)
        try:
            return max(b, self.floor() * f)
        except Exception:
            return b

    def reset(self):
        """清空窗口（换设备、换房间、重新开麦时调用）。"""
        self._hist.clear()

    # ------------------------------------------------------------ 辅助
    def __len__(self):
        return len(self._hist)

    def __repr__(self):
        return ("NoiseFloor(size=%d, n=%d, floor=%.5f, threshold=%.5f)"
                % (self._hist.maxlen, len(self._hist), self.floor(),
                   self.threshold()))

    @property
    def factor(self):
        return self._factor

    @property
    def base(self):
        return self._base


def frame_energies(audio, rate=16000, frame_ms=20.0):
    """把音频切成帧并返回每帧 RMS，方便调用方自己看分布。"""
    a = np.asarray(audio, dtype=np.float64).ravel() if audio is not None else None
    if a is None or a.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame = max(32, int(round(rate * frame_ms / 1000.0)))
    n = a.size // frame
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    body = a[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(body * body, axis=1))
