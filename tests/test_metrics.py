"""端到端指标测试：处理完之后，到底有没有变好。

单元测试保证「每一小步算得对」，这个文件保证「拼起来真的有用」——
这是两件不同的事。上一版引擎就是每步都对、整体把语音压没了。

指标定义：
    底噪下降率 = 1 - 处理后静音段 RMS / 处理前静音段 RMS
    语音保留率 = 语音段中未被改动的样本占比（相对误差 < 1e-6）
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hae_nsg as H  # noqa: E402

RATE = 16000


def build_scene(kind, rate=RATE):
    """造一段「底噪 + 说话」的场景。

    kind:
        near  贴麦说话，底噪很小
        far   四米外说话，声音衰减到 1/20，底噪不变
        quiet 极小声，需要强制放大
        noisy 底噪很大的嘈杂环境
    """
    rng = np.random.RandomState(7)
    dur = 3.0
    t = np.arange(int(dur * rate)) / rate

    # 语音：三个音节，带包络与共振峰，中间有停顿
    speech = np.zeros_like(t)
    for start, length in ((0.3, 0.55), (1.1, 0.5), (2.0, 0.6)):
        i0 = int(start * rate)
        i1 = int((start + length) * rate)
        seg = t[i0:i1] - start
        env = np.sin(np.pi * seg / length) ** 0.7          # 音节包络
        voice = (np.sin(2 * np.pi * 165 * seg)
                 + 0.55 * np.sin(2 * np.pi * 480 * seg)
                 + 0.30 * np.sin(2 * np.pi * 1150 * seg)
                 + 0.15 * np.sin(2 * np.pi * 2400 * seg))
        speech[i0:i1] = voice * env

    amp = {"near": 0.25, "far": 0.012, "quiet": 0.004, "noisy": 0.25}[kind]
    floor = {"near": 0.002, "far": 0.002, "quiet": 0.002, "noisy": 0.030}[kind]

    noise = rng.randn(t.size) * floor
    x = speech * amp + noise
    return x.astype(np.float32)


def silent_slice(kind):
    """返回 (起点, 终点) 索引，选一个明确的纯底噪区间。"""
    # 三段话分别在 0.3-0.85、1.1-1.6、2.0-2.6，其余是纯底噪
    a = 0.9 if kind != "noisy" else 0.9
    return int(a * RATE), int((a + 0.15) * RATE)


def speech_slice():
    return int(0.35 * RATE), int(0.80 * RATE)


class TestNoiseReduction(unittest.TestCase):
    def test_floor_drops(self):
        for kind in ("near", "far", "noisy"):
            x = build_scene(kind)
            y = H.process(x, kind if kind != "noisy" else "far")
            s0, s1 = silent_slice(kind)
            before = H.rms(x[s0:s1])
            after = H.rms(y[s0:s1])
            drop = 1.0 - after / max(before, 1e-9)
            self.assertGreater(drop, 0.30, "%s 底噪只降了 %.0f%%" % (kind, drop * 100))

    def test_no_vad_like_damage_on_speech(self):
        # 语音段必须基本原样：允许极少量样本因噪声门判错被压，但不许整段缩放
        x = build_scene("near")
        y = H.process(x, "near")
        s0, s1 = speech_slice()
        seg_x, seg_y = x[s0:s1], y[s0:s1]
        scale = float(H.rms(seg_y)) / max(float(H.rms(seg_x)), 1e-9)
        self.assertGreater(scale, 0.85, "语音段整体被压到了 %.2f 倍" % scale)


class TestGain(unittest.TestCase):
    def test_far_scene_reaches_target(self):
        # target_rms 是对「语音部分」说的。静音段被故意压掉，所以整段 RMS
        # 必然低于目标值，用整段去比会永远差一截。
        x = build_scene("far")
        y = H.process(x, "far")
        s0, s1 = speech_slice()
        self.assertAlmostEqual(float(H.rms(y[s0:s1])), H.FAR.target_rms, delta=0.06)

    def test_far_is_meaningfully_louder(self):
        x = build_scene("far")
        y = H.process(x, "far")
        gain_db = float(H.to_db(H.rms(y))) - float(H.to_db(H.rms(x)))
        self.assertGreater(gain_db, 15.0, "远场只抬了 %.1f dB" % gain_db)

    def test_near_untouched_when_already_loud(self):
        x = build_scene("near")
        y = H.process(x, "near")
        delta = abs(float(H.to_db(H.rms(y))) - float(H.to_db(H.rms(x))))
        self.assertLess(delta, 6.0)

    def test_quiet_needs_aggressive(self):
        x = build_scene("quiet")
        normal = H.process(x, "far")
        loud = H.process(x, "aggressive")
        self.assertGreater(float(H.rms(loud)), float(H.rms(normal)))

    def test_denoise_keeps_volume(self):
        x = build_scene("near")
        y = H.process(x, "denoise")
        delta = abs(float(H.to_db(H.rms(y))) - float(H.to_db(H.rms(x))))
        self.assertLess(delta, 1.0)


class TestSafety(unittest.TestCase):
    def test_never_clips(self):
        for kind in ("near", "far", "quiet", "noisy"):
            for name in H.profile_names():
                y = H.process(build_scene(kind), name)
                self.assertLessEqual(float(H.peak(y)), 1.0,
                                     "%s/%s 输出爆了" % (kind, name))

    def test_output_is_finite(self):
        for kind in ("near", "far", "quiet", "noisy"):
            y = H.process(build_scene(kind), "far")
            self.assertTrue(np.isfinite(y).all())

    def test_dtype_is_float32(self):
        y = H.process(build_scene("near"), "near")
        self.assertEqual(y.dtype, np.float32)

    def test_idempotent_on_level(self):
        # 处理过一遍的音频再处理一次，电平不该再涨一大截
        x = build_scene("far")
        once = H.process(x, "far")
        twice = H.process(once, "far")
        delta_db = abs(float(H.to_db(H.rms(twice))) - float(H.to_db(H.rms(once))))
        self.assertLess(delta_db, 3.0, "二次处理又抬了 %.1f dB" % delta_db)

    def test_length_preserved(self):
        x = build_scene("near")
        for name in H.profile_names():
            self.assertEqual(H.process(x, name).size, x.size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
