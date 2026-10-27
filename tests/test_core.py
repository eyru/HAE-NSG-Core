"""核心算法单元测试。

跑法：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hae_nsg as H  # noqa: E402


def tone(freq=440.0, dur=1.0, rate=16000, amp=0.3):
    n = int(dur * rate)
    return (np.sin(2 * np.pi * freq * np.arange(n) / rate) * amp).astype(np.float32)


def noise(dur=1.0, rate=16000, amp=0.01, seed=1):
    rng = np.random.RandomState(seed)
    return (rng.randn(int(dur * rate)) * amp).astype(np.float32)


def speech_like(rate=16000):
    """有起伏的类语音信号：低频包络 * 宽带成分。"""
    n = rate
    t = np.arange(n) / rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    base = np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 900 * t)
    return (base * env * 0.25).astype(np.float32)


class TestProfiles(unittest.TestCase):
    def test_names(self):
        self.assertEqual(H.profile_names(), ["aggressive", "denoise", "far", "near"])

    def test_get(self):
        for n in H.profile_names():
            self.assertEqual(H.get_profile(n).name, n)

    def test_get_case_insensitive(self):
        self.assertEqual(H.get_profile(" FAR ").name, "far")

    def test_get_unknown_raises(self):
        # 不静默回退：拼错的档位必须报出来，否则用户会以为参数生效了
        with self.assertRaises(ValueError):
            H.get_profile("nope")

    def test_near_no_preemphasis(self):
        self.assertEqual(H.NEAR.pre_emph, 0.0)

    def test_far_is_louder_than_near(self):
        self.assertGreater(H.FAR.target_rms, H.NEAR.target_rms)
        self.assertGreater(H.FAR.max_gain, H.NEAR.max_gain)

    def test_aggressive_highest_gain(self):
        self.assertGreater(H.AGGRESSIVE.max_gain, H.FAR.max_gain)

    def test_denoise_does_not_change_volume(self):
        self.assertEqual(H.DENOISE_ONLY.target_rms, 0.0)
        self.assertEqual(H.DENOISE_ONLY.pre_emph, 0.0)


class TestMetrics(unittest.TestCase):
    def test_rms_empty(self):
        self.assertEqual(H.rms([]), 0.0)

    def test_rms_full_scale_sine(self):
        # 满幅正弦的 RMS 是 1/sqrt(2)
        a = tone(amp=1.0)
        self.assertAlmostEqual(float(H.rms(a)), 1.0 / math.sqrt(2), places=3)

    def test_rms_dc(self):
        a = np.full(1000, 0.5, dtype=np.float32)
        self.assertAlmostEqual(float(H.rms(a)), 0.5, places=6)

    def test_peak_empty(self):
        self.assertEqual(H.peak([]), 0.0)

    def test_peak(self):
        self.assertAlmostEqual(float(H.peak(np.array([0.1, -0.7, 0.3], np.float32))), 0.7, places=6)

    def test_to_db_known(self):
        self.assertAlmostEqual(float(H.to_db(1.0)), 0.0, places=6)
        self.assertAlmostEqual(float(H.to_db(0.5)), -6.0206, places=3)

    def test_to_db_zero_is_floor(self):
        self.assertEqual(float(H.to_db(0.0)), -120.0)

    def test_percentile_median(self):
        # 经典插值中位数
        self.assertAlmostEqual(float(H.percentile([1, 2, 3, 4], 50)), 2.5)
        self.assertAlmostEqual(float(H.percentile([1, 2, 3], 50)), 2.0)

    def test_percentile_extremes(self):
        self.assertAlmostEqual(float(H.percentile([5, 1, 9], 0)), 1.0)
        self.assertAlmostEqual(float(H.percentile([5, 1, 9], 100)), 9.0)

    def test_percentile_empty(self):
        self.assertEqual(float(H.percentile([], 50)), 0.0)

    def test_percentile_single(self):
        self.assertEqual(float(H.percentile([7.0], 90)), 7.0)


class TestStages(unittest.TestCase):
    """注意：处理函数是纯函数，返回新数组，不原地改输入。"""

    def test_pre_emphasis_disabled(self):
        a = tone()
        out = H.pre_emphasis(a, 0.0, 0.0)
        np.testing.assert_allclose(a, out, atol=1e-9)

    def test_pre_emphasis_boosts_high_freq(self):
        # 预加重的本质是高通，高频成分应比低频被放大得更多（看谱不看电平）
        low = tone(200.0, amp=0.2)
        high = tone(3000.0, amp=0.2)
        lo = H.pre_emphasis(low, 0.55, 0.38)
        hi = H.pre_emphasis(high, 0.55, 0.38)
        # 电平被归一了，比较「改动量」：高频的波形变化远大于低频
        d_low = float(np.mean(np.abs(lo - low))) / 0.2
        d_high = float(np.mean(np.abs(hi - high))) / 0.2
        self.assertGreater(d_high, d_low)

    def test_pre_emphasis_keeps_level(self):
        # 预加重只改频谱形状，不该改音量；否则后级增益要白补一份
        a = tone(200.0, amp=0.2)
        out = H.pre_emphasis(a, 0.55, 0.38)
        self.assertAlmostEqual(float(H.rms(out)), float(H.rms(a)), delta=1e-4)

    def test_soft_limit_noop_below_knee(self):
        a = tone(amp=0.5)
        out = H.soft_limit(a, 0.97)
        np.testing.assert_allclose(a, out, atol=1e-6)

    def test_soft_limit_clamps_peak(self):
        a = tone(amp=1.0)
        out = H.soft_limit(a, 0.85)
        self.assertLessEqual(float(H.peak(out)), 0.86)

    def test_normalize_gain_boosts_quiet(self):
        a = tone(amp=0.02)
        out = H.normalize_gain(a, 0.2, 100.0)
        self.assertAlmostEqual(float(H.rms(out)), 0.2, places=3)

    def test_normalize_gain_never_attenuates(self):
        # 已经够响就不动，避免近场录音被压小
        a = tone(amp=0.6)
        before = float(H.rms(a))
        out = H.normalize_gain(a, 0.12, 6.0)
        self.assertAlmostEqual(float(H.rms(out)), before, places=6)

    def test_normalize_gain_respects_cap(self):
        a = tone(amp=0.001)
        out = H.normalize_gain(a, 0.5, 5.0)
        self.assertLessEqual(float(H.rms(out)), 0.001 * 5.0 + 1e-4)

    def test_normalize_gain_clamps_peak(self):
        a = tone(amp=0.05)
        out = H.normalize_gain(a, 0.6, 100.0)
        self.assertLessEqual(float(H.peak(out)), 0.9701)


class TestNoiseGate(unittest.TestCase):
    def _mixed(self):
        """前 0.3s 纯底噪，后面是平稳的响亮信号（代表语音）。"""
        rate = 16000
        quiet = noise(0.3, rate, 0.004, seed=3)
        loud = tone(300.0, dur=0.7, rate=rate, amp=0.3)
        return np.concatenate([quiet, loud]).astype(np.float32), rate

    def test_noise_floor_touched(self):
        # 底噪段能量应下降
        a, rate = self._mixed()
        head = int(0.3 * rate)
        before = float(H.rms(a[:head]))
        out = H.noise_gate(a, rate)
        after = float(H.rms(out[:head]))
        self.assertLess(after, before * 0.75)

    def test_speech_preserved(self):
        # 语音段一个样本都不许动
        a, rate = self._mixed()
        head = int(0.3 * rate)
        out = H.noise_gate(a, rate)
        np.testing.assert_allclose(a[head:], out[head:], atol=1e-6)

    def test_steady_signal_not_gated(self):
        # 稳态纯音：能量处处相同，「第 10 分位门限」会横切在信号中间。
        # 这种情况必须整体跳过，不能把内容当噪声压掉。
        a = tone(amp=0.05)
        out = H.noise_gate(a, 16000)
        self.assertAlmostEqual(float(H.rms(out)), float(H.rms(a)), delta=1e-4)

    def test_short_input_noop(self):
        a = tone(dur=0.02)
        out = H.noise_gate(a, 16000)
        np.testing.assert_allclose(a, out, atol=1e-7)


class TestProcess(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(H.process([], "far").size, 0)

    def test_dirty_inputs_do_not_raise(self):
        cases = [
            [],
            np.zeros(3, np.float32),
            np.array([np.nan, 0.1, np.inf], np.float32),
            np.zeros(16000, np.float32),
        ]
        for c in cases:
            for name in H.profile_names():
                H.process(np.asarray(c, np.float32).copy(), name)

    def test_far_reaches_target(self):
        a = tone(amp=0.02)
        b = H.process(a, "far")
        self.assertAlmostEqual(float(H.rms(b)), H.FAR.target_rms, delta=0.05)

    def test_aggressive_louder_than_far(self):
        a = tone(amp=0.01)
        far = H.process(a.copy(), "far")
        agg = H.process(a.copy(), "aggressive")
        self.assertGreater(float(H.rms(agg)), float(H.rms(far)))

    def test_denoise_keeps_level(self):
        a = speech_like()
        b = H.process(a.copy(), "denoise")
        self.assertAlmostEqual(float(H.rms(b)), float(H.rms(a)), delta=0.01)

    def test_output_never_clips(self):
        a = tone(amp=0.95)
        b = H.process(a, "aggressive")
        self.assertLessEqual(float(H.peak(b)), 1.0)

    def test_process_accepts_profile_object(self):
        a = tone(amp=0.02)
        b = H.process(a, H.FAR)
        self.assertAlmostEqual(float(H.rms(b)), H.FAR.target_rms, delta=0.05)

    def test_float64_input_ok(self):
        a = tone(amp=0.02).astype(np.float64)
        b = H.process(a, "far")
        self.assertTrue(np.isfinite(b).all())


class TestNoiseFloor(unittest.TestCase):
    def test_empty_floor(self):
        self.assertEqual(H.NoiseFloor().floor(), 0.0)

    def test_needs_two_samples(self):
        with self.assertRaises(ValueError):
            H.NoiseFloor(size=1)

    def test_floor_tracks_quiet(self):
        nf = H.NoiseFloor()
        for _ in range(50):
            nf.update(0.01)
        self.assertAlmostEqual(nf.floor(), 0.01, places=6)

    def test_floor_ignores_spikes(self):
        nf = H.NoiseFloor()
        for _ in range(100):
            nf.update(0.01)
        for _ in range(5):
            nf.update(0.9)
        self.assertLess(nf.floor(), 0.02)

    def test_threshold_has_base(self):
        nf = H.NoiseFloor(base=0.02)
        self.assertGreaterEqual(nf.threshold(), 0.02)

    def test_threshold_scales_with_floor(self):
        nf = H.NoiseFloor(base=0.001, factor=3.0)
        for _ in range(40):
            nf.update(0.01)
        self.assertAlmostEqual(nf.threshold(), 0.03, places=4)

    def test_reset(self):
        nf = H.NoiseFloor()
        for _ in range(20):
            nf.update(0.01)
        nf.reset()
        self.assertEqual(nf.floor(), 0.0)

    def test_update_audio(self):
        nf = H.NoiseFloor()
        nf.update_audio(noise(0.5, 16000, 0.01))
        self.assertGreater(nf.floor(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
