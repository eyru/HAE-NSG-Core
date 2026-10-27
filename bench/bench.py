"""性能与效果基准。

跑法：

    python bench/bench.py

输出两段内容：
1. 速度：每个预设处理 1 秒音频要多久（越小于 1 秒越好，说明是实时可用的）。
2. 效果：用合成信号量化「语音保真」和「底噪压低」两件事。

这里用的是合成信号，所以数字只反映算法本身的取舍，不等于真实录音上的
识别率提升。真实效果请用你自己的录音跑 examples/process_wav.py 对比听感。
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hae_nsg  # noqa: E402

RATE = 16000


def synth_voice(seconds=3.0, level=0.05, rate=RATE):
    """造一段「像说话」的信号：基频滑动 + 几个谐波 + 包络起伏。"""
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float64) / rate
    f0 = 120.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    wave = np.zeros(n)
    for k, amp in enumerate((1.0, 0.55, 0.32, 0.18, 0.11), start=1):
        wave += amp * np.sin(k * phase)
    # 4 Hz 的音节包络，模拟说话时的一顿一顿
    env = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    wave = wave * env
    wave = wave / (np.max(np.abs(wave)) + 1e-12) * level
    return wave.astype(np.float32)


def synth_noise(seconds=3.0, level=0.008, rate=RATE, seed=7):
    """白噪声 + 一点低频嗡嗡声，比纯白噪声更接近真实底噪。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    white = rng.standard_normal(n) * level
    t = np.arange(n) / rate
    hum = 0.5 * level * np.sin(2 * np.pi * 50.0 * t)
    return (white + hum).astype(np.float32)


def snr_db(signal, noise):
    """按 RMS 算信噪比。signal 和 noise 必须等长同域。"""
    ps = float(np.mean(np.asarray(signal, dtype=np.float64) ** 2)) + 1e-20
    pn = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2)) + 1e-20
    return 10.0 * np.log10(ps / pn)


def bench_speed():
    print("=" * 62)
    print("速度：处理 1 秒音频需要多少毫秒")
    print("=" * 62)
    audio = synth_voice(seconds=3.0)
    for name in hae_nsg.profile_names():
        profile = hae_nsg.get_profile(name)
        # 先跑一遍预热，避免把首次调用的一次性开销算进去
        hae_nsg.process(audio, profile, RATE)
        t0 = time.perf_counter()
        hae_nsg.process(audio, profile, RATE)
        cost = (time.perf_counter() - t0) * 1000.0
        per_second = cost / 3.0
        print("  %-10s  3 秒音频耗时 %6.2f ms  折合每秒 %5.2f ms  RTF %.4f"
              % (name, cost, per_second, per_second / 1000.0))
    print()


def bench_quality():
    """分别看「有声段」和「纯噪声段」的电平变化。

    不能把语音和噪声拆开各自过一遍 process 再合起来算——增益是跟着信号
    电平走的非线性环节，拆开处理得到的数字没有意义。所以这里造一段
    「前半有声、后半只有底噪」的信号，整段处理，再分段统计。
    """
    print("=" * 62)
    print("效果：前半段有语音、后半段只有底噪，处理后各自变成多少")
    print("=" * 62)

    half = 1.5
    voice = np.concatenate([synth_voice(half, level=0.05),
                            np.zeros(int(half * RATE), dtype=np.float32)])
    noise = synth_noise(half * 2, level=0.008)
    mixed = (voice + noise).astype(np.float32)

    n_voice = int(half * RATE)
    silence_slice = slice(n_voice, len(mixed))
    voice_slice = slice(0, n_voice)

    def seg_rms(x, sl):
        seg = np.asarray(x[sl], dtype=np.float64)
        return float(np.sqrt(np.mean(seg ** 2))) + 1e-20

    base_v = seg_rms(mixed, voice_slice)
    base_s = seg_rms(mixed, silence_slice)

    for name in hae_nsg.profile_names():
        out = hae_nsg.process(mixed, hae_nsg.get_profile(name), RATE)
        v_gain = 20.0 * np.log10(seg_rms(out, voice_slice) / base_v)
        s_gain = 20.0 * np.log10(seg_rms(out, silence_slice) / base_s)
        net = v_gain - s_gain
        print("  %-10s  有声段 %+6.1f dB   底噪段 %+6.1f dB   "
              "净改善 %+5.1f dB   峰值 %.3f"
              % (name, v_gain, s_gain, net, float(np.max(np.abs(out)))))
    print("  （净改善 = 有声段提升 − 底噪段提升，正数表示人声离底噪更远了）")
    print()


def bench_far_field():
    print("=" * 62)
    print("远场：很小的声音（-46 dBFS）能抬到多少")
    print("=" * 62)
    quiet = synth_voice(seconds=2.0, level=0.005)
    for name in hae_nsg.profile_names():
        profile = hae_nsg.get_profile(name)
        out = hae_nsg.process(quiet, profile, RATE)
        before = float(np.sqrt(np.mean(quiet.astype(np.float64) ** 2)))
        after = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        gain = after / (before + 1e-20)
        print("  %-10s  RMS %.5f -> %.5f  放大 %.1f 倍（%+.1f dB）  峰值 %.3f"
              % (name, before, after, gain, 20.0 * np.log10(gain),
                 float(np.max(np.abs(out)))))
    print()


if __name__ == "__main__":
    bench_speed()
    bench_quality()
    bench_far_field()
