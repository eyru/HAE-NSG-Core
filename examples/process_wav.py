"""最小示例：把一个 wav 处理一遍，写出左右声道对照。

左声道 = 处理前，右声道 = 处理后。用播放器听的时候可以关掉一个声道对比。

    python examples/process_wav.py 你的录音.wav
    python examples/process_wav.py 你的录音.wav far
    python examples/process_wav.py 你的录音.wav aggressive out.wav
"""

import os
import sys

# 允许不安装、直接从仓库根目录跑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hae_nsg


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("可用的预设：%s" % ", ".join(hae_nsg.profile_names()))
        return 2

    src = sys.argv[1]
    profile = sys.argv[2] if len(sys.argv) > 2 else "far"
    if len(sys.argv) > 3:
        dst = sys.argv[3]
    else:
        base, ext = os.path.splitext(src)
        dst = base + "_" + profile + (ext or ".wav")

    info = hae_nsg.read_info(src)
    print("输入  %s" % src)
    print("      采样率 %d Hz，%d 声道，%.2f 秒"
          % (info.rate, info.channels, info.samples / float(info.rate)))
    if info.channels != 1:
        print("      注意：本库只处理单声道，多声道会被当成单声道算")

    audio, rate = hae_nsg.read_wav(src)
    before_rms = hae_nsg.rms(audio)
    before_peak = hae_nsg.peak(audio)

    out = hae_nsg.process(audio, profile, rate=rate)

    after_rms = hae_nsg.rms(out)
    after_peak = hae_nsg.peak(out)
    print("预设  %s" % profile)
    print("      RMS  %.5f -> %.5f  (%+.1f dB)"
          % (before_rms, after_rms, hae_nsg.to_db(after_rms / (before_rms + 1e-20))))
    print("      峰值 %.3f -> %.3f" % (before_peak, after_peak))

    # 左右声道对照：更方便直接听差别
    hae_nsg.write_wav_stereo_compare(dst, audio, out, rate)
    print("输出  %s（左=处理前，右=处理后）" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
