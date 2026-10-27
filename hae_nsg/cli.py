"""命令行：给一个 WAV，输出处理好的一版，可选并排 A/B 对比文件。

    python -m hae_nsg.cli in.wav -o out.wav --profile far
    python -m hae_nsg.cli in.wav -o out.wav --ab
    python -m hae_nsg.cli --print-profiles
"""

from __future__ import annotations

import argparse
import os
import sys

from .core import peak, process, rms, to_db, profile_names
from .wav import read_wav, write_wav, write_wav_stereo_compare


def _fmt(x):
    return "%.1f dB" % x


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hae_nsg",
        description="HAE-NSG Core：降噪 + 增益前处理",
    )
    ap.add_argument("input", nargs="?", help="输入 WAV")
    ap.add_argument("-o", "--output", help="输出 WAV（默认在输入名后加 _nsg）")
    ap.add_argument("-p", "--profile", default="far",
                    choices=profile_names(), help="参数预设（默认 far）")
    ap.add_argument("--ab", action="store_true",
                    help="额外输出一个左右声道对比文件：左=原声，右=处理后")
    ap.add_argument("--print-profiles", action="store_true",
                    help="列出所有预设后退出")
    args = ap.parse_args(argv)

    if args.print_profiles:
        for n in profile_names():
            print(n)
        return 0

    if not args.input:
        ap.error("缺少输入文件")

    audio, rate = read_wav(args.input)
    out = process(audio, args.profile, rate=rate)

    dst = args.output
    if not dst:
        base, ext = os.path.splitext(args.input)
        dst = base + "_nsg" + (ext or ".wav")
    write_wav(dst, out, rate)

    print("输入  %s" % args.input)
    print("  时长 %.2fs  采样率 %d  RMS %s  峰值 %s"
          % (audio.size / float(rate), rate,
             _fmt(to_db(rms(audio))), _fmt(to_db(peak(audio)))))
    print("输出  %s  (profile=%s)" % (dst, args.profile))
    print("  RMS %s  峰值 %s"
          % (_fmt(to_db(rms(out))), _fmt(to_db(peak(out)))))

    if args.ab:
        base, ext = os.path.splitext(dst)
        ab = base + "_ab.wav"
        write_wav_stereo_compare(ab, audio, out, rate=rate)
        print("对比  %s  （左=原声，右=处理后）" % ab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
