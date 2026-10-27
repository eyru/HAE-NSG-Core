"""Python 与 C++ 逐样本对拍。

两个实现是同一套算法的两份手写代码，最容易出的问题是「改一边忘一边」：
参数差一点、循环边界差一个、浮点顺序差一步，单看都能跑通，结果却不一样。
这个脚本把同一段输入喂给两边，要求每个输出样本的差都小于 1e-6。

跑法：
    python tests/run_parity.py            # 自动编译（需要 MSVC）后对拍
    python tests/run_parity.py --no-build # 复用已编译的 tests/parity 目录
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hae_nsg as H  # noqa: E402

BUILD_DIR = os.path.join(ROOT, "tests", "_parity_build")
EXE = os.path.join(BUILD_DIR, "parity_main.exe")

VS_VCVARS = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    r"\VC\Auxiliary\Build\vcvars64.bat"
)

TOL = 1e-6


def build():
    """用 cl 编译对拍程序。没有 MSVC 就返回 False，让调用方优雅跳过。"""
    if not os.path.isfile(VS_VCVARS):
        return False
    os.makedirs(BUILD_DIR, exist_ok=True)
    bat = os.path.join(BUILD_DIR, "build.bat")
    with open(bat, "w", encoding="mbcs", newline="\r\n") as f:
        f.write("@echo off\n")
        f.write('call "%s" >nul\n' % VS_VCVARS)
        f.write('cd /d "%s"\n' % ROOT)
        f.write(
            'cl /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /DNOMINMAX '
            '/D_CRT_SECURE_NO_WARNINGS /I include '
            '/Fe:"%s" /Fo:"%s\\\\" tests\\parity_main.cpp\n'
            % (EXE, BUILD_DIR)
        )
    p = subprocess.run(["cmd", "/c", bat], capture_output=True)
    ok = os.path.isfile(EXE)
    if not ok:
        sys.stderr.write((p.stdout + p.stderr).decode("mbcs", "replace")[-1500:])
    return ok


def signals():
    """一组覆盖各种情况的测试信号。"""
    rate = 16000
    rng = np.random.RandomState(20261028)
    out = {}

    # 1. 纯正弦
    t = np.arange(rate) / rate
    out["sine"] = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    # 2. 白噪声
    out["noise"] = (rng.randn(rate) * 0.02).astype(np.float32)

    # 3. 静音 + 语音（最接近真实用法）
    quiet = (rng.randn(int(0.5 * rate)) * 0.003).astype(np.float32)
    voice = (np.sin(2 * np.pi * 180 * t[: int(0.5 * rate)])
             + 0.5 * np.sin(2 * np.pi * 900 * t[: int(0.5 * rate)])) * 0.05
    out["mixed"] = np.concatenate([quiet, voice.astype(np.float32)])

    # 4. 极小声（考增益上限）
    out["tiny"] = (rng.randn(rate) * 0.0008).astype(np.float32)

    # 5. 接近满幅（考软限幅）
    out["loud"] = (np.sin(2 * np.pi * 220 * t) * 0.98).astype(np.float32)

    # 6. 全零（考除零保护）
    out["zero"] = np.zeros(rate, dtype=np.float32)

    # 7. 超短（考帧数不足）
    out["short"] = (np.sin(2 * np.pi * 500 * np.arange(200) / rate) * 0.2).astype(np.float32)

    # 8. 含静音间隙的三段话
    gap = np.zeros(int(0.3 * rate), dtype=np.float32)
    word = (np.sin(2 * np.pi * 250 * t[: int(0.25 * rate)]) * 0.08).astype(np.float32)
    out["gapped"] = np.concatenate([word, gap, word, gap, word])

    return out


def compare(tmp):
    cases = signals()
    failures = []
    checked = 0
    for name, sig in cases.items():
        for profile in H.profile_names():
            tag = "%s/%s" % (name, profile)
            src = os.path.join(tmp, "in.raw")
            dst = os.path.join(tmp, "out.raw")
            sig.astype(np.float32).tofile(src)

            p = subprocess.run([EXE, src, dst, profile, "16000"],
                               capture_output=True)
            if p.returncode != 0:
                failures.append("%s: exe 返回 %d %s"
                                % (tag, p.returncode,
                                   p.stderr.decode("mbcs", "replace").strip()))
                continue

            cpp = np.fromfile(dst, dtype=np.float32)
            py = H.process(sig.astype(np.float32), profile, rate=16000)

            if cpp.size != py.size:
                failures.append("%s: 长度 %d vs %d" % (tag, cpp.size, py.size))
                continue

            diff = np.max(np.abs(cpp.astype(np.float64) - py.astype(np.float64))) if py.size else 0.0
            checked += 1
            if diff > TOL:
                failures.append("%s: 最大差 %.3e（容差 %.0e）" % (tag, diff, TOL))

    return checked, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build or not os.path.isfile(EXE):
        if not build():
            print("跳过：找不到 MSVC（%s）" % VS_VCVARS)
            return 0

    # 工作目录放在工程内（D 盘），不放 %TEMP%（C 盘）：
    # 新编译的 exe 写 C 盘临时目录会被 ERRNO 5 拒绝，写自己旁边就正常。
    tmp = os.path.join(BUILD_DIR, "tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        checked, failures = compare(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("对拍失败 %d / %d：" % (len(failures), len(failures) + checked))
        for f in failures:
            print("  " + f)
        return 1

    print("对拍通过：%d 组，逐样本差均 < %.0e" % (checked, TOL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
