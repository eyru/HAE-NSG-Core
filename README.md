# HAE-NSG Core

**Hearizon Audiology Engine — Noise-Suppression Gain Core**

语音识别之前的降噪与增益环节：把「听不清」的录音变成 ASR 听得清的录音，同时不损伤本来
就清楚的录音。

纯 numpy 实现，无模型、无训练、无外部依赖。一段 3 秒音频耗时 **0.5–1.5 毫秒**。

```python
import hae_nsg

audio, rate = hae_nsg.read_wav("meeting.wav")
out = hae_nsg.process(audio, "far", rate=rate)   # 远场预设
hae_nsg.write_wav("meeting_clean.wav", out, rate)
```

---

## 这个项目解决什么问题

语音识别在两种情况下会显著变差：

1. **声音太小**。笔记本内置麦克风在四五米外收到的语音电平只有 -40 dBFS 上下，
   ASR 的输入归一化不足以把它救回来。
2. **底噪太吵**。空调、风扇、机箱、远处人声把语音淹掉，VAD 反复误触发。

常见做法是先降噪再整体放大。这个顺序有个陷阱：**放大是线性的，底噪会跟着一起被放大**。
先把底噪压到 0.3 倍、再整体乘 20，净效果是底噪反而变响 6 倍——测量上「降了 10 dB」，
听感上更吵。

HAE-NSG 的做法是让增益**只作用在语音帧上**，静音帧走独立路径：

```
语音帧：× gain            （gain 可以到 40 倍甚至 120 倍）
静音帧：× gate_attenuation（固定 0.30，与 gain 无关）
```

所以无论把语音抬高多少倍，底噪的绝对电平都不会超过输入。这才是「放大人声、不放大噪声」。

---

## 四个预设

| 预设 | 适用场景 | 目标电平 | 增益上限 | 高频补偿 |
|---|---|---|---|---|
| `near` | 贴着电脑说话 | 0.12 | 6× | 无 |
| `far` | 四五米以上 | 0.22 | 40× | 有 |
| `aggressive` | 麦克风很远 / 说话很轻 | 0.38 | 120× | 有 |
| `denoise` | 只降噪，不改音量 | — | 1× | 无 |

用合成信号实测（前半段语音+底噪、后半段只有底噪，16 kHz）：

| 预设 | 有声段 | 底噪段 | 净改善 | 输出峰值 |
|---|---|---|---|---|
| `near` | +13.3 dB | −10.5 dB | **+23.8 dB** | 0.339 |
| `far` | +18.1 dB | −9.0 dB | **+27.1 dB** | 0.649 |
| `aggressive` | +21.7 dB | −8.7 dB | **+30.4 dB** | 0.742 |
| `denoise` | −0.3 dB | −10.5 dB | **+10.2 dB** | 0.071 |

「净改善」= 有声段提升 − 底噪段提升，正数表示人声离底噪更远了。峰值都在 0.75 以下，
不会削顶。

> 这是合成信号上的数字，只反映算法本身的取舍，**不等于**真实录音上的识别率提升。
> 真实效果请用自己的录音跑一遍对比听感。

---

## 安装

```bash
pip install numpy
git clone https://github.com/eyru/HAE-NSG-Core.git
cd HAE-NSG-Core
pip install -e .
```

只有 numpy 一个依赖。

---

## 用法

### 作为库

```python
import hae_nsg

# 从文件
audio, rate = hae_nsg.read_wav("in.wav")
out = hae_nsg.process(audio, "far", rate=rate)
hae_nsg.write_wav("out.wav", out, rate)

# 直接给 numpy 数组（float32 / float64，单声道，范围 -1..1）
import numpy as np
x = np.zeros(16000, dtype=np.float32)
y = hae_nsg.process(x, "near", rate=16000)

# 自定义参数
from hae_nsg import Profile
mine = Profile(name="custom", target_rms=0.30, max_gain=50.0,
               pre_emph=0.6, pre_emph_mix=0.38,
               gate_enabled=True, gate_frame_ms=20.0,
               gate_attenuation=0.25, limit_knee=0.88)
y = hae_nsg.process(x, mine, rate=16000)
```

`process` **返回新数组**，不修改输入。

### 命令行

```bash
hae-nsg in.wav                          # 默认 far 预设，输出 in_nsg.wav
hae-nsg in.wav -o out.wav -p near       # 指定输出与预设
hae-nsg in.wav --ab -o ab.wav -p far    # 左右声道对照（左=处理前，右=处理后）
hae-nsg --print-profiles                # 列出所有预设
```

### 在 C++ 里用

`include/hae_nsg.h` 是同一套算法的头文件实现（single-header，C++17，不依赖 numpy）。
两边输出逐样本一致，误差 < 1e-6，`tests/run_parity.py` 会自动验证。

```cpp
#include "hae_nsg.h"

std::vector<float> audio = /* 16 kHz 单声道 */;
hae::nsg::process(audio, *hae::nsg::profileByName("far"), 16000);
```

---

## 建议接在 VAD 之前

HAE-NSG 只做「让声音更清楚」，不做切句。推荐链路：

```
麦克风 → HAE-NSG（本库）→ VAD 切句 → ASR
```

先增益再送 VAD 的好处是：小声音被抬起来之后，VAD 的能量门限才有意义。
反过来先 VAD 再增益，远场说话会被直接判成静音丢掉。

配套的 VAD 在 [HAE-VAD-Core](https://github.com/eyru/HAE-VAD-Core)。

---

## 参数怎么调

| 参数 | 作用 | 调大会怎样 |
|---|---|---|
| `target_rms` | 想把语音抬到多大 | 增益更大，同时更容易碰到 `max_gain` |
| `max_gain` | 增益硬上限 | 能救更小的声音，但底噪也会被抬 |
| `pre_emph` | 高频补偿强度（0–0.95） | 齿音更清楚，过大时声音发尖 |
| `pre_emph_mix` | 补偿混回原始的比例 | 同上，但更平缓 |
| `gate_attenuation` | 静音帧保留多少能量 | 越大底噪越响；太小会有「抽气感」 |
| `limit_knee` | 软限幅拐点 | 越小越不容易削顶，但大音量会被压 |

`target_rms` 是目标不是强制：**只放大不缩小**，已经够响的音频原样返回。
所以近场录音误用远场预设也不会被压小，最多是被多补一点高频。

---

## 实测速度

3 秒 16 kHz 音频，单线程，CPython 3.12 + numpy：

| 预设 | 耗时 | 折合每秒音频 | 实时率 RTF |
|---|---|---|---|
| `denoise` | 0.53 ms | 0.18 ms | 0.0002 |
| `near` | 0.84 ms | 0.28 ms | 0.0003 |
| `far` | 1.25 ms | 0.42 ms | 0.0004 |
| `aggressive` | 1.49 ms | 0.50 ms | 0.0005 |

比 ASR 推理快两三个数量级，实时流里可以放心地每句都跑。

---

## 和别的降噪方案比

这个库**不是**通用降噪器，定位很窄：

| 方案 | 定位 | 代价 |
|---|---|---|
| **HAE-NSG** | 为 ASR 前置设计：抬语音 + 压静音段 | 不做频域降噪，稳态噪声压不掉 |
| 谱减法 / `noisereduce` | 频域稳态噪声抑制 | 需要噪声估计，容易产生音乐噪声；实测在语音上会削掉辅音 |
| RNNoise | 神经网络降噪 | 需要模型文件，CPU 开销大一个量级，48 kHz 原生 |
| WebRTC NS | 通话降噪 | 为 16 kHz 窄带通话调优，对 ASR 增益无帮助 |

如果你的问题是「背景一直有空调声」，谱减法更合适；如果问题是「人离得远、声音小、VAD 抓不到」，
这个库更合适。两者也可以串起来用。

> 关于谱减法：本项目最早的版本试过，在真实录音上把信噪比从 15.5 dB 打到 3.83 dB——
> 测量假象叠加语音损伤。原因和详细数据写在 [docs/ALGORITHM.md](docs/ALGORITHM.md)。

---

## 已知局限

- **不做频域降噪**。稳态噪声（空调、风扇）只能靠静音帧门限压掉一部分，语音段内部的噪声
  原样保留。要更强需要接谱减法或神经网络降噪。
- **单声道、假设 16 kHz 或 48 kHz**。其他采样率能用，但 `gate_frame_ms` 对应的帧长会变。
- **不处理立体声**。传双声道会被当成单声道算，结果是错的。
- **不能救回被削顶的录音**。输入峰值到 1.0 的素材，信息已经丢了。
- **没有 VAD**。整段所有帧都会被判断，静音段是「按能量判断」的，不是语义判断。

---

## 测试

```bash
python -m unittest discover -s tests -v     # 60 项单元测试
python tests/run_parity.py                  # Python 与 C++ 逐样本对拍
python bench/bench.py                       # 速度与效果基准
```

对拍需要 MSVC（`cl.exe`）。没装的话 `run_parity.py` 会跳过并说明原因。

---

## 许可证

MIT。见 [LICENSE](LICENSE)。

---

## 关于

Hearizon Audiology Engine 是 [Hearizon Sync](https://github.com/eyru/HearizonSync)
的听力学引擎部分，抽出来单独维护，方便别的项目直接用。

作者：何瑞程（北京语言大学 · 海南国际学院），联系 1585253788@qq.com。
