// HAE-NSG Core —— 降噪与增益前端（C++ 单头实现）
//
//     Hearizon Audiology Engine — Noise-Suppression Gain Core
//
// 用法：
//     #include "hae_nsg.h"
//     std::vector<float> audio = ...;          // -1..1 单声道
//     hae::nsg::process(audio, hae::nsg::kFar, 16000);
//
// 这个文件和 Python 版是同一套算法、同一套参数、同一个浮点顺序。
// tests/parity 会拿同一段输入跑两边，要求逐样本差 < 1e-6。
//
// 依赖：只有标准库。C++17。
//
// SPDX-License-Identifier: MIT

#ifndef HAE_NSG_H
#define HAE_NSG_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace hae {
namespace nsg {

// ---------------------------------------------------------------- 参数预设
struct Profile {
    const char* name;
    double target_rms;       // 目标电平，<=0 表示不做增益
    double max_gain;         // 增益硬上限
    double pre_emph;         // 预加重系数，0 表示不做高频补偿
    double pre_emph_mix;     // 补偿量与原始信号的混合比例
    bool gate_enabled;       // 是否启用帧级噪声门
    double gate_frame_ms;    // 噪声门分析帧长
    double gate_attenuation; // 噪声帧保留多少能量
    double limit_knee;       // 软限幅拐点
};

//: 近场：贴着电脑说话，只做轻度归一 + 噪声门
inline constexpr Profile kNear{
    "near", 0.12, 6.0, 0.0, 0.0, true, 20.0, 0.30, 0.97};

//: 远场：四五米以上，补高频 + 强增益 + 软限幅
inline constexpr Profile kFar{
    "far", 0.22, 40.0, 0.55, 0.38, true, 20.0, 0.30, 0.85};

//: 强制放大：麦克风很远、说话很轻（底噪会一起变大）
inline constexpr Profile kAggressive{
    "aggressive", 0.38, 120.0, 0.65, 0.38, true, 20.0, 0.30, 0.85};

//: 只降噪不改音量
inline constexpr Profile kDenoiseOnly{
    "denoise", 0.0, 1.0, 0.0, 0.0, true, 20.0, 0.30, 0.97};

namespace detail {

constexpr double kEps = 1e-12;
constexpr double kDefaultGateRatio = 1.1;   // 门限 = 噪声底 * 该倍数
constexpr double kClampForGain = 0.97;      // 增益后峰值上限
constexpr double kClampForGate = 0.98;      // 噪声门后峰值上限

inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace detail

// ---------------------------------------------------------------- 基础度量
inline double rms(const std::vector<float>& a) {
    if (a.empty()) return 0.0;
    double sum = 0.0;
    for (float v : a) {
        const double d = static_cast<double>(v);
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(a.size()));
}

inline double peak(const std::vector<float>& a) {
    double p = 0.0;
    for (float v : a) {
        const double d = std::fabs(static_cast<double>(v));
        if (d > p) p = d;
    }
    return p;
}

inline double toDb(double level, double floor_db = -120.0) {
    if (!(level > 0.0)) return floor_db;
    const double db = 20.0 * std::log10(level);
    return db < floor_db ? floor_db : db;
}

// ---------------------------------------------------------------- 分位数
// 线性插值分位数，公式与 Python 版逐字对应：
//   pos = (n-1) * p / 100,  lo = floor(pos),  hi = ceil(pos)
//   v = s[lo] + (s[hi] - s[lo]) * (pos - lo)
inline double percentile(std::vector<double>& values, double p) {
    const std::size_t n = values.size();
    if (n == 0) return 0.0;
    if (n == 1) return values[0];
    const double q = detail::clamp(p, 0.0, 100.0);
    std::sort(values.begin(), values.end());
    const double pos = static_cast<double>(n - 1) * q / 100.0;
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    const double frac = pos - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * frac;
}

// ---------------------------------------------------------------- 处理单元
// 一阶高通与原信号按比例混合，补回远场丢失的辅音。
// 出口会把电平缩回输入电平：预加重只该改频谱形状，不该改音量。
inline void preEmphasis(std::vector<float>& a, double coeff, double mix) {
    if (a.empty()) return;
    if (coeff <= 0.0 || mix <= 0.0) return;
    const std::size_t n = a.size();
    std::vector<double> hi(n);
    hi[0] = static_cast<double>(a[0]);          // hi[0] 保持原样
    for (std::size_t i = 1; i < n; ++i) {
        hi[i] = static_cast<double>(a[i]) - coeff * static_cast<double>(a[i - 1]);
    }

    std::vector<double> y(n);
    double s0 = 0.0;
    double s1 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double orig = static_cast<double>(a[i]);
        const double v = (1.0 - mix) * orig + mix * hi[i];
        y[i] = v;
        s0 += orig * orig;
        s1 += v * v;
    }

    const double r0 = std::sqrt(s0 / static_cast<double>(n));
    const double r1 = std::sqrt(s1 / static_cast<double>(n));
    const double k = (r1 > detail::kEps && r0 > detail::kEps) ? (r0 / r1) : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(y[i] * k);
    }
}

// 峰值软限幅：超过拐点用 tanh 压缩，不产生硬削波的方波谐波。
inline void softLimit(std::vector<float>& a, double knee) {
    if (a.empty() || knee <= 0.0) return;
    const double pk = peak(a);
    if (pk <= knee) return;
    for (float& v : a) {
        const double d = static_cast<double>(v);
        v = static_cast<float>(std::tanh(d / knee) * knee);
    }
}

// 把整段拉到目标电平。只放大不缩小。
inline void normalizeGain(std::vector<float>& a, double target, double max_gain) {
    if (a.empty()) return;
    const double cur = rms(a);
    if (cur <= detail::kEps) return;
    double gain = target / cur;
    if (gain <= 1.0) return;
    if (gain > max_gain) gain = max_gain;
    for (float& v : a) {
        v = static_cast<float>(static_cast<double>(v) * gain);
    }
    const double pk = peak(a);
    if (pk > detail::kClampForGain) {
        const double k = detail::kClampForGain / pk;
        for (float& v : a) {
            v = static_cast<float>(static_cast<double>(v) * k);
        }
    }
}

// 把帧能量分成「静音簇」和「语音簇」。
// 为什么不用固定分位数：第 10 分位落在静音帧自身的分布里，门限会横切在
// 静音段中间——一半静音帧被判静音、一半漏掉，漏掉的那部分随后被增益放大
// 几十倍，底噪反而比处理前更响。一维两簇 k-means 只要几轮迭代，
// 找到的阈值落在两簇之间而不是某一簇内部。
struct SplitResult {
    bool ok = false;
    double threshold = 0.0;
    double low = 0.0;
    double high = 0.0;
};

inline SplitResult splitEnergy(std::vector<double> energy) {
    SplitResult r;
    const std::size_t n = energy.size();
    if (n < 3) return r;

    std::vector<double> sorted = energy;
    const double lo = percentile(sorted, 10.0);
    sorted = energy;
    const double hi = percentile(sorted, 90.0);
    if (hi <= lo * 1.5) return r;   // 两簇分不开

    double thr = (lo + hi) / 2.0;
    double c_lo = lo;
    double c_hi = hi;
    for (int iter = 0; iter < 12; ++iter) {
        double sum_lo = 0.0, sum_hi = 0.0;
        std::size_t n_lo = 0, n_hi = 0;
        for (double e : energy) {
            if (e < thr) { sum_lo += e; ++n_lo; }
            else         { sum_hi += e; ++n_hi; }
        }
        if (n_lo == 0 || n_hi == 0) return r;
        c_lo = sum_lo / static_cast<double>(n_lo);
        c_hi = sum_hi / static_cast<double>(n_hi);
        const double next = (c_lo + c_hi) / 2.0;
        const bool done = std::fabs(next - thr) < 1e-9;
        thr = next;
        if (done) break;
    }
    if (c_hi < c_lo * 2.0) return r;   // 中心离得太近，降噪只会锯信号
    r.ok = true;
    r.threshold = thr;
    r.low = c_lo;
    r.high = c_hi;
    return r;
}

// 标出静音帧，返回样本级掩码（空表示不该做门控）。
inline std::vector<char> quietMask(const std::vector<float>& a, int rate,
                                   double frame_ms) {
    std::vector<char> mask;
    if (a.empty()) return mask;
    std::size_t frame = static_cast<std::size_t>(
        std::llround(static_cast<double>(rate) * frame_ms / 1000.0));
    if (frame < 32) frame = 32;
    const std::size_t n = a.size() / frame;
    if (n < 2) return mask;

    std::vector<double> energy(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < frame; ++j) {
            const double d = static_cast<double>(a[i * frame + j]);
            sum += d * d;
        }
        energy[i] = std::sqrt(sum / static_cast<double>(frame));
    }
    const SplitResult sp = splitEnergy(energy);
    if (!sp.ok) return mask;

    mask.assign(a.size(), 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (energy[i] < sp.threshold) {
            for (std::size_t j = 0; j < frame; ++j) mask[i * frame + j] = 1;
        }
    }
    return mask;
}

// 帧级噪声门：只压静音段与稳定底噪，语音帧一个样本都不动。
//
// 为什么不用谱减法：谱减在低 SNR 下会产生音乐噪声（musical noise），
// 听感上比原来的底噪更难受，而且会吃掉句尾的气声。帧级门限虽然粗糙，
// 但它对语音是零损伤的——这一点在 ASR 前端比「降得干净」更重要。
//
// 判决用两簇聚类而不是固定分位数，理由见 splitEnergy。
inline void noiseGate(std::vector<float>& a, int rate = 16000,
                      double frame_ms = 20.0, double attenuation = 0.30) {
    if (a.empty()) return;
    const std::vector<char> mask = quietMask(a, rate, frame_ms);
    if (mask.empty()) return;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (mask[i]) a[i] = static_cast<float>(static_cast<double>(a[i]) * attenuation);
    }
    // 不做峰值保护：门控只会把样本乘小，输入峰值多大输出就多大。
    // 在这里统一缩放会把语音帧也改掉，「语音零损伤」的承诺就破了。
}


// ---------------------------------------------------------------- 完整链路
// 完整链路：高频补偿 → 增益归一 → 静音段压制 → 软限幅。
//
// 为什么把增益和降噪放在一起算，而不是「先降噪再整体放大」：
//
//   先降噪再放大，底噪会被放大倍数原样抬回去。20 倍增益下先把底噪压到
//   0.3 倍、再整体乘 20，净效果是底噪变响 6 倍——测量上「降了 10 dB」，
//   实际听起来更吵。所以这里让增益只作用在语音帧上，静音帧走独立路径：
//   它的最终倍数固定为 gate_attenuation，与增益无关。
//   这才是「放大人声、不放大噪声」。
inline void process(std::vector<float>& a, const Profile& p, int rate = 16000) {
    if (a.empty()) return;

    preEmphasis(a, p.pre_emph, p.pre_emph_mix);

    std::vector<char> mask;
    if (p.gate_enabled) mask = quietMask(a, rate, p.gate_frame_ms);
    const bool hasMask = !mask.empty();

    double gain = 1.0;
    if (p.target_rms > 0.0) {
        // 增益按「语音部分」的电平算，不算静音段，否则静音越多增益算得越大
        double cur = 0.0;
        if (hasMask) {
            double sum = 0.0;
            std::size_t cnt = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (!mask[i]) {
                    const double d = static_cast<double>(a[i]);
                    sum += d * d;
                    ++cnt;
                }
            }
            cur = cnt > 0 ? std::sqrt(sum / static_cast<double>(cnt)) : 0.0;
        } else {
            cur = rms(a);
        }
        if (cur > detail::kEps) {
            const double g = p.target_rms / cur;
            if (g > 1.0) gain = std::min(g, p.max_gain);
        }
    }

    for (float& v : a) v = static_cast<float>(static_cast<double>(v) * gain);

    if (hasMask) {
        // 静音段的最终倍数固定为 gate_attenuation，与增益无关。
        const double k = p.gate_attenuation / std::max(gain, 1.0);
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (mask[i]) a[i] = static_cast<float>(static_cast<double>(a[i]) * k);
        }
    }

    softLimit(a, p.limit_knee);
}


// 按名字取预设，找不到返回 nullptr（不静默回退到默认值）。
inline const Profile* profileByName(const char* name) {
    if (name == nullptr) return nullptr;
    const std::string_view s(name);
    if (s == "near") return &kNear;
    if (s == "far") return &kFar;
    if (s == "aggressive") return &kAggressive;
    if (s == "denoise") return &kDenoiseOnly;
    return nullptr;
}

// ---------------------------------------------------------------- 噪声底
// 滑窗噪声底估计：取低分位数，避免被偶发响声带偏。
class NoiseFloor {
public:
    explicit NoiseFloor(std::size_t size = 180, double pct = 20.0,
                        double factor = 2.4, double base = 0.008)
        : hist_(), size_(size), pct_(pct), factor_(factor), base_(base) {
        if (size_ < 2) size_ = 2;
    }

    double update(double value) {
        if (!std::isfinite(value) || value < 0.0) return floor();
        hist_.push_back(value);
        while (hist_.size() > size_) hist_.pop_front();
        return floor();
    }

    double updateAudio(const std::vector<float>& a, int rate = 16000,
                       double frame_ms = 20.0) {
        if (a.empty()) return floor();
        std::size_t frame = static_cast<std::size_t>(
            std::llround(static_cast<double>(rate) * frame_ms / 1000.0));
        if (frame < 32) frame = 32;
        const std::size_t n = a.size() / frame;
        if (n == 0) return floor();
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < frame; ++j) {
                const double d = static_cast<double>(a[i * frame + j]);
                sum += d * d;
            }
            update(std::sqrt(sum / static_cast<double>(frame)));
        }
        return floor();
    }

    double floor() const {
        if (hist_.empty()) return 0.0;
        std::vector<double> v(hist_.begin(), hist_.end());
        return percentile(v, pct_);
    }

    double threshold(double base = -1.0, double factor = -1.0) const {
        const double b = base < 0.0 ? base_ : base;
        const double f = factor < 0.0 ? factor_ : factor;
        const double t = floor() * f;
        return t > b ? t : b;
    }

    void reset() { hist_.clear(); }

    std::size_t size() const { return hist_.size(); }

private:
    std::deque<double> hist_;
    std::size_t size_;
    double pct_;
    double factor_;
    double base_;
};

}  // namespace nsg
}  // namespace hae

#endif  // HAE_NSG_H
