// Python 与 C++ 对拍用的驱动程序。
//
// 读一段 float32 单声道裸数据，跑 hae::nsg::process，再写回去。
// 旁边的 run_parity.py 会拿同样的输入跑 Python 版，逐样本比对。
//
// 文件读写直接用 Win32 API：CRT 的 fopen 既打不开中文路径，在这台机器上
// 连纯英文路径的 "wb" 都返回失败，用它做对拍工具只会平白引入噪音。
//
// 用法：parity_main <in.raw> <out.raw> <profile> <rate>
// 退出码：0 成功，2 参数错，3 未知 profile，4 文件读写失败

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hae_nsg.h"

static std::wstring widen(const char* s) {
    if (s == nullptr) return std::wstring();
    const int n = ::MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (n <= 1) return std::wstring();
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_ACP, 0, s, -1, &w[0], n);
    while (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

static bool readRaw(const std::wstring& path, std::vector<float>& out) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(h, &size)) {
        ::CloseHandle(h);
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(size.QuadPart);
    out.resize(bytes / sizeof(float));
    std::size_t done = 0;
    while (done < bytes) {
        DWORD got = 0;
        const DWORD want = static_cast<DWORD>(bytes - done);
        if (!::ReadFile(h, reinterpret_cast<char*>(out.data()) + done, want,
                        &got, nullptr) || got == 0) {
            break;
        }
        done += got;
    }
    ::CloseHandle(h);
    out.resize(done / sizeof(float));
    return true;
}

static bool writeRaw(const std::wstring& path, const std::vector<float>& in) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "CreateFileW failed: %lu\n",
                     static_cast<unsigned long>(::GetLastError()));
        return false;
    }
    const std::size_t bytes = in.size() * sizeof(float);
    std::size_t done = 0;
    while (done < bytes) {
        DWORD put = 0;
        const DWORD want = static_cast<DWORD>(bytes - done);
        if (!::WriteFile(h, reinterpret_cast<const char*>(in.data()) + done,
                         want, &put, nullptr) || put == 0) {
            ::CloseHandle(h);
            return false;
        }
        done += put;
    }
    ::CloseHandle(h);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: parity_main <in.raw> <out.raw> <profile> <rate>\n");
        return 2;
    }
    const std::wstring in_path = widen(argv[1]);
    const std::wstring out_path = widen(argv[2]);
    const char* profile_name = argv[3];
    const int rate = std::atoi(argv[4]);

    const hae::nsg::Profile* profile = hae::nsg::profileByName(profile_name);
    if (profile == nullptr) {
        std::fprintf(stderr, "unknown profile: %s\n", profile_name);
        return 3;
    }

    std::vector<float> audio;
    if (!readRaw(in_path, audio)) {
        std::fprintf(stderr, "cannot read input\n");
        return 4;
    }

    hae::nsg::process(audio, *profile, rate);

    if (!writeRaw(out_path, audio)) {
        std::fprintf(stderr, "cannot write output\n");
        return 4;
    }

    std::printf("%zu %f %f\n", audio.size(),
                hae::nsg::rms(audio), hae::nsg::peak(audio));
    return 0;
}
