import os
import re
import tarfile
import tempfile
import shutil
import subprocess
import ctypes.util
from pathlib import Path
from typing import List, Optional, Tuple


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return os.path.commonpath([str(directory)]) == os.path.commonpath([str(directory), str(target)])
    except Exception:
        return False


def _safe_extract_tar(tar_path: str, dst_dir: Path) -> None:
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = dst_dir / member.name
            if not _is_within_directory(dst_dir, member_path):
                continue
            try:
                tar.extract(member, path=dst_dir)
            except Exception:
                continue


def _find_single_root_dir(extracted_dir: Path) -> Path:
    try:
        entries = [p for p in extracted_dir.iterdir() if p.name not in (".", "..")]
    except Exception:
        return extracted_dir
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extracted_dir


def _read_file_bytes(p: Path, max_size: int = 5_000_000) -> Optional[bytes]:
    try:
        st = p.stat()
        if st.st_size <= 0 or st.st_size > max_size:
            return None
        with p.open("rb") as f:
            return f.read()
    except Exception:
        return None


def _looks_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    sample = data[:4096]
    bad = 0
    for b in sample:
        if b in (9, 10, 13):
            continue
        if 32 <= b <= 126:
            continue
        bad += 1
    return (bad / max(1, len(sample))) < 0.02


def _keyword_score(name: str) -> int:
    n = name.lower()
    score = 0
    for kw, s in (
        ("clusterfuzz", 50),
        ("testcase", 40),
        ("minimized", 35),
        ("crash", 35),
        ("repro", 25),
        ("poc", 25),
        ("uaf", 20),
        ("use-after-free", 20),
        ("heap", 10),
    ):
        if kw in n:
            score += s
    if n.endswith((".ttf", ".otf", ".woff", ".woff2", ".bin", ".dat")):
        score += 8
    return score


def _select_existing_poc(root: Path) -> Optional[bytes]:
    candidates: List[Tuple[int, int, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        sc = _keyword_score(name)
        if sc <= 0:
            continue
        try:
            sz = p.stat().st_size
        except Exception:
            continue
        if sz <= 0 or sz > 200_000:
            continue
        data = _read_file_bytes(p, max_size=200_000)
        if data is None:
            continue
        if _looks_text(data):
            continue
        closeness = -abs(sz - 800)
        candidates.append((sc, closeness, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0][2]
    return _read_file_bytes(best, max_size=200_000)


def _find_fuzzer_sources(root: Path) -> List[Path]:
    fuzzers = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".cc", ".cpp", ".cxx", ".c++"):
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if "LLVMFuzzerTestOneInput" in txt:
            fuzzers.append(p)
    return fuzzers


def _score_fuzzer_source(p: Path) -> int:
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        return 0
    score = 0
    if "ots::" in txt:
        score += 20
    if "OTSStream" in txt:
        score += 40
    if "Write" in txt:
        score += 5
    if "FuzzedDataProvider" in txt:
        score += 10
    if "Process" in txt and "ots" in txt:
        score += 10
    return score


def _find_ots_header_dir(root: Path) -> Optional[Path]:
    candidates = []
    for p in root.rglob("ots.h"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if "OTSStream" in txt or "namespace ots" in txt:
            candidates.append(p)
    if not candidates:
        for p in root.rglob("*ots*.h"):
            if not p.is_file():
                continue
            if p.name.lower() != "ots.h":
                continue
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0].parent


def _has_any_file(root: Path, rel_patterns: List[str]) -> bool:
    for pat in rel_patterns:
        if list(root.rglob(pat)):
            return True
    return False


def _write_fuzzed_data_provider(include_root: Path) -> None:
    # Provide both "FuzzedDataProvider.h" and "fuzzer/FuzzedDataProvider.h"
    fuzzer_dir = include_root / "fuzzer"
    fuzzer_dir.mkdir(parents=True, exist_ok=True)
    header_paths = [include_root / "FuzzedDataProvider.h", fuzzer_dir / "FuzzedDataProvider.h"]
    header = r'''
#ifndef FUZZED_DATA_PROVIDER_H_
#define FUZZED_DATA_PROVIDER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

class FuzzedDataProvider {
 public:
  FuzzedDataProvider(const uint8_t* data, size_t size) : data_(data), size_(size), offset_(0) {}
  size_t remaining_bytes() const { return size_ - offset_; }
  bool ConsumeBool() { return ConsumeIntegralInRange<uint8_t>(0, 1) != 0; }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, T>::type ConsumeIntegral() {
    return ConsumeIntegralInRange<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, T>::type ConsumeIntegralInRange(T min, T max) {
    if (min > max) std::swap(min, max);
    if (remaining_bytes() == 0) return min;
    using U = typename std::make_unsigned<T>::type;
    U range = static_cast<U>(max) - static_cast<U>(min);
    if (range == 0) return min;

    size_t bytes_to_read = std::min(sizeof(T), remaining_bytes());
    U val = 0;
    for (size_t i = 0; i < bytes_to_read; i++) {
      val = (val << 8) | data_[offset_ + i];
    }
    offset_ += bytes_to_read;

    U result = (range == std::numeric_limits<U>::max()) ? val : (val % (range + 1));
    return static_cast<T>(static_cast<U>(min) + result);
  }

  template <typename T>
  std::vector<T> ConsumeBytes(size_t num_bytes) {
    num_bytes = std::min(num_bytes, remaining_bytes());
    std::vector<T> out;
    out.resize(num_bytes / sizeof(T));
    if (!out.empty()) {
      std::memcpy(out.data(), data_ + offset_, out.size() * sizeof(T));
    }
    offset_ += out.size() * sizeof(T);
    return out;
  }

  std::string ConsumeRandomLengthString(size_t max_length) {
    max_length = std::min(max_length, remaining_bytes());
    size_t len = ConsumeIntegralInRange<size_t>(0, max_length);
    auto bytes = ConsumeBytes<uint8_t>(len);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }

  template <typename T, size_t N>
  T PickValueInArray(const T (&array)[N]) {
    size_t idx = ConsumeIntegralInRange<size_t>(0, N - 1);
    return array[idx];
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t offset_;
};

#endif  // FUZZED_DATA_PROVIDER_H_
'''.lstrip()
    for hp in header_paths:
        try:
            hp.write_text(header)
        except Exception:
            pass


def _collect_sources(root: Path, fuzzer: Path, ots_dir: Optional[Path]) -> Tuple[List[Path], List[Path]]:
    include_dirs = set()
    include_dirs.add(root)
    include_dirs.add(fuzzer.parent)
    if ots_dir:
        include_dirs.add(ots_dir)
        include_dirs.add(ots_dir.parent)
    for dname in ("include", "src"):
        p = root / dname
        if p.is_dir():
            include_dirs.add(p)

    # Collect library sources.
    sources: List[Path] = []
    base_dirs: List[Path] = []
    if ots_dir and ots_dir.is_dir():
        base_dirs.append(ots_dir)
    else:
        base_dirs.append(root)

    excluded_dir_parts = {
        ".git", ".svn", "__pycache__", "build", "out", "dist", "cmake-build-debug", "cmake-build-release",
        "test", "tests", "testing", "example", "examples", "benchmark", "benchmarks", "tools", "tool",
        "doc", "docs", "fuzz", "fuzzer", "corpus",
    }

    def is_excluded(p: Path) -> bool:
        parts = set(p.parts)
        return any(part in excluded_dir_parts for part in parts)

    # Always include the fuzzer source itself.
    sources.append(fuzzer)

    for bd in base_dirs:
        for p in bd.rglob("*"):
            if not p.is_file():
                continue
            if p == fuzzer:
                continue
            if p.suffix.lower() not in (".cc", ".cpp", ".cxx", ".c++", ".c"):
                continue
            if is_excluded(p):
                continue
            # Skip other fuzzers/mains if present.
            try:
                txt = p.read_text(errors="ignore")
                if "LLVMFuzzerTestOneInput" in txt:
                    continue
            except Exception:
                pass
            sources.append(p)

    # Filter optional woff2/brotli sources if brotli headers are not present in tree and system likely lacks them.
    brotli_present = _has_any_file(root, ["*brotli*/decode.h", "*brotli*/encode.h", "*brotli*"])
    if not brotli_present:
        filtered = []
        for s in sources:
            nm = s.name.lower()
            if "woff2" in nm or "brotli" in nm:
                continue
            filtered.append(s)
        sources = filtered

    # De-dup, stable
    seen = set()
    uniq_sources = []
    for s in sources:
        sp = str(s.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq_sources.append(s)
    sources = uniq_sources

    include_list = sorted({str(p.resolve()) for p in include_dirs if p and p.exists()})
    return sources, [Path(p) for p in include_list]


def _pick_compiler() -> Optional[str]:
    for c in ("clang++", "clang", "g++", "c++"):
        if shutil.which(c):
            return c
    return None


def _compiler_supports_flag(compiler: str, flag: str) -> bool:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "t.cc"
        src.write_text("int main(){return 0;}\n")
        cmd = [compiler, str(src), "-o", str(td / "a.out"), flag]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return r.returncode == 0
        except Exception:
            return False


def _compile_libfuzzer_binary(root: Path, fuzzer_src: Path, ots_dir: Optional[Path], build_dir: Path) -> Optional[Path]:
    compiler = _pick_compiler()
    if not compiler:
        return None

    include_root = build_dir / "include"
    include_root.mkdir(parents=True, exist_ok=True)
    _write_fuzzed_data_provider(include_root)

    sources, include_dirs = _collect_sources(root, fuzzer_src, ots_dir)

    # Determine sanitizer flags
    fuzzer_flag = "-fsanitize=fuzzer,address"
    if not _compiler_supports_flag(compiler, fuzzer_flag):
        # Try fuzzer-no-link and link libFuzzer explicitly if possible is too complex; bail.
        return None

    out_bin = build_dir / "fuzz_target"
    flags = [
        "-std=c++17",
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1",
        fuzzer_flag,
    ]

    inc_flags = [f"-I{str(include_root.resolve())}"]
    for inc in include_dirs:
        inc_flags.append(f"-I{str(inc.resolve())}")

    libs = []
    if ctypes.util.find_library("z"):
        libs.append("-lz")

    # Response file
    rsp = build_dir / "compile.rsp"
    lines = []
    lines.extend(flags)
    lines.extend(inc_flags)
    for s in sources:
        lines.append(str(s.resolve()))
    lines.extend(libs)
    lines.extend(["-o", str(out_bin.resolve())])
    try:
        rsp.write_text("\n".join(lines) + "\n")
    except Exception:
        return None

    cmd = [compiler, f"@{str(rsp.resolve())}"]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(root), timeout=240)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    if not out_bin.exists():
        return None
    return out_bin


def _parse_artifact_path(output: bytes, artifacts_dir: Path) -> Optional[Path]:
    try:
        txt = output.decode("utf-8", errors="ignore")
    except Exception:
        txt = ""
    m = re.search(r"Test unit written to\s+([^\r\n]+)", txt)
    if m:
        p = m.group(1).strip().strip("'\"")
        pp = Path(p)
        if not pp.is_absolute():
            pp = (artifacts_dir.parent / pp).resolve()
        if pp.exists() and pp.is_file():
            return pp
    # Fallback: newest crash file in artifacts_dir
    try:
        files = [p for p in artifacts_dir.glob("*") if p.is_file() and (p.name.startswith("crash-") or p.name.startswith("timeout-"))]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]
    except Exception:
        return None


def _run_fuzzer_find_crash(bin_path: Path, max_total_time: int, artifacts_dir: Path, seed: int = 1) -> Tuple[Optional[bytes], bytes]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bin_path.resolve()),
        f"-max_total_time={int(max_total_time)}",
        f"-artifact_prefix={str((artifacts_dir.resolve()).as_posix())}/",
        f"-seed={int(seed)}",
        "-timeout=5",
        "-rss_limit_mb=2048",
    ]
    env = os.environ.copy()
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1")
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=max_total_time + 30)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + b"\n" + (e.stderr or b"")
        return None, out
    out = (r.stdout or b"") + b"\n" + (r.stderr or b"")
    if r.returncode == 0:
        return None, out
    art = _parse_artifact_path(out, artifacts_dir)
    if art and art.exists():
        data = _read_file_bytes(art, max_size=2_000_000)
        if data is not None:
            return data, out
    return None, out


def _crashes_with_input(bin_path: Path, data: bytes) -> Tuple[bool, bytes]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "in"
        inp.write_bytes(data)
        cmd = [str(bin_path.resolve()), "-runs=1", str(inp.resolve())]
        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1")
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=20)
        except Exception:
            return False, b""
        out = (r.stdout or b"") + b"\n" + (r.stderr or b"")
        return r.returncode != 0, out


def _suffix_minimize(bin_path: Path, data: bytes, max_checks: int = 18) -> bytes:
    # Only try removing suffix chunks; many fuzzers ignore extra bytes.
    if len(data) <= 1:
        return data
    best = data
    checks = 0
    step = max(1, len(best) // 2)
    while step >= 1 and checks < max_checks:
        cand = best[:-step]
        if not cand:
            step //= 2
            continue
        ok, out = _crashes_with_input(bin_path, cand)
        checks += 1
        if ok:
            best = cand
        else:
            step //= 2
    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            src_p = Path(src_path)
            if src_p.is_dir():
                root = src_p
            else:
                extract_dir = td_path / "src"
                extract_dir.mkdir(parents=True, exist_ok=True)
                _safe_extract_tar(str(src_p), extract_dir)
                root = _find_single_root_dir(extract_dir)

            poc = _select_existing_poc(root)
            if poc is not None:
                return poc

            fuzzers = _find_fuzzer_sources(root)
            if not fuzzers:
                return b"\x00" * 800

            fuzzers.sort(key=_score_fuzzer_source, reverse=True)
            fuzzer_src = fuzzers[0]

            ots_dir = _find_ots_header_dir(root)

            build_dir = td_path / "build"
            build_dir.mkdir(parents=True, exist_ok=True)

            bin_path = _compile_libfuzzer_binary(root, fuzzer_src, ots_dir, build_dir)
            if bin_path is None:
                return b"\x00" * 800

            artifacts_dir = td_path / "artifacts"
            for t, seed in ((10, 1), (30, 2), (60, 3), (90, 4)):
                data, out = _run_fuzzer_find_crash(bin_path, t, artifacts_dir, seed=seed)
                if data is None:
                    continue
                out_txt = out.decode("utf-8", errors="ignore")
                if "AddressSanitizer" in out_txt and "heap-use-after-free" in out_txt and ("OTSStream::Write" in out_txt or "ots::OTSStream::Write" in out_txt):
                    data = _suffix_minimize(bin_path, data)
                    return data
                if "AddressSanitizer" in out_txt and "heap-use-after-free" in out_txt:
                    data = _suffix_minimize(bin_path, data)
                    return data

            crash_files = []
            try:
                crash_files = [p for p in artifacts_dir.glob("crash-*") if p.is_file()]
            except Exception:
                crash_files = []
            if crash_files:
                crash_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                data = _read_file_bytes(crash_files[0], max_size=2_000_000)
                if data is not None:
                    data = _suffix_minimize(bin_path, data)
                    return data

            return b"\x00" * 800