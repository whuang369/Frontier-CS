import os
import sys
import tarfile
import subprocess
import tempfile
import time
import random
import re
import shutil
import stat
import codecs
from typing import List, Tuple, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            self._extract_tarball(src_path, tmpdir)
            root = self._find_project_root(tmpdir)
            self._build_project(root)
            binaries = self._find_executables(root)
            if not binaries:
                return self._default_poc()

            seeds = self._gather_seeds(root)
            crash = self._find_crash_input(binaries, seeds)
            if crash is None:
                # No crash found; return a reasonable LSAT-style guess as fallback
                return self._fallback_lsat_poc()

            bin_path, crash_input = crash
            minimized = self._minimize_input(bin_path, crash_input)
            return minimized
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _extract_tarball(self, src_path: str, dst: str) -> None:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(dst)
        except Exception:
            pass

    def _find_project_root(self, tmpdir: str) -> str:
        try:
            entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
        except Exception:
            return tmpdir
        if len(entries) == 1:
            single = os.path.join(tmpdir, entries[0])
            if os.path.isdir(single):
                return single
        return tmpdir

    def _build_project(self, root: str) -> None:
        # Priority: build.sh > CMakeLists.txt > configure > Makefile > fallback compile
        build_sh = os.path.join(root, "build.sh")
        if os.path.isfile(build_sh):
            self._run_cmd(["bash", "build.sh"], cwd=root, timeout=120)
            return

        cmakelists = os.path.join(root, "CMakeLists.txt")
        if os.path.isfile(cmakelists):
            build_dir = os.path.join(root, "build")
            os.makedirs(build_dir, exist_ok=True)
            if self._run_cmd(["cmake", ".."], cwd=build_dir, timeout=60):
                self._run_cmd(["cmake", "--build", ".", "-j8"], cwd=build_dir, timeout=120)
            return

        configure = os.path.join(root, "configure")
        if os.path.isfile(configure) and os.access(configure, os.X_OK):
            if self._run_cmd(["bash", "configure"], cwd=root, timeout=60):
                self._run_cmd(["make", "-j8"], cwd=root, timeout=120)
            return

        makefile = None
        for name in ("Makefile", "makefile"):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                makefile = path
                break
        if makefile is not None:
            self._run_cmd(["make", "-j8"], cwd=root, timeout=120)
            return

        # Fallback: try to compile all .c files into a single binary
        self._fallback_build(root)

    def _run_cmd(self, cmd: List[str], cwd: Optional[str] = None, timeout: int = 60) -> bool:
        try:
            subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            return True
        except Exception:
            return False

    def _fallback_build(self, root: str) -> None:
        c_files: List[str] = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".c"):
                    c_files.append(os.path.join(dirpath, f))
        if not c_files:
            return
        out = os.path.join(root, "a.out")
        # Try cc first, then gcc
        compilers = ["cc", "gcc"]
        for cc in compilers:
            cmd = [cc, "-O0", "-g", "-o", out] + c_files
            if self._run_cmd(cmd, cwd=root, timeout=120):
                break

    def _find_executables(self, root: str) -> List[str]:
        executables: List[str] = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                path = os.path.join(dirpath, f)
                if not os.path.isfile(path):
                    continue
                try:
                    st = os.stat(path)
                except Exception:
                    continue
                if not (st.st_mode & stat.S_IXUSR):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in (".sh", ".py", ".pl", ".rb", ".so", ".a", ".dll", ".dylib", ".jar"):
                    continue
                try:
                    with open(path, "rb") as fp:
                        header = fp.read(4)
                except Exception:
                    continue
                if not (header.startswith(b"\x7fELF") or header.startswith(b"MZ")):
                    continue
                executables.append(path)
        # Sort to ensure deterministic order
        executables.sort()
        return executables

    def _gather_seeds(self, root: str) -> List[bytes]:
        seeds: List[bytes] = []

        # Basic seeds
        seeds.append(b"")
        seeds.append(b"\n")

        # Some generic LSAT-flavored seeds (always add, cheap and could help)
        seeds.extend(self._lsat_specific_seeds())

        # Collect string literals from source files
        string_literals: List[bytes] = []
        pattern = re.compile(rb'"([^"\\]*(?:\\.[^"\\]*)*)"')
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith((".c", ".h", ".cpp", ".hpp", ".cc", ".cxx")):
                    continue
                path = os.path.join(dirpath, f)
                try:
                    with open(path, "rb") as fp:
                        data = fp.read()
                except Exception:
                    continue
                for m in pattern.finditer(data):
                    lit_bytes = m.group(1)
                    decoded = self._decode_c_string(lit_bytes)
                    if decoded is None:
                        continue
                    if not decoded:
                        continue
                    if len(decoded) > 80:
                        continue
                    # Printable-ish
                    if not all(32 <= b < 127 or b in (9, 10, 13) for b in decoded):
                        continue
                    string_literals.append(decoded)

        # Deduplicate while preserving order
        seen = set()
        for s in string_literals:
            if s in seen:
                continue
            seen.add(s)
            seeds.append(s + (b"" if s.endswith(b"\n") else b"\n"))

        # Heuristic: LSAT-related combined seeds
        lsat_tokens = [s for s in seen if b"lsat" in s.lower() or b"LSAT" in s]
        proj_tokens = [s for s in seen if b"proj" in s.lower()]
        combined: List[bytes] = []
        for t1 in lsat_tokens[:10]:
            combined.append(t1 + b"\n")
        for t1 in lsat_tokens[:10]:
            for t2 in proj_tokens[:10]:
                line = t1 + b" " + t2 + b"\n"
                combined.append(line)
                if len(combined) >= 40:
                    break
            if len(combined) >= 40:
                break
        seeds.extend(combined)

        # Limit total seeds to avoid spending too much time
        max_seeds = 200
        if len(seeds) > max_seeds:
            seeds = seeds[:max_seeds]

        # Ensure at least one non-empty seed for mutation
        if all(len(s) == 0 for s in seeds):
            seeds.append(b"A\n")
        return seeds

    def _decode_c_string(self, lit: bytes) -> Optional[bytes]:
        try:
            # Interpret using unicode_escape for C-style backslash escapes
            text = lit.decode("latin1")
            decoded = codecs.decode(text, "unicode_escape")
            return decoded.encode("latin1", "ignore")
        except Exception:
            return None

    def _lsat_specific_seeds(self) -> List[bytes]:
        seeds: List[bytes] = []
        base = b"+proj=lsat +lat_1=0 +lat_2=0 +lat_0=0 +lon_0=0 "
        sats = [1, 0, 2, 3, 4, 5, -1]
        paths = [1, 0, 10, 60, 128, 233, 251, -1]
        for s in sats:
            for p in paths:
                line = base + f"+sat={s} +path={p}".encode("ascii") + b"\n"
                seeds.append(line)
                if len(seeds) >= 40:
                    return seeds
        return seeds

    def _find_crash_input(
        self, binaries: List[str], seeds: List[bytes]
    ) -> Optional[Tuple[str, bytes]]:
        if not binaries:
            return None

        # Base seeds for mutation (non-empty preferred)
        base_seeds = [s for s in seeds if s]
        if not base_seeds:
            base_seeds = [b"A\n"]

        start = time.time()
        total_budget = 10.0  # seconds for fuzzing
        deadline = start + total_budget

        # Run all seeds first on each binary
        for bin_path in binaries:
            for s in seeds:
                if time.time() > deadline:
                    return None
                if self._run_test(bin_path, s):
                    return bin_path, s

        # Random mutation-based fuzzing
        # We keep the same overall deadline across binaries
        while time.time() < deadline:
            for bin_path in binaries:
                if time.time() >= deadline:
                    break
                candidate = self._mutate(random.choice(base_seeds))
                if self._run_test(bin_path, candidate):
                    return bin_path, candidate
        return None

    def _is_sanitizer_crash(self, stderr: bytes) -> bool:
        if not stderr:
            return False
        markers = [
            b"ERROR: AddressSanitizer",
            b"AddressSanitizer",
            b"heap-use-after-free",
            b"ERROR: LeakSanitizer",
            b"runtime error:",
        ]
        return any(m in stderr for m in markers)

    def _run_test(self, binary: str, data: bytes) -> bool:
        try:
            res = subprocess.run(
                [binary],
                input=data,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=0.5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        return self._is_sanitizer_crash(res.stderr or b"")

    def _mutate(self, seed: bytes, max_len: int = 64) -> bytes:
        # If seed is too long, truncate for performance
        if len(seed) > max_len:
            seed = seed[:max_len]
        data = bytearray(seed)
        # Randomly decide to generate entirely new random input
        if not data or random.random() < 0.1:
            length = random.randint(1, max_len)
            return bytes(random.getrandbits(8) for _ in range(length))

        num_ops = random.randint(1, 3)
        for _ in range(num_ops):
            op = random.randint(0, 2)
            if op == 0 and data:
                # Flip a byte
                idx = random.randrange(len(data))
                data[idx] = random.getrandbits(8)
            elif op == 1 and len(data) < max_len:
                # Insert a random byte
                idx = random.randrange(len(data) + 1)
                data.insert(idx, random.getrandbits(8))
            elif op == 2 and len(data) > 1:
                # Delete a byte
                idx = random.randrange(len(data))
                del data[idx]
        return bytes(data)

    def _minimize_input(self, binary: str, data: bytes) -> bytes:
        best = data
        start = time.time()
        time_budget = 5.0
        i = 0
        while i < len(best) and (time.time() - start) < time_budget:
            candidate = best[:i] + best[i + 1 :]
            if not candidate:
                i += 1
                continue
            if self._run_test(binary, candidate):
                best = candidate
                # Do not increment i; check same index again after shrinking
            else:
                i += 1
        return best

    def _default_poc(self) -> bytes:
        # Generic small input as absolute fallback
        return b"A" * 16

    def _fallback_lsat_poc(self) -> bytes:
        # LSAT-flavored guess (around 38 bytes length)
        s = b"+proj=lsat +sat=1 +path=1\n"
        return s