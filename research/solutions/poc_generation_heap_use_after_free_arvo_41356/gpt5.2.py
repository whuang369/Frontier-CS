import os
import re
import io
import sys
import time
import shlex
import hashlib
import random
import tarfile
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


def _safe_extract_tar(tar_path: str, dest_dir: str) -> None:
    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not is_within_directory(dest_dir, member_path):
                continue
            tf.extract(member, dest_dir)


def _list_files(root: str) -> List[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.join(dp, f))
    return out


def _read_file_bytes(path: str, max_bytes: int = 1_000_000) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes)
    except Exception:
        return b""


def _read_file_text(path: str, max_chars: int = 1_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_chars)
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _is_elf_executable(path: str) -> bool:
    try:
        if not os.path.isfile(path):
            return False
        st = os.stat(path)
        if st.st_size < 4:
            return False
        if not os.access(path, os.X_OK):
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:
        return False


def _run_cmd(cmd: List[str], cwd: str, env: dict, timeout: float) -> Tuple[int, bytes, bytes]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, b"", (e.stdout or b"") + (e.stderr or b"")
    except Exception as e:
        return 127, b"", str(e).encode("utf-8", errors="ignore")


def _build_with_cmake(src_root: str, build_root: str, use_asan: bool) -> bool:
    os.makedirs(build_root, exist_ok=True)
    env = os.environ.copy()
    cflags = env.get("CFLAGS", "")
    cxxflags = env.get("CXXFLAGS", "")
    ldflags = env.get("LDFLAGS", "")
    san_flags = ""
    if use_asan:
        san_flags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all"
    else:
        san_flags = "-O2 -g"
    env["CFLAGS"] = (cflags + " " + san_flags).strip()
    env["CXXFLAGS"] = (cxxflags + " " + san_flags).strip()
    env["LDFLAGS"] = (ldflags + " " + ("-fsanitize=address,undefined" if use_asan else "")).strip()

    cmake_args = [
        "cmake",
        "-S",
        src_root,
        "-B",
        build_root,
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_C_FLAGS={env['CFLAGS']}",
        f"-DCMAKE_CXX_FLAGS={env['CXXFLAGS']}",
        f"-DCMAKE_EXE_LINKER_FLAGS={env['LDFLAGS']}",
    ]
    rc, out, err = _run_cmd(cmake_args, cwd=src_root, env=env, timeout=180.0)
    if rc != 0:
        return False

    build_args = ["cmake", "--build", build_root, "-j", str(max(1, os.cpu_count() or 1))]
    rc, out, err = _run_cmd(build_args, cwd=src_root, env=env, timeout=240.0)
    return rc == 0


def _build_with_make(src_root: str, use_asan: bool) -> bool:
    env = os.environ.copy()
    san_flags = ""
    if use_asan:
        san_flags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all"
    else:
        san_flags = "-O2 -g"
    env["CFLAGS"] = (env.get("CFLAGS", "") + " " + san_flags).strip()
    env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " " + san_flags).strip()
    env["LDFLAGS"] = (env.get("LDFLAGS", "") + " " + ("-fsanitize=address,undefined" if use_asan else "")).strip()

    rc, out, err = _run_cmd(["make", "-j", str(max(1, os.cpu_count() or 1))], cwd=src_root, env=env, timeout=300.0)
    return rc == 0


def _has_cmake(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "CMakeLists.txt"))


def _has_make(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "Makefile")) or os.path.isfile(os.path.join(root, "makefile"))


def _detect_project_root(extracted_dir: str) -> str:
    entries = [os.path.join(extracted_dir, p) for p in os.listdir(extracted_dir)]
    dirs = [p for p in entries if os.path.isdir(p)]
    if len(dirs) == 1:
        return dirs[0]
    return extracted_dir


def _find_cmake_targets(root: str, max_files: int = 200) -> List[str]:
    targets = []
    cmake_files = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f == "CMakeLists.txt":
                cmake_files.append(os.path.join(dp, f))
                if len(cmake_files) >= max_files:
                    break
        if len(cmake_files) >= max_files:
            break

    add_exec_re = re.compile(r"add_executable\s*\(\s*([A-Za-z0-9_\-\.]+)", re.IGNORECASE)
    for p in cmake_files:
        txt = _read_file_text(p, 300_000)
        for m in add_exec_re.finditer(txt):
            t = m.group(1).strip()
            if t and t not in targets:
                targets.append(t)
    return targets


def _find_built_executable(build_root: str, preferred_names: List[str]) -> List[str]:
    exes = []
    for dp, dn, fn in os.walk(build_root):
        if "CMakeFiles" in dp:
            continue
        for f in fn:
            p = os.path.join(dp, f)
            if _is_elf_executable(p):
                exes.append(p)
    # Prefer named executables
    preferred = []
    preferred_set = set(preferred_names)
    for p in exes:
        bn = os.path.basename(p)
        if bn in preferred_set:
            preferred.append(p)
    others = [p for p in exes if p not in preferred]

    def score(path: str) -> Tuple[int, int]:
        try:
            st = os.stat(path)
            sz = st.st_size
            mt = int(st.st_mtime)
        except Exception:
            sz, mt = 0, 0
        bn = os.path.basename(path).lower()
        penalty = 0
        for bad in ("test", "unit", "bench", "fuzz", "example", "demo"):
            if bad in bn:
                penalty += 1
        return (-penalty, sz + (mt // 10))

    preferred.sort(key=score, reverse=True)
    others.sort(key=score, reverse=True)
    return preferred + others


def _normalize_usage_text(b: bytes) -> str:
    s = b.decode("utf-8", errors="ignore")
    s = s.replace("\r\n", "\n")
    return s


def _looks_like_usage(out: bytes, err: bytes) -> bool:
    s = (_normalize_usage_text(out) + "\n" + _normalize_usage_text(err)).lower()
    if "usage:" in s or "usage " in s:
        return True
    if "help" in s and ("-h" in s or "--help" in s):
        return True
    return False


@dataclass
class _Invocation:
    argv: List[str]
    mode: str  # "stdin" or "file"


class _TargetRunner:
    def __init__(self, exe_path: str, workdir: str):
        self.exe_path = exe_path
        self.workdir = workdir
        self.env = os.environ.copy()
        self.env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")
        self.env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:abort_on_error=1:symbolize=0")
        self.invocation = self._detect_invocation()

    def _try_run(self, inv: _Invocation, data: bytes, timeout: float = 1.0) -> Tuple[int, bytes, bytes]:
        if inv.mode == "stdin":
            try:
                p = subprocess.run(
                    inv.argv,
                    cwd=self.workdir,
                    env=self.env,
                    input=data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                return 124, e.stdout or b"", e.stderr or b""
            except Exception as e:
                return 127, b"", str(e).encode("utf-8", errors="ignore")
        else:
            with tempfile.NamedTemporaryFile(prefix="poc_", delete=False) as tf:
                tmp = tf.name
                tf.write(data)
            try:
                argv = inv.argv + [tmp]
                p = subprocess.run(
                    argv,
                    cwd=self.workdir,
                    env=self.env,
                    input=b"",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                return 124, e.stdout or b"", e.stderr or b""
            except Exception as e:
                return 127, b"", str(e).encode("utf-8", errors="ignore")
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def _detect_invocation(self) -> _Invocation:
        # Try stdin with no args
        inv_stdin = _Invocation([self.exe_path], "stdin")
        rc, out, err = self._try_run(inv_stdin, b"", timeout=0.8)
        if rc == 0 and not _looks_like_usage(out, err):
            return inv_stdin

        # Try file mode: common patterns
        # We create a minimal file and try passing it in various ways
        base = [self.exe_path]
        patterns = [
            (base, "file"),
            (base + ["-f"], "file"),
            (base + ["--file"], "file"),
            (base + ["-i"], "file"),
            (base + ["--input"], "file"),
        ]
        probe = b"\n"
        best = None
        for argv, mode in patterns:
            inv = _Invocation(argv, mode)
            rc2, out2, err2 = self._try_run(inv, probe, timeout=0.8)
            if not _looks_like_usage(out2, err2):
                best = inv
                if rc2 == 0:
                    return best
        if best is not None:
            return best

        # If still looks like usage, default to stdin (harness often feeds stdin)
        return inv_stdin

    @staticmethod
    def _is_sanitizer_crash(rc: int, out: bytes, err: bytes) -> bool:
        if rc == 124:
            return False
        s = (out + b"\n" + err).decode("utf-8", errors="ignore").lower()
        indicators = [
            "addresssanitizer",
            "heap-use-after-free",
            "use-after-free",
            "double-free",
            "attempting double-free",
            "free(): double free detected",
            "invalid free",
            "asan:",
            "runtime error:",
        ]
        if any(k in s for k in indicators):
            return True
        if rc < 0:
            return True
        if rc != 0 and ("abort" in s or "corrupted" in s or "malloc" in s and "error" in s):
            return True
        return False

    def crashes(self, data: bytes, timeout: float = 1.0) -> bool:
        rc, out, err = self._try_run(self.invocation, data, timeout=timeout)
        return self._is_sanitizer_crash(rc, out, err)


def _gather_seeds(root: str) -> List[bytes]:
    seeds = []
    # Common textual seeds
    seeds.extend([
        b"",
        b"\n",
        b"0\n",
        b"1\n",
        b"{}\n",
        b"[]\n",
        b"()\n",
        b'{"a":1}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":[1,2,3]}\n',
        b'{"a":{"b":{"c":{"d":1}}}}\n',
        b"<a></a>\n",
        b"(a (b (c)))\n",
        b"a=b\n",
        b"1+2+3\n",
        b"node 1\nadd 1 1\n",
    ])
    # Scan for sample inputs
    exts = {
        ".txt", ".json", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
        ".dat", ".bin", ".input", ".in", ".sample", ".test", ".case", ".poc"
    }
    bad_exts = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".py", ".sh", ".md",
        ".cmake", ".a", ".so", ".o", ".obj", ".png", ".jpg", ".jpeg", ".gif", ".pdf"
    }
    name_hints = ("poc", "crash", "fail", "uaf", "double", "asan", "repro", "corpus", "seed")
    files = _list_files(root)
    files.sort(key=lambda p: (len(p), p))
    for p in files:
        bn = os.path.basename(p).lower()
        _, ext = os.path.splitext(bn)
        if ext in bad_exts:
            continue
        try:
            st = os.stat(p)
        except Exception:
            continue
        if st.st_size <= 0 or st.st_size > 8192:
            continue
        if ext in exts or any(h in bn for h in name_hints):
            b = _read_file_bytes(p, 8192)
            if b and b not in seeds:
                seeds.append(b)
        if len(seeds) >= 80:
            break

    # Deduplicate preserving order
    seen = set()
    uniq = []
    for s in seeds:
        h = hashlib.sha1(s).digest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(s)
    return uniq


def _is_text_like(data: bytes) -> bool:
    if not data:
        return True
    printable = 0
    for b in data:
        if b in (9, 10, 13) or 32 <= b <= 126:
            printable += 1
    return printable / max(1, len(data)) >= 0.85


def _mutate(rng: random.Random, data: bytes, max_len: int = 512) -> bytes:
    if data is None:
        data = b""
    if len(data) == 0:
        data = b"\n"
    text_like = _is_text_like(data)
    ops = ["flip", "del", "ins", "rep", "dup", "splice"]
    op = rng.choice(ops)
    b = bytearray(data)

    def rand_bytes(n: int) -> bytes:
        if text_like:
            alphabet = b" \t\r\n{}[]():,;\"'\\<>/=+-_*&|!@#$%^~`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            return bytes(rng.choice(alphabet) for _ in range(n))
        return bytes(rng.randrange(0, 256) for _ in range(n))

    if op == "flip":
        for _ in range(rng.randint(1, 4)):
            if not b:
                break
            i = rng.randrange(0, len(b))
            bit = 1 << rng.randrange(0, 8)
            b[i] ^= bit
    elif op == "del":
        if len(b) > 1:
            i = rng.randrange(0, len(b))
            j = rng.randrange(i + 1, min(len(b) + 1, i + 1 + rng.randint(1, max(1, len(b) // 3))))
            del b[i:j]
    elif op == "ins":
        n = rng.randint(1, 16 if text_like else 32)
        i = rng.randrange(0, len(b) + 1)
        ins = rand_bytes(n)
        b[i:i] = ins
    elif op == "rep":
        if b:
            i = rng.randrange(0, len(b))
            n = rng.randint(1, min(16 if text_like else 32, len(b) - i))
            b[i:i + n] = rand_bytes(n)
    elif op == "dup":
        if b:
            i = rng.randrange(0, len(b))
            n = rng.randint(1, min(32, len(b) - i))
            chunk = b[i:i + n]
            j = rng.randrange(0, len(b) + 1)
            b[j:j] = chunk
    elif op == "splice":
        # Add structured tokens
        tokens = [
            b"{", b"}", b"[", b"]", b"(", b")", b":", b",", b"\"", b"\\", b"\n",
            b"<", b">", b"</", b"/>", b"=", b"0", b"1"
        ]
        i = rng.randrange(0, len(b) + 1)
        t = rng.choice(tokens)
        b[i:i] = t

    if len(b) > max_len:
        # Truncate but keep head and tail segments
        keep = max_len
        if keep < 8:
            b = b[:keep]
        else:
            head = keep // 2
            tail = keep - head
            b = b[:head] + b[-tail:]
    return bytes(b)


def _ddmin(data: bytes, crashes_fn, time_limit: float) -> bytes:
    start = time.time()
    if not data:
        return data
    if not crashes_fn(data):
        return data
    n = 2
    cur = data
    while len(cur) >= 2 and time.time() - start < time_limit:
        length = len(cur)
        if n > length:
            break
        chunk_size = (length + n - 1) // n
        removed_any = False
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            a = i * chunk_size
            b = min(length, (i + 1) * chunk_size)
            if a >= b:
                continue
            trial = cur[:a] + cur[b:]
            if trial and crashes_fn(trial):
                cur = trial
                n = max(2, n - 1)
                removed_any = True
                break
            elif not trial and crashes_fn(trial):
                cur = trial
                n = 2
                removed_any = True
                break
        if not removed_any:
            if n >= length:
                break
            n = min(length, n * 2)
    return cur


class Solution:
    def solve(self, src_path: str) -> bytes:
        rng = random.Random(0xC0FFEE)
        with tempfile.TemporaryDirectory(prefix="arvo_src_") as td:
            _safe_extract_tar(src_path, td)
            root = _detect_project_root(td)

            # Build project
            build_ok = False
            build_dir = os.path.join(root, "build_asan")
            if _has_cmake(root):
                build_ok = _build_with_cmake(root, build_dir, use_asan=True)
                if not build_ok:
                    # Fallback without sanitizers
                    build_ok = _build_with_cmake(root, build_dir, use_asan=False)
            elif _has_make(root):
                build_ok = _build_with_make(root, use_asan=True)
                if not build_ok:
                    build_ok = _build_with_make(root, use_asan=False)
                build_dir = root
            else:
                build_dir = root

            preferred_targets = _find_cmake_targets(root) if _has_cmake(root) else []
            exe_candidates = []
            if build_ok and os.path.isdir(build_dir):
                exe_candidates = _find_built_executable(build_dir, preferred_targets)
            if not exe_candidates and build_ok and build_dir != root:
                exe_candidates = _find_built_executable(root, preferred_targets)

            # If no exe found, return a best-effort guess
            if not exe_candidates:
                return b'{"a":1,"a":2,"b":{"c":[1,2,3,4]}}\n'

            seeds = _gather_seeds(root)

            # Try candidates
            best_crash = None
            best_runner = None

            # Cache crashes per executable
            for exe in exe_candidates[:6]:
                runner = _TargetRunner(exe, os.path.dirname(exe))
                cache: Dict[bytes, bool] = {}

                def crashes_cached(d: bytes) -> bool:
                    h = hashlib.sha1(d).digest()
                    v = cache.get(h, None)
                    if v is not None:
                        return v
                    v = runner.crashes(d, timeout=1.0)
                    cache[h] = v
                    return v

                # Quick seed check
                for s in seeds:
                    if crashes_cached(s):
                        best_crash = s
                        best_runner = (runner, crashes_cached)
                        break
                if best_crash is not None:
                    break

                # Fuzzing loop
                start = time.time()
                budget = 18.0
                corpus = seeds[:]
                if not corpus:
                    corpus = [b"\n"]

                iters = 0
                while time.time() - start < budget:
                    iters += 1
                    base = corpus[rng.randrange(0, len(corpus))]
                    child = _mutate(rng, base, max_len=512)
                    if crashes_cached(child):
                        best_crash = child
                        best_runner = (runner, crashes_cached)
                        break
                    # Occasionally add to corpus
                    if len(corpus) < 64 and rng.random() < 0.25:
                        corpus.append(child)
                if best_crash is not None:
                    break

            if best_crash is None or best_runner is None:
                # Fallback
                return b'{"a":1,"a":2}\n{"b":3}\n{"c":{"d":4}}\n'

            runner, crashes_cached = best_runner

            # Minimize
            # Tight time limits to keep solve bounded
            if len(best_crash) > 80:
                best_crash = _ddmin(best_crash, crashes_cached, time_limit=20.0)
            else:
                best_crash = _ddmin(best_crash, crashes_cached, time_limit=8.0)

            # Try simple trimming for text-like
            if _is_text_like(best_crash):
                trimmed = best_crash
                for _ in range(200):
                    if len(trimmed) <= 1:
                        break
                    # Remove trailing whitespace/newlines
                    t2 = trimmed.rstrip(b" \t\r\n")
                    if t2 != trimmed and crashes_cached(t2):
                        trimmed = t2
                        continue
                    # Remove leading whitespace/newlines
                    t2 = trimmed.lstrip(b" \t\r\n")
                    if t2 != trimmed and crashes_cached(t2):
                        trimmed = t2
                        continue
                    break
                best_crash = trimmed

            return best_crash