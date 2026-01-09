import os
import re
import io
import sys
import tarfile
import zipfile
import time
import random
import shutil
import tempfile
import hashlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set


class Solution:
    def solve(self, src_path: str) -> bytes:
        start_time = time.time()
        time_budget = 95.0

        def time_left() -> float:
            return time_budget - (time.time() - start_time)

        def within_budget(min_left: float = 0.0) -> bool:
            return time_left() > min_left

        def safe_mkdir(p: str) -> None:
            os.makedirs(p, exist_ok=True)

        def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
            base = os.path.realpath(path)
            members = []
            for m in tar.getmembers():
                member_path = os.path.realpath(os.path.join(path, m.name))
                if member_path == base or member_path.startswith(base + os.sep):
                    members.append(m)
            tar.extractall(path=path, members=members)

        def extract_src(src: str) -> str:
            if os.path.isdir(src):
                return os.path.realpath(src)
            tmp = tempfile.mkdtemp(prefix="arvo_src_")
            sp = os.path.realpath(src)
            lower = sp.lower()
            try:
                if lower.endswith(".zip"):
                    with zipfile.ZipFile(sp, "r") as zf:
                        zf.extractall(tmp)
                else:
                    with tarfile.open(sp, "r:*") as tf:
                        safe_extract_tar(tf, tmp)
            except Exception:
                return os.path.realpath(src)

            entries = [os.path.join(tmp, e) for e in os.listdir(tmp)]
            dirs = [e for e in entries if os.path.isdir(e)]
            files = [e for e in entries if os.path.isfile(e)]
            if len(dirs) == 1 and not files:
                return os.path.realpath(dirs[0])
            return os.path.realpath(tmp)

        def read_text_files(root: str, max_bytes_per_file: int = 2_000_000) -> List[Tuple[str, str]]:
            exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl", ".inc", ".m", ".mm", ".rs", ".go", ".java", ".cs", ".py", ".txt", ".md"}
            out = []
            for p in Path(root).rglob("*"):
                if not p.is_file():
                    continue
                if p.stat().st_size > max_bytes_per_file:
                    continue
                if p.suffix.lower() not in exts and p.name not in {"Makefile", "CMakeLists.txt"}:
                    continue
                try:
                    data = p.read_bytes()
                except Exception:
                    continue
                try:
                    txt = data.decode("utf-8", "ignore")
                except Exception:
                    continue
                out.append((str(p), txt))
            return out

        def detect_build_system(root: str) -> str:
            if os.path.isfile(os.path.join(root, "CMakeLists.txt")):
                return "cmake"
            if os.path.isfile(os.path.join(root, "Makefile")) or os.path.isfile(os.path.join(root, "makefile")):
                return "make"
            return "manual"

        def run_cmd(cmd: List[str], cwd: str, env: dict, timeout: int) -> Tuple[int, bytes, bytes]:
            try:
                p = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                out = e.stdout or b""
                err = e.stderr or b""
                return 124, out, err
            except Exception as e:
                return 127, b"", str(e).encode("utf-8", "ignore")

        def build_project(root: str) -> Optional[str]:
            if not within_budget(20.0):
                return None

            env = os.environ.copy()
            env.setdefault("ASAN_OPTIONS", "abort_on_error=1:disable_coredump=1:detect_leaks=0:allocator_may_return_null=1")
            env.setdefault("UBSAN_OPTIONS", "abort_on_error=1:disable_coredump=1:print_stacktrace=1")
            env.setdefault("LSAN_OPTIONS", "detect_leaks=0")
            cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address"
            cxxflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address -std=c++17"
            ldflags = "-fsanitize=address"

            buildsys = detect_build_system(root)
            build_dir = os.path.join(root, "build_arvo")
            safe_mkdir(build_dir)

            def find_elf_execs(search_root: str) -> List[str]:
                execs = []
                for p in Path(search_root).rglob("*"):
                    if not p.is_file():
                        continue
                    sp = str(p)
                    if "/CMakeFiles/" in sp.replace("\\", "/"):
                        continue
                    if p.suffix.lower() in {".o", ".a", ".so", ".dylib", ".dll", ".obj", ".lib"}:
                        continue
                    try:
                        st = p.stat()
                    except Exception:
                        continue
                    if st.st_size < 10_000:
                        continue
                    if not os.access(sp, os.X_OK):
                        continue
                    try:
                        b = p.open("rb").read(4)
                    except Exception:
                        continue
                    if b != b"\x7fELF":
                        continue
                    execs.append(sp)
                execs.sort(key=lambda x: (-(os.path.getsize(x) if os.path.exists(x) else 0), x))
                return execs

            def pick_runnable(exes: List[str]) -> Optional[str]:
                for e in exes[:25]:
                    if not within_budget(10.0):
                        break
                    try:
                        p = subprocess.run([e, "--help"], cwd=os.path.dirname(e), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1.0, check=False)
                        if p.returncode in (0, 1, 2) or p.returncode < 0:
                            return e
                    except Exception:
                        continue
                return exes[0] if exes else None

            if buildsys == "cmake":
                cmake_cmd = [
                    "cmake",
                    "-S",
                    root,
                    "-B",
                    build_dir,
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                    f"-DCMAKE_C_FLAGS={cflags}",
                    f"-DCMAKE_CXX_FLAGS={cxxflags}",
                    f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}",
                    f"-DCMAKE_SHARED_LINKER_FLAGS={ldflags}",
                ]
                rc, out, err = run_cmd(cmake_cmd, root, env, timeout=max(10, min(120, int(max(10.0, time_left())))))
                if rc == 0 and within_budget(10.0):
                    build_cmd = ["cmake", "--build", build_dir, "-j", str(min(8, os.cpu_count() or 2))]
                    run_cmd(build_cmd, root, env, timeout=max(10, min(180, int(max(10.0, time_left())))))
                exes = find_elf_execs(build_dir) + find_elf_execs(root)
                return pick_runnable(exes)

            if buildsys == "make":
                env2 = env.copy()
                env2["CFLAGS"] = (env2.get("CFLAGS", "") + " " + cflags).strip()
                env2["CXXFLAGS"] = (env2.get("CXXFLAGS", "") + " " + cxxflags).strip()
                env2["LDFLAGS"] = (env2.get("LDFLAGS", "") + " " + ldflags).strip()
                make_cmd = ["make", "-j", str(min(8, os.cpu_count() or 2))]
                run_cmd(make_cmd, root, env2, timeout=max(10, min(180, int(max(10.0, time_left())))))
                exes = find_elf_execs(root)
                return pick_runnable(exes)

            # manual
            sources_c = []
            sources_cpp = []
            for p in Path(root).rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() in {".c"}:
                    sources_c.append(str(p))
                elif p.suffix.lower() in {".cc", ".cpp", ".cxx", ".mm"}:
                    sources_cpp.append(str(p))
            if not sources_c and not sources_cpp:
                return None
            out_exe = os.path.join(build_dir, "a.out")
            safe_mkdir(build_dir)
            if sources_cpp:
                cmd = [os.environ.get("CXX", "g++")] + cxxflags.split() + ["-I", root, "-o", out_exe] + sources_cpp + ldflags.split()
            else:
                cmd = [os.environ.get("CC", "gcc")] + cflags.split() + ["-I", root, "-o", out_exe] + sources_c + ldflags.split()
            run_cmd(cmd, root, env, timeout=max(10, min(180, int(max(10.0, time_left())))))
            if os.path.exists(out_exe) and os.access(out_exe, os.X_OK):
                return out_exe
            exes = find_elf_execs(build_dir) + find_elf_execs(root)
            return pick_runnable(exes)

        def scan_small_input_files(root: str) -> List[Tuple[str, bytes]]:
            hits = []
            keywords = ("poc", "crash", "uaf", "double", "free", "repro", "bad", "trigger", "asan", "heap")
            for p in Path(root).rglob("*"):
                if not p.is_file():
                    continue
                try:
                    st = p.stat()
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 4096:
                    continue
                name = p.name.lower()
                parent = str(p.parent).lower()
                score = 0
                for k in keywords:
                    if k in name:
                        score += 5
                    if k in parent:
                        score += 1
                if any(t in parent for t in ("test", "corpus", "seed", "input", "case", "repro", "fuzz")):
                    score += 2
                if p.suffix.lower() in {".in", ".txt", ".dat", ".poc", ".seed", ".crash", ".input"}:
                    score += 1
                if score <= 0:
                    continue
                try:
                    data = p.read_bytes()
                except Exception:
                    continue
                hits.append((score, str(p), data))
            hits.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
            return [(p, d) for _, p, d in hits]

        def extract_strings_and_cmds(texts: List[Tuple[str, str]]) -> Tuple[Set[str], Set[str], str]:
            all_txt = "\n".join(t for _, t in texts)
            string_lits = set()

            # String literals
            for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', all_txt):
                s = m.group(1)
                s = s.encode("utf-8", "ignore").decode("unicode_escape", "ignore")
                if 1 <= len(s) <= 48:
                    string_lits.add(s)

            # Command tokens likely used in comparisons
            cmds = set()
            for m in re.finditer(r'==\s*"([A-Za-z0-9_\-]{1,24})"', all_txt):
                cmds.add(m.group(1))
            for m in re.finditer(r'strcmp\s*\(\s*[^,]+,\s*"([A-Za-z0-9_\-]{1,24})"\s*\)\s*==\s*0', all_txt):
                cmds.add(m.group(1))
            for m in re.finditer(r'!=\s*"([A-Za-z0-9_\-]{1,24})"', all_txt):
                cmds.add(m.group(1))
            for m in re.finditer(r'"(add|ADD|Add|insert|INSERT|push|PUSH|append|APPEND|new|NEW|create|CREATE|node|NODE|root|ROOT|del|DEL|delete|DELETE|remove|REMOVE)"', all_txt):
                cmds.add(m.group(1))
            return string_lits, cmds, all_txt

        def infer_invocation_methods(all_src_text: str) -> List[str]:
            src = all_src_text
            fileish = False
            if "argv[1]" in src or "argc" in src and ("< 2" in src or "<2" in src or "!= 2" in src or "!=2" in src):
                fileish = True
            if "ifstream" in src or "fopen(argv[1]" in src or "std::ifstream" in src:
                fileish = True
            methods = []
            if fileish:
                methods.extend(["file", "stdin"])
            else:
                methods.extend(["stdin", "file"])
            methods.append("dashfile")
            # de-dup
            out = []
            for m in methods:
                if m not in out:
                    out.append(m)
            return out

        def make_candidate_inputs(string_lits: Set[str], cmds: Set[str], all_src_text: str) -> List[bytes]:
            cands: List[bytes] = []

            def add(b: bytes) -> None:
                if b is None:
                    return
                if len(b) == 0:
                    return
                cands.append(b)

            # If JSON seems used
            if "nlohmann::json" in all_src_text or "rapidjson" in all_src_text or "json::" in all_src_text or "nlohmann" in all_src_text:
                keys = set(re.findall(r'\[\s*"([A-Za-z0-9_\-]{1,32})"\s*\]', all_src_text))
                # Generic duplicates
                if "nodes" in keys and "id" in keys:
                    add(b'{"nodes":[{"id":0},{"id":0}]}\n')
                if "children" in keys and "id" in keys:
                    add(b'{"id":0,"children":[{"id":1},{"id":1}]}\n')
                add(b'{"a":[0,0]}\n')
                add(b'[{"id":0},{"id":0}]\n')

            # Try to find a canonical "add"
            add_cmds = [c for c in cmds if c.lower() in ("add", "insert", "append", "push")]
            new_cmds = [c for c in cmds if c.lower() in ("new", "create", "node", "alloc", "make", "root", "init")]
            del_cmds = [c for c in cmds if c.lower() in ("del", "delete", "remove", "free", "drop", "erase")]

            def choose_best(options: List[str], default: str) -> str:
                if not options:
                    return default
                # Prefer lowercase token if present
                for opt in options:
                    if opt == default:
                        return opt
                for opt in options:
                    if opt.lower() == default:
                        return opt
                return sorted(options, key=lambda x: (len(x), x))[0]

            add_cmd = choose_best(add_cmds, "add")
            new_cmd = choose_best(new_cmds, "new")
            del_cmd = choose_best(del_cmds, "del")

            # Command style candidates
            # Setup + duplicate add
            add(f"{new_cmd} 0\n{new_cmd} 1\n{add_cmd} 0 1\n{add_cmd} 0 1\n".encode("utf-8"))
            add(f"{new_cmd} 0\n{add_cmd} 0 1\n{add_cmd} 0 1\n".encode("utf-8"))
            add(f"{add_cmd} 0 1\n{add_cmd} 0 1\n".encode("utf-8"))
            add(f"{add_cmd} a b\n{add_cmd} a b\n".encode("utf-8"))
            add(f"{add_cmd} 0 0\n{add_cmd} 0 0\n".encode("utf-8"))
            add(f"{add_cmd}\n{add_cmd}\n".encode("utf-8"))

            # Add with big sizes (exception via bad_alloc/length_error)
            bigs = [
                "18446744073709551615",
                "9223372036854775807",
                "4294967295",
                "2147483647",
                "2147483648",
                "-1",
                "-2147483648",
                "9999999999999999999999999999",
            ]
            for b in bigs[:6]:
                add(f"{add_cmd} 0 {b}\n{add_cmd} 0 {b}\n".encode("utf-8"))
                add(f"{new_cmd} 0\n{add_cmd} 0 {b}\n".encode("utf-8"))

            # Edge-list numeric formats
            add(b"2\n0 1\n0 1\n")
            add(b"3\n0 1\n0 2\n0 2\n")
            add(b"4\n0 1\n0 2\n0 3\n0 3\n")

            # With huge N to trigger allocation paths in parsers
            add(b"1000000000\n0 1\n")
            add(b"4294967295\n0 1\n")
            add(b"-1\n0 1\n")

            # Try with delete in between (UAF patterns)
            add(f"{new_cmd} 0\n{new_cmd} 1\n{add_cmd} 0 1\n{del_cmd} 1\n{add_cmd} 0 1\n".encode("utf-8"))
            add(f"{new_cmd} 0\n{new_cmd} 1\n{add_cmd} 0 1\n{add_cmd} 0 1\n{del_cmd} 1\n".encode("utf-8"))

            # Some generic "key=value" style
            add(b"op=add\nx=0\ny=1\nop=add\nx=0\ny=1\n")
            add(b"ADD 0 1\nADD 0 1\n")
            add(b"add,0,1\nadd,0,1\n")

            # Pull a few likely keywords to create a line-based blob
            likely = []
            for s in sorted(string_lits, key=lambda x: (len(x), x)):
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,15}", s) and s.lower() in ("add", "new", "node", "create", "insert", "remove", "delete", "del", "root", "init", "set", "put"):
                    likely.append(s)
            if likely:
                toks = " ".join(likely[:6]).encode("utf-8") + b"\n"
                add(toks + toks)

            # De-dup and keep short first
            uniq = {}
            for x in cands:
                uniq[x] = True
            out = list(uniq.keys())
            out.sort(key=lambda x: (len(x), x))
            return out

        def is_crash_result(rc: int, out: bytes, err: bytes) -> bool:
            if rc == 0:
                return False
            blob = (out or b"") + b"\n" + (err or b"")
            lower = blob.lower()
            crash_markers = [
                b"addresssanitizer",
                b"undefinedbehaviorsanitizer",
                b"ubsan",
                b"heap-use-after-free",
                b"use-after-free",
                b"double free",
                b"attempting double-free",
                b"free(): double free",
                b"double free or corruption",
                b"invalid pointer",
                b"corrupted size vs. prev_size",
                b"malloc():",
                b"glibc detected",
                b"segmentation fault",
                b"stack trace",
                b"sanitizer",
            ]
            for m in crash_markers:
                if m in lower:
                    return True
            # signal-based termination often negative
            if rc < 0:
                return True
            # Heuristic: abort often 134
            if rc in (134, 139):
                return True
            return False

        class TargetRunner:
            def __init__(self, exe: str, methods: List[str]):
                self.exe = exe
                self.methods = methods
                self.env = os.environ.copy()
                self.env.setdefault("ASAN_OPTIONS", "abort_on_error=1:disable_coredump=1:detect_leaks=0:allocator_may_return_null=1")
                self.env.setdefault("UBSAN_OPTIONS", "abort_on_error=1:disable_coredump=1:print_stacktrace=1")
                self.env.setdefault("LSAN_OPTIONS", "detect_leaks=0")
                self.cache: Dict[Tuple[str, str], bool] = {}
                self.last_method: Optional[str] = None

            def test(self, data: bytes, timeout: float = 0.8) -> bool:
                if not data:
                    return False
                h = hashlib.sha1(data).hexdigest()
                # Prefer last successful method
                methods = self.methods[:]
                if self.last_method and self.last_method in methods:
                    methods.remove(self.last_method)
                    methods.insert(0, self.last_method)
                for m in methods:
                    key = (m, h)
                    if key in self.cache:
                        if self.cache[key]:
                            self.last_method = m
                            return True
                        continue
                    ok = self._run_method(m, data, timeout=timeout)
                    self.cache[key] = ok
                    if ok:
                        self.last_method = m
                        return True
                return False

            def _run_method(self, method: str, data: bytes, timeout: float) -> bool:
                try:
                    if method == "stdin":
                        p = subprocess.run(
                            [self.exe],
                            input=data,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=timeout,
                            env=self.env,
                            check=False,
                        )
                        return is_crash_result(p.returncode, p.stdout, p.stderr)
                    elif method == "file":
                        with tempfile.NamedTemporaryFile(prefix="arvo_in_", delete=False) as tf:
                            tf.write(data)
                            tf.flush()
                            name = tf.name
                        try:
                            p = subprocess.run(
                                [self.exe, name],
                                input=b"",
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout,
                                env=self.env,
                                check=False,
                            )
                            return is_crash_result(p.returncode, p.stdout, p.stderr)
                        finally:
                            try:
                                os.unlink(name)
                            except Exception:
                                pass
                    elif method == "dashfile":
                        with tempfile.NamedTemporaryFile(prefix="arvo_in_", delete=False) as tf:
                            tf.write(data)
                            tf.flush()
                            name = tf.name
                        try:
                            # Try reading from stdin when "-" passed
                            p = subprocess.run(
                                [self.exe, "-"],
                                input=data,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout,
                                env=self.env,
                                check=False,
                            )
                            if is_crash_result(p.returncode, p.stdout, p.stderr):
                                return True
                            # Try program that reads from file but treats "-" specially; also try giving a file anyway
                            p2 = subprocess.run(
                                [self.exe, name],
                                input=b"",
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout,
                                env=self.env,
                                check=False,
                            )
                            return is_crash_result(p2.returncode, p2.stdout, p2.stderr)
                        finally:
                            try:
                                os.unlink(name)
                            except Exception:
                                pass
                except subprocess.TimeoutExpired as e:
                    out = e.stdout or b""
                    err = e.stderr or b""
                    # Treat timeouts as non-crash to avoid false positives
                    return False
                except Exception:
                    return False
                return False

        def ddmin_bytes(data: bytes, test_fn, min_len: int = 1, max_rounds: int = 200, min_time_left: float = 10.0) -> bytes:
            if not data or len(data) <= min_len:
                return data
            if not test_fn(data):
                return data

            rounds = 0
            n = 2
            cur = data
            while len(cur) >= 2 and rounds < max_rounds and within_budget(min_time_left):
                rounds += 1
                length = len(cur)
                if n > length:
                    break
                chunk = length // n
                if chunk == 0:
                    break
                reduced = False
                for i in range(n):
                    if not within_budget(min_time_left):
                        break
                    start = i * chunk
                    end = length if i == n - 1 else (i + 1) * chunk
                    cand = cur[:start] + cur[end:]
                    if len(cand) < min_len:
                        continue
                    if test_fn(cand):
                        cur = cand
                        n = max(2, n - 1)
                        reduced = True
                        break
                if not reduced:
                    if n >= length:
                        break
                    n = min(length, n * 2)
            return cur

        def ddmin_lines(data: bytes, test_fn, min_time_left: float = 10.0) -> bytes:
            if not test_fn(data):
                return data
            try:
                txt = data.decode("utf-8")
            except Exception:
                return data
            if "\n" not in txt:
                return data
            lines = txt.splitlines(True)
            if len(lines) <= 1:
                return data

            def rebuild(ls: List[str]) -> bytes:
                return "".join(ls).encode("utf-8", "ignore")

            cur_lines = lines
            n = 2
            rounds = 0
            while len(cur_lines) >= 2 and rounds < 200 and within_budget(min_time_left):
                rounds += 1
                L = len(cur_lines)
                if n > L:
                    break
                chunk = L // n
                if chunk == 0:
                    break
                reduced = False
                for i in range(n):
                    if not within_budget(min_time_left):
                        break
                    start = i * chunk
                    end = L if i == n - 1 else (i + 1) * chunk
                    cand_lines = cur_lines[:start] + cur_lines[end:]
                    if not cand_lines:
                        continue
                    cand = rebuild(cand_lines)
                    if test_fn(cand):
                        cur_lines = cand_lines
                        n = max(2, n - 1)
                        reduced = True
                        break
                if not reduced:
                    if n >= L:
                        break
                    n = min(L, n * 2)
            return rebuild(cur_lines)

        # Main solve steps
        root = extract_src(src_path)
        texts = read_text_files(root)
        string_lits, cmds, all_src_text = extract_strings_and_cmds(texts)
        methods = infer_invocation_methods(all_src_text)

        exe = build_project(root)
        if exe is None:
            # Best-effort fallback: try return likely poc file or heuristic input
            smalls = scan_small_input_files(root)
            if smalls:
                return smalls[0][1]
            return b"2\n0 1\n0 1\n"

        runner = TargetRunner(exe, methods)

        # Try existing small inputs in repository first
        best: Optional[bytes] = None
        if within_budget(20.0):
            for path, data in scan_small_input_files(root)[:50]:
                if not within_budget(10.0):
                    break
                if runner.test(data, timeout=1.2):
                    best = data
                    break

        # Try generated heuristic candidates
        if best is None and within_budget(20.0):
            cands = make_candidate_inputs(string_lits, cmds, all_src_text)
            for data in cands[:200]:
                if not within_budget(10.0):
                    break
                if runner.test(data, timeout=1.0):
                    best = data
                    break

        # Light fuzz if needed
        if best is None and within_budget(25.0):
            dict_tokens = []
            for s in sorted(cmds, key=lambda x: (len(x), x)):
                if 1 <= len(s) <= 16 and re.fullmatch(r"[A-Za-z0-9_\-]+", s):
                    dict_tokens.append(s.encode("utf-8"))
            if not dict_tokens:
                dict_tokens = [b"add", b"new", b"node", b"create", b"del", b"remove"]

            seeds = [
                b"2\n0 1\n0 1\n",
                b"add 0 1\nadd 0 1\n",
                b"new 0\nnew 1\nadd 0 1\nadd 0 1\n",
                b'{"nodes":[{"id":0},{"id":0}]}\n',
            ]
            seeds = [s for s in seeds if s]
            if seeds:
                best_seed = seeds[0]
            else:
                best_seed = b"2\n0 1\n0 1\n"

            def mutate(data: bytes) -> bytes:
                r = random.random()
                if r < 0.35:
                    tok = random.choice(dict_tokens)
                    pos = random.randint(0, len(data))
                    return data[:pos] + tok + (b" " if random.random() < 0.5 else b"\n") + data[pos:]
                if r < 0.55:
                    # duplicate a slice
                    if len(data) < 2:
                        return data + data
                    a = random.randint(0, len(data) - 1)
                    b = random.randint(a + 1, min(len(data), a + 1 + random.randint(1, 32)))
                    pos = random.randint(0, len(data))
                    return data[:pos] + data[a:b] + data[pos:]
                if r < 0.75:
                    # insert number
                    nums = [b"0", b"1", b"-1", b"2147483647", b"4294967295", b"18446744073709551615"]
                    num = random.choice(nums)
                    pos = random.randint(0, len(data))
                    return data[:pos] + num + (b" " if random.random() < 0.7 else b"\n") + data[pos:]
                else:
                    # delete a range
                    if len(data) <= 1:
                        return data
                    a = random.randint(0, len(data) - 1)
                    b = random.randint(a + 1, min(len(data), a + 1 + random.randint(1, 32)))
                    return data[:a] + data[b:]

            random.seed(0x41356)
            tries = 0
            while within_budget(10.0) and tries < 800:
                tries += 1
                base = random.choice(seeds) if seeds else best_seed
                cand = base
                for _ in range(random.randint(1, 6)):
                    cand = mutate(cand)
                if len(cand) > 8192:
                    cand = cand[:8192]
                if runner.test(cand, timeout=0.8):
                    best = cand
                    break

        if best is None:
            return b"2\n0 1\n0 1\n"

        # Minimize
        if within_budget(20.0):
            best = best.rstrip(b"\x00")
            if runner.test(best, timeout=1.2):
                best2 = ddmin_lines(best, lambda x: runner.test(x, timeout=1.2), min_time_left=12.0)
                if best2 and runner.test(best2, timeout=1.2):
                    best = best2
                best3 = ddmin_bytes(best, lambda x: runner.test(x, timeout=1.2), min_len=1, max_rounds=250, min_time_left=10.0)
                if best3 and runner.test(best3, timeout=1.2):
                    best = best3

        return best if isinstance(best, (bytes, bytearray)) else bytes(best)