import os
import re
import tarfile
import tempfile
import shutil
import subprocess
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class Solution:
    def solve(self, src_path: str) -> bytes:
        rnd = random.Random(0xC0FFEE)

        def safe_extract(tar: tarfile.TarFile, path: str) -> None:
            base = os.path.realpath(path)
            for member in tar.getmembers():
                member_path = os.path.realpath(os.path.join(path, member.name))
                if not (member_path == base or member_path.startswith(base + os.sep)):
                    continue
            tar.extractall(path)

        def find_project_root(extract_dir: str) -> str:
            entries = [os.path.join(extract_dir, x) for x in os.listdir(extract_dir)]
            dirs = [p for p in entries if os.path.isdir(p)]
            if len(dirs) == 1 and not any(os.path.isfile(p) for p in entries):
                return dirs[0]
            return extract_dir

        def list_files(root: str):
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    yield os.path.join(dp, fn)

        def read_text(path: str, limit: int = 2_000_000) -> str:
            try:
                with open(path, "rb") as f:
                    data = f.read(limit)
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return ""

        def has_binary_io(sources_text: str) -> bool:
            pats = [
                r"\.read\s*\(",
                r"istream::read\s*\(",
                r"fread\s*\(",
                r"read\s*\(\s*\w+\s*,",
                r"reinterpret_cast\s*<\s*char\s*\*\s*>",
                r"sizeof\s*\(",
            ]
            score = 0
            for p in pats:
                if re.search(p, sources_text):
                    score += 1
            return score >= 3

        def extract_cmd_tokens(sources_text: str):
            tokens = set()

            for m in re.finditer(r'==\s*"([^"\\\n\r]{1,24})"', sources_text):
                tokens.add(m.group(1))
            for m in re.finditer(r'"([^"\\\n\r]{1,24})"\s*==', sources_text):
                tokens.add(m.group(1))
            for m in re.finditer(r"strcmp\s*\(\s*[^,]+,\s*\"([^\"\\\n\r]{1,24})\"\s*\)", sources_text):
                tokens.add(m.group(1))
            for m in re.finditer(r"case\s+'([^'\\\n\r])'\s*:", sources_text):
                tokens.add(m.group(1))

            # Common command-ish identifiers
            for m in re.finditer(r"\b(add|ADD|insert|INSERT|push|PUSH|node|NODE|tree|TREE|del|DEL|remove|REMOVE)\b", sources_text):
                tokens.add(m.group(1))

            good = []
            for t in tokens:
                if not t:
                    continue
                if len(t) > 24:
                    continue
                if any(c in t for c in "\t\r"):
                    continue
                if all(32 <= ord(c) < 127 for c in t):
                    good.append(t)
            good = sorted(set(good), key=lambda x: (len(x), x))
            return good

        def extract_seed_inputs(root: str):
            seeds = []
            exts_bad = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".md", ".rst", ".cmake", ".make", ".o", ".a", ".so", ".dll", ".dylib", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".tgz", ".xz", ".bz2"}
            name_good = ("sample", "test", "input", "poc", "corpus", "seed", "example", "demo")
            for p in list_files(root):
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 1_000_000:
                    continue
                ext = Path(p).suffix.lower()
                if ext in exts_bad:
                    continue
                bn = os.path.basename(p).lower()
                if not any(k in bn for k in name_good):
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except Exception:
                    continue
                if not data:
                    continue
                # Prefer mostly printable seeds
                printable = sum(1 for b in data[:4096] if b in b"\n\r\t" or 32 <= b < 127)
                ratio = printable / max(1, min(len(data), 4096))
                if ratio < 0.75:
                    continue
                seeds.append(data)
            seeds.sort(key=len)
            return seeds[:20]

        def detect_file_mode(sources_text: str) -> bool:
            if re.search(r"\bargv\s*\[\s*1\s*\]", sources_text) and re.search(r"\b(ifstream|fopen)\b", sources_text):
                return True
            if re.search(r"\bargc\s*>\s*1\b", sources_text) and re.search(r"\bargv\s*\[\s*1\s*\]", sources_text):
                return True
            return False

        def build_direct(root: str, out_exe: str) -> bool:
            srcs = []
            headers_dirs = set()
            mains = []
            for p in list_files(root):
                ext = Path(p).suffix.lower()
                if ext in {".cpp", ".cc", ".cxx"}:
                    srcs.append(p)
                    txt = read_text(p, limit=400_000)
                    if re.search(r"\bint\s+main\s*\(", txt):
                        mains.append(p)
                elif ext in {".h", ".hpp", ".hh"}:
                    headers_dirs.add(os.path.dirname(p))
            if not srcs:
                return False

            chosen_main = None
            if mains:
                def main_score(path: str):
                    lp = path.lower()
                    s = 0
                    if "test" in lp or "unittest" in lp or "benchmark" in lp:
                        s += 10
                    if os.path.basename(lp) in ("main.cpp", "main.cc", "main.cxx"):
                        s -= 5
                    if "/src/" in lp or "\\src\\" in lp:
                        s -= 2
                    s += len(path)
                    return s
                mains_sorted = sorted(mains, key=main_score)
                chosen_main = mains_sorted[0]
                srcs2 = []
                for p in srcs:
                    if p == chosen_main:
                        srcs2.append(p)
                    else:
                        t = read_text(p, limit=400_000)
                        if re.search(r"\bint\s+main\s*\(", t):
                            continue
                        srcs2.append(p)
                srcs = srcs2

            inc_dirs = set(headers_dirs)
            inc_dirs.add(root)
            # also add all unique parent dirs up to depth 4 to help includes
            for p in srcs[:200]:
                d = os.path.dirname(p)
                inc_dirs.add(d)
                pd = d
                for _ in range(3):
                    np = os.path.dirname(pd)
                    if np and np != pd:
                        inc_dirs.add(np)
                    pd = np

            inc_args = []
            for d in sorted(inc_dirs):
                inc_args.append("-I" + d)

            base_cmd = ["g++", "-std=c++17", "-O0", "-g", "-fno-omit-frame-pointer"]
            # Try ASan first, then without
            for san in (True, False):
                cmd = list(base_cmd)
                if san:
                    cmd += ["-fsanitize=address", "-fsanitize=undefined"]
                cmd += inc_args
                cmd += srcs
                cmd += ["-o", out_exe, "-pthread"]
                try:
                    r = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                except Exception:
                    continue
                if r.returncode == 0 and os.path.exists(out_exe):
                    return True
            return False

        def build_cmake(root: str, build_dir: str) -> bool:
            cmakelists = os.path.join(root, "CMakeLists.txt")
            if not os.path.exists(cmakelists):
                return False
            os.makedirs(build_dir, exist_ok=True)
            flags = "-O0 -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
            cfg_cmd = ["cmake", "-S", root, "-B", build_dir, f"-DCMAKE_CXX_FLAGS={flags}"]
            try:
                r1 = subprocess.run(cfg_cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            except Exception:
                return False
            if r1.returncode != 0:
                return False
            build_cmd = ["cmake", "--build", build_dir, "-j", str(min(8, os.cpu_count() or 2))]
            try:
                r2 = subprocess.run(build_cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            except Exception:
                return False
            return r2.returncode == 0

        def build_make(root: str) -> bool:
            makefile = None
            for name in ("Makefile", "makefile", "GNUmakefile"):
                p = os.path.join(root, name)
                if os.path.exists(p):
                    makefile = p
                    break
            if not makefile:
                return False
            env = os.environ.copy()
            env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " -O0 -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined").strip()
            try:
                r = subprocess.run(["make", "-j", str(min(8, os.cpu_count() or 2))], cwd=root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            except Exception:
                return False
            return r.returncode == 0

        def find_executables(search_root: str):
            exes = []
            for p in list_files(search_root):
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                if not os.path.isfile(p):
                    continue
                if st.st_size < 10_000:
                    continue
                if os.access(p, os.X_OK):
                    bn = os.path.basename(p)
                    if bn.endswith((".o", ".a", ".so", ".dylib", ".dll")):
                        continue
                    exes.append((st.st_mtime, st.st_size, p))
            exes.sort(reverse=True)
            return [p for _, _, p in exes]

        def parse_cmake_targets(root: str):
            cmakelists = os.path.join(root, "CMakeLists.txt")
            if not os.path.exists(cmakelists):
                return []
            txt = read_text(cmakelists, limit=1_000_000)
            targets = re.findall(r"add_executable\s*\(\s*([A-Za-z0-9_\-\.]+)", txt)
            out = []
            for t in targets:
                if t.lower().startswith(("test", "unit", "bench")):
                    continue
                out.append(t)
            return out

        def pick_executable(root: str, build_dir: str, direct_exe: str):
            # Prefer direct_exe if exists
            if direct_exe and os.path.exists(direct_exe) and os.access(direct_exe, os.X_OK):
                return direct_exe

            # Prefer CMake targets in build dir
            targets = parse_cmake_targets(root)
            if targets and os.path.isdir(build_dir):
                for t in targets:
                    cand = os.path.join(build_dir, t)
                    if os.path.exists(cand) and os.access(cand, os.X_OK):
                        return cand
                    cand2 = os.path.join(build_dir, "bin", t)
                    if os.path.exists(cand2) and os.access(cand2, os.X_OK):
                        return cand2

            # Scan build dir first, then root
            for base in (build_dir, root):
                if base and os.path.isdir(base):
                    exes = find_executables(base)
                    if exes:
                        # avoid obvious tool executables
                        for p in exes:
                            bn = os.path.basename(p).lower()
                            if bn in ("cmake", "make"):
                                continue
                            if "test" in bn or "unittest" in bn or "bench" in bn:
                                continue
                            return p
                        return exes[0]
            return None

        def crash_signature(stderr: bytes, returncode: int) -> bool:
            if returncode == 0:
                return False
            s = stderr.lower()
            patterns = [
                b"heap-use-after-free",
                b"use-after-free",
                b"attempting double-free",
                b"double free",
                b"double-free",
                b"free(): double free detected",
                b"asan: heap-use-after-free",
                b"addresssanitizer: heap-use-after-free",
                b"addresssanitizer: attempting double-free",
            ]
            if any(p in s for p in patterns):
                return True
            # Abort/SEGV with allocator message
            if returncode < 0 or returncode in (134, 139):
                if b"free()" in s or b"malloc" in s or b"corrupt" in s or b"sanitizer" in s:
                    return True
            return False

        def run_target(exe: str, cwd: str, data: bytes, file_mode: bool, timeout: float = 0.5):
            env = os.environ.copy()
            env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "")
            if env["ASAN_OPTIONS"]:
                env["ASAN_OPTIONS"] += ":"
            env["ASAN_OPTIONS"] += "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1"
            env["UBSAN_OPTIONS"] = "halt_on_error=1:abort_on_error=1:print_stacktrace=1"

            if file_mode:
                tf = None
                try:
                    fd, path = tempfile.mkstemp(prefix="poc_", suffix=".bin", dir=cwd)
                    os.close(fd)
                    with open(path, "wb") as f:
                        f.write(data)
                    tf = path
                    r = subprocess.run([exe, path], cwd=cwd, input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=env, timeout=timeout)
                    return crash_signature(r.stderr, r.returncode), r
                except subprocess.TimeoutExpired:
                    return False, None
                except Exception:
                    return False, None
                finally:
                    if tf:
                        try:
                            os.unlink(tf)
                        except Exception:
                            pass
            else:
                try:
                    r = subprocess.run([exe], cwd=cwd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=env, timeout=timeout)
                    return crash_signature(r.stderr, r.returncode), r
                except subprocess.TimeoutExpired:
                    return False, None
                except Exception:
                    return False, None

        def ddmin(data: bytes, is_crash_func, time_limit: float):
            start = time.time()
            if not data:
                return data
            n = 2
            best = data
            while len(best) >= 2 and time.time() - start < time_limit:
                chunk = max(1, len(best) // n)
                if chunk <= 0:
                    break
                reduced = False
                for i in range(n):
                    if time.time() - start >= time_limit:
                        break
                    a = i * chunk
                    b = len(best) if i == n - 1 else (i + 1) * chunk
                    trial = best[:a] + best[b:]
                    if trial and is_crash_func(trial):
                        best = trial
                        n = max(2, n - 1)
                        reduced = True
                        break
                if not reduced:
                    if n >= len(best):
                        break
                    n = min(len(best), n * 2)
            # Byte-level simplification
            if time.time() - start < time_limit:
                ba = bytearray(best)
                # Try replace bytes with newline/space/0
                for i in range(len(ba)):
                    if time.time() - start >= time_limit:
                        break
                    orig = ba[i]
                    for repl in (0x0A, 0x20, 0x00, ord('0'), ord('A')):
                        if repl == orig:
                            continue
                        ba[i] = repl
                        trial = bytes(ba)
                        if is_crash_func(trial):
                            best = trial
                            break
                        ba[i] = orig
                # Try truncation
                for cut in range(len(best), 0, -1):
                    if time.time() - start >= time_limit:
                        break
                    trial = best[:cut]
                    if is_crash_func(trial):
                        best = trial
                    else:
                        break
            return best

        def mutate_text(data: bytes, tokens):
            s = data.decode("utf-8", errors="ignore")
            if not s:
                s = ""
            lines = s.splitlines(True)
            def rand_int_str():
                choices = [
                    "0", "1", "-1", "2", "3", "7", "8", "9", "10",
                    "127", "128", "255", "256", "511", "512", "1023", "1024",
                    "2047", "2048", "4095", "4096", "65535", "65536",
                    "2147483647", "-2147483648", "4294967295", "9223372036854775807",
                    "9999999999", "-9999999999"
                ]
                return rnd.choice(choices)

            def rand_tok():
                if tokens:
                    return rnd.choice(tokens)
                return rnd.choice(["add", "ADD", "insert", "node", "push", "set"])

            op = rnd.randrange(0, 9)
            if op == 0:
                # append a command-like line
                cmd = rand_tok()
                line = cmd
                argc = rnd.randrange(0, 4)
                for _ in range(argc):
                    if rnd.random() < 0.8:
                        line += " " + rand_int_str()
                    else:
                        line += " " + rand_tok()
                line += "\n"
                lines.append(line)
            elif op == 1 and lines:
                # duplicate a line
                i = rnd.randrange(0, len(lines))
                lines.insert(i, lines[i])
            elif op == 2 and lines:
                # delete a line
                i = rnd.randrange(0, len(lines))
                del lines[i]
            elif op == 3:
                # add a count header
                body = "".join(lines)
                header = str(rnd.randrange(0, 64)) + "\n"
                return (header + body).encode("utf-8", errors="ignore")
            elif op == 4 and lines:
                # modify a line by injecting number
                i = rnd.randrange(0, len(lines))
                l = lines[i]
                parts = re.split(r"(\s+)", l.rstrip("\n"))
                if parts:
                    pos = rnd.randrange(0, len(parts)+1)
                    parts.insert(pos, " " + rand_int_str() + " ")
                    lines[i] = "".join(parts).strip() + "\n"
            elif op == 5:
                # create many repeated add-like commands to hit constraints
                cmd = None
                for t in tokens:
                    if re.search(r"add|insert|push", t, re.IGNORECASE):
                        cmd = t
                        break
                if cmd is None:
                    cmd = "add"
                k = rnd.randrange(6, 40)
                a = rand_int_str()
                b = rand_int_str()
                rep = "".join([f"{cmd} {a} {b}\n" for _ in range(k)])
                return rep.encode("utf-8", errors="ignore")
            elif op == 6:
                # JSON-ish attempt
                cmd = None
                for t in tokens:
                    if re.search(r"add|insert|push", t, re.IGNORECASE):
                        cmd = t
                        break
                if cmd is None:
                    cmd = "add"
                k = rnd.randrange(4, 20)
                ops = []
                for _ in range(k):
                    ops.append(f'{{"op":"{cmd}","a":{rand_int_str()},"b":{rand_int_str()}}}')
                js = '{"ops":[' + ",".join(ops) + ']}\n'
                return js.encode("utf-8", errors="ignore")
            elif op == 7:
                # random token soup
                k = rnd.randrange(10, 80)
                out = []
                for _ in range(k):
                    if rnd.random() < 0.7:
                        out.append(rand_tok())
                    else:
                        out.append(rand_int_str())
                return (" ".join(out) + "\n").encode("utf-8", errors="ignore")
            else:
                # slight char-level mutation
                b = bytearray(data if data else b"\n")
                if b:
                    for _ in range(rnd.randrange(1, 5)):
                        i = rnd.randrange(0, len(b))
                        b[i] = rnd.randrange(0x20, 0x7F)
                return bytes(b)

            out = "".join(lines)
            if not out.endswith("\n"):
                out += "\n"
            return out.encode("utf-8", errors="ignore")

        def mutate_binary(data: bytes):
            b = bytearray(data if data else b"\x00")
            op = rnd.randrange(0, 7)
            if op == 0 and len(b) > 1:
                # truncate
                newlen = rnd.randrange(1, len(b))
                b = b[:newlen]
            elif op == 1:
                # append random bytes
                n = rnd.randrange(1, 32)
                b.extend(rnd.randrange(0, 256) for _ in range(n))
            elif op == 2 and len(b) > 0:
                # flip bits
                for _ in range(rnd.randrange(1, 8)):
                    i = rnd.randrange(0, len(b))
                    b[i] ^= 1 << rnd.randrange(0, 8)
            elif op == 3 and len(b) > 4:
                # overwrite a 32-bit value with special
                i = rnd.randrange(0, len(b) - 4)
                val = rnd.choice([0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff, 0x10000, 0x40000000])
                b[i:i+4] = val.to_bytes(4, "little", signed=False)
            elif op == 4 and len(b) > 0:
                # delete a chunk
                a = rnd.randrange(0, len(b))
                c = rnd.randrange(1, min(16, len(b) - a) + 1)
                del b[a:a+c]
                if not b:
                    b = bytearray(b"\x00")
            elif op == 5:
                # repeat pattern
                pat = bytes(rnd.randrange(0, 256) for _ in range(rnd.randrange(1, 8)))
                b = bytearray(pat * rnd.randrange(1, 32))
            else:
                # random fill
                n = rnd.randrange(1, 128)
                b = bytearray(rnd.randrange(0, 256) for _ in range(n))
            return bytes(b)

        tmp = tempfile.mkdtemp(prefix="arvo_poc_")
        try:
            # Extract
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    safe_extract(tar, tmp)
            except Exception:
                # If tar cannot be opened, treat src_path as a directory
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, tmp, dirs_exist_ok=True)
                else:
                    return b"A" * 60

            root = find_project_root(tmp)

            # Collect sources
            src_text_parts = []
            src_files = []
            for p in list_files(root):
                if Path(p).suffix.lower() in {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".c"}:
                    src_files.append(p)
            for p in src_files[:400]:
                src_text_parts.append(read_text(p, limit=600_000))
            sources_text = "\n".join(src_text_parts)

            tokens = extract_cmd_tokens(sources_text)
            file_mode = detect_file_mode(sources_text)
            binary_io = has_binary_io(sources_text)

            # Build
            build_dir = os.path.join(root, "build_poc")
            direct_exe = os.path.join(root, "poc_bin")
            built = False
            # Try cmake, make, direct
            if os.path.exists(os.path.join(root, "CMakeLists.txt")):
                if build_cmake(root, build_dir):
                    built = True
            if not built:
                if build_make(root):
                    built = True
            if not built:
                if build_direct(root, direct_exe):
                    built = True

            exe = pick_executable(root, build_dir, direct_exe)
            if not exe or not os.path.exists(exe):
                return b"A" * 60

            # Basic targeted candidates
            candidates = []

            # Seed: repeat likely add/insert
            add_tok = None
            for t in tokens:
                if re.search(r"\badd\b", t, re.IGNORECASE) or re.search(r"insert|push", t, re.IGNORECASE):
                    add_tok = t
                    break
            if add_tok is None:
                add_tok = "add"

            # candidate 1: many repeated same add to trigger duplicate/limit
            rep = "".join([f"{add_tok} 0 0\n" for _ in range(32)]).encode("utf-8")
            candidates.append(rep)

            # candidate 2: header + repeated
            rep2 = (f"32\n" + "".join([f"{add_tok} 0 0\n" for _ in range(32)])).encode("utf-8")
            candidates.append(rep2)

            # candidate 3: token soup with numbers
            soup = (" ".join([add_tok, "0", "0"] * 20) + "\n").encode("utf-8")
            candidates.append(soup)

            # candidate 4: JSON-ish
            ops = ",".join([f'{{"op":"{add_tok}","a":0,"b":0}}' for _ in range(24)])
            candidates.append((f'{{"ops":[{ops}]}}\n').encode("utf-8"))

            # Seed inputs from repo
            seeds = extract_seed_inputs(root)
            candidates.extend(seeds[:10])

            if not candidates:
                candidates = [b"\n", b"0\n", b"1\n", rep, soup]

            # Crash finder
            start_time = time.time()
            total_budget = 55.0
            fuzz_budget = 35.0
            minimize_budget = 15.0
            thread_workers = min(8, (os.cpu_count() or 2))

            def is_good_crash(data: bytes) -> bool:
                crashed, _ = run_target(exe, root, data, file_mode=file_mode, timeout=0.6)
                return crashed

            found = None

            # First, try the basic candidates quickly
            for c in candidates[:20]:
                if time.time() - start_time > total_budget:
                    break
                if is_good_crash(c):
                    found = c
                    break

            # Fuzz if not found
            if found is None and time.time() - start_time < total_budget:
                corpus = list(candidates[:10])
                if not corpus:
                    corpus = [b"\n"]
                # Ensure non-empty
                corpus = [c for c in corpus if c]
                if not corpus:
                    corpus = [b"\n"]

                def gen_candidate():
                    base = rnd.choice(corpus)
                    if binary_io:
                        if rnd.random() < 0.25:
                            return mutate_binary(b"")
                        return mutate_binary(base)
                    else:
                        if rnd.random() < 0.2:
                            return mutate_text(b"", tokens)
                        return mutate_text(base, tokens)

                batch_size = thread_workers * 2
                iters = 0
                while time.time() - start_time < min(total_budget, fuzz_budget) and iters < 2500 and found is None:
                    batch = [gen_candidate() for _ in range(batch_size)]
                    iters += batch_size
                    with ThreadPoolExecutor(max_workers=thread_workers) as ex:
                        futs = {ex.submit(run_target, exe, root, b, file_mode, 0.6): b for b in batch}
                        for fut in as_completed(futs):
                            b = futs[fut]
                            try:
                                crashed, r = fut.result()
                            except Exception:
                                continue
                            if crashed:
                                found = b
                                break
                    if found is not None:
                        break
                    # Add some to corpus to diversify (small ones preferred)
                    batch.sort(key=len)
                    for b in batch[:3]:
                        corpus.append(b)
                    corpus.sort(key=len)
                    corpus = corpus[:40]

            if found is None:
                # Fallback: fixed length 60 bytes
                return (b"A" * 59) + b"\n"

            # Minimize
            remaining = total_budget - (time.time() - start_time)
            if remaining > 2.0:
                mm_time = min(minimize_budget, remaining - 1.0)
                found = ddmin(found, is_good_crash, time_limit=mm_time)

            # Keep reasonable size
            if len(found) > 4096:
                found = found[:4096]
            if not found:
                found = b"\n"
            return found
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass