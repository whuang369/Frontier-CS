import os
import tarfile
import zipfile
import tempfile
import subprocess
import random
import string
from typing import Optional, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        random.seed(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._extract_archive(src_path, tmpdir)
            root = self._detect_root_dir(tmpdir)

            # Try dynamic fuzzing using locally built binary
            try:
                poc = self._generate_poc_by_fuzz(root)
                if poc is not None:
                    return poc
            except Exception:
                pass

            # Fallback: static guess
            return self._default_poc()

    # ---------------- Archive handling ----------------

    def _extract_archive(self, src_path: str, dest_dir: str) -> None:
        if tarfile.is_tarfile(src_path):
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            continue
                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tf, dest_dir)
        elif zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path, "r") as zf:
                zf.extractall(dest_dir)
        else:
            raise ValueError("Unsupported archive format")

    def _detect_root_dir(self, base_dir: str) -> str:
        entries = [os.path.join(base_dir, e) for e in os.listdir(base_dir) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(entries[0]):
            return entries[0]
        return base_dir

    # ---------------- Build & executable discovery ----------------

    def _generate_poc_by_fuzz(self, root: str) -> Optional[bytes]:
        # Try to build the project; ignore errors
        try:
            self._try_build_project(root)
        except Exception:
            pass

        executables = self._find_executables(root)
        if not executables:
            return None

        # Prioritize likely harness binaries by name
        executables = self._sort_executables_by_likelihood(executables)

        for exe in executables:
            for mode in ("stdin", "file"):
                try:
                    poc = self._fuzz_executable(exe, mode)
                    if poc is not None:
                        return poc
                except Exception:
                    continue
        return None

    def _try_build_project(self, root: str) -> None:
        env = os.environ.copy()
        extra_flags = "-fsanitize=address -g -O1 -fno-omit-frame-pointer"
        for var in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
            if var in env and env[var]:
                env[var] = env[var] + " " + extra_flags
            else:
                env[var] = extra_flags

        def run_cmd(cmd, cwd, timeout):
            try:
                subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    check=True,
                )
                return True
            except Exception:
                return False

        # 1. Any build.sh in project (root or one-level subdirs)
        for dirpath, dirnames, filenames in os.walk(root):
            if "build.sh" in filenames:
                if run_cmd(["bash", "build.sh"], dirpath, timeout=40):
                    return
            # avoid going too deep
            if dirpath != root:
                dirnames[:] = []

        # 2. CMake at root
        if os.path.isfile(os.path.join(root, "CMakeLists.txt")):
            build_dir = os.path.join(root, "build_asan")
            os.makedirs(build_dir, exist_ok=True)
            if run_cmd(
                ["cmake", "-DCMAKE_BUILD_TYPE=Debug", ".."],
                build_dir,
                timeout=40,
            ):
                run_cmd(["cmake", "--build", ".", "-j4"], build_dir, timeout=60)
                return

        # 3. Autotools configure
        configure_path = os.path.join(root, "configure")
        if os.path.isfile(configure_path) and os.access(configure_path, os.X_OK):
            if run_cmd([configure_path], root, timeout=40):
                run_cmd(["make", "-j4"], root, timeout=60)
                return

        # 4. Simple Makefile
        if os.path.isfile(os.path.join(root, "Makefile")):
            run_cmd(["make", "-j4"], root, timeout=60)

    def _find_executables(self, root: str) -> List[str]:
        execs: List[str] = []
        skip_dirs = {".git", ".svn", ".hg", "__pycache__", "build", "build_asan", "cmake-build-debug", "cmake-build-release"}

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    st = os.stat(path)
                except FileNotFoundError:
                    continue
                if not (st.st_mode & 0o111):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext in (".sh", ".py", ".pl", ".rb", ".lua", ".bat", ".cmd", ".ps1", ".so", ".a", ".o", ".dll", ".dylib"):
                    continue
                try:
                    with open(path, "rb") as f:
                        head = f.read(4)
                    if head.startswith(b"#!"):
                        continue
                except Exception:
                    continue

                execs.append(path)
        return execs

    def _sort_executables_by_likelihood(self, execs: List[str]) -> List[str]:
        keywords_high = ["vuln", "asan", "uaf", "heap", "driver", "main", "test", "poc", "node"]
        keywords_low = ["cmake", "example", "demo"]

        def score(path: str) -> int:
            name = os.path.basename(path).lower()
            s = 0
            for kw in keywords_high:
                if kw in name:
                    s += 10
            for kw in keywords_low:
                if kw in name:
                    s += 2
            if "." not in name:
                s += 3
            return -s  # sort ascending => larger s first

        return sorted(execs, key=score)

    # ---------------- Fuzzing ----------------

    def _fuzz_executable(self, exe_path: str, mode: str) -> Optional[bytes]:
        max_tries = 40
        timeout_per_run = 0.25

        for data in self._candidate_inputs(max_tries):
            if mode == "stdin":
                try:
                    proc = subprocess.run(
                        [exe_path],
                        input=data,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout_per_run,
                    )
                except subprocess.TimeoutExpired:
                    continue
            else:  # mode == "file"
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(data)
                    tf.flush()
                    tmpname = tf.name
                try:
                    proc = subprocess.run(
                        [exe_path, tmpname],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout_per_run,
                    )
                except subprocess.TimeoutExpired:
                    os.unlink(tmpname)
                    continue
                finally:
                    try:
                        os.unlink(tmpname)
                    except OSError:
                        pass

            if self._is_heap_bug_crash(proc.returncode, proc.stdout, proc.stderr):
                return data

        return None

    def _candidate_inputs(self, max_tries: int):
        seeds: List[bytes] = []

        # Simple structured seeds with large integers and tokens
        seeds.append(b"A" * 60)
        seeds.append(b"0 " * 30)
        seeds.append(b"999999999 " * 8)
        seeds.append(b"2147483647 " * 8)
        seeds.append(b"4294967295 " * 8)
        seeds.append(b"-1 -1 -1 -1 -1 -1 -1 -1\n")
        seeds.append(b"add node 2147483647 2147483647 2147483647 2147483647\n")
        seeds.append(b"node add 4294967295 4294967295 4294967295 4294967295\n")
        seeds.append(b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[\n")
        seeds.append(b"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n")
        seeds.append(b"((((((((((((((((((((((((((((((((((((((((((((((((((((((((\n")
        seeds.append(b"))))))))))))))))))))))))))))))))))))))))))))))))))))))))\n")
        seeds.append(b'{"node": {"add": [999999999,999999999,999999999]}}\n')
        seeds.append(b"<node><add>999999999</add></node>\n")
        seeds.append(b"<node><add>4294967295</add></node>\n")
        seeds.append(b"NODE ADD 1000000000 1000000000 1000000000 1000000000\n")

        tries = 0
        for s in seeds:
            if tries >= max_tries:
                return
            tries += 1
            yield s

        # Randomized candidates
        def rand_ascii(n: int) -> bytes:
            chars = string.ascii_letters + string.digits + string.punctuation + " \n\t"
            return "".join(random.choice(chars) for _ in range(n)).encode("ascii", "ignore")

        while tries < max_tries:
            tries += 1
            t = random.randint(0, 4)
            if t == 0:
                length = random.randint(1, 80)
                yield rand_ascii(length)
            elif t == 1:
                # Sequence of large integers
                count = random.randint(1, 10)
                nums = []
                for _ in range(count):
                    if random.random() < 0.7:
                        nums.append(str(random.randint(10**8, 2**31 - 1)))
                    else:
                        nums.append(str(random.randint(0, 2**31 - 1)))
                text = " ".join(nums) + "\n"
                yield text.encode("ascii")
            elif t == 2:
                # "node add" style command with big numbers
                count = random.randint(2, 8)
                nums = [str(random.randint(10**8, 2**31 - 1)) for _ in range(count)]
                text = "node add " + " ".join(nums) + "\n"
                yield text.encode("ascii")
            elif t == 3:
                # Deeply nested parentheses/brackets
                depth = random.randint(10, 40)
                open_ch = random.choice(["(", "[", "{"])
                close_ch = { "(": ")", "[": "]", "{": "}" }[open_ch]
                text = open_ch * depth + close_ch * (depth - random.randint(0, 3)) + "\n"
                yield text.encode("ascii")
            else:
                # Slight mutation of an existing seed
                base = random.choice(seeds)
                ba = bytearray(base)
                for _ in range(random.randint(1, max(1, len(ba) // 4))):
                    idx = random.randrange(len(ba))
                    ba[idx] = random.randint(0, 255)
                yield bytes(ba)

    def _is_heap_bug_crash(self, returncode: int, stdout: bytes, stderr: bytes) -> bool:
        text = (stdout + stderr).decode("latin1", errors="ignore")

        if "AddressSanitizer" in text:
            # Any ASan error is likely a memory bug
            return True

        keywords = [
            "double free",
            "double-free",
            "free(): invalid",
            "free(): double",
            "corrupted size vs. prev_size",
            "invalid pointer",
            "heap-use-after-free",
        ]
        for k in keywords:
            if k in text:
                return True
        return False

    # ---------------- Fallback PoC ----------------

    def _default_poc(self) -> bytes:
        # Generic structured input with large integers and "node add" tokens
        text = "node add 2147483647 2147483647 2147483647 2147483647 2147483647\n"
        return text.encode("ascii")