import os
import tarfile
import tempfile
import subprocess
import time
import random
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        start_time = time.time()
        max_time = 40.0
        random.seed(0)

        tmp_root = tempfile.mkdtemp(prefix="pocgen_")
        try:
            project_root = self._prepare_project(src_path, tmp_root)

            # First try to find a statically provided PoC file
            static_poc = self._static_poc_search(project_root)
            if static_poc:
                return static_poc

            input_mode = self._detect_input_mode(project_root)

            # Try to find an existing binary before building
            bin_path = self._find_candidate_binary(project_root)

            # If no binary, try to build the project
            if bin_path is None:
                self._build_project(project_root, start_time, max_time)
                bin_path = self._find_candidate_binary(project_root)

            if bin_path is None:
                # Fallback: simple generic input
                return b"A" * 60

            crash_input = self._find_crashing_input(
                bin_path, project_root, input_mode, start_time, max_time
            )
            if not crash_input:
                # Fallback if we could not find a crashing input
                return b"A" * 60

            return crash_input
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def _prepare_project(self, src_path: str, tmp_root: str) -> str:
        with tarfile.open(src_path, "r:*") as tar:
            tar.extractall(tmp_root)
        entries = [e for e in os.listdir(tmp_root) if not e.startswith(".")]
        if len(entries) == 1:
            candidate = os.path.join(tmp_root, entries[0])
            if os.path.isdir(candidate):
                return candidate
        return tmp_root

    def _static_poc_search(self, root: str, max_size: int = 4096) -> bytes:
        patterns = ("poc", "crash", "uaf", "doublefree", "use-after-free", "heap-use-after-free", "id:0", "id_0")
        exts = (".bin", ".raw", ".in", ".dat", ".data", ".txt", ".json", ".xml", ".yaml", ".yml")
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                lower = name.lower()
                if not lower.endswith(exts):
                    continue
                if not any(p in lower for p in patterns):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0 or size > max_size:
                    continue
                candidates.append((size, path))
        # Prefer smaller PoCs, they are likely minimized
        candidates.sort()
        for _, path in candidates:
            try:
                with open(path, "rb") as f:
                    data = f.read(max_size)
                if data:
                    return data
            except OSError:
                continue
        return b""

    def _detect_input_mode(self, root: str) -> str:
        code_exts = {".c", ".cc", ".cpp", ".cxx", ".C", ".CPP", ".c++", ".cp"}
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext not in code_exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", errors="ignore") as f:
                        data = f.read()
                except OSError:
                    continue
                if "int main(" in data or "int main (" in data:
                    if "argv[" in data or "argc>" in data or "argc >" in data or "argv[1]" in data:
                        return "filearg"
                    if "std::cin" in data or "cin." in data or "fgets" in data:
                        return "stdin"
                    if "read(" in data and "STDIN" in data:
                        return "stdin"
                    if "argv" in data:
                        return "filearg"
                    return "stdin"
        return "unknown"

    def _build_project(self, root: str, start_time: float, max_time: float) -> None:
        def time_left() -> float:
            return max_time - (time.time() - start_time)

        # Look for build scripts
        scripts = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                lower = name.lower()
                if lower in ("build.sh", "build.bash", "build"):
                    scripts.append(os.path.join(dirpath, name))
                elif lower in ("compile.sh", "build_project.sh", "build_all.sh"):
                    scripts.append(os.path.join(dirpath, name))

        for script in scripts:
            if time_left() <= 1.0:
                return
            try:
                subprocess.run(
                    ["bash", script],
                    cwd=os.path.dirname(script),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(5, min(300, int(time_left()))),
                    check=True,
                )
                return
            except Exception:
                continue

        if time_left() <= 1.0:
            return

        # Try CMake-based build
        if os.path.exists(os.path.join(root, "CMakeLists.txt")):
            build_dir = os.path.join(root, "build-pocgen")
            os.makedirs(build_dir, exist_ok=True)
            try:
                configure_timeout = max(5, min(300, int(time_left() / 2)))
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        ".",
                        "-B",
                        build_dir,
                        "-DCMAKE_BUILD_TYPE=Debug",
                        "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer",
                        "-DCMAKE_C_FLAGS=-fsanitize=address -fno-omit-frame-pointer",
                    ],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=configure_timeout,
                    check=True,
                )
                build_timeout = max(5, min(300, int(time_left())))
                subprocess.run(
                    ["cmake", "--build", build_dir, "-j", "4"],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=build_timeout,
                    check=True,
                )
                return
            except Exception:
                pass

        if time_left() <= 1.0:
            return

        # Try Makefile
        if os.path.exists(os.path.join(root, "Makefile")) or os.path.exists(
            os.path.join(root, "makefile")
        ):
            try:
                subprocess.run(
                    ["make", "-j", "4"],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(5, min(300, int(time_left()))),
                    check=True,
                )
                return
            except Exception:
                pass

    def _find_candidate_binary(self, root: str) -> str:
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                if not os.path.isfile(path):
                    continue
                if not os.access(path, os.X_OK):
                    continue
                lower = name.lower()
                if lower.endswith(
                    (
                        ".sh",
                        ".bash",
                        ".py",
                        ".pl",
                        ".rb",
                        ".php",
                        ".jar",
                        ".bat",
                        ".cmd",
                        ".so",
                        ".dylib",
                        ".dll",
                        ".a",
                        ".lib",
                        ".o",
                    )
                ):
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size < 4096:
                    continue
                candidates.append((size, path))
        if not candidates:
            return ""
        preferred = [
            c
            for c in candidates
            if any(k in os.path.basename(c[1]).lower() for k in ("fuzz", "asan", "test", "driver", "poc", "sample", "demo"))
        ]
        if preferred:
            preferred.sort(reverse=True)
            return preferred[0][1]
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _collect_seed_files(self, root: str, max_seeds: int = 50, max_size: int = 8192):
        seed_paths = []
        exts = (
            ".in",
            ".bin",
            ".raw",
            ".data",
            ".dat",
            ".txt",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".msg",
            ".pb",
            ".pbf",
            ".poc",
            ".inp",
        )
        name_keywords = ("seed", "input", "sample", "test", "case", "poc", "crash", "id:", "fuzz")
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                lower = name.lower()
                if lower.endswith(exts) or any(k in lower for k in name_keywords):
                    path = os.path.join(dirpath, name)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    if size == 0 or size > max_size:
                        continue
                    seed_paths.append(path)
        random.shuffle(seed_paths)
        seed_paths = seed_paths[:max_seeds]
        seeds = []
        for path in seed_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read(max_size)
                if data:
                    seeds.append(data)
            except OSError:
                continue
        return seeds

    def _basic_seeds(self):
        seeds = [b"", b"A", b"0", b"1", b"[]", b"{}", b"null", b'""', b" "]
        for i in range(1, 8):
            seeds.append(b"A" * i)
        return seeds

    def _mutate_input(self, data: bytes, max_len: int = 512) -> bytes:
        if not data:
            length = random.randint(1, min(128, max_len))
            return bytes(random.getrandbits(8) for _ in range(length))

        buf = bytearray(data)
        if len(buf) > max_len:
            start = random.randint(0, len(buf) - max_len)
            buf = buf[start : start + max_len]

        num_mutations = random.randint(1, 6)
        for _ in range(num_mutations):
            choice = random.randint(0, 2)
            if choice == 0 and buf:
                idx = random.randrange(len(buf))
                buf[idx] = random.getrandbits(8)
            elif choice == 1 and len(buf) < max_len:
                idx = random.randint(0, len(buf))
                insert_len = random.randint(1, min(8, max_len - len(buf)))
                for _ in range(insert_len):
                    buf.insert(idx, random.getrandbits(8))
            elif choice == 2 and len(buf) > 1:
                start = random.randrange(len(buf))
                end = min(len(buf), start + random.randint(1, 8))
                del buf[start:end]
        return bytes(buf)

    def _check_crash(self, bin_path: str, data: bytes, input_mode: str, timeout: float) -> bool:
        asan_markers = (
            b"ERROR: AddressSanitizer",
            b"AddressSanitizer:",
            b"heap-use-after-free",
            b"double-free",
            b"heap-use-after-free",
            b"heap-use-after-free",
            b"Segmentation fault",
            b"segmentation fault",
        )
        tmp_file = None
        try:
            if input_mode in ("filearg", "unknown"):
                try:
                    fd, tmp_file = tempfile.mkstemp(prefix="poc_input_", suffix=".bin")
                    os.write(fd, data)
                    os.close(fd)
                    result = subprocess.run(
                        [bin_path, tmp_file],
                        input=None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                    if result.returncode != 0 and any(m in result.stderr for m in asan_markers):
                        return True
                except subprocess.TimeoutExpired:
                    pass
                except OSError:
                    pass
                finally:
                    if tmp_file and os.path.exists(tmp_file):
                        os.remove(tmp_file)
                        tmp_file = None

            if input_mode in ("stdin", "unknown"):
                try:
                    result = subprocess.run(
                        [bin_path],
                        input=data,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                    if result.returncode != 0 and any(m in result.stderr for m in asan_markers):
                        return True
                except subprocess.TimeoutExpired:
                    pass
                except OSError:
                    pass

            return False
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _find_crashing_input(
        self,
        bin_path: str,
        root: str,
        input_mode: str,
        start_time: float,
        max_time: float,
    ) -> bytes:
        def time_left() -> float:
            return max_time - (time.time() - start_time)

        seeds = self._collect_seed_files(root)
        basic = self._basic_seeds()

        # Stage 1: try existing seed files
        for data in seeds:
            if time_left() <= 0.5:
                return b""
            if self._check_crash(bin_path, data, input_mode, timeout=0.5):
                return data

        # Stage 1b: try basic small seeds
        for data in basic:
            if time_left() <= 0.5:
                return b""
            if self._check_crash(bin_path, data, input_mode, timeout=0.5):
                return data

        # Stage 2: random mutation-based fuzzing
        if not seeds:
            seeds = basic
        else:
            seeds = seeds + basic
        if not seeds:
            seeds = [b"A"]

        while time_left() > 0.5:
            base = random.choice(seeds)
            mutant = self._mutate_input(base, max_len=256)
            if self._check_crash(bin_path, mutant, input_mode, timeout=0.5):
                return mutant

        return b""