import os
import tarfile
import tempfile
import subprocess
import shutil
import time
import random
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        workdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            self._extract_tarball(src_path, workdir)
            root = self._detect_root(workdir)
            binary = self._build_binary(root)
            if binary is not None and os.path.exists(binary):
                poc = self._search_poc(binary)
                if poc is not None and isinstance(poc, (bytes, bytearray)):
                    return bytes(poc)
            return self._fallback_poc()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _extract_tarball(self, src_path: str, dest: str) -> None:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(dest)
        except tarfile.ReadError:
            # If it's not a tarball for some reason, just ignore; fallback PoC will be used.
            pass

    def _detect_root(self, workdir: str) -> str:
        try:
            entries = [os.path.join(workdir, e) for e in os.listdir(workdir)]
        except FileNotFoundError:
            return workdir
        dirs = [e for e in entries if os.path.isdir(e)]
        if len(dirs) == 1:
            return dirs[0]
        return workdir

    def _pick_compiler(self) -> str:
        for c in ("clang", "gcc", "cc"):
            if shutil.which(c):
                return c
        return "cc"

    def _build_binary(self, root: str) -> str | None:
        cc = self._pick_compiler()
        env_base = os.environ.copy()
        env_base.setdefault("CC", cc)
        cxx = None
        if cc.endswith("clang"):
            cxx = cc + "++"
        elif cc.endswith("gcc"):
            cxx = "g++"
        else:
            cxx = cc + "++"
        env_base.setdefault("CXX", cxx)
        extra_flags = " -g -O1 -fsanitize=address"
        env_base["CFLAGS"] = env_base.get("CFLAGS", "") + extra_flags
        env_base["CXXFLAGS"] = env_base.get("CXXFLAGS", "") + extra_flags
        env_base["LDFLAGS"] = env_base.get("LDFLAGS", "") + " -fsanitize=address"

        # 1) Try build.sh scripts (OSS-Fuzz style)
        build_sh_list: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            if "build.sh" in filenames:
                build_sh_list.append(os.path.join(dirpath, "build.sh"))
        build_sh_list.sort(key=len)
        for build_sh in build_sh_list:
            bdir = os.path.dirname(build_sh)
            out_dir = os.path.join(bdir, "out")
            os.makedirs(out_dir, exist_ok=True)
            env = env_base.copy()
            env.setdefault("OUT", out_dir)
            try:
                subprocess.run(
                    ["bash", build_sh],
                    cwd=bdir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=240,
                    check=False,
                )
            except Exception:
                continue
            bins = self._find_elf_binaries(out_dir)
            if not bins:
                bins = self._find_elf_binaries(bdir)
            if bins:
                return self._choose_binary(bins)

        # 2) Try Makefile
        make_dirs: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in ("Makefile", "makefile"):
                if name in filenames:
                    make_dirs.append(dirpath)
                    break
        make_dirs.sort(key=len)
        for mdir in make_dirs:
            env = env_base.copy()
            try:
                subprocess.run(
                    ["make", "-j", "4"],
                    cwd=mdir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=240,
                    check=False,
                )
            except Exception:
                continue
            bins = self._find_elf_binaries(mdir)
            if bins:
                return self._choose_binary(bins)

        # 3) Try CMake if present
        cmake_paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            if "CMakeLists.txt" in filenames:
                cmake_paths.append(dirpath)
        cmake_paths.sort(key=len)
        cmake_exe = shutil.which("cmake")
        if cmake_exe:
            for cdir in cmake_paths:
                build_dir = os.path.join(cdir, "build")
                os.makedirs(build_dir, exist_ok=True)
                env = env_base.copy()
                cfg_cmd = [
                    cmake_exe,
                    "-S",
                    ".",
                    "-B",
                    build_dir,
                    "-DCMAKE_BUILD_TYPE=Debug",
                    "-DCMAKE_C_FLAGS=-g -O1 -fsanitize=address",
                    "-DCMAKE_CXX_FLAGS=-g -O1 -fsanitize=address",
                ]
                try:
                    subprocess.run(
                        cfg_cmd,
                        cwd=cdir,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=240,
                        check=False,
                    )
                    subprocess.run(
                        [cmake_exe, "--build", build_dir, "-j", "4"],
                        cwd=cdir,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=240,
                        check=False,
                    )
                except Exception:
                    continue
                bins = self._find_elf_binaries(build_dir)
                if bins:
                    return self._choose_binary(bins)

        # 4) Naively compile all .c files
        c_files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".c"):
                    c_files.append(os.path.join(dirpath, f))
        if c_files:
            out_bin = os.path.join(root, "a.out")
            cmd = [cc, "-g", "-O1", "-fsanitize=address", "-o", out_bin] + c_files
            env = env_base.copy()
            try:
                subprocess.run(
                    cmd,
                    cwd=root,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=240,
                    check=False,
                )
                if os.path.exists(out_bin):
                    return out_bin
            except Exception:
                pass

        return None

    def _find_elf_binaries(self, root: str) -> list[str]:
        if not os.path.isdir(root):
            return []
        bins: list[str] = []
        lib_exts = {".a", ".so", ".o", ".lo", ".la", ".dylib"}
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in lib_exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "rb") as f:
                        magic = f.read(4)
                    if magic != b"\x7fELF":
                        continue
                except Exception:
                    continue
                bins.append(path)
        return bins

    def _choose_binary(self, bins: list[str]) -> str:
        if not bins:
            return ""
        def score(path: str) -> tuple[int, int, int]:
            name = os.path.basename(path).lower()
            pri = 0
            keywords = ["fuzz", "poc", "uaf", "bug", "test", "main", "xml"]
            for idx, kw in enumerate(keywords):
                if kw in name:
                    pri -= (len(keywords) - idx)
            return pri, len(name), len(path)
        bins_sorted = sorted(bins, key=score)
        return bins_sorted[0]

    def _initial_seeds(self) -> list[bytes]:
        seeds: list[bytes] = []
        seeds.extend(
            [
                b"",
                b"\x00",
                b"\xff",
                b"\x00" * 4,
                b"\xff" * 4,
                b"A",
                b"B" * 4,
                b"\x00\x01\x02\x03",
            ]
        )
        # Enumerate all 1-byte inputs to cover simple branch selectors
        for i in range(256):
            seeds.append(bytes([i]))
        # Some XML / encoding-related seeds
        xml_seeds = [
            b"<root/>",
            b"<r></r>",
            b"<?xml version='1.0'?>",
            b"<?xml version='1.0' encoding='UTF-8'?><r/>",
            b"<?xml version='1.0' encoding='ISO-8859-1'?><r/>",
            b"<?xml version='1.0' encoding='UTF-16'?><r/>",
            b"<?xml version='1.0' encoding='ASCII'?><r/>",
            b"<a>\n<b/>\n</a>",
            b"UTF-8",
            b"ISO-8859-1",
            b"UTF-16",
            b"ASCII",
        ]
        seeds.extend(xml_seeds)
        return seeds

    def _generate_random_input(self, seeds: list[bytes]) -> bytes:
        max_len = 128
        if seeds and random.random() < 0.7:
            base = random.choice(seeds)
            b = bytearray(base)
            if not b:
                b = bytearray(os.urandom(random.randint(1, 16)))
            for _ in range(random.randint(1, 3)):
                op = random.randrange(4)
                if op == 0 and len(b) > 0:
                    idx = random.randrange(len(b))
                    b[idx] ^= 1 << random.randrange(8)
                elif op == 1:  # insert
                    pos = random.randrange(len(b) + 1)
                    for _ in range(random.randint(1, 4)):
                        b.insert(pos, random.getrandbits(8))
                elif op == 2 and len(b) > 1:  # delete
                    pos = random.randrange(len(b))
                    del b[pos : min(pos + random.randint(1, 4), len(b))]
                elif op == 3 and seeds:
                    other = random.choice(seeds)
                    if other:
                        pos = random.randrange(len(b) + 1)
                        ins = other[: random.randint(1, len(other))]
                        b = b[:pos] + ins + b[pos:]
            if len(b) == 0:
                b = bytearray(os.urandom(1))
            if len(b) > max_len:
                b = b[:max_len]
            return bytes(b)
        else:
            length = random.randint(1, max_len)
            return os.urandom(length)

    def _run_target(self, binary: str, data: bytes, mode: str) -> bool:
        try:
            if mode == "stdin":
                proc = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=2,
                    check=False,
                )
            else:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        f.write(data)
                        tmp_path = f.name
                    proc = subprocess.run(
                        [binary, tmp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=2,
                        check=False,
                    )
                finally:
                    if tmp_path is not None:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

        stderr = proc.stderr or b""
        if (
            b"ERROR: AddressSanitizer" in stderr
            or b"runtime error:" in stderr
            or b"ERROR: LeakSanitizer" in stderr
        ):
            return True
        return False

    def _search_poc(self, binary: str) -> bytes | None:
        max_total_time = 20.0
        start = time.time()
        seeds = self._initial_seeds()
        # Try both stdin and file-argument modes
        for mode in ("stdin", "arg"):
            mode_start = time.time()
            seed_idx = 0
            while time.time() - start < max_total_time:
                if seed_idx < len(seeds):
                    data = seeds[seed_idx]
                    seed_idx += 1
                else:
                    data = self._generate_random_input(seeds)
                if self._run_target(binary, data, mode):
                    return data
                if time.time() - mode_start > max_total_time / 2:
                    break
        return None

    def _fallback_poc(self) -> bytes:
        # A structured XML with explicit encoding to exercise libxml2 I/O paths.
        return b"<?xml version='1.0' encoding='UTF-8'?><poc>test</poc>"