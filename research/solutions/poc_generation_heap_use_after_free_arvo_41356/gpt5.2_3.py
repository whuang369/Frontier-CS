import os
import re
import io
import sys
import tarfile
import time
import random
import shutil
import stat
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RunConfig:
    exe: str
    mode: str  # 'filearg', 'stdin', 'libfuzzer'
    base_args: List[str]
    cwd: str
    env: dict
    timeout: float


class Solution:
    def solve(self, src_path: str) -> bytes:
        random.seed(0xC0FFEE)

        with tempfile.TemporaryDirectory(prefix="arvo_poc_") as td:
            root = self._prepare_source(src_path, td)

            # If the repo already contains an obvious crash/poc input, use it
            existing = self._find_existing_poc(root)
            if existing is not None:
                return existing

            tokens = self._extract_tokens(root)

            # Try building and letting libFuzzer find an artifact if possible
            poc = self._try_libfuzzer_end_to_end(root, td)
            if poc is not None:
                return self._strip_and_limit(poc)

            build_dir = os.path.join(td, "build")
            os.makedirs(build_dir, exist_ok=True)

            exe = self._build_any(root, build_dir)
            if exe is None:
                # Fallback: return something deterministic
                return b"A" * 60

            cfg = self._determine_run_config(exe, td, prefer_libfuzzer=True)
            if cfg is None:
                return b"A" * 60

            # Try repo seeds
            seeds = self._collect_seed_inputs(root)
            for s in seeds:
                if self._is_uaf_or_double_free(cfg, s):
                    return self._minimize(cfg, s)

            # Fuzz quickly
            poc = self._quick_fuzz(cfg, tokens, seeds)
            if poc is not None:
                return self._minimize(cfg, poc)

            # Last resort: try some structured attempts
            for s in self._structured_attempts(tokens):
                if self._is_uaf_or_double_free(cfg, s):
                    return self._minimize(cfg, s)

            return b"A" * 60

    def _strip_and_limit(self, b: bytes) -> bytes:
        if not b:
            return b
        if len(b) > 4096:
            b = b[:4096]
        return b

    def _prepare_source(self, src_path: str, td: str) -> str:
        if os.path.isdir(src_path):
            return src_path

        extract_dir = os.path.join(td, "src")
        os.makedirs(extract_dir, exist_ok=True)

        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        with tarfile.open(src_path, "r:*") as tar:
            members = tar.getmembers()
            for m in members:
                dest = os.path.join(extract_dir, m.name)
                if not is_within_directory(extract_dir, dest):
                    continue
            safe_members = []
            for m in members:
                dest = os.path.join(extract_dir, m.name)
                if is_within_directory(extract_dir, dest):
                    safe_members.append(m)
            tar.extractall(extract_dir, members=safe_members)

        # If tarball contained a single top-level directory, use it
        entries = [os.path.join(extract_dir, x) for x in os.listdir(extract_dir)]
        dirs = [p for p in entries if os.path.isdir(p)]
        files = [p for p in entries if os.path.isfile(p)]
        if len(dirs) == 1 and not files:
            return dirs[0]
        return extract_dir

    def _find_existing_poc(self, root: str) -> Optional[bytes]:
        keywords = ("crash", "poc", "uaf", "double", "dfree", "useafter", "use-after", "heap-use")
        preferred_ext = (".bin", ".dat", ".in", ".input", ".txt", ".yaml", ".yml", ".json", ".xml")

        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", "build", "cmake-build-debug", "cmake-build-release"):
                dirnames[:] = []
                continue
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".md", ".rst", ".py", ".cmake")):
                    continue
                if any(k in lfn for k in keywords) or lfn.endswith(preferred_ext):
                    path = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > 1_000_000:
                        continue
                    score = 0
                    if any(k in lfn for k in keywords):
                        score += 50
                    if lfn.endswith(preferred_ext):
                        score += 10
                    score -= min(100, st.st_size // 64)
                    candidates.append((score, path))

        candidates.sort(reverse=True)
        for _, path in candidates[:50]:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except OSError:
                continue
        return None

    def _extract_tokens(self, root: str) -> List[bytes]:
        tokens = set()
        base = [
            b"{", b"}", b"[", b"]", b"(", b")", b"<", b">", b":", b",", b";", b"\n", b"\r\n", b"\t", b" ",
            b"0", b"1", b"-1", b"2", b"3", b"9", b"10", b"16", b"32", b"64", b"128",
            b"255", b"256", b"1024", b"4096", b"65535", b"65536",
            b"2147483647", b"4294967295",
            b"true", b"false", b"null",
            b"---\n", b"...\n", b"- ", b": ", b"&a ", b"*a ", b"!!map ", b"!!seq ",
            b"\"\"", b"''", b"\"", b"\\", b"\\u0000",
        ]
        for t in base:
            tokens.add(t)

        src_exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
        string_re = re.compile(r'"([^"\\\n\r]{1,24})"|\'([^\'\\\n\r]{1,24})\'')
        ident_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{1,24})\b")

        limit_files = 250
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", "build"):
                dirnames[:] = []
                continue
            for fn in filenames:
                if not fn.lower().endswith(src_exts):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "rb") as f:
                        data = f.read(200_000)
                except OSError:
                    continue
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                for m in string_re.finditer(text):
                    s = m.group(1) or m.group(2) or ""
                    s = s.strip()
                    if 1 <= len(s) <= 24:
                        if all(32 <= ord(ch) < 127 for ch in s):
                            tokens.add(s.encode("ascii", errors="ignore"))

                # Some identifiers help for text formats / commands
                for m in ident_re.finditer(text):
                    w = m.group(1)
                    if len(w) >= 2 and w.lower() in ("add", "node", "parse", "load", "read", "write", "map", "seq", "key", "value"):
                        tokens.add(w.encode("ascii", errors="ignore"))

                count += 1
                if count >= limit_files:
                    break
            if count >= limit_files:
                break

        out = list(tokens)
        out.sort(key=lambda x: (len(x), x))
        return out

    def _run_cmd(self, cmd: List[str], cwd: str, env: dict, timeout: float) -> Tuple[int, bytes, bytes]:
        try:
            p = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or b"", e.stderr or b""
        except Exception:
            return 125, b"", b""

    def _is_elf_executable(self, path: str) -> bool:
        try:
            st = os.stat(path)
        except OSError:
            return False
        if not stat.S_ISREG(st.st_mode):
            return False
        if not os.access(path, os.X_OK):
            return False
        if st.st_size < 4:
            return False
        try:
            with open(path, "rb") as f:
                hdr = f.read(4)
            return hdr == b"\x7fELF"
        except OSError:
            return False

    def _scan_executables(self, root: str) -> List[str]:
        exes = []
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git",):
                dirnames[:] = []
                continue
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if self._is_elf_executable(path):
                    exes.append(path)
        return exes

    def _build_any(self, root: str, build_dir: str) -> Optional[str]:
        # If there is already an executable in the tree (sometimes), try it
        pre = self._scan_executables(root)
        pre = self._rank_exes(pre)
        for e in pre[:5]:
            cfg = self._determine_run_config(e, build_dir, prefer_libfuzzer=True)
            if cfg is not None:
                return e

        # Try cmake
        exe = self._try_cmake(root, build_dir)
        if exe is not None:
            return exe

        # Try make
        exe = self._try_make(root)
        if exe is not None:
            return exe

        # Naive compile
        exe = self._try_naive_compile(root, build_dir)
        return exe

    def _rank_exes(self, exes: List[str]) -> List[str]:
        def score(p: str) -> Tuple[int, int, str]:
            name = os.path.basename(p).lower()
            s = 0
            for kw, w in (("fuzz", 50), ("poc", 40), ("target", 30), ("test", 5), ("tool", 1)):
                if kw in name:
                    s += w
            # prefer smaller binaries
            try:
                sz = os.stat(p).st_size
            except OSError:
                sz = 1 << 30
            return (s, -min(sz, 1 << 29), p)
        return sorted(exes, key=score, reverse=True)

    def _try_cmake(self, root: str, build_dir: str) -> Optional[str]:
        if not os.path.exists(os.path.join(root, "CMakeLists.txt")):
            return None
        if shutil.which("cmake") is None:
            return None

        san_cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address -fno-sanitize-recover=all -Wno-error"
        san_ldflags = "-fsanitize=address"
        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")

        cfg_cmd = [
            "cmake",
            "-S", root,
            "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_C_FLAGS={san_cflags}",
            f"-DCMAKE_CXX_FLAGS={san_cflags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={san_ldflags}",
            "-DBUILD_TESTING=OFF",
        ]
        rc, _, _ = self._run_cmd(cfg_cmd, cwd=build_dir, env=env, timeout=120.0)
        if rc != 0:
            return None

        build_cmd = ["cmake", "--build", build_dir, "-j", str(min(8, os.cpu_count() or 2))]
        rc, _, _ = self._run_cmd(build_cmd, cwd=build_dir, env=env, timeout=240.0)
        if rc != 0:
            return None

        exes = self._scan_executables(build_dir)
        exes = self._rank_exes(exes)
        for e in exes[:20]:
            cfg = self._determine_run_config(e, build_dir, prefer_libfuzzer=True)
            if cfg is not None:
                return e
        return None

    def _try_make(self, root: str) -> Optional[str]:
        if not os.path.exists(os.path.join(root, "Makefile")) and not os.path.exists(os.path.join(root, "makefile")):
            return None
        if shutil.which("make") is None:
            return None

        san_cflags = "-O1 -g -fno-omit-frame-pointer -fsanitize=address -fno-sanitize-recover=all -Wno-error"
        san_ldflags = "-fsanitize=address"
        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")
        env["CFLAGS"] = (env.get("CFLAGS", "") + " " + san_cflags).strip()
        env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " " + san_cflags).strip()
        env["LDFLAGS"] = (env.get("LDFLAGS", "") + " " + san_ldflags).strip()

        cmd = ["make", "-j", str(min(8, os.cpu_count() or 2))]
        rc, _, _ = self._run_cmd(cmd, cwd=root, env=env, timeout=240.0)
        if rc != 0:
            return None

        exes = self._scan_executables(root)
        exes = self._rank_exes(exes)
        for e in exes[:20]:
            cfg = self._determine_run_config(e, root, prefer_libfuzzer=True)
            if cfg is not None:
                return e
        return None

    def _collect_sources(self, root: str) -> Tuple[List[str], List[str], List[str]]:
        c_src = []
        cpp_src = []
        all_src = []
        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", "build", "cmake-build-debug", "cmake-build-release"):
                dirnames[:] = []
                continue
            for fn in filenames:
                lfn = fn.lower()
                path = os.path.join(dirpath, fn)
                if lfn.endswith(".c"):
                    c_src.append(path)
                    all_src.append(path)
                elif lfn.endswith((".cc", ".cpp", ".cxx")):
                    cpp_src.append(path)
                    all_src.append(path)
        return c_src, cpp_src, all_src

    def _has_fuzzer_entry(self, root: str) -> bool:
        _, _, all_src = self._collect_sources(root)
        for p in all_src[:500]:
            try:
                with open(p, "rb") as f:
                    data = f.read(200_000)
            except OSError:
                continue
            if b"LLVMFuzzerTestOneInput" in data:
                return True
        return False

    def _select_main_file(self, sources: List[str]) -> Optional[str]:
        best = None
        best_score = -1
        main_re = re.compile(r"\bint\s+main\s*\(")
        for p in sources:
            try:
                with open(p, "rb") as f:
                    data = f.read(200_000)
            except OSError:
                continue
            try:
                t = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "main" not in t:
                continue
            m = len(main_re.findall(t))
            if m <= 0:
                continue
            score = m * 10
            score += 2 if "argv" in t else 0
            score += 2 if "stdin" in t else 0
            score += 1 if "fopen" in t else 0
            # Prefer files in src/ or tools/
            low = p.lower()
            if "/src/" in low or low.endswith("/src/main.cpp") or low.endswith("/src/main.cc"):
                score += 3
            if score > best_score:
                best_score = score
                best = p
        return best

    def _try_naive_compile(self, root: str, build_dir: str) -> Optional[str]:
        gcc = shutil.which("gcc") or shutil.which("cc")
        gpp = shutil.which("g++") or shutil.which("c++")
        if gcc is None or gpp is None:
            return None

        c_src, cpp_src, all_src = self._collect_sources(root)
        if not all_src:
            return None

        has_fuzz = self._has_fuzzer_entry(root)

        # Determine entrypoint strategy
        main_file = self._select_main_file(all_src)

        # Exclude extra mains
        def file_contains_main(p: str) -> bool:
            try:
                with open(p, "rb") as f:
                    data = f.read(200_000)
            except OSError:
                return False
            return re.search(rb"\bint\s+main\s*\(", data) is not None

        compile_sources = []
        if has_fuzz and main_file is None:
            # Build with our driver main calling LLVMFuzzerTestOneInput
            compile_sources = [p for p in all_src if not file_contains_main(p)]
            driver = os.path.join(build_dir, "driver.cpp")
            with open(driver, "w", encoding="utf-8") as f:
                f.write(
                    r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size);

int main(int argc, char** argv) {
    std::vector<uint8_t> buf;
    if (argc >= 2) {
        std::ifstream ifs(argv[1], std::ios::binary);
        if (!ifs.good()) return 0;
        ifs.seekg(0, std::ios::end);
        std::streamoff n = ifs.tellg();
        if (n < 0) n = 0;
        ifs.seekg(0, std::ios::beg);
        buf.resize((size_t)n);
        if (n > 0) ifs.read((char*)buf.data(), n);
    } else {
        // Read stdin
        std::vector<char> tmp((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        buf.assign(tmp.begin(), tmp.end());
    }
    (void)LLVMFuzzerTestOneInput(buf.data(), buf.size());
    return 0;
}
'''
                )
            compile_sources.append(driver)
        else:
            if main_file is None:
                return None
            compile_sources = [p for p in all_src if not file_contains_main(p) or p == main_file]
            # Ensure main_file included
            if main_file not in compile_sources:
                compile_sources.append(main_file)

        include_dirs = set()
        for p in compile_sources:
            include_dirs.add(os.path.dirname(p))
        for cand in ("include", "includes", "inc", "src"):
            ip = os.path.join(root, cand)
            if os.path.isdir(ip):
                include_dirs.add(ip)

        inc_flags = []
        for d in sorted(include_dirs):
            inc_flags += ["-I", d]

        cflags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address", "-fno-sanitize-recover=all", "-Wno-error"]
        cxxflags = cflags + ["-std=c++17"]
        ldflags = ["-fsanitize=address"]

        obj_dir = os.path.join(build_dir, "obj")
        os.makedirs(obj_dir, exist_ok=True)
        objs = []

        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")

        for p in compile_sources:
            base = os.path.basename(p)
            obj = os.path.join(obj_dir, base + ".o")
            if p.lower().endswith(".c"):
                cmd = [gcc, "-c", p, "-o", obj] + cflags + inc_flags
            else:
                cmd = [gpp, "-c", p, "-o", obj] + cxxflags + inc_flags
            rc, _, _ = self._run_cmd(cmd, cwd=build_dir, env=env, timeout=240.0)
            if rc != 0:
                return None
            objs.append(obj)

        out = os.path.join(build_dir, "poc_target")
        cmd = [gpp, "-o", out] + objs + ldflags
        rc, _, _ = self._run_cmd(cmd, cwd=build_dir, env=env, timeout=240.0)
        if rc != 0:
            return None
        try:
            st = os.stat(out)
            os.chmod(out, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
        return out

    def _determine_run_config(self, exe: str, workdir: str, prefer_libfuzzer: bool) -> Optional[RunConfig]:
        if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            return None

        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")
        env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=0")
        env.setdefault("MSAN_OPTIONS", "halt_on_error=1")
        env.setdefault("LSAN_OPTIONS", "detect_leaks=0")

        tmp_inp = os.path.join(workdir, "probe_input")
        try:
            with open(tmp_inp, "wb") as f:
                f.write(b"")
        except OSError:
            return None

        def looks_like_usage(out: bytes) -> bool:
            s = (out[:4000]).decode("latin1", errors="ignore").lower()
            if "usage:" in s:
                return True
            if "unknown option" in s:
                return True
            if "unrecognized option" in s:
                return True
            return False

        probe_timeout = 1.0

        # Try libFuzzer single-run mode
        if prefer_libfuzzer:
            rc, so, se = self._run_cmd([exe, "-runs=1", tmp_inp], cwd=workdir, env=env, timeout=probe_timeout)
            combined = so + se
            if rc in (0, 1) and not looks_like_usage(combined):
                return RunConfig(exe=exe, mode="libfuzzer", base_args=["-runs=1"], cwd=workdir, env=env, timeout=2.0)

        # Try file argument mode
        rc, so, se = self._run_cmd([exe, tmp_inp], cwd=workdir, env=env, timeout=probe_timeout)
        combined = so + se
        if not looks_like_usage(combined):
            return RunConfig(exe=exe, mode="filearg", base_args=[], cwd=workdir, env=env, timeout=2.0)

        # Try stdin mode
        try:
            p = subprocess.run([exe], cwd=workdir, env=env, input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=probe_timeout)
            combined = p.stdout + p.stderr
            if not looks_like_usage(combined):
                return RunConfig(exe=exe, mode="stdin", base_args=[], cwd=workdir, env=env, timeout=2.0)
        except Exception:
            pass

        # If nothing looks good, still allow filearg as last resort
        return RunConfig(exe=exe, mode="filearg", base_args=[], cwd=workdir, env=env, timeout=2.0)

    def _exec_with_input(self, cfg: RunConfig, data: bytes) -> Tuple[int, bytes, bytes]:
        inp_path = os.path.join(cfg.cwd, "cur_input")
        try:
            with open(inp_path, "wb") as f:
                f.write(data)
        except OSError:
            return 125, b"", b""

        if cfg.mode == "libfuzzer":
            cmd = [cfg.exe] + cfg.base_args + [inp_path]
            return self._run_cmd(cmd, cwd=cfg.cwd, env=cfg.env, timeout=cfg.timeout)
        elif cfg.mode == "filearg":
            cmd = [cfg.exe] + cfg.base_args + [inp_path]
            return self._run_cmd(cmd, cwd=cfg.cwd, env=cfg.env, timeout=cfg.timeout)
        elif cfg.mode == "stdin":
            try:
                p = subprocess.run([cfg.exe] + cfg.base_args, cwd=cfg.cwd, env=cfg.env, input=data,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=cfg.timeout)
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                return 124, e.stdout or b"", e.stderr or b""
            except Exception:
                return 125, b"", b""
        else:
            cmd = [cfg.exe] + cfg.base_args + [inp_path]
            return self._run_cmd(cmd, cwd=cfg.cwd, env=cfg.env, timeout=cfg.timeout)

    def _is_uaf_or_double_free(self, cfg: RunConfig, data: bytes) -> bool:
        rc, so, se = self._exec_with_input(cfg, data)
        if rc == 0:
            return False
        s = (so + se).decode("latin1", errors="ignore")
        if "AddressSanitizer" not in s and "ERROR: LeakSanitizer" not in s:
            # allow glibc abort messages sometimes
            sl = s.lower()
            if "double free" in sl or "use after free" in sl or "use-after-free" in sl:
                return True
            return False
        sl = s.lower()
        if "double-free" in sl or "attempting double-free" in sl:
            return True
        if "heap-use-after-free" in sl or "use-after-free" in sl:
            return True
        return False

    def _collect_seed_inputs(self, root: str) -> List[bytes]:
        seeds = []
        good_dirs = ("corpus", "seed", "seeds", "test", "tests", "testcases", "examples", "example", "data", "inputs", "fuzz")
        bad_ext = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".md", ".rst", ".py", ".cmake", ".o", ".a", ".so", ".dylib")

        for dirpath, dirnames, filenames in os.walk(root):
            dn = os.path.basename(dirpath).lower()
            if dn in (".git", "build"):
                dirnames[:] = []
                continue

            dir_score = 1
            lowpath = dirpath.lower()
            if any(f"/{d}/" in (lowpath + "/") for d in good_dirs) or any(d == dn for d in good_dirs):
                dir_score = 3

            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(bad_ext):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 8192:
                    continue
                try:
                    with open(path, "rb") as f:
                        b = f.read()
                except OSError:
                    continue
                if not b:
                    continue
                # Avoid accidentally collecting binaries/tools
                if b.startswith(b"\x7fELF"):
                    continue
                seeds.append((dir_score, len(b), b))

        # Add some synthetic seeds
        synth = [
            b"",
            b"\n",
            b"0\n",
            b"1\n",
            b"-1\n",
            b"{}\n",
            b"[]\n",
            b"{\"a\":1}\n",
            b"---\na: b\n",
            b"- a\n- b\n- c\n",
            b"(&a)\n",
        ]
        for b in synth:
            seeds.append((2, len(b), b))

        seeds.sort(key=lambda x: (-x[0], x[1]))
        out = []
        seen = set()
        for _, _, b in seeds:
            if b in seen:
                continue
            seen.add(b)
            out.append(b)
            if len(out) >= 60:
                break
        return out

    def _mutate(self, data: bytes, tokens: List[bytes], max_len: int) -> bytes:
        if data is None:
            data = b""
        b = bytearray(data)

        def rand_token() -> bytes:
            if not tokens:
                return bytes([random.randrange(256)])
            t = tokens[random.randrange(len(tokens))]
            if random.random() < 0.15 and len(t) > 0 and all(32 <= x < 127 for x in t):
                # sometimes add separators around ascii tokens
                sep = random.choice([b"", b" ", b"\n", b":", b",", b";"])
                return sep + t + sep
            return t

        ops = random.randint(1, 6)
        for _ in range(ops):
            if not b and random.random() < 0.6:
                op = "insert"
            else:
                op = random.choice(["insert", "delete", "flip", "repeat", "setbyte", "splice"])

            if op == "insert":
                t = rand_token()
                pos = random.randrange(len(b) + 1) if b else 0
                if len(b) + len(t) <= max_len:
                    b[pos:pos] = t
            elif op == "delete" and b:
                a = random.randrange(len(b))
                l = random.randrange(1, min(len(b) - a, 16) + 1)
                del b[a:a + l]
            elif op == "flip" and b:
                i = random.randrange(len(b))
                bit = 1 << random.randrange(8)
                b[i] ^= bit
            elif op == "setbyte" and b:
                i = random.randrange(len(b))
                b[i] = random.randrange(256)
            elif op == "repeat" and b:
                a = random.randrange(len(b))
                l = random.randrange(1, min(len(b) - a, 32) + 1)
                seg = b[a:a + l]
                pos = random.randrange(len(b) + 1)
                if len(b) + len(seg) <= max_len:
                    b[pos:pos] = seg
            elif op == "splice":
                t = rand_token()
                if not t:
                    continue
                if b:
                    a = random.randrange(len(b))
                    l = random.randrange(0, min(len(b) - a, len(t), 16) + 1)
                    b[a:a + l] = t[:l] if l else t[:1]
                else:
                    if len(t) <= max_len:
                        b.extend(t)

        # Occasionally generate fresh random
        if random.random() < 0.08:
            ln = random.randrange(0, min(max_len, 256) + 1)
            return bytes(random.getrandbits(8) for _ in range(ln))

        if len(b) > max_len:
            b = b[:max_len]
        return bytes(b)

    def _structured_attempts(self, tokens: List[bytes]) -> List[bytes]:
        attempts = []
        # Some YAML/JSON-ish structures; Node::* suggests tree building
        attempts.append(b"---\na:\n  b: c\n")
        attempts.append(b"---\na: [1,2,3,4,5,6,7,8,9]\n")
        attempts.append(b"{\"a\":[1,2,3,4,5,6,7,8,9]}\n")
        attempts.append(b"---\na: &a {b: c}\nd: *a\n")
        attempts.append(b"---\na: &a [1,2,3]\nb: *a\n")
        attempts.append(b"---\n" + b"a: {b: {c: {d: {e: {f: 1}}}}}\n")
        attempts.append(b"[" + b",".join([b"0"] * 200) + b"]\n")
        attempts.append(b"{" + b",".join([b"\"k\":0"] * 200) + b"}\n")
        # Use tokens to build a pseudo-structured text
        ascii_tokens = [t for t in tokens if t and all(32 <= x < 127 for x in t) and len(t) <= 12]
        if ascii_tokens:
            for _ in range(20):
                parts = []
                for __ in range(random.randint(5, 20)):
                    parts.append(random.choice(ascii_tokens))
                    parts.append(random.choice([b" ", b"\n", b":", b",", b";"]))
                a = b"".join(parts)[:512]
                attempts.append(a)
        # Also some binary-ish patterns
        attempts.append(b"\x00" * 64)
        attempts.append(b"\xff" * 64)
        attempts.append(b"\x00\x00\x00\x00" + b"\xff\xff\xff\xff" + b"\x7f\xff\xff\xff" + b"A" * 48)
        return attempts

    def _quick_fuzz(self, cfg: RunConfig, tokens: List[bytes], seeds: List[bytes]) -> Optional[bytes]:
        start = time.time()
        budget = 18.0

        corpus = list(seeds) if seeds else [b"", b"\n", b"{}\n", b"---\na: b\n"]
        # Ensure some diversity
        corpus += [b"A" * 32, b"0" * 32, b"---\n" + b"- a\n" * 20, b"{" + b"\"a\":0," * 30 + b"\"z\":0}\n"]
        corpus = [c for c in corpus if c is not None]

        max_len = 2048

        while time.time() - start < budget:
            parent = random.choice(corpus)
            cand = self._mutate(parent, tokens, max_len)

            if self._is_uaf_or_double_free(cfg, cand):
                return cand

            # Keep some candidates to diversify
            if len(corpus) < 80 and random.random() < 0.08:
                corpus.append(cand)

        return None

    def _minimize(self, cfg: RunConfig, data: bytes) -> bytes:
        if not data:
            return data
        if not self._is_uaf_or_double_free(cfg, data):
            return data

        b = data

        # Trim ends
        changed = True
        while changed and len(b) > 1:
            changed = False
            # Try trimming from end
            for cut in [len(b) - 1, max(0, len(b) - 2), max(0, len(b) - 4), max(0, len(b) - 8)]:
                if cut <= 0:
                    continue
                nb = b[:cut]
                if self._is_uaf_or_double_free(cfg, nb):
                    b = nb
                    changed = True
                    break
            if changed:
                continue
            # Try trimming from start
            for cut in [1, 2, 4, 8]:
                if cut >= len(b):
                    continue
                nb = b[cut:]
                if self._is_uaf_or_double_free(cfg, nb):
                    b = nb
                    changed = True
                    break

        # ddmin-like chunk removal
        n = 2
        start_time = time.time()
        time_budget = 12.0
        while len(b) >= 2 and time.time() - start_time < time_budget:
            chunk = (len(b) + n - 1) // n
            if chunk <= 0:
                break
            reduced = False
            i = 0
            while i < len(b) and time.time() - start_time < time_budget:
                nb = b[:i] + b[i + chunk:]
                if nb and self._is_uaf_or_double_free(cfg, nb):
                    b = nb
                    reduced = True
                    n = max(2, n - 1)
                    break
                i += chunk
            if not reduced:
                if n >= len(b):
                    break
                n = min(len(b), n * 2)

        return b

    def _try_libfuzzer_end_to_end(self, root: str, td: str) -> Optional[bytes]:
        # If there's a fuzzer entry and clang is available, try building with -fsanitize=fuzzer,address and let it find an artifact.
        if not self._has_fuzzer_entry(root):
            return None
        clangpp = shutil.which("clang++")
        if clangpp is None:
            return None

        c_src, cpp_src, all_src = self._collect_sources(root)
        if not all_src:
            return None

        # Exclude any main definitions to avoid conflicts with libFuzzer main
        def file_contains_main(p: str) -> bool:
            try:
                with open(p, "rb") as f:
                    data = f.read(200_000)
            except OSError:
                return False
            return re.search(rb"\bint\s+main\s*\(", data) is not None

        fsrc = [p for p in all_src if not file_contains_main(p)]
        if not fsrc:
            return None

        build_dir = os.path.join(td, "libfuzzer_build")
        os.makedirs(build_dir, exist_ok=True)
        obj_dir = os.path.join(build_dir, "obj")
        os.makedirs(obj_dir, exist_ok=True)

        include_dirs = set()
        for p in fsrc:
            include_dirs.add(os.path.dirname(p))
        for cand in ("include", "includes", "inc", "src"):
            ip = os.path.join(root, cand)
            if os.path.isdir(ip):
                include_dirs.add(ip)
        inc_flags = []
        for d in sorted(include_dirs):
            inc_flags += ["-I", d]

        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:symbolize=0")

        cxxflags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=fuzzer,address", "-fno-sanitize-recover=all", "-Wno-error", "-std=c++17"]
        objs = []
        for p in fsrc:
            base = os.path.basename(p)
            obj = os.path.join(obj_dir, base + ".o")
            cmd = [clangpp, "-c", p, "-o", obj] + cxxflags + inc_flags
            rc, _, _ = self._run_cmd(cmd, cwd=build_dir, env=env, timeout=240.0)
            if rc != 0:
                return None
            objs.append(obj)

        exe = os.path.join(build_dir, "fuzzer")
        link_cmd = [clangpp, "-o", exe] + objs + ["-fsanitize=fuzzer,address"]
        rc, _, _ = self._run_cmd(link_cmd, cwd=build_dir, env=env, timeout=240.0)
        if rc != 0:
            return None
        try:
            st = os.stat(exe)
            os.chmod(exe, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass

        artifacts = os.path.join(build_dir, "artifacts")
        corpus = os.path.join(build_dir, "corpus")
        os.makedirs(artifacts, exist_ok=True)
        os.makedirs(corpus, exist_ok=True)

        # Give it a short time to find the vulnerability
        cmd = [exe, f"-artifact_prefix={artifacts}/", "-max_total_time=12", corpus]
        self._run_cmd(cmd, cwd=build_dir, env=env, timeout=30.0)

        # Collect crash artifacts
        try:
            files = [os.path.join(artifacts, x) for x in os.listdir(artifacts)]
        except OSError:
            files = []
        crash_files = []
        for p in files:
            bn = os.path.basename(p).lower()
            if bn.startswith("crash-") or bn.startswith("timeout-") or bn.startswith("leak-") or bn.startswith("oom-"):
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if 0 < st.st_size <= 1_000_000:
                    crash_files.append((st.st_size, p))
        crash_files.sort()
        for _, p in crash_files[:5]:
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            # Validate it triggers the specific sanitizer class
            cfg = self._determine_run_config(exe, build_dir, prefer_libfuzzer=True)
            if cfg and self._is_uaf_or_double_free(cfg, data):
                return data
        return None