import os
import tarfile
import tempfile
import subprocess
import re
import struct
import random
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        workspace = None
        try:
            source_root = src_path
            if not os.path.isdir(src_path):
                workspace = tempfile.mkdtemp(prefix="poc_tic30_")
                with tarfile.open(src_path, "r:*") as tar:
                    tar.extractall(path=workspace)
                source_root = workspace

            project_root = self._find_project_root_with_build_sh(source_root) or source_root

            binary_path = self._build_and_find_binary(project_root)
            if not binary_path:
                return b"A" * 10

            input_mode = self._detect_input_mode(project_root)

            tic30_opc_path = self._find_file(project_root, "tic30-opc.c")
            opcodes = self._parse_opcodes(tic30_opc_path) if tic30_opc_path else []

            if opcodes:
                poc = self._search_with_opcodes(binary_path, project_root, input_mode, opcodes)
                if poc is not None:
                    return poc

            poc = self._random_fuzz_search(binary_path, project_root, input_mode)
            if poc is not None:
                return poc

            return b"A" * 10
        except Exception:
            return b"A" * 10
        finally:
            if workspace is not None and os.path.isdir(workspace):
                try:
                    shutil.rmtree(workspace)
                except Exception:
                    pass

    # ---------------- Helper methods ----------------

    def _find_project_root_with_build_sh(self, root: str) -> str | None:
        for dirpath, _, filenames in os.walk(root):
            if "build.sh" in filenames:
                return dirpath
        return None

    def _build_and_find_binary(self, project_root: str) -> str | None:
        build_sh = os.path.join(project_root, "build.sh")
        if not os.path.isfile(build_sh):
            return None

        try:
            subprocess.run(
                ["bash", "build.sh"],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=False,
            )
        except Exception:
            pass

        binary = self._binary_from_build_sh(project_root, build_sh)
        if binary:
            return binary

        execs = []
        for dirpath, _, filenames in os.walk(project_root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not os.path.isfile(path):
                    continue
                if not (st.st_mode & 0o111):
                    continue
                if name.endswith((".sh", ".py")):
                    continue
                execs.append((st.st_mtime, path))
        if not execs:
            return None
        execs.sort(reverse=True)
        return execs[0][1]

    def _binary_from_build_sh(self, project_root: str, build_sh: str) -> str | None:
        try:
            text = open(build_sh, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            return None
        text = re.sub(r"\\\n", " ", text)
        outs = re.findall(r"-o\s+([^\s]+)", text)
        candidates = []
        for out in outs:
            out = out.strip("\"'")
            if not out or "$" in out or "`" in out or "(" in out or ")" in out:
                continue
            if any(out.endswith(ext) for ext in (".a", ".so", ".dylib", ".o", ".obj")):
                continue
            path = os.path.join(project_root, out)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                candidates.append(path)
        if candidates:
            return candidates[-1]
        return None

    def _find_file(self, root: str, name: str) -> str | None:
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
        return None

    def _parse_opcodes(self, path: str) -> list[int]:
        try:
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            return []
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//.*", "", text)
        pattern = re.compile(
            r'\{\s*"[^"]*"\s*,\s*'
            r'(0x[0-9A-Fa-f]+[uUlL]*|\d+[uUlL]*)\s*,\s*'
            r'(0x[0-9A-Fa-f]+[uUlL]*|\d+[uUlL]*)',
            re.M,
        )
        opcodes = set()
        for m in pattern.finditer(text):
            opcode_str = m.group(1)
            opcode_str = re.sub(r"[uUlL]+$", "", opcode_str)
            try:
                if opcode_str.lower().startswith("0x"):
                    val = int(opcode_str, 16)
                else:
                    val = int(opcode_str, 10)
                opcodes.add(val)
            except ValueError:
                continue
        return sorted(opcodes)

    def _detect_input_mode(self, root: str) -> str:
        # Returns 'stdin', 'file', or 'both'
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith((".c", ".cc", ".cpp")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, "r", encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                if "int main(" not in text:
                    continue
                if "argv[1]" in text or "argc <" in text or "argc <= " in text or "argc != " in text:
                    return "file"
                if "stdin" in text or "fread(" in text or "read(0" in text:
                    return "stdin"
        return "both"

    def _is_crash(self, result: subprocess.CompletedProcess) -> bool:
        code = result.returncode
        if code is None:
            return False
        if code < 0:
            return True
        output = (result.stdout or b"") + (result.stderr or b"")
        txt = output.decode("latin1", errors="ignore").lower()
        crash_keywords = [
            "addresssanitizer",
            "heap-buffer-overflow",
            "stack-buffer-overflow",
            "stack-buffer-overrun",
            "runtime error",
            "segmentation fault",
            "core dumped",
            "stack smashing detected",
            "asan:",
        ]
        return any(k in txt for k in crash_keywords)

    def _run_candidate(self, prog: str, data: bytes, workdir: str, mode: str, timeout: float = 0.5) -> bool:
        if mode in ("stdin", "both"):
            try:
                r = subprocess.run(
                    [prog],
                    input=data,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                if self._is_crash(r):
                    return True
            except Exception:
                pass

        if mode in ("file", "both"):
            fd, path = tempfile.mkstemp(dir=workdir, suffix=".bin")
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            try:
                r = subprocess.run(
                    [prog, path],
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                if self._is_crash(r):
                    os.remove(path)
                    return True
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass

        return False

    def _search_with_opcodes(
        self,
        prog: str,
        workdir: str,
        mode: str,
        opcodes: list[int],
    ) -> bytes | None:
        if not opcodes:
            return None
        max_ops = min(len(opcodes), 512)
        words = opcodes[:max_ops]
        random.shuffle(words)
        for word in words:
            for endian in ("<", ">"):
                try:
                    b = struct.pack(endian + "I", word & 0xFFFFFFFF)
                except struct.error:
                    continue
                for mult in (1, 2, 3):
                    data = b * mult
                    if self._run_candidate(prog, data, workdir, mode):
                        return data
        return None

    def _random_fuzz_search(self, prog: str, workdir: str, mode: str) -> bytes | None:
        for _ in range(256):
            size = random.randint(6, 20)
            data = os.urandom(size)
            if self._run_candidate(prog, data, workdir, mode):
                return data
        # Some structured patterns
        patterns = [
            b"\x00" * 10,
            b"\xff" * 10,
            bytes(range(10)),
            bytes(reversed(range(10))),
            b"\x00\xff" * 5,
        ]
        for data in patterns:
            if self._run_candidate(prog, data, workdir, mode):
                return data
        return None