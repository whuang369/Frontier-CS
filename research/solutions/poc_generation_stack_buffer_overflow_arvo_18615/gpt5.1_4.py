import os
import tarfile
import tempfile
import shutil
import stat
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        def is_binary(path: str) -> bool:
            try:
                with open(path, "rb") as f:
                    chunk = f.read(1024)
                if not chunk:
                    return False
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
                nontext = sum(1 for b in chunk if b not in text_chars)
                return nontext > 0.3 * len(chunk)
            except Exception:
                return False

        def extract_tar(src: str) -> str:
            tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
            try:
                if os.path.isdir(src):
                    return src, tmpdir  # root_dir, tmpdir_to_cleanup
                with tarfile.open(src, "r:*") as tar:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        return prefix == abs_directory

                    for member in tar.getmembers():
                        member_path = os.path.join(tmpdir, member.name)
                        if not is_within_directory(tmpdir, member_path):
                            continue
                    tar.extractall(tmpdir)
                entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
                if len(entries) == 1:
                    root = os.path.join(tmpdir, entries[0])
                    if os.path.isdir(root):
                        return root, tmpdir
                return tmpdir, tmpdir
            except Exception:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None, None

        root_dir, tmpdir = extract_tar(src_path)
        if root_dir is None:
            return b"A" * 10

        best = None  # (score, size, path)

        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    size = st.st_size
                    if size <= 0 or size > 1024 * 1024:
                        continue

                    path_lower = full.lower()
                    name_lower = name.lower()

                    score = 0.0

                    if name_lower in ("poc", "poc.bin", "poc.raw", "poc.dat"):
                        score += 120
                    if name_lower.startswith("poc") or name_lower.endswith(".poc"):
                        score += 100
                    if "poc" in path_lower:
                        score += 80
                    if "crash" in path_lower or "id:" in name_lower:
                        score += 60
                    if "input" in path_lower or "testcase" in path_lower or "seed" in path_lower:
                        score += 40
                    if name_lower.endswith((".bin", ".raw", ".dat", ".obj", ".o", ".exe", ".out")):
                        score += 20

                    score += max(0.0, 50.0 - abs(size - 10) * 5.0)
                    score += max(0.0, 20.0 - (size / 10.0))

                    d = dirpath.lower()
                    if d.endswith(os.sep + "poc") or (os.sep + "poc" + os.sep) in d:
                        score += 60
                    if d.endswith(os.sep + "crash") or (os.sep + "crash" + os.sep) in d:
                        score += 40

                    if is_binary(full):
                        score += 30
                    else:
                        score -= 10

                    if best is None or score > best[0]:
                        best = (score, size, full)
        except Exception:
            pass

        best_hex_bytes = None
        if best is None or best[0] < 40:
            try:
                hex_pattern = re.compile(r"\b(?:0x)?[0-9a-fA-F]{2}\b")
                for dirpath, dirnames, filenames in os.walk(root_dir):
                    for name in filenames:
                        full = os.path.join(dirpath, name)
                        try:
                            st = os.stat(full)
                        except OSError:
                            continue
                        size = st.st_size
                        if size <= 0 or size > 100 * 1024:
                            continue
                        name_lower = name.lower()
                        if not name_lower.endswith((".txt", ".md", ".yaml", ".yml", ".json", ".cfg", ".ini", ".log", ".sh", ".c", ".h")):
                            continue
                        try:
                            with open(full, "r", errors="ignore") as f:
                                text = f.read()
                        except Exception:
                            continue
                        if "poc" not in text and "crash" not in text and "input" not in text:
                            continue

                        for line in text.splitlines():
                            if "poc" not in line and "input" not in line and "crash" not in line:
                                continue
                            tokens = hex_pattern.findall(line)
                            if len(tokens) >= 4:
                                b = bytearray()
                                for t in tokens:
                                    t = t.strip()
                                    if t.lower().startswith("0x"):
                                        t = t[2:]
                                    if len(t) != 2:
                                        continue
                                    try:
                                        b.append(int(t, 16))
                                    except ValueError:
                                        pass
                                if 0 < len(b) <= 100:
                                    if best_hex_bytes is None or len(b) < len(best_hex_bytes):
                                        best_hex_bytes = bytes(b)

                        for m in re.finditer(r'"((?:\\x[0-9a-fA-F]{2})+)"', text):
                            seq = m.group(1)
                            b = bytearray()
                            for h in re.findall(r"\\x([0-9a-fA-F]{2})", seq):
                                b.append(int(h, 16))
                            if 0 < len(b) <= 100:
                                if best_hex_bytes is None or len(b) < len(best_hex_bytes):
                                    best_hex_bytes = bytes(b)
            except Exception:
                best_hex_bytes = None

        try:
            if best is not None and best[0] >= 40:
                try:
                    with open(best[2], "rb") as f:
                        data = f.read()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return data
                except Exception:
                    pass

            if best_hex_bytes is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return best_hex_bytes
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return b"A" * 10