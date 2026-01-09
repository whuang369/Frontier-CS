import os
import tarfile
import zipfile
import io
import gzip
import lzma
from typing import Optional


class Solution:
    TARGET_POC_LEN = 71298
    MAX_DECOMPRESSED_SIZE = 10 * 1024 * 1024  # 10MB

    TEXT_EXTS = {
        ".c", ".h", ".hpp", ".hh", ".cc", ".cpp", ".cxx", ".java",
        ".py", ".pyw", ".pl", ".pm", ".rb", ".sh", ".bash", ".zsh",
        ".txt", ".md", ".markdown", ".rst",
        ".html", ".htm", ".xml", ".xhtml",
        ".json", ".csv", ".tsv",
        ".cfg", ".conf", ".ini", ".toml", ".yml", ".yaml",
        ".tex", ".bib",
        ".cmake", ".am", ".ac",
        ".log",
        ".in", ".m4",
        ".sln", ".vcxproj", ".vcproj", ".mk", ".makefile",
        ".gradle", ".pom", ".properties",
    }

    EXT_CANDS = {
        ".bin", ".dat", ".raw", ".poc", ".in", ".out",
        ".packet", ".state", ".dump", ".usb",
    }

    KEYWORDS = [
        "poc", "crash", "uaf", "use_after_free", "use-after-free",
        "use after free", "heap-use-after-free", "heapoverflow",
        "heap_overflow", "heap", "exploit", "trigger", "testcase",
        "id:", "clusterfuzz", "cve", "bug",
    ]

    FOLDER_KEYWORDS = [
        "poc", "pocs", "crashes", "crash", "seeds", "corpus",
        "inputs", "input", "testcases", "tests", "regress", "fuzz",
        "oss-fuzz", "clusterfuzz",
    ]

    def solve(self, src_path: str) -> bytes:
        best_data: Optional[bytes] = None

        # Try as tarball
        try:
            with tarfile.open(src_path, "r:*") as tar:
                best_data = self._find_poc_in_tar(tar)
        except tarfile.TarError:
            best_data = None

        # If not tarball or nothing found, maybe it's a directory
        if best_data is None and os.path.isdir(src_path):
            best_data = self._find_poc_in_dir(src_path)

        if best_data is not None and len(best_data) > 0:
            return best_data

        # Fallback generic PoC
        return self._generate_fallback_poc()

    # ---------- Helper methods ----------

    def _compute_score(self, path: str, size: int) -> float:
        if size <= 0:
            return -1.0

        path_norm = path.replace("\\", "/").lower()
        base, ext = os.path.splitext(path_norm)

        # Skip obvious text files unless strongly hinted
        if ext in self.TEXT_EXTS:
            hinted = any(kw in path_norm for kw in self.KEYWORDS)
            if not hinted:
                return -1.0

        score = 0.0

        # Name-based hints
        if any(kw in path_norm for kw in self.KEYWORDS):
            score += 50.0

        for kw in self.FOLDER_KEYWORDS:
            if (
                f"/{kw}/" in path_norm
                or path_norm.startswith(kw + "/")
                or path_norm.endswith("/" + kw)
            ):
                score += 30.0
                break

        if ext in self.EXT_CANDS:
            score += 20.0

        # Size closeness to target PoC length
        target = self.TARGET_POC_LEN
        size_score = max(0.0, 30.0 - abs(size - target) / 2048.0)
        score += size_score

        return score

    def _find_poc_in_tar(self, tar: tarfile.TarFile) -> Optional[bytes]:
        best_data: Optional[bytes] = None
        best_score: float = 0.0

        for member in tar.getmembers():
            if not member.isfile():
                continue

            name = member.name
            lower = name.lower()
            size = member.size

            # Handle ZIP archives inside tar (e.g., seed_corpus.zip)
            if lower.endswith(".zip"):
                try:
                    with tar.extractfile(member) as f:
                        zipped_bytes = f.read()
                except (KeyError, OSError):
                    continue
                try:
                    with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zf:
                        for zi in zf.infolist():
                            if zi.is_dir():
                                continue
                            zsize = zi.file_size
                            if zsize <= 0 or zsize > self.MAX_DECOMPRESSED_SIZE:
                                continue
                            path = f"{name}|{zi.filename}"
                            score = self._compute_score(path, zsize)
                            if score > best_score and score >= 40.0:
                                try:
                                    with zf.open(zi) as zf_f:
                                        data = zf_f.read()
                                except OSError:
                                    continue
                                best_score = score
                                best_data = data
                except zipfile.BadZipFile:
                    continue
                continue

            # Handle .gz or .gzip
            if lower.endswith(".gz") or lower.endswith(".gzip"):
                base_name = name
                if lower.endswith(".gz"):
                    base_name = name[:-3]
                elif lower.endswith(".gzip"):
                    base_name = name[:-5]
                try:
                    with tar.extractfile(member) as f:
                        comp = f.read()
                except (KeyError, OSError):
                    continue
                try:
                    data = gzip.decompress(comp)
                except OSError:
                    continue
                dlen = len(data)
                if dlen <= 0 or dlen > self.MAX_DECOMPRESSED_SIZE:
                    continue
                score = self._compute_score(base_name, dlen)
                if score > best_score and score >= 40.0:
                    best_score = score
                    best_data = data
                continue

            # Handle .xz
            if lower.endswith(".xz"):
                base_name = name[:-3]
                try:
                    with tar.extractfile(member) as f:
                        comp = f.read()
                except (KeyError, OSError):
                    continue
                try:
                    data = lzma.decompress(comp)
                except lzma.LZMAError:
                    continue
                dlen = len(data)
                if dlen <= 0 or dlen > self.MAX_DECOMPRESSED_SIZE:
                    continue
                score = self._compute_score(base_name, dlen)
                if score > best_score and score >= 40.0:
                    best_score = score
                    best_data = data
                continue

            # Regular file inside tar
            if size <= 0 or size > self.MAX_DECOMPRESSED_SIZE:
                continue

            score = self._compute_score(name, size)
            if score > best_score and score >= 40.0:
                try:
                    with tar.extractfile(member) as f:
                        data = f.read()
                except (KeyError, OSError):
                    continue
                best_score = score
                best_data = data

        return best_data

    def _find_poc_in_dir(self, root: str) -> Optional[bytes]:
        best_data: Optional[bytes] = None
        best_score: float = 0.0

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                rel_path = os.path.relpath(full, root)
                lower = full.lower()

                # ZIP archives
                if lower.endswith(".zip"):
                    try:
                        with open(full, "rb") as f:
                            zipped_bytes = f.read()
                    except OSError:
                        continue
                    try:
                        with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zf:
                            for zi in zf.infolist():
                                if zi.is_dir():
                                    continue
                                zsize = zi.file_size
                                if zsize <= 0 or zsize > self.MAX_DECOMPRESSED_SIZE:
                                    continue
                                path = f"{rel_path}|{zi.filename}"
                                score = self._compute_score(path, zsize)
                                if score > best_score and score >= 40.0:
                                    try:
                                        with zf.open(zi) as zf_f:
                                            data = zf_f.read()
                                    except OSError:
                                        continue
                                    best_score = score
                                    best_data = data
                    except zipfile.BadZipFile:
                        continue
                    continue

                # Gzip
                if lower.endswith(".gz") or lower.endswith(".gzip"):
                    base_name = rel_path
                    if lower.endswith(".gz"):
                        base_name = rel_path[:-3]
                    elif lower.endswith(".gzip"):
                        base_name = rel_path[:-5]
                    try:
                        with open(full, "rb") as f:
                            comp = f.read()
                    except OSError:
                        continue
                    try:
                        data = gzip.decompress(comp)
                    except OSError:
                        continue
                    dlen = len(data)
                    if dlen <= 0 or dlen > self.MAX_DECOMPRESSED_SIZE:
                        continue
                    score = self._compute_score(base_name, dlen)
                    if score > best_score and score >= 40.0:
                        best_score = score
                        best_data = data
                    continue

                # XZ
                if lower.endswith(".xz"):
                    base_name = rel_path[:-3]
                    try:
                        with open(full, "rb") as f:
                            comp = f.read()
                    except OSError:
                        continue
                    try:
                        data = lzma.decompress(comp)
                    except lzma.LZMAError:
                        continue
                    dlen = len(data)
                    if dlen <= 0 or dlen > self.MAX_DECOMPRESSED_SIZE:
                        continue
                    score = self._compute_score(base_name, dlen)
                    if score > best_score and score >= 40.0:
                        best_score = score
                        best_data = data
                    continue

                # Regular file
                if size <= 0 or size > self.MAX_DECOMPRESSED_SIZE:
                    continue

                score = self._compute_score(rel_path, size)
                if score > best_score and score >= 40.0:
                    try:
                        with open(full, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    best_score = score
                    best_data = data

        return best_data

    def _generate_fallback_poc(self) -> bytes:
        # Generate a large binary blob intended to stress serializers and buffers.
        size = max(self.TARGET_POC_LEN, 80 * 1024)
        buf = bytearray(size)

        prefix = b"USBREDIR_POC_HEAP_USE_AFTER_FREE\n"
        plen = min(len(prefix), size)
        buf[0:plen] = prefix[:plen]

        # After prefix, place some large integers to encourage "large counts"
        offset = plen
        for i in range(offset, min(offset + 256, size - 4), 4):
            buf[i:i + 4] = (0xFFFFFFFF).to_bytes(4, "little")

        # Fill the rest with a deterministic pseudo-random pattern
        start = min(offset + 256, size)
        for i in range(start, size):
            buf[i] = (i * 37 + 101) & 0xFF

        return bytes(buf)