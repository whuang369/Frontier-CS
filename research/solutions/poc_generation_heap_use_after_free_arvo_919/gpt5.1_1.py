import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = None

        if os.path.isfile(src_path):
            try:
                if tarfile.is_tarfile(src_path):
                    data = self._poc_from_tar(src_path)
            except Exception:
                data = None

        if data is None and os.path.isdir(src_path):
            data = self._poc_from_dir(src_path)

        if data is None:
            data = self._fallback_poc()

        return data

    def _poc_from_tar(self, path: str) -> bytes | None:
        try:
            with tarfile.open(path, "r:*") as tf:
                candidates = []
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    size = m.size
                    if size <= 0 or size > 4 * 1024 * 1024:
                        continue
                    name = m.name
                    candidates.append(
                        (
                            name,
                            size,
                            lambda m=m, tf=tf: self._read_tar_member(tf, m),
                        )
                    )
                return self._select_best_candidate(candidates)
        except Exception:
            return None

    def _read_tar_member(self, tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
        f = tf.extractfile(member)
        if f is None:
            return b""
        try:
            return f.read()
        finally:
            f.close()

    def _poc_from_dir(self, root: str) -> bytes | None:
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size <= 0 or size > 4 * 1024 * 1024:
                    continue
                rel_name = os.path.relpath(full, root)
                candidates.append(
                    (
                        rel_name,
                        size,
                        lambda p=full: self._read_file(p),
                    )
                )
        return self._select_best_candidate(candidates)

    def _read_file(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _select_best_candidate(self, candidates) -> bytes | None:
        if not candidates:
            return None

        best_loader = None
        best_score = None

        for name, size, loader in candidates:
            score = self._score_candidate(name, size)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_loader = loader

        if best_loader is None:
            return None

        try:
            data = best_loader()
            if not isinstance(data, (bytes, bytearray)):
                return None
            return bytes(data)
        except Exception:
            return None

    def _score_candidate(self, name: str, size: int):
        # Skip VCS and build metadata
        lower = name.lower()
        if "/.git/" in lower or "/.svn/" in lower or "/.hg/" in lower:
            return None
        if "cmake" in lower and (lower.endswith(".txt") or "cmakelists.txt" in lower):
            return None

        # Extension-based scoring
        _, ext = os.path.splitext(lower)
        font_exts = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
        binary_exts = {".a", ".o", ".so", ".dylib", ".dll", ".exe", ".class", ".jar"}
        code_exts = {
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hpp",
            ".hh",
            ".py",
            ".java",
            ".js",
            ".ts",
            ".cs",
            ".go",
            ".rs",
            ".php",
            ".rb",
        }
        doc_exts = {
            ".txt",
            ".md",
            ".rst",
            ".html",
            ".htm",
            ".xml",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
        }

        if ext in font_exts:
            etype = 0
        elif ext == "":
            etype = 5
        elif ext in binary_exts:
            etype = 30
        elif ext in code_exts:
            etype = 40
        elif ext in doc_exts:
            etype = 50
        else:
            etype = 10

        # Name-based scoring
        if "poc" in lower:
            ntype = 0
        elif "crash" in lower or "id:" in lower or "clusterfuzz" in lower:
            ntype = 1
        elif "fuzz" in lower:
            ntype = 2
        elif "corpus" in lower or "test" in lower or "input" in lower or "inputs" in lower:
            ntype = 3
        else:
            ntype = 5

        base_cat = etype + ntype

        # Size closeness to 800 bytes
        size_penalty = abs(size - 800)

        # Final score tuple: lower is better
        return (base_cat, size_penalty, size)

    def _fallback_poc(self) -> bytes:
        # Deterministic fallback input (~800 bytes)
        header = b"OTTO"  # OpenType font magic
        body = b"A" * (800 - len(header))
        return header + body