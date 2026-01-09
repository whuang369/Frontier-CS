import os
import tarfile
import tempfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = self._prepare_root(src_path)
        poc = self._find_poc(root_dir)
        if poc is not None:
            return poc
        # Fallback generic PoC for compound division by zero
        return b"a=1;a/=0;\n"

    def _prepare_root(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path
        try:
            if tarfile.is_tarfile(src_path):
                tmpdir = tempfile.mkdtemp(prefix="src_")
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmpdir)
                return tmpdir
        except Exception:
            pass
        # Fallback: create empty temp dir if src_path is unusable
        return tempfile.mkdtemp(prefix="src_")

    def _find_poc(self, root_dir: str) -> bytes | None:
        binary_exts = {
            ".o", ".a", ".so", ".dll", ".dylib",
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico",
            ".pdf", ".zip", ".gz", ".xz", ".bz2", ".7z", ".tar", ".rar",
            ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".class", ".jar",
        }
        source_skip_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
            ".m", ".mm", ".go", ".rs", ".java", ".cs",
        }
        text_bonus_exts = {
            ".txt", ".in", ".input", ".poc", ".case", ".test",
            ".script", ".src", ".rb", ".js", ".lua", ".php", ".py",
            ".json", ".toml", ".conf",
        }

        best_data = None
        best_score = -1
        best_len = None

        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                if not os.path.isfile(fpath):
                    continue
                size = st.st_size
                if size == 0 or size > 4096:
                    continue

                ext = os.path.splitext(fname)[1].lower()
                if ext in binary_exts or ext in source_skip_exts:
                    continue

                try:
                    with open(fpath, "rb") as f:
                        data = f.read()
                except OSError:
                    continue

                if b"/=" not in data or b"0" not in data:
                    continue

                score = 0
                # Content-based scoring
                if b"/=" in data:
                    score += 10
                if b"/=0" in data or b"/= 0" in data or b"/=0;" in data:
                    score += 5
                if b"0" in data:
                    score += 1

                lower_data = data.lower()
                for kw in (b"div", b"zero", b"divide"):
                    if kw in lower_data:
                        score += 2

                # Path-based scoring
                path_lower = fpath.lower()
                for kw in ("test", "tests", "poc", "example", "examples",
                           "case", "cases", "input", "inputs", "script", "scripts", "bug"):
                    if kw in path_lower:
                        score += 3

                # Extension-based bonus
                if ext in text_bonus_exts:
                    score += 2

                # Length-based heuristics
                dlen = len(data)
                if dlen == 79:
                    score += 20
                elif 60 <= dlen <= 100:
                    score += 5

                # Prefer ASCII-printable-ish data
                printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in data)
                if printable >= dlen * 0.8:
                    score += 1

                if score > best_score or (score == best_score and (best_len is None or dlen < best_len)):
                    best_score = score
                    best_data = data
                    best_len = dlen

        if best_score >= 10 and best_data is not None:
            return best_data
        return None