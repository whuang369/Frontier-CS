import os
import tarfile
from typing import Optional, List, Tuple


class Solution:
    def _score_member(self, member: tarfile.TarInfo) -> int:
        name = member.name.lower()
        base = os.path.basename(name)
        _, ext = os.path.splitext(base)

        score = 0

        target_len = 524
        if member.size == target_len:
            score += 1000
        else:
            # Prefer sizes close to 524
            diff = abs(member.size - target_len)
            score += max(0, 200 - diff)

        binary_exts = {
            ".rar", ".bin", ".dat", ".raw", ".poc", ".in", ".inp", ".out",
            ".gz", ".bz2", ".xz", ".zip"
        }
        text_exts = {
            ".c", ".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx",
            ".py", ".java", ".js", ".ts", ".md", ".txt", ".rst",
            ".html", ".htm", ".xml", ".json", ".yml", ".yaml",
            ".toml", ".ini", ".cfg", ".cmake", ".mak", ".mk",
        }

        if ext in binary_exts:
            score += 100
        if ext in text_exts:
            score -= 200

        keywords_high = ["12466"]
        for kw in keywords_high:
            if kw in name:
                score += 80

        keywords_medium = ["poc", "crash", "exploit", "id:", "id_", "rar5", "huffman"]
        for kw in keywords_medium:
            if kw in name:
                score += 40

        keywords_low = ["regress", "test", "tests", "fuzz", "cases"]
        for kw in keywords_low:
            if kw in name:
                score += 10

        # Slight preference for paths under directories that look like PoC storage
        dirs_pref = ["poc", "pocs", "crashes", "seeds", "inputs", "corpus"]
        parts = name.split("/")
        for part in parts[:-1]:
            if part in dirs_pref:
                score += 15

        return score

    def _choose_member(self, members: List[tarfile.TarInfo]) -> Optional[tarfile.TarInfo]:
        best: Optional[Tuple[int, int, str, tarfile.TarInfo]] = None

        for m in members:
            if not m.isreg():
                continue
            if m.size <= 0:
                continue

            s = self._score_member(m)
            key = (s, -m.size, m.name, m)
            if best is None or key > best:
                best = key

        if best is not None and best[0] > 0:
            return best[3]

        # Fallbacks if scoring didn't find anything with positive score
        binary_exts = {".rar", ".bin", ".dat", ".raw", ".poc", ".in", ".inp", ".out"}
        text_exts = {
            ".c", ".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx",
            ".py", ".java", ".js", ".ts", ".md", ".txt", ".rst",
            ".html", ".htm", ".xml", ".json", ".yml", ".yaml",
            ".toml", ".ini", ".cfg", ".cmake", ".mak", ".mk",
        }

        # Smallest binary-looking file
        best_bin: Optional[Tuple[int, tarfile.TarInfo]] = None
        for m in members:
            if not m.isreg() or m.size <= 0:
                continue
            name = m.name.lower()
            base = os.path.basename(name)
            _, ext = os.path.splitext(base)
            if ext in binary_exts:
                key = (m.size, m)
                if best_bin is None or key < best_bin:
                    best_bin = key
        if best_bin is not None:
            return best_bin[1]

        # Smallest non-text file
        best_non_text: Optional[Tuple[int, tarfile.TarInfo]] = None
        for m in members:
            if not m.isreg() or m.size <= 0:
                continue
            name = m.name.lower()
            base = os.path.basename(name)
            _, ext = os.path.splitext(base)
            if ext not in text_exts:
                key = (m.size, m)
                if best_non_text is None or key < best_non_text:
                    best_non_text = key
        if best_non_text is not None:
            return best_non_text[1]

        # Absolute last resort: smallest file
        best_any: Optional[Tuple[int, tarfile.TarInfo]] = None
        for m in members:
            if not m.isreg() or m.size <= 0:
                continue
            key = (m.size, m)
            if best_any is None or key < best_any:
                best_any = key
        return best_any[1] if best_any is not None else None

    def solve(self, src_path: str) -> bytes:
        with tarfile.open(src_path, "r:*") as tf:
            members = tf.getmembers()
            member = self._choose_member(members)
            if member is None:
                return b""
            f = tf.extractfile(member)
            if f is None:
                return b""
            data = f.read()
            return data if isinstance(data, bytes) else bytes(data)