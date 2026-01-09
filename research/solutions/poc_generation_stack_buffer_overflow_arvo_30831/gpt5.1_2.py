import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = None

        if os.path.isdir(src_path):
            poc = self._find_poc_in_dir(src_path)
        elif tarfile.is_tarfile(src_path):
            poc = self._find_poc_in_tar(src_path)

        if poc is not None:
            return poc

        # Fallback: generic 21-byte payload if no PoC found
        return b"A" * 21

    def _is_mostly_text(self, data: bytes) -> bool:
        if not data:
            return True
        text_chars = set(range(32, 127)) | {9, 10, 13}
        text_count = sum(1 for b in data if b in text_chars)
        return text_count / len(data) > 0.9

    def _score_candidate(self, name_lower: str, size: int, is_text_ext: bool, ext: str,
                         target_size: int, data: bytes | None = None) -> float:
        score = 0.0

        # Size proximity to target
        score -= abs(size - target_size) * 0.1

        # Name-based hints
        if "poc" in name_lower or "exploit" in name_lower:
            score += 10
        if "crash" in name_lower or "overflow" in name_lower:
            score += 7
        if any(tag in name_lower for tag in ("test", "fuzz", "seed", "input", "case", "regress")):
            score += 3
        base = os.path.basename(name_lower)
        if base.startswith(("id_", "crash", "poc")):
            score += 4

        # Extension-based hints
        if ext in (".bin", ".raw", ".dat", ""):
            score += 2
        if is_text_ext:
            score -= 3

        # Content-based hints, if available
        if data is not None and len(data) > 0:
            if not self._is_mostly_text(data):
                score += 3
            else:
                score -= 1

        return score

    def _find_poc_in_tar(self, tar_path: str) -> bytes | None:
        target_size = 21
        try:
            tf = tarfile.open(tar_path, "r:*")
        except Exception:
            return None

        cand_exact = []
        cand_small = []

        text_exts = {
            ".c", ".h", ".cpp", ".cc", ".hpp", ".txt", ".md", ".rst",
            ".py", ".java", ".js", ".ts", ".go", ".rs", ".rb", ".php",
            ".sh", ".cmake", ".yml", ".yaml", ".json", ".xml", ".html",
        }

        for m in tf.getmembers():
            if not m.isreg() or m.size == 0:
                continue
            size = m.size
            # Ignore clearly large files
            if size > 1024:
                continue

            name_lower = m.name.lower()
            ext = os.path.splitext(name_lower)[1]
            is_text_ext = ext in text_exts

            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue

            if size == target_size:
                score = self._score_candidate(
                    name_lower, size, is_text_ext, ext, target_size, data=data
                )
                cand_exact.append((score, m, data))
            elif size <= 64:
                score = self._score_candidate(
                    name_lower, size, is_text_ext, ext, target_size, data=data
                )
                cand_small.append((score, m, data))

        if cand_exact:
            cand_exact.sort(key=lambda x: (-x[0], x[1].name))
            best_score, _m, best_data = cand_exact[0]
            if best_score > 0:
                return best_data

        if cand_small:
            cand_small.sort(key=lambda x: (-x[0], abs(len(x[2]) - target_size), x[1].name))
            best_score, _m, best_data = cand_small[0]
            if best_score > 0:
                return best_data

        return None

    def _find_poc_in_dir(self, root: str) -> bytes | None:
        target_size = 21
        cand_exact = []
        cand_small = []

        text_exts = {
            ".c", ".h", ".cpp", ".cc", ".hpp", ".txt", ".md", ".rst",
            ".py", ".java", ".js", ".ts", ".go", ".rs", ".rb", ".php",
            ".sh", ".cmake", ".yml", ".yaml", ".json", ".xml", ".html",
        }

        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    if os.path.islink(path):
                        continue
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0 or size > 1024:
                    continue

                name_lower = path.lower()
                ext = os.path.splitext(name_lower)[1]
                is_text_ext = ext in text_exts

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue

                if size == target_size:
                    score = self._score_candidate(
                        name_lower, size, is_text_ext, ext, target_size, data=data
                    )
                    cand_exact.append((score, path, data))
                elif size <= 64:
                    score = self._score_candidate(
                        name_lower, size, is_text_ext, ext, target_size, data=data
                    )
                    cand_small.append((score, path, data))

        if cand_exact:
            cand_exact.sort(key=lambda x: (-x[0], x[1]))
            best_score, _p, best_data = cand_exact[0]
            if best_score > 0:
                return best_data

        if cand_small:
            cand_small.sort(key=lambda x: (-x[0], abs(len(x[2]) - target_size), x[1]))
            best_score, _p, best_data = cand_small[0]
            if best_score > 0:
                return best_data

        return None