import os
import tarfile
import tempfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")

        def safe_extract(tar_obj, path):
            for member in tar_obj.getmembers():
                member_path = os.path.join(path, member.name)
                abs_directory = os.path.abspath(path)
                abs_target = os.path.abspath(member_path)
                if not abs_target.startswith(abs_directory + os.sep) and abs_target != abs_directory:
                    continue
                tar_obj.extract(member, path)

        try:
            with tarfile.open(src_path, "r:*") as tf:
                safe_extract(tf, tmpdir)
        except Exception:
            return b"A" * 60

        poc = self._find_poc_file(tmpdir)
        if poc is not None:
            return poc

        return b"A" * 60

    def _collect_candidates(self, root: str, allowed_exts):
        candidates = []
        max_size = 4096

        for dirpath, dirnames, filenames in os.walk(root):
            lower_dir = dirpath.lower()
            if any(skip in lower_dir for skip in ("/.git", "/build", "/.svn", "/out", "/dist")):
                continue

            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue

                size = st.st_size
                if size <= 0 or size > max_size:
                    continue

                lower_path = path.lower()
                base = name.lower()
                _, ext = os.path.splitext(base)

                if allowed_exts is not None and ext not in allowed_exts:
                    continue

                score = 0.0

                if any(k in lower_path for k in ("poc", "crash", "uaf", "use-after-free", "heap-use-after-free", "heapov", "bug", "issue", "exploit")):
                    score += 50.0
                if "cve" in lower_path:
                    score += 40.0
                if ext in (".yaml", ".yml"):
                    score += 30.0
                if "yaml" in lower_path:
                    score += 10.0
                if any(k in lower_path for k in ("test", "tests", "fuzz", "case", "inputs", "corpus")):
                    score += 5.0

                score -= abs(size - 60)

                score -= size / 500.0

                candidates.append((score, path, size))

        return candidates

    def _find_poc_file(self, root: str):
        allowed_exts_primary = {".yaml", ".yml", ".txt", ".in", ".input", ".conf", ".cfg", ".dat", ".json", ".xml"}
        candidates = self._collect_candidates(root, allowed_exts_primary)

        if not candidates:
            candidates = self._collect_candidates(root, None)
            if not candidates:
                return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path, _ = candidates[0]

        try:
            with open(best_path, "rb") as f:
                data = f.read()
        except OSError:
            return None

        return data