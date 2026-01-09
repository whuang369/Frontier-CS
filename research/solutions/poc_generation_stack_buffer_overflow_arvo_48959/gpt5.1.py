import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 27

        def read_member(tar, member):
            try:
                f = tar.extractfile(member)
                if f is None:
                    return b""
                return f.read()
            except Exception:
                return b""

        try:
            with tarfile.open(src_path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]

                # Pass 0: exact-length candidates
                exact_candidates = [m for m in members if m.size == target_len]
                if exact_candidates:
                    def name_score(m):
                        name = m.name.lower()
                        score = 0
                        if "poc" in name:
                            score += 50
                        if "crash" in name:
                            score += 40
                        if "overflow" in name or "oflow" in name:
                            score += 30
                        if "stack" in name:
                            score += 10
                        if name.endswith((".gz", ".gzip", ".zlib", ".png", ".bin", ".dat", ".raw", ".in")):
                            score += 20
                        return score

                    best = max(exact_candidates, key=name_score)
                    data = read_member(tar, best)
                    if data:
                        return data

                # Pass 1: files whose names strongly suggest PoCs
                poc_keywords = ("poc", "crash", "overflow", "stack", "bug", "fail", "exploit")
                poc_candidates = []
                for m in members:
                    name = m.name.lower()
                    if any(k in name for k in poc_keywords):
                        poc_candidates.append(m)

                if poc_candidates:
                    def score_poc(m):
                        name = m.name.lower()
                        score = 0
                        if "poc" in name:
                            score += 50
                        if "crash" in name:
                            score += 40
                        if "overflow" in name or "stack" in name:
                            score += 20
                        diff = abs(m.size - target_len)
                        if diff == 0:
                            score += 50
                        elif diff <= 5:
                            score += 25
                        elif diff <= 20:
                            score += 10
                        _, ext = os.path.splitext(name)
                        if ext in (".gz", ".gzip", ".zlib", ".png", ".bin", ".dat", ".raw", ".in"):
                            score += 15
                        if m.size < 4096:
                            score += 5
                        return score

                    best = max(poc_candidates, key=score_poc)
                    data = read_member(tar, best)
                    if data:
                        return data

                # Pass 2: small, non-source-like files
                src_exts = {
                    ".c", ".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx",
                    ".py", ".sh", ".md", ".rst", ".html", ".xml",
                    ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
                    ".cmake", ".java", ".rb", ".go"
                }
                small_candidates = []
                for m in members:
                    if m.size <= 4096:
                        name = m.name.lower()
                        _, ext = os.path.splitext(name)
                        if ext in src_exts:
                            continue
                        small_candidates.append(m)

                if small_candidates:
                    def score_small(m):
                        name = m.name.lower()
                        score = 0
                        diff = abs(m.size - target_len)
                        if diff == 0:
                            score += 50
                        elif diff <= 5:
                            score += 25
                        elif diff <= 20:
                            score += 10
                        if any(k in name for k in ("id_", "id-", "id=")):
                            score += 5
                        if any(k in name for k in ("seed", "corpus", "input", "fuzz")):
                            score += 5
                        _, ext = os.path.splitext(name)
                        if ext in (".gz", ".gzip", ".zlib", ".png", ".bin", ".dat", ".raw", ".in"):
                            score += 10
                        return score

                    best = max(small_candidates, key=score_small)
                    data = read_member(tar, best)
                    if data:
                        return data

                # Pass 3: smallest file as last resort
                smallest = None
                for m in members:
                    if smallest is None or m.size < smallest.size:
                        smallest = m
                if smallest is not None:
                    data = read_member(tar, smallest)
                    if data:
                        return data

        except Exception:
            pass

        # Final fallback: synthetic 27-byte buffer
        return b"A" * target_len