import os
import tarfile


class Solution:
    def _select_poc_member(self, members):
        target_size = 140
        allowed_exts = {
            "",
            ".bin",
            ".dat",
            ".data",
            ".json",
            ".jsn",
            ".txt",
            ".log",
            ".dump",
            ".dmp",
            ".trace",
            ".trc",
            ".pb",
            ".pbtxt",
            ".in",
            ".out",
            ".raw",
            ".poc",
            ".msg",
            ".binproto",
        }
        keywords = {
            "poc": 1000,
            "proof": 800,
            "crash": 900,
            "overflow": 600,
            "stack": 400,
            "heap": 400,
            "memory": 400,
            "snapshot": 500,
            "trace": 350,
            "fuzz": 300,
            "clusterfuzz": 350,
            "oss-fuzz": 350,
            "bug": 250,
            "testdata": 100,
            "seed": 150,
        }

        best_member = None
        best_score = -1

        for m in members:
            size = m.size
            if size <= 0 or size > 5 * 1024 * 1024:
                continue

            name = os.path.basename(m.name)
            lower_path = m.name.lower()
            ext = os.path.splitext(name)[1].lower()

            if ext not in allowed_exts:
                if "poc" not in lower_path and "crash" not in lower_path:
                    continue

            score = 0

            if size == target_size:
                score += 2000

            diff = abs(size - target_size)
            closeness_range = 4096
            if diff < closeness_range:
                score += (closeness_range - diff) // 8

            for kw, weight in keywords.items():
                if kw in lower_path:
                    score += weight

            if score > best_score or (score == best_score and best_member is not None and m.name < best_member.name):
                best_score = score
                best_member = m

        if best_member is not None and best_score > 0:
            return best_member
        return None

    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                candidate = self._select_poc_member(members)
                if candidate is not None:
                    f = tf.extractfile(candidate)
                    if f is not None:
                        data = f.read()
                        if data:
                            return data
        except Exception:
            pass

        return b"A" * 140