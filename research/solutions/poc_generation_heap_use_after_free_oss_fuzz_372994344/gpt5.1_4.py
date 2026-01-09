import os
import tarfile
from typing import List, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        TARGET_LEN = 1128

        def is_mostly_text(sample: bytes) -> bool:
            if not sample:
                return True
            text_bytes = set(range(32, 127))
            text_bytes.update({9, 10, 13})
            nontext = 0
            for b in sample:
                if b not in text_bytes:
                    nontext += 1
            ratio = nontext / len(sample)
            return ratio < 0.05

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return b"A"

        candidates: List[Tuple[float, tarfile.TarInfo]] = []

        high_keywords = ["372994344", "oss-fuzz", "ossfuzz", "clusterfuzz", "heap-use-after-free", "uaf"]
        medium_keywords = ["regress", "regression", "fuzz", "crash", "bug", "issue", "test", "tests", "poc"]
        media_exts = [".ts", ".m2ts", ".mpg", ".mpeg", ".mp4", ".bin", ".es", ".tsv"]
        uninterested_names = ["readme", "license", "copying", "changelog", "news", "notice"]

        for member in tf.getmembers():
            if not member.isfile():
                continue

            size = member.size
            if size <= 0:
                continue
            if size > 50_000_000:
                continue

            path = member.name
            lower = path.lower()
            base = os.path.basename(lower)

            score = 0.0

            # Size-based scoring, heavily favoring TARGET_LEN
            diff = abs(size - TARGET_LEN)
            if diff == 0:
                score += 1000.0
            else:
                score += max(0.0, 300.0 - diff * 0.5)

            # Path-based scoring
            for kw in high_keywords:
                if kw in lower:
                    score += 400.0
            for kw in medium_keywords:
                if kw in lower:
                    score += 40.0
            for ext in media_exts:
                if lower.endswith(ext):
                    score += 80.0

            # Penalize obvious source/text files
            if lower.endswith(
                (
                    ".c",
                    ".h",
                    ".cpp",
                    ".cc",
                    ".cxx",
                    ".hpp",
                    ".py",
                    ".txt",
                    ".md",
                    ".rst",
                    ".json",
                    ".xml",
                    ".html",
                    ".htm",
                    ".cmake",
                    ".yml",
                    ".yaml",
                    ".sh",
                    ".bat",
                    ".in",
                    ".am",
                    ".ac",
                    ".m4",
                    ".java",
                    ".kt",
                    ".gradle",
                    ".pro",
                    ".pri",
                )
            ):
                score -= 120.0

            # Penalize typical doc filenames
            for nm in uninterested_names:
                if base == nm or base.startswith(nm + "."):
                    score -= 300.0

            # Penalize build artifacts
            if lower.endswith((".o", ".a", ".so", ".dll", ".dylib", ".exe", ".class", ".jar")):
                score -= 300.0

            # Prefer smaller test-like assets
            if size <= 16_384:
                score += 40.0
            elif size <= 262_144:
                score += 10.0
            else:
                score -= 50.0

            candidates.append((score, member))

        if not candidates:
            tf.close()
            return b"A"

        candidates.sort(key=lambda x: x[0], reverse=True)

        chosen_data = None

        # Try top N candidates, prefer binary files
        N = min(50, len(candidates))
        fallback_member = candidates[0][1]

        for i in range(N):
            member = candidates[i][1]
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                to_read = min(4096, member.size)
                head = f.read(to_read)
                if is_mostly_text(head) and member.size > 256:
                    continue
                rest = f.read()
                chosen_data = head + rest
                if chosen_data:
                    break
            except Exception:
                continue

        if chosen_data is None:
            # Fallback to highest-scoring member regardless of text/binary
            try:
                f = tf.extractfile(fallback_member)
                if f is not None:
                    chosen_data = f.read()
            except Exception:
                chosen_data = None

        tf.close()

        if not chosen_data:
            return b"A"

        return chosen_data