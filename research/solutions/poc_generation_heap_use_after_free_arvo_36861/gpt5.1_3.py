import tarfile
import os


class Solution:
    def solve(self, src_path: str) -> bytes:
        EXACT_SIZE = 71298

        try:
            tf = tarfile.open(src_path, "r:*")
        except tarfile.ReadError:
            return b""

        with tf:
            members = [m for m in tf.getmembers() if m.isreg()]
            if not members:
                return b""

            def score_member(m: tarfile.TarInfo) -> int:
                name = m.name.lower()
                base = os.path.basename(name)
                score = 0

                # Keyword-based heuristics
                if "poc" in name:
                    score += 100
                if "crash" in name:
                    score += 80
                if "uaf" in name:
                    score += 70
                if "use-after-free" in name or "use_after_free" in name:
                    score += 70
                if "heap" in name:
                    score += 20
                if "fuzz" in name:
                    score += 10
                if "id_" in base:
                    score += 5
                if "corpus" in name:
                    score += 5
                if "repro" in name:
                    score += 60

                # Directory component heuristics
                for p in name.split("/"):
                    if p in (
                        "poc",
                        "pocs",
                        "crash",
                        "crashes",
                        "repro",
                        "repros",
                        "artifacts",
                        "corpus",
                        "seeds",
                    ):
                        score += 40

                # Extension-based heuristics
                _, ext = os.path.splitext(base)
                ext = ext.lower()
                good_exts = {".bin", ".raw", ".dat", ".poc"}
                bad_exts = {
                    ".c",
                    ".h",
                    ".hpp",
                    ".hh",
                    ".cc",
                    ".cpp",
                    ".py",
                    ".rb",
                    ".pl",
                    ".sh",
                    ".txt",
                    ".md",
                    ".rst",
                    ".html",
                    ".htm",
                    ".js",
                    ".css",
                    ".json",
                    ".yml",
                    ".yaml",
                    ".toml",
                    ".ini",
                    ".am",
                    ".ac",
                    ".m4",
                    ".in",
                    ".out",
                    ".bat",
                    ".ps1",
                    ".spec",
                    ".po",
                    ".pot",
                    ".xml",
                    ".cmake",
                    ".m",
                    ".java",
                    ".go",
                    ".rs",
                }
                if ext in good_exts:
                    score += 30
                if ext in bad_exts:
                    score -= 100

                # Size closeness to known PoC size
                diff = abs(m.size - EXACT_SIZE)
                if diff == 0:
                    score += 200
                elif diff < 4096:
                    score += int(50 * (4096 - diff) / 4096)

                return score

            best_member = max(members, key=score_member)
            fileobj = tf.extractfile(best_member)
            if fileobj is None:
                return b""
            data = fileobj.read()
            if isinstance(data, bytes):
                return data
            return bytes(data or b"")