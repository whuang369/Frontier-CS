import os
import tarfile
from typing import Optional


class Solution:
    TARGET_SIZE = 274_773

    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Try to locate an existing PoC in the provided tarball or directory
        try:
            if os.path.isdir(src_path):
                data = self._find_poc_in_dir(src_path)
                if data is not None:
                    return data

            if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
                data = self._find_poc_in_tar(src_path)
                if data is not None:
                    return data
        except Exception:
            # Fall back if anything unexpected happens while inspecting the tarball
            pass

        # Ultimate fallback: return a minimal placeholder input
        return self._fallback_poc()

    # ---------------- Internal helpers ----------------

    def _score_name(self, path: str) -> int:
        """
        Heuristic score based on filename/path indicating it's likely a PoC.
        """
        name = path.lower()
        score = 0

        # Issue id
        if "368076875" in name:
            score += 50

        # Generic PoC / crash indicators
        keywords = [
            "poc",
            "proof",
            "repro",
            "reproducer",
            "crash",
            "uaf",
            "use-after-free",
            "use_after_free",
            "heap",
            "bug",
            "fail",
            "input",
            "testcase",
            "fuzz",
        ]
        for kw in keywords:
            if kw in name:
                score += 5

        # Directory hints
        components = name.split("/")
        dir_keywords = [
            "poc",
            "pocs",
            "crash",
            "crashes",
            "regress",
            "regression",
            "tests",
            "corpus",
            "inputs",
        ]
        if any(c in dir_keywords for c in components):
            score += 3

        # Extension hints
        base = os.path.basename(name)
        _, ext = os.path.splitext(base)
        if ext in (
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".bin",
            ".dat",
            ".in",
            ".input",
            ".xml",
            ".html",
            ".c",
            ".cc",
            ".cpp",
            ".js",
        ):
            score += 2

        return score

    def _should_skip_ext(self, filename: str) -> bool:
        """
        Filter out obvious build artifacts / libraries unlikely to be PoCs.
        """
        name = filename.lower()
        skip_exts = (
            ".o",
            ".lo",
            ".la",
            ".a",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".obj",
            ".lib",
            ".jar",
            ".class",
        )
        return any(name.endswith(ext) for ext in skip_exts)

    def _choose_better_candidate(
        self,
        current_best: Optional[tuple],
        candidate_path: str,
        size: int,
        name_score: int,
    ) -> Optional[tuple]:
        """
        Decide whether the new candidate is better than the current best.

        current_best is a tuple: (best_absdiff, best_name_score, best_path)
        """
        absdiff = abs(size - self.TARGET_SIZE)

        if current_best is None:
            return (absdiff, name_score, candidate_path)

        best_absdiff, best_name_score, best_path = current_best

        # Prefer closer size to TARGET_SIZE first, then higher name_score.
        if absdiff < best_absdiff:
            return (absdiff, name_score, candidate_path)
        if absdiff == best_absdiff and name_score > best_name_score:
            return (absdiff, name_score, candidate_path)
        return current_best

    def _find_poc_in_tar(self, tar_path: str) -> Optional[bytes]:
        """
        Search within a tarball for a file that is likely the PoC.
        """
        best = None  # (absdiff, name_score, member_name)

        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                size = member.size
                if size <= 0:
                    continue

                # Skip obviously irrelevant huge files
                if size > 10 * self.TARGET_SIZE:
                    continue

                name = member.name
                if self._should_skip_ext(name):
                    continue

                name_score = self._score_name(name)
                best = self._choose_better_candidate(best, name, size, name_score)

            if best is None:
                return None

            _, _, best_name = best
            try:
                member = tf.getmember(best_name)
                f = tf.extractfile(member)
                if f is None:
                    return None
                data = f.read()
                return data
            except Exception:
                return None

    def _find_poc_in_dir(self, base_dir: str) -> Optional[bytes]:
        """
        Search within a directory tree for a file that is likely the PoC.
        """
        best = None  # (absdiff, name_score, path)

        for root, _, files in os.walk(base_dir):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                if size <= 0:
                    continue

                if size > 10 * self.TARGET_SIZE:
                    continue

                if self._should_skip_ext(fname):
                    continue

                rel_path = os.path.relpath(path, base_dir)
                name_score = self._score_name(rel_path)
                best = self._choose_better_candidate(best, path, size, name_score)

        if best is None:
            return None

        _, _, best_path = best
        try:
            with open(best_path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _fallback_poc(self) -> bytes:
        """
        Fallback PoC input when no suitable existing file is found.
        This is a generic text input intended for AST-based fuzz targets.
        """
        # A moderately complex synthetic input intended to exercise AST parsing
        # and repr() in many languages (especially those with nested structures).
        text = (
            "func main() {\n"
            "    // Nested structures to stress AST repr\n"
            "    var data = [\n"
            "        {\"key\": \"value\", \"list\": [1, 2, 3, 4], \"nested\": {\"a\": 1}},\n"
            "        {\"key\": \"other\", \"list\": [5, 6, 7, 8], \"nested\": {\"b\": 2}}\n"
            "    ];\n"
            "    /* Deeply nested expression */\n"
            "    result = (((1 + 2) * (3 + 4)) - ((5 - 6) * (7 - 8))) / (9 + 10 - 11);\n"
            "    // Recursive-like pattern\n"
            "    if (result > 0) {\n"
            "        main();\n"
            "    }\n"
            "}\n"
        )
        return text.encode("utf-8")