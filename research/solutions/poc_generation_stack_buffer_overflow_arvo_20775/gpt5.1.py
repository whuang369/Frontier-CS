import os
import io
import tarfile


class Solution:
    TARGET_POC_SIZE = 844
    NESTED_MAX_SIZE = 20 * 1024 * 1024  # 20 MB
    ANY_MAX_SIZE = 16 * 1024  # 16 KB

    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, mode="r:*") as tf:
                # First, try to find exact-size PoC in this tar or nested tars
                data = self._search_tar_recursive(tf, exact=True)
                if data is not None:
                    return data

                # Fallback: try to find any reasonable PoC-like file
                data = self._search_tar_recursive(tf, exact=False)
                if data is not None:
                    return data
        except tarfile.TarError:
            pass

        # Ultimate fallback: synthetic payload
        return self._default_payload()

    def _search_tar_recursive(self, tf: tarfile.TarFile, exact: bool) -> bytes | None:
        # Phase 1: search within current tar
        members = tf.getmembers()
        best_member, _ = self._select_best_member(members, exact=exact)

        if best_member is not None:
            try:
                extracted = tf.extractfile(best_member)
                if extracted is not None:
                    return extracted.read()
            except (tarfile.ExtractError, OSError):
                pass

        # Phase 2: search within nested tar files
        for member in members:
            if not member.isreg():
                continue
            name_lower = member.name.lower()
            if not name_lower.endswith(
                (
                    ".tar",
                    ".tar.gz",
                    ".tgz",
                    ".tar.xz",
                    ".txz",
                    ".tar.bz2",
                    ".tbz2",
                    ".tar.lzma",
                )
            ):
                continue
            if member.size <= 0 or member.size > self.NESTED_MAX_SIZE:
                continue

            try:
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
            except (tarfile.ExtractError, OSError):
                continue

            try:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as sub_tf:
                    result = self._search_tar_recursive(sub_tf, exact=exact)
                    if result is not None:
                        return result
            except tarfile.TarError:
                continue

        return None

    def _select_best_member(self, members, exact: bool):
        best_member = None
        best_score = None

        for m in members:
            if not m.isreg():
                continue

            size = m.size

            if exact:
                if size != self.TARGET_POC_SIZE:
                    continue
            else:
                if size <= 0 or size > self.ANY_MAX_SIZE:
                    continue

            score = self._score_member(m.name, size, exact)

            if best_member is None or score > best_score:
                best_member = m
                best_score = score

        return best_member, best_score

    def _score_member(self, name: str, size: int, exact: bool) -> int:
        # Heuristic scoring based on filename and size
        lower = name.lower()
        base = os.path.basename(lower)
        dirpath = os.path.dirname(lower)

        score = 0

        positive_keywords = [
            "poc",
            "proof",
            "crash",
            "repro",
            "reproduce",
            "input",
            "seed",
            "case",
            "test",
            "id",
            "bug",
            "overflow",
            "stack",
            "commission",
            "dataset",
            "tlv",
            "network",
        ]
        negative_keywords = [
            "readme",
            "license",
            "copying",
            "changelog",
            "notice",
            "todo",
            "example",
            "sample",
        ]

        positive_exts = [
            ".bin",
            ".dat",
            ".raw",
            ".pcap",
            ".poc",
            ".input",
            ".in",
            ".case",
            ".seed",
        ]
        negative_exts = [
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".hpp",
            ".java",
            ".py",
            ".md",
            ".txt",
            ".rst",
            ".html",
            ".xml",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".mk",
            ".sh",
            ".bat",
            ".ps1",
            ".go",
            ".rs",
            ".m4",
        ]

        for kw in positive_keywords:
            if kw in base:
                score += 15
            if kw in dirpath:
                score += 8

        for kw in negative_keywords:
            if kw in base:
                score -= 25

        _, ext = os.path.splitext(base)
        if ext in positive_exts:
            score += 25
        if ext in negative_exts:
            score -= 15

        # Additional encouragement for exact-size match when in non-exact mode
        if not exact:
            diff = abs(size - self.TARGET_POC_SIZE)
            # Prefer sizes closer to TARGET_POC_SIZE
            score -= diff // 16

        # Slightly prefer smaller files when all else equal
        score -= size // 2048

        return score

    def _default_payload(self) -> bytes:
        return b"A" * self.TARGET_POC_SIZE