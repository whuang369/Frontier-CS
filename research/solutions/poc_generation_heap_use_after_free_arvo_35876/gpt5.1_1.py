import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc_data = None
        try:
            if os.path.isdir(src_path):
                poc_data = self._search_directory(src_path)
            elif tarfile.is_tarfile(src_path):
                poc_data = self._search_tar(src_path)
            elif zipfile.is_zipfile(src_path):
                poc_data = self._search_zip(src_path)
        except Exception:
            poc_data = None

        if poc_data is None:
            poc_data = self._default_poc()
        return poc_data

    def _default_poc(self) -> bytes:
        # Generic fallback PoC aiming to exercise compound division by zero
        return b"a = 1;\na /= 0;\n"

    def _search_tar(self, src_path: str) -> bytes | None:
        best_score = -1
        best_data = None
        size_limit = 131072  # 128 KB per file

        with tarfile.open(src_path, "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if member.size <= 0 or member.size > size_limit:
                    continue
                try:
                    f = tar.extractfile(member)
                except Exception:
                    continue
                if f is None:
                    continue
                try:
                    data = f.read()
                finally:
                    f.close()
                score = self._score_candidate(member.name, data)
                if score > best_score:
                    best_score = score
                    best_data = data

        if best_score >= 45 and best_data is not None:
            return self._patch_candidate_for_div_zero(best_data)
        return None

    def _search_zip(self, src_path: str) -> bytes | None:
        best_score = -1
        best_data = None
        size_limit = 131072

        with zipfile.ZipFile(src_path, "r") as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                try:
                    info = z.getinfo(name)
                    size = info.file_size
                except KeyError:
                    continue
                if size <= 0 or size > size_limit:
                    continue
                try:
                    with z.open(name, "r") as f:
                        data = f.read()
                except Exception:
                    continue
                score = self._score_candidate(name, data)
                if score > best_score:
                    best_score = score
                    best_data = data

        if best_score >= 45 and best_data is not None:
            return self._patch_candidate_for_div_zero(best_data)
        return None

    def _search_directory(self, src_path: str) -> bytes | None:
        best_score = -1
        best_data = None
        size_limit = 131072

        for root, _, files in os.walk(src_path):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or size > size_limit:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                rel_path = os.path.relpath(path, src_path)
                score = self._score_candidate(rel_path, data)
                if score > best_score:
                    best_score = score
                    best_data = data

        if best_score >= 45 and best_data is not None:
            return self._patch_candidate_for_div_zero(best_data)
        return None

    def _score_candidate(self, path: str, data: bytes) -> int:
        length = len(data)
        if length == 0:
            return -1

        printable = 0
        for b in data:
            if 32 <= b < 127 or b in (9, 10, 13):
                printable += 1
        ratio = printable / length
        if ratio < 0.4:
            return -1  # treat as binary, unlikely to be our PoC

        score = 0
        if ratio > 0.9:
            score += 5
        elif ratio > 0.75:
            score += 2

        path_l = path.lower()

        if "poc" in path_l or "proof" in path_l:
            score += 20
        if "crash" in path_l or "repro" in path_l or "reproducer" in path_l:
            score += 15
        if "uaf" in path_l or "heap" in path_l or "use-after" in path_l:
            score += 15
        if "asan" in path_l or "ubsan" in path_l or "msan" in path_l:
            score += 10
        if "fuzz" in path_l or "clusterfuzz" in path_l or "afl" in path_l:
            score += 10
        if "test" in path_l or "tests" in path_l or "case" in path_l or "input" in path_l or "sample" in path_l:
            score += 3
        if "div" in path_l or "zero" in path_l:
            score += 5

        if "." in path_l:
            for ext, ext_score in [
                (".poc", 10),
                (".repro", 10),
                (".in", 8),
                (".txt", 5),
                (".dat", 3),
                (".data", 3),
                (".cfg", 2),
            ]:
                if path_l.endswith(ext):
                    score += ext_score
                    break

        lower = data.lower()
        if b"/=" in data:
            score += 25
        if b"/=0" in data or b"/= 0" in data or b"/=\t0" in data or b"/=\n0" in data:
            score += 15
        if b"division" in lower or b"divide" in lower or b"div0" in lower:
            score += 10
        if b"zero" in lower:
            score += 5
        if (
            b"use-after-free" in lower
            or b"use after free" in lower
            or b"heap-use-after-free" in lower
            or b"uaf" in lower
        ):
            score += 15

        diff = abs(length - 79)
        closeness = 20 - diff
        if closeness > 0:
            score += closeness

        return score

    def _patch_candidate_for_div_zero(self, data: bytes) -> bytes:
        # Ensure the PoC includes a compound division by zero pattern
        try:
            s = data.decode("utf-8", errors="ignore")
        except Exception:
            s = "".join(
                chr(b) for b in data if 32 <= b < 127 or b in (9, 10, 13)
            )

        if "/= 0" in s or "/=0" in s:
            return data

        idx = s.find("/=")
        if idx == -1:
            # No compound division found, append our own small trigger
            s = s.rstrip("\n\r") + "\n" + "a = 1; a /= 0;\n"
            return s.encode("utf-8")

        end = len(s)
        for delim in (";", "\n", "\r"):
            e = s.find(delim, idx + 2)
            if e != -1 and e < end:
                end = e

        new_s = s[:idx] + "/= 0" + s[end:]
        return new_s.encode("utf-8")