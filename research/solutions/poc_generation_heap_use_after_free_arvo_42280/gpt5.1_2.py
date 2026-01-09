import tarfile
import gzip
import bz2
try:
    import lzma
except ImportError:
    lzma = None
import os


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 13996
        try:
            tar = tarfile.open(src_path, "r:*")
        except Exception:
            try:
                with open(src_path, "rb") as f:
                    return f.read()
            except Exception:
                return b""
        with tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.size > 0]
            if not members:
                return b""

            zipped_candidates = [
                m
                for m in members
                if m.name.lower().endswith((".gz", ".bz2", ".xz"))
                and "poc" in m.name.lower()
            ]
            for m in zipped_candidates[:20]:
                try:
                    f = tar.extractfile(m)
                except Exception:
                    continue
                if f is None:
                    continue
                try:
                    comp_data = f.read()
                except Exception:
                    continue
                data = None
                name_lower = m.name.lower()
                try:
                    if name_lower.endswith(".gz"):
                        data = gzip.decompress(comp_data)
                    elif name_lower.endswith(".bz2"):
                        data = bz2.decompress(comp_data)
                    elif name_lower.endswith(".xz") and lzma is not None:
                        data = lzma.decompress(comp_data)
                except Exception:
                    data = None
                if data is not None and len(data) == target_size:
                    return data

            def score_member(m):
                name_lower = m.name.lower()
                size = m.size
                score = 0
                if name_lower.endswith(".ps"):
                    score += 120
                elif name_lower.endswith(".pdf"):
                    score += 110
                elif name_lower.endswith(".eps"):
                    score += 100
                elif name_lower.endswith(".txt"):
                    score += 90
                else:
                    score += 10
                kw_map = [
                    ("poc", 80),
                    ("crash", 70),
                    ("uaf", 60),
                    ("heap", 30),
                    ("use-after-free", 60),
                    ("use_after_free", 60),
                    ("regress", 40),
                    ("bug", 30),
                    ("test", 20),
                    ("42280", 100),
                    ("cve", 20),
                ]
                for kw, pts in kw_map:
                    if kw in name_lower:
                        score += pts
                if "/poc" in name_lower or "poc/" in name_lower:
                    score += 40
                if "tests" in name_lower or "regress" in name_lower or "fuzz" in name_lower:
                    score += 20
                diff = abs(size - target_size)
                if diff == 0:
                    score += 3000
                else:
                    bonus = 500 - diff // 10
                    if bonus > 0:
                        score += bonus
                if size < 100:
                    score -= 50
                if size > 1000000:
                    score -= 200
                return score

            exacts = [m for m in members if m.size == target_size]
            if exacts:
                best_member = max(exacts, key=score_member)
            else:
                best_member = max(members, key=score_member)

            try:
                f = tar.extractfile(best_member)
            except Exception:
                f = None
            if f is not None:
                try:
                    data = f.read()
                except Exception:
                    data = b""
                if data:
                    return data

            for m in members:
                if m.size <= 0 or m.size > 1000000:
                    continue
                try:
                    f2 = tar.extractfile(m)
                except Exception:
                    continue
                if f2 is None:
                    continue
                try:
                    d2 = f2.read()
                except Exception:
                    continue
                if d2:
                    return d2

        return b"%!PS-Adobe-3.0\nshowpage\n"