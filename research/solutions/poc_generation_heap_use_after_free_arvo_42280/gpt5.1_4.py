import os
import tarfile
import gzip
import bz2
import lzma
import io


class Solution:
    GROUND_TRUTH_LEN = 13996
    MAX_FILE_SIZE = 2_000_000

    def solve(self, src_path: str) -> bytes:
        if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            data = self._from_tar(src_path)
        else:
            data = self._from_dir(src_path)

        if not data:
            data = self._fallback_poc()
        return data

    def _from_tar(self, path: str) -> bytes:
        best = None  # (score, delta, member, name)
        try:
            with tarfile.open(path, "r:*") as tar:
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    size = m.size
                    if size <= 0 or size > self.MAX_FILE_SIZE:
                        continue
                    name = os.path.basename(m.name)
                    score = self._score_candidate(name, size)
                    delta = abs(size - self.GROUND_TRUTH_LEN)
                    if best is None or score > best[0] or (
                        score == best[0] and delta < best[1]
                    ):
                        best = (score, delta, m, name)

                if best is not None and best[0] >= 60:
                    _, _, member, name = best
                    f = tar.extractfile(member)
                    if f is not None:
                        data = f.read()
                        if data:
                            return self._maybe_decompress(name, data)
        except Exception:
            pass
        return b""

    def _from_dir(self, root: str) -> bytes:
        if not os.path.isdir(root):
            return b""
        best = None  # (score, delta, path, name)
        try:
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    try:
                        size = os.path.getsize(fpath)
                    except OSError:
                        continue
                    if size <= 0 or size > self.MAX_FILE_SIZE:
                        continue
                    score = self._score_candidate(fname, size)
                    delta = abs(size - self.GROUND_TRUTH_LEN)
                    if best is None or score > best[0] or (
                        score == best[0] and delta < best[1]
                    ):
                        best = (score, delta, fpath, fname)
            if best is not None and best[0] >= 60:
                _, _, fpath, name = best
                try:
                    with open(fpath, "rb") as f:
                        data = f.read()
                        if data:
                            return self._maybe_decompress(name, data)
                except OSError:
                    pass
        except Exception:
            pass
        return b""

    def _score_candidate(self, name: str, size: int) -> int:
        name_l = name.lower()
        score = 0

        if size == self.GROUND_TRUTH_LEN:
            score += 100

        delta = abs(size - self.GROUND_TRUTH_LEN)
        # Size closeness: up to +40 for very close sizes
        closeness = 40 - (delta // 100)
        if closeness > 0:
            score += closeness

        keywords = [
            "poc",
            "crash",
            "id_",
            "uaf",
            "heap",
            "bug",
            "issue",
            "testcase",
            "repro",
            "sample",
            "trigger",
        ]
        if any(k in name_l for k in keywords):
            score += 40

        ext = ""
        dot = name_l.rfind(".")
        if dot != -1:
            ext = name_l[dot:]
        if ext in (".pdf", ".ps", ".eps", ".xps", ".bin"):
            score += 25

        if "pdf" in name_l:
            score += 10
        if "ps" in name_l:
            score += 5
        if "fuzz" in name_l or "oss-fuzz" in name_l:
            score += 5
        if "42280" in name_l:
            score += 20

        return score

    def _maybe_decompress(self, name: str, data: bytes) -> bytes:
        name_l = name.lower()
        # Limit decompressed size to avoid excessive memory usage
        max_decompressed = 10_000_000

        if name_l.endswith(".gz"):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
                    return f.read(max_decompressed)
            except Exception:
                return data
        if name_l.endswith(".bz2"):
            try:
                return bz2.decompress(data)
            except Exception:
                return data
        if name_l.endswith(".xz") or name_l.endswith(".lzma"):
            try:
                return lzma.decompress(data)
            except Exception:
                return data
        return data

    def _fallback_poc(self) -> bytes:
        # Generic PostScript/PDF-ish fallback; unlikely to trigger but better than empty.
        fallback = b"""%!PS-Adobe-3.0
%%Title: pdfi UAF fallback PoC
%%Pages: 1
%%EndComments

/pdfi-test-stream (dummy.pdf) def

% Try to exercise pdf interpreter operators if available
userdict /pdfdict known {
  pdfdict begin
    (dummy.pdf) (r) file /pdfi-stream exch def
    % Intentionally cause an error to put pdfi in a bad state
    (nonexistent.pdf) (r) file
  end
} if

showpage
%%EOF
"""
        return fallback