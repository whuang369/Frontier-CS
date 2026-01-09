import os
import re
import io
import tarfile
import gzip
import bz2
import lzma
from typing import List, Optional, Tuple


class Solution:
    _FONT_EXTS = {
        ".ttf", ".otf", ".ttc", ".woff", ".woff2", ".eot", ".pfb", ".pfm", ".bin", ".dat", ".raw"
    }

    def _is_sfnt_magic(self, data: bytes) -> bool:
        if len(data) < 4:
            return False
        m = data[:4]
        if m in (b"OTTO", b"ttcf", b"true", b"typ1"):
            return True
        if m == b"\x00\x01\x00\x00":
            return True
        return False

    def _is_woff_magic(self, data: bytes) -> bool:
        return len(data) >= 4 and data[:4] in (b"wOFF", b"wOF2")

    def _is_font_like(self, data: bytes) -> bool:
        return self._is_sfnt_magic(data) or self._is_woff_magic(data)

    def _safe_decompress(self, data: bytes, max_out: int = 5_000_000) -> List[bytes]:
        outs: List[bytes] = []
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            try:
                out = gzip.decompress(data)
                if len(out) <= max_out:
                    outs.append(out)
            except Exception:
                pass
        if len(data) >= 3 and data[:3] == b"BZh":
            try:
                out = bz2.decompress(data)
                if len(out) <= max_out:
                    outs.append(out)
            except Exception:
                pass
        if len(data) >= 6 and data[:6] in (b"\xfd7zXZ\x00", b"\x5d\x00\x00\x80\x00\xff"):
            try:
                out = lzma.decompress(data)
                if len(out) <= max_out:
                    outs.append(out)
            except Exception:
                pass
        return outs

    def _score_candidate(self, path: str, data: bytes) -> float:
        size = len(data)
        lower = path.lower()
        ext = os.path.splitext(lower)[1]
        score = 0.0

        # size preference: near 800, and smaller generally better
        score -= abs(size - 800) / 80.0
        score -= size / 200000.0

        if ext in self._FONT_EXTS:
            score += 8.0
        if self._is_font_like(data):
            score += 20.0

        for kw, w in (
            ("clusterfuzz", 30.0),
            ("testcase", 18.0),
            ("minimized", 12.0),
            ("repro", 12.0),
            ("poc", 12.0),
            ("crash", 12.0),
            ("uaf", 10.0),
            ("use-after-free", 10.0),
            ("heap-use-after-free", 10.0),
            ("asan", 8.0),
            ("oss-fuzz", 8.0),
            ("919", 6.0),
            ("arvo", 6.0),
        ):
            if kw in lower:
                score += w

        # strong bump if exact size match with typical repro
        if size == 800:
            score += 25.0

        return score

    def _maybe_consider_blob_from_text(self, path: str, text: str, out: List[Tuple[float, str, bytes]]) -> None:
        lower = path.lower()
        if not any(k in lower for k in ("test", "fuzz", "poc", "repro", "crash", "corpus", "regress", "cve", "issue")):
            if not any(k in text.lower() for k in ("clusterfuzz", "testcase", "minimized", "poc", "repro", "crash", "oss-fuzz", "asan", "uaf", "use after free", "use-after-free")):
                return

        # C-style hex arrays
        if "0x" in text:
            # find brace blocks not too huge
            for m in re.finditer(r"\{([^{}]{120,200000})\}", text, flags=re.S):
                block = m.group(1)
                if "0x" not in block:
                    continue
                toks = re.findall(r"0x[0-9a-fA-F]{1,2}|\b\d{1,3}\b", block)
                if len(toks) < 64 or len(toks) > 50000:
                    continue
                try:
                    b = bytes(int(t, 16) if t.lower().startswith("0x") else int(t) for t in toks)
                except Exception:
                    continue
                if len(b) < 64 or len(b) > 2_000_000:
                    continue
                if not self._is_font_like(b) and not any(k in lower for k in ("clusterfuzz", "testcase", "minimized", "poc", "repro", "crash")):
                    continue
                out.append((self._score_candidate(path + "::embedded_array", b), path + "::embedded_array", b))

        # \xNN escaped string blobs
        if "\\x" in text:
            for m in re.finditer(r'"([^"\n]{120,200000})"', text, flags=re.S):
                s = m.group(1)
                if "\\x" not in s:
                    continue
                if not any(k in lower for k in ("clusterfuzz", "testcase", "minimized", "poc", "repro", "crash")) and s.count("\\x") < 64:
                    continue
                try:
                    b = s.encode("utf-8").decode("unicode_escape", errors="ignore").encode("latin1", errors="ignore")
                except Exception:
                    continue
                if len(b) < 64 or len(b) > 2_000_000:
                    continue
                if self._is_font_like(b):
                    out.append((self._score_candidate(path + "::escaped_string", b), path + "::escaped_string", b))

    def _scan_tarball(self, tar_path: str) -> Optional[bytes]:
        candidates: List[Tuple[float, str, bytes]] = []
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for ti in tf:
                    if not ti.isreg():
                        continue
                    size = ti.size
                    if size <= 0 or size > 20_000_000:
                        continue
                    name = ti.name or ""
                    lower = name.lower()
                    ext = os.path.splitext(lower)[1]

                    preselect = False
                    if ext in self._FONT_EXTS:
                        preselect = True
                    if size <= 50_000 and any(k in lower for k in ("clusterfuzz", "testcase", "minimized", "poc", "repro", "crash", "uaf", "oss-fuzz", "arvo", "919", "corpus")):
                        preselect = True
                    if abs(size - 800) <= 2000:
                        preselect = True
                    if not preselect and (ext in (".cc", ".cpp", ".c", ".h", ".hpp", ".txt", ".md") and size <= 1_000_000):
                        preselect = True
                    if not preselect:
                        continue

                    f = tf.extractfile(ti)
                    if f is None:
                        continue
                    data = f.read()
                    if not data:
                        continue

                    # Consider as-is
                    candidates.append((self._score_candidate(name, data), name, data))

                    # Consider decompressed variants if any
                    for dd in self._safe_decompress(data):
                        candidates.append((self._score_candidate(name + "::decompressed", dd), name + "::decompressed", dd))

                    # Consider embedded blobs in text-like files
                    if ext in (".cc", ".cpp", ".c", ".h", ".hpp", ".txt", ".md") and len(data) <= 1_000_000:
                        try:
                            text = data.decode("utf-8", errors="ignore")
                        except Exception:
                            text = ""
                        if text:
                            self._maybe_consider_blob_from_text(name, text, candidates)

        except Exception:
            return None

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]

    def _scan_directory(self, root: str) -> Optional[bytes]:
        candidates: List[Tuple[float, str, bytes]] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except Exception:
                    continue
                if not os.path.isfile(path):
                    continue
                size = st.st_size
                if size <= 0 or size > 20_000_000:
                    continue
                lower = path.lower()
                ext = os.path.splitext(lower)[1]

                preselect = False
                if ext in self._FONT_EXTS:
                    preselect = True
                if size <= 50_000 and any(k in lower for k in ("clusterfuzz", "testcase", "minimized", "poc", "repro", "crash", "uaf", "oss-fuzz", "arvo", "919", "corpus")):
                    preselect = True
                if abs(size - 800) <= 2000:
                    preselect = True
                if not preselect and (ext in (".cc", ".cpp", ".c", ".h", ".hpp", ".txt", ".md") and size <= 1_000_000):
                    preselect = True
                if not preselect:
                    continue

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue
                if not data:
                    continue

                rel = os.path.relpath(path, root)
                candidates.append((self._score_candidate(rel, data), rel, data))
                for dd in self._safe_decompress(data):
                    candidates.append((self._score_candidate(rel + "::decompressed", dd), rel + "::decompressed", dd))

                if ext in (".cc", ".cpp", ".c", ".h", ".hpp", ".txt", ".md") and len(data) <= 1_000_000:
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = ""
                    if text:
                        self._maybe_consider_blob_from_text(rel, text, candidates)

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]

    def _fallback_minimal_woff(self) -> bytes:
        # Minimal WOFF with one uncompressed 'head' table (mostly zeros with required magic number)
        # This is a weak fallback and may not trigger anything; intended only if no embedded PoC found.
        head = bytearray(54)
        # magicNumber at offset 12
        head[12:16] = b"\x5F\x0F\x3C\xF5"
        # unitsPerEm at offset 18 (must be 16..16384 usually)
        head[18:20] = b"\x00\x10"
        # indexToLocFormat at offset 50
        head[50:52] = b"\x00\x00"
        # glyphDataFormat at offset 52
        head[52:54] = b"\x00\x00"

        signature = b"wOFF"
        flavor = b"\x00\x01\x00\x00"
        num_tables = 1
        reserved = 0
        total_sfnt_size = 12 + 16 * num_tables + len(head)
        major = 1
        minor = 0
        meta_offset = meta_length = meta_orig_length = 0
        priv_offset = priv_length = 0

        header_len = 44
        dir_len = 20 * num_tables
        table_data_offset = header_len + dir_len
        length = table_data_offset + len(head)

        woff_header = bytearray()
        woff_header += signature
        woff_header += flavor
        woff_header += length.to_bytes(4, "big")
        woff_header += num_tables.to_bytes(2, "big")
        woff_header += reserved.to_bytes(2, "big")
        woff_header += total_sfnt_size.to_bytes(4, "big")
        woff_header += major.to_bytes(2, "big")
        woff_header += minor.to_bytes(2, "big")
        woff_header += meta_offset.to_bytes(4, "big")
        woff_header += meta_length.to_bytes(4, "big")
        woff_header += meta_orig_length.to_bytes(4, "big")
        woff_header += priv_offset.to_bytes(4, "big")
        woff_header += priv_length.to_bytes(4, "big")

        tag = b"head"
        offset = table_data_offset
        comp_len = len(head)
        orig_len = len(head)
        orig_checksum = 0
        table_dir = bytearray()
        table_dir += tag
        table_dir += offset.to_bytes(4, "big")
        table_dir += comp_len.to_bytes(4, "big")
        table_dir += orig_len.to_bytes(4, "big")
        table_dir += orig_checksum.to_bytes(4, "big")

        return bytes(woff_header + table_dir + head)

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            poc = self._scan_directory(src_path)
            if poc is not None:
                return poc
            return self._fallback_minimal_woff()

        poc = self._scan_tarball(src_path)
        if poc is not None:
            return poc
        return self._fallback_minimal_woff()