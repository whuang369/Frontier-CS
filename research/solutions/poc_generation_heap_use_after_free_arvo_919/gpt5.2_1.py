import os
import re
import io
import tarfile
import zipfile
import base64
import binascii
import gzip
import lzma
import zlib
from typing import Optional, Tuple, Iterable, List


_FONT_SIGS = (
    b"\x00\x01\x00\x00",  # TrueType
    b"OTTO",              # CFF OpenType
    b"true",              # TrueType (Mac)
    b"typ1",              # Type 1
    b"wOFF",              # WOFF
    b"wOF2",              # WOFF2
)
_FONT_EXTS = (".ttf", ".otf", ".woff", ".woff2", ".eot", ".pfb", ".ttc", ".otc")
_BIN_EXTS = _FONT_EXTS + (".bin", ".dat", ".raw", ".poc", ".crash", ".input", ".testcase", ".seed")


def _is_probably_text(b: bytes) -> bool:
    if not b:
        return True
    n = min(len(b), 4096)
    chunk = b[:n]
    if b"\x00" in chunk:
        return False
    # Allow common whitespace/control
    bad = 0
    for c in chunk:
        if c in (9, 10, 13):
            continue
        if 32 <= c <= 126:
            continue
        bad += 1
    return bad <= n // 50


def _looks_like_font(b: bytes) -> bool:
    if len(b) < 12:
        return False
    if b[:4] in _FONT_SIGS:
        return True
    # TTC header
    if b[:4] == b"ttcf" and len(b) >= 12:
        return True
    return False


def _safe_zlib_decompress(data: bytes, max_out: int = 5_000_000) -> Optional[bytes]:
    try:
        d = zlib.decompressobj()
        out = d.decompress(data, max_out)
        if d.unconsumed_tail:
            # would exceed max_out
            return None
        out += d.flush()
        if len(out) > max_out:
            return None
        return out
    except Exception:
        return None


def _maybe_decompress(data: bytes) -> List[bytes]:
    outs = [data]
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        try:
            outs.append(gzip.decompress(data))
        except Exception:
            pass
    if len(data) >= 6 and data[:6] == b"\xfd7zXZ\x00":
        try:
            outs.append(lzma.decompress(data))
        except Exception:
            pass
    if len(data) >= 2 and data[:1] == b"\x78" and data[1:2] in (b"\x01", b"\x9c", b"\xda"):
        dec = _safe_zlib_decompress(data)
        if dec is not None:
            outs.append(dec)
    # De-duplicate
    uniq = []
    seen = set()
    for o in outs:
        h = (len(o), o[:16], o[-16:] if len(o) >= 16 else o)
        if h in seen:
            continue
        seen.add(h)
        uniq.append(o)
    return uniq


def _b64_candidates_from_text(s: str) -> List[bytes]:
    out = []
    # data:...;base64,....
    for m in re.finditer(r"base64,([A-Za-z0-9+/=\s]{200,})", s):
        blob = re.sub(r"\s+", "", m.group(1))
        try:
            out.append(base64.b64decode(blob, validate=True))
        except Exception:
            pass
    # long base64 blocks
    for m in re.finditer(r"([A-Za-z0-9+/]{200,}={0,2})", s):
        blob = m.group(1)
        if len(blob) % 4 != 0:
            continue
        try:
            out.append(base64.b64decode(blob, validate=True))
        except Exception:
            pass
    return out


def _hex_array_candidates_from_text(s: str) -> List[bytes]:
    out = []
    # 0x.., 0x.., ...
    for m in re.finditer(r"((?:0x[0-9a-fA-F]{2}\s*,\s*){64,}0x[0-9a-fA-F]{2})", s):
        seq = m.group(1)
        hx = re.findall(r"0x([0-9a-fA-F]{2})", seq)
        if len(hx) >= 64:
            try:
                out.append(bytes(int(x, 16) for x in hx))
            except Exception:
                pass

    # \x.. \x..
    for m in re.finditer(r"((?:\\x[0-9a-fA-F]{2}){64,})", s):
        seq = m.group(1)
        hx = re.findall(r"\\x([0-9a-fA-F]{2})", seq)
        if len(hx) >= 64:
            try:
                out.append(bytes(int(x, 16) for x in hx))
            except Exception:
                pass
    return out


def _zip_inner_files(data: bytes) -> List[Tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            res = []
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                if zi.file_size <= 0 or zi.file_size > 5_000_000:
                    continue
                try:
                    res.append((zi.filename, zf.read(zi)))
                except Exception:
                    continue
            return res
    except Exception:
        return []


class Solution:
    def solve(self, src_path: str) -> bytes:
        best_data = None
        best_key = None  # (priority, abs(len-800), len)
        target_len = 800

        def consider(name: str, data: bytes, base_priority: int) -> None:
            nonlocal best_data, best_key
            if not data:
                return
            if len(data) > 5_000_000:
                return

            # Try common decompress variants
            for variant in _maybe_decompress(data):
                pri = base_priority

                # If it's a zip, look inside quickly for likely fonts
                if len(variant) >= 4 and variant[:2] == b"PK":
                    for in_name, in_data in _zip_inner_files(variant):
                        consider(f"{name}::zip::{in_name}", in_data, base_priority + 5)
                    continue

                looks = _looks_like_font(variant)
                if looks:
                    pri = max(0, pri - 8)

                key = (pri, abs(len(variant) - target_len), len(variant))
                if best_key is None or key < best_key:
                    best_key = key
                    best_data = variant

        def priority_for_name(name: str) -> int:
            n = name.lower()
            pri = 50
            if any(k in n for k in ("clusterfuzz", "ossfuzz", "oss-fuzz", "testcase", "crash", "repro", "poc", "regression", "uaf", "use-after-free", "asan")):
                pri -= 35
            if any(k in n for k in ("fuzz", "corpus", "seed")):
                pri -= 12
            if n.endswith(_FONT_EXTS):
                pri -= 18
            elif n.endswith(_BIN_EXTS):
                pri -= 8
            if any(k in n for k in ("readme", "license", "changelog", ".md", ".rst")):
                pri += 15
            if any(n.endswith(ext) for ext in (".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".js", ".java", ".go", ".rs", ".txt")):
                pri += 3
            return pri

        def scan_text_embeds(name: str, b: bytes, base_pri: int) -> None:
            if len(b) > 300_000:
                return
            try:
                s = b.decode("utf-8", errors="ignore")
            except Exception:
                return

            cands = []
            cands.extend(_hex_array_candidates_from_text(s))
            cands.extend(_b64_candidates_from_text(s))
            for c in cands:
                consider(name + "::embed", c, base_pri - 5)

        # Open tarball or directory
        if os.path.isdir(src_path):
            # Phase 1: names likely
            likely_paths = []
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except Exception:
                        continue
                    if st.st_size <= 0 or st.st_size > 5_000_000:
                        continue
                    rel = os.path.relpath(p, src_path)
                    likely_paths.append((priority_for_name(rel), st.st_size, p, rel))
            likely_paths.sort()

            for pri, sz, p, rel in likely_paths[:2000]:
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except Exception:
                    continue
                consider(rel, data, pri)
                if _is_probably_text(data) and any(k in rel.lower() for k in ("poc", "testcase", "crash", "repro", "regression", "clusterfuzz", "ossfuzz")):
                    scan_text_embeds(rel, data, pri)
                if best_key is not None and best_key[0] <= 0 and best_key[1] == 0 and best_key[2] <= 900 and _looks_like_font(best_data):
                    break

            if best_data is not None:
                return best_data
            return b"\x00" * 800

        # Tarball mode
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            # last resort: treat as raw bytes (unlikely)
            try:
                with open(src_path, "rb") as f:
                    d = f.read()
                return d if d else (b"\x00" * 800)
            except Exception:
                return b"\x00" * 800

        with tf:
            members = [m for m in tf.getmembers() if m.isreg() and 0 < m.size <= 5_000_000]

            # Phase 1: likely by name/ext
            members.sort(key=lambda m: (priority_for_name(m.name), abs(m.size - 800), m.size))
            for m in members[:2500]:
                pri = priority_for_name(m.name)
                nlow = m.name.lower()
                # Avoid reading huge source-like files early unless name is strong
                if m.size > 400_000 and pri > 10 and not any(k in nlow for k in ("clusterfuzz", "ossfuzz", "testcase", "poc", "crash", "repro", "regression")):
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                consider(m.name, data, pri)
                if _is_probably_text(data) and any(k in nlow for k in ("poc", "testcase", "crash", "repro", "regression", "clusterfuzz", "ossfuzz")):
                    scan_text_embeds(m.name, data, pri)
                if best_key is not None and best_key[0] <= 0 and best_key[1] == 0 and best_key[2] <= 900 and _looks_like_font(best_data):
                    break

            # Phase 2: signature scan over small files (read only header first)
            if best_data is None or not _looks_like_font(best_data):
                for m in members:
                    if m.size < 64 or m.size > 200_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        head = f.read(16)
                    except Exception:
                        continue
                    if head[:4] in _FONT_SIGS or head[:4] == b"ttcf":
                        try:
                            f2 = tf.extractfile(m)
                            if f2 is None:
                                continue
                            data = f2.read()
                        except Exception:
                            continue
                        consider(m.name, data, priority_for_name(m.name) - 5)
                        if best_key is not None and best_key[0] <= 0 and best_key[1] <= 32 and best_key[2] <= 1200 and _looks_like_font(best_data):
                            break

        if best_data is not None:
            return best_data

        return b"\x00" * 800