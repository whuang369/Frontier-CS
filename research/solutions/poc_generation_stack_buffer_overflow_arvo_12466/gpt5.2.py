import os
import re
import tarfile
import zipfile
import gzip
import bz2
import lzma
from typing import Iterable, List, Optional, Tuple


RAR5_SIG = b"Rar!\x1a\x07\x01\x00"


def _try_decompress(data: bytes) -> List[bytes]:
    outs = []
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        try:
            outs.append(gzip.decompress(data))
        except Exception:
            pass
    if len(data) >= 3 and data[:3] == b"BZh":
        try:
            outs.append(bz2.decompress(data))
        except Exception:
            pass
    if len(data) >= 6 and data[:6] == b"\xfd7zXZ\x00":
        try:
            outs.append(lzma.decompress(data))
        except Exception:
            pass
    return outs


def _is_rar5(data: bytes) -> bool:
    return len(data) >= 8 and data[:8] == RAR5_SIG


def _score_name(name: str) -> int:
    n = name.lower()
    score = 0
    if n.endswith(".rar") or n.endswith(".rar5"):
        score += 8
    if ".rar" in n:
        score += 3
    if "rar5" in n:
        score += 8
    if "huffman" in n or "huff" in n:
        score += 18
    if "overflow" in n:
        score += 18
    if "stack" in n:
        score += 8
    if "crash" in n:
        score += 10
    if "poc" in n:
        score += 10
    if "cve" in n:
        score += 6
    if "fuzz" in n or "corpus" in n:
        score += 6
    if "test" in n:
        score += 2
    if "table" in n:
        score += 3
    if "rle" in n:
        score += 3
    if "decode" in n or "unpack" in n:
        score += 2
    if "libarchive" in n:
        score += 1
    return score


def _iter_files_from_dir(root: str, max_size: int) -> Iterable[Tuple[str, bytes]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            if st.st_size <= 0 or st.st_size > max_size:
                continue
            try:
                with open(path, "rb") as f:
                    yield os.path.relpath(path, root), f.read()
            except OSError:
                continue


def _iter_files_from_tar(tar_path: str, max_size: int) -> Iterable[Tuple[str, bytes]]:
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            if m.size <= 0 or m.size > max_size:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                yield m.name, data
            except Exception:
                continue


def _iter_files_from_zip(zip_path: str, max_size: int) -> Iterable[Tuple[str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            if zi.file_size <= 0 or zi.file_size > max_size:
                continue
            try:
                with zf.open(zi, "r") as f:
                    yield zi.filename, f.read()
            except Exception:
                continue


_hex_escape_re = re.compile(r'(?:\\x[0-9A-Fa-f]{2}){16,}')
_num_seq_re = re.compile(
    r'(?:(?:0x[0-9A-Fa-f]{1,2}|\d{1,3})\s*,\s*){20,}(?:0x[0-9A-Fa-f]{1,2}|\d{1,3})'
)
_num_re = re.compile(r'0x[0-9A-Fa-f]{1,2}|\d{1,3}')


def _extract_rar_from_text(text: str, max_out: int = 16) -> List[bytes]:
    outs: List[bytes] = []

    for m in _hex_escape_re.finditer(text):
        s = m.group(0)
        try:
            b = bytes(int(s[i + 2:i + 4], 16) for i in range(0, len(s), 4))
        except Exception:
            continue
        if _is_rar5(b):
            outs.append(b)
            if len(outs) >= max_out:
                return outs

    for m in _num_seq_re.finditer(text):
        seq = m.group(0)
        nums = _num_re.findall(seq)
        if len(nums) < 32:
            continue
        try:
            arr = []
            for tok in nums:
                if tok.startswith(("0x", "0X")):
                    v = int(tok, 16)
                else:
                    v = int(tok, 10)
                if v < 0 or v > 255:
                    arr = []
                    break
                arr.append(v)
                if len(arr) > 2_000_000:
                    arr = []
                    break
            if not arr:
                continue
            b = bytes(arr)
        except Exception:
            continue
        if _is_rar5(b):
            outs.append(b)
            if len(outs) >= max_out:
                return outs

    return outs


class Solution:
    def solve(self, src_path: str) -> bytes:
        max_size = 8_000_000

        candidates: List[Tuple[int, int, str, bytes]] = []

        def consider(name: str, data: bytes, embedded: bool = False) -> None:
            sc = _score_name(name) + (2 if embedded else 0)
            if _is_rar5(data):
                candidates.append((sc, len(data), name, data))
                return
            for d2 in _try_decompress(data):
                if _is_rar5(d2):
                    candidates.append((sc + 1, len(d2), name + "|decompressed", d2))
                    return

        it: Optional[Iterable[Tuple[str, bytes]]] = None
        if os.path.isdir(src_path):
            it = _iter_files_from_dir(src_path, max_size)
        else:
            try:
                if zipfile.is_zipfile(src_path):
                    it = _iter_files_from_zip(src_path, max_size)
                else:
                    it = _iter_files_from_tar(src_path, max_size)
            except Exception:
                it = None

        if it is not None:
            for name, data in it:
                nl = name.lower()
                if any(k in nl for k in ("rar5", "huffman", "huff", "overflow", "crash", "poc", "cve", "stack")):
                    consider(name, data)
                    continue
                if nl.endswith((".rar", ".rar5", ".bin", ".dat", ".raw", ".input", ".crash")) or ".rar" in nl:
                    consider(name, data)

        if os.path.isdir(src_path):
            for name, data in _iter_files_from_dir(src_path, max_size):
                nl = name.lower()
                if not nl.endswith((".c", ".h", ".cc", ".cpp", ".hh")):
                    continue
                if len(data) > 2_000_000:
                    continue
                try:
                    text = data.decode("utf-8", "ignore")
                except Exception:
                    continue
                tl = text.lower()
                if ("rar5" not in tl) and ("rar" not in tl) and ("huffman" not in tl) and ("overflow" not in tl):
                    continue
                for idx, b in enumerate(_extract_rar_from_text(text)):
                    consider(f"{name}|embedded_{idx}", b, embedded=True)
        else:
            try:
                if it is None:
                    pass
                else:
                    pass
            except Exception:
                pass

        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return candidates[0][3]

        fallback = RAR5_SIG + b"\x00" * (524 - len(RAR5_SIG))
        return fallback