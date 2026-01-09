import struct
import zlib


class BitWriter:
    def __init__(self):
        self.bytes = bytearray()
        self.bitbuf = 0
        self.bitcount = 0

    def write_bits(self, value: int, nbits: int) -> None:
        while nbits > 0:
            space = 8 - self.bitcount
            take = space if space < nbits else nbits
            mask = (1 << take) - 1
            self.bitbuf |= ((value & mask) << self.bitcount)
            self.bitcount += take
            value >>= take
            nbits -= take
            if self.bitcount == 8:
                self.bytes.append(self.bitbuf)
                self.bitbuf = 0
                self.bitcount = 0

    def flush_zero(self) -> None:
        if self.bitcount > 0:
            self.bytes.append(self.bitbuf)
            self.bitbuf = 0
            self.bitcount = 0


def build_canonical_codes(lengths):
    if not lengths:
        return [], []
    max_len = max(lengths)
    if max_len == 0:
        return [0] * len(lengths), lengths[:]
    bl_count = [0] * (max_len + 1)
    for l in lengths:
        if l > 0:
            bl_count[l] += 1
    next_code = [0] * (max_len + 1)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = [0] * len(lengths)
    for sym, l in enumerate(lengths):
        if l > 0:
            codes[sym] = next_code[l]
            next_code[l] += 1
    return codes, lengths[:]


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Parameters for dynamic Huffman block
        HLIT = 31   # 257 + 31 = 288 literal/length codes
        HDIST = 31  # 1 + 31 = 32 distance codes
        HCLEN = 15  # 4 + 15 = 19 code-length codes

        # Code-length alphabet lengths (19 symbols, 0..18)
        # We'll have only symbols 0 and 1 with length 1; others 0.
        cl_lens = [0] * 19
        cl_lens[0] = 1
        cl_lens[1] = 1

        cl_codes, _ = build_canonical_codes(cl_lens)

        # Literal/length code lengths (288 symbols: 0..287)
        lit_count = 257 + HLIT
        lit_lens = [0] * lit_count
        lit_lens[65] = 1    # literal 'A'
        lit_lens[256] = 1   # end-of-block

        # Distance code lengths (32 symbols: 0..31)
        dist_count = 1 + HDIST
        dist_lens = [0] * dist_count
        dist_lens[0] = 1    # only distance symbol 0 is used

        lit_codes, _ = build_canonical_codes(lit_lens)
        dist_codes, _ = build_canonical_codes(dist_lens)  # not actually used, but built for correctness

        # Start writing Deflate bitstream (raw, to be wrapped in gzip)
        w = BitWriter()

        # BFINAL=1 (last block), BTYPE=2 (dynamic Huffman)
        w.write_bits(1, 1)   # BFINAL
        w.write_bits(2, 2)   # BTYPE = 2 (10b)

        # HLIT, HDIST, HCLEN
        w.write_bits(HLIT, 5)
        w.write_bits(HDIST, 5)
        w.write_bits(HCLEN, 4)

        # Order in which code-length code lengths are stored
        order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                 11, 4, 12, 3, 13, 2, 14, 1, 15]

        # Write code-length code lengths (first HCLEN+4 entries in 'order')
        for i in range(HCLEN + 4):
            sym = order[i]
            length = cl_lens[sym]
            w.write_bits(length, 3)  # each is 3 bits

        # Now encode literal/length and distance code lengths using the CL Huffman tree
        all_lens = lit_lens + dist_lens  # total HLIT+257 + HDIST+1 entries

        for l in all_lens:
            # Direct mapping: symbol = length (0..15)
            sym = l
            code = cl_codes[sym]
            bits = cl_lens[sym]
            if bits == 0:
                # Should not happen with our chosen lengths (only 0 or 1, and both defined),
                # but guard just in case.
                continue
            w.write_bits(code, bits)

        # Compressed data: literal 'A', then end-of-block
        # Literal 'A' (65)
        w.write_bits(lit_codes[65], lit_lens[65])
        # End-of-block (256)
        w.write_bits(lit_codes[256], lit_lens[256])

        # Flush remaining bits to full bytes
        w.flush_zero()
        deflate_data = bytes(w.bytes)

        # Build gzip container around the deflate stream
        uncompressed = b"A"
        crc = zlib.crc32(uncompressed) & 0xFFFFFFFF
        isize = len(uncompressed) & 0xFFFFFFFF

        header = bytearray()
        header += b"\x1f\x8b"          # ID1, ID2
        header.append(8)              # CM = deflate
        header.append(0)              # FLG
        header += b"\x00\x00\x00\x00"  # MTIME
        header.append(0)              # XFL
        header.append(255)            # OS = unknown

        footer = struct.pack("<II", crc, isize)

        return bytes(header) + deflate_data + footer