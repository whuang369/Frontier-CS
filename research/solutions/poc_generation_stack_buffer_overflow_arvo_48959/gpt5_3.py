import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Bit writer for DEFLATE (LSB-first within each byte)
        class BitWriter:
            def __init__(self):
                self.buf = bytearray()
                self.cur = 0
                self.bitpos = 0

            def write_bits(self, value, nbits):
                v = value
                for _ in range(nbits):
                    bit = v & 1
                    v >>= 1
                    self.cur |= (bit << self.bitpos)
                    self.bitpos += 1
                    if self.bitpos == 8:
                        self.buf.append(self.cur)
                        self.cur = 0
                        self.bitpos = 0

            def get_bytes(self):
                if self.bitpos != 0:
                    self.buf.append(self.cur)
                    self.cur = 0
                    self.bitpos = 0
                return bytes(self.buf)

        # Build canonical Huffman codes given code lengths
        def build_huffman_codes(lengths):
            if not lengths:
                return [], []
            max_bits = max(lengths) if lengths else 0
            bl_count = [0] * (max_bits + 1 if max_bits >= 0 else 1)
            for l in lengths:
                if l > 0:
                    if l >= len(bl_count):
                        bl_count.extend([0] * (l - len(bl_count) + 1))
                    bl_count[l] += 1
            next_code = [0] * (len(bl_count))
            code = 0
            # ensure index 0 exists
            if len(bl_count) == 0:
                bl_count = [0]
                next_code = [0]
            for bits in range(1, len(bl_count)):
                prev = bl_count[bits - 1] if (bits - 1) >= 0 and (bits - 1) < len(bl_count) else 0
                code = (code + prev) << 1
                if bits < len(next_code):
                    next_code[bits] = code
                else:
                    next_code.append(code)
            codes = [0] * len(lengths)
            for n, l in enumerate(lengths):
                if l != 0:
                    codes[n] = next_code[l]
                    next_code[l] += 1
            return codes, lengths

        # Construct DEFLATE dynamic block that uses HCLEN=15 (19 code length codes)
        b = BitWriter()
        # BFINAL=1, BTYPE=2 (10 binary, written LSB-first)
        b.write_bits(1, 1)  # BFINAL
        b.write_bits(2, 2)  # BTYPE=10

        # HLIT=0 (257 lit/len codes), HDIST=0 (1 distance code), HCLEN=15 (19 code length codes)
        b.write_bits(0, 5)   # HLIT
        b.write_bits(0, 5)   # HDIST
        b.write_bits(15, 4)  # HCLEN -> 19 code length codes

        # Code length code order
        order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]

        # We need code length code tree with just two symbols: 1 and 18 (length 1 each), others 0
        cl_lengths_map = {1: 1, 18: 1}
        cl_lengths = [0]*19
        for k, v in cl_lengths_map.items():
            cl_lengths[k] = v

        # Write code length code lengths in the specified order (each 3 bits)
        for idx in order:
            b.write_bits(cl_lengths[idx], 3)

        # Build code length code Huffman codes
        cl_codes, cl_lens = build_huffman_codes(cl_lengths)
        code_1 = cl_codes[1]
        len_1 = cl_lens[1]
        code_18 = cl_codes[18]
        len_18 = cl_lens[18]

        # Now encode the literal/length and distance code lengths using the above code length code tree.
        # For HLIT=0 -> 257 literal/length codes. We'll set first 256 zeros via two 18 repeats, then '1' for symbol 256.
        # 18: repeat zero 11-138 times, with 7 extra bits (repeat_count - 11).
        # First 138 zeros
        b.write_bits(code_18, len_18)
        b.write_bits(138 - 11, 7)
        # Next 118 zeros
        b.write_bits(code_18, len_18)
        b.write_bits(118 - 11, 7)
        # Then set code length 1 for symbol 256
        b.write_bits(code_1, len_1)

        # Distance codes HDIST=0 -> 1 distance code; set its code length to 1
        b.write_bits(code_1, len_1)

        # Build the actual literal/length Huffman tree with only symbol 256 having length 1
        ll_lengths = [0]*257
        ll_lengths[256] = 1
        ll_codes, ll_lens = build_huffman_codes(ll_lengths)
        # Symbol 256 code and length
        eob_code = ll_codes[256]
        eob_len = ll_lens[256]

        # Emit end-of-block symbol (256)
        b.write_bits(eob_code, eob_len)

        deflate_data = b.get_bytes()

        # Build minimal GZIP: header + deflate_data + CRC32 + ISIZE (for empty payload)
        gz = bytearray()
        # GZIP header
        gz.extend(b'\x1f\x8b')      # ID1 ID2
        gz.append(8)                # CM = DEFLATE
        gz.append(0)                # FLG = 0
        gz.extend(b'\x00\x00\x00\x00')  # MTIME
        gz.append(0)                # XFL
        gz.append(0)                # OS
        # Compressed data
        gz.extend(deflate_data)
        # CRC32 and ISIZE of uncompressed data (empty)
        gz.extend(struct.pack('<I', 0))  # CRC32
        gz.extend(struct.pack('<I', 0))  # ISIZE

        return bytes(gz)